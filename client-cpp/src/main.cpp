// edge-client — EdgeServe 的 C++17 流式客户端（D5）
//
// 设计要点（面试可讲）：
//   1. 用 cpp-httplib 的 content_receiver 拿「原始字节回调」，而不是等整个 body 回来
//      → 真正的流式，不受缓冲影响；
//   2. 自己实现 SSE 行缓冲：按 '\n' 切行、剥 '\r'、处理 ':' 注释与 'data:' 前缀、
//      遇 [DONE] 主动收流 → 一块 token 可能被 TCP 拆开，半行必须重新拼接；
//   3. UTF-8 安全：只在行边界切分，多字节汉字永远不会被切坏；
//   4. 连接失败重连一次，[DONE] 后主动 return false 断开（省一个 RTT 的等待）；
//   5. 本地打点 TTFT / TPOT / decode tok/s，与服务端 timings 对账。
//
// 用法见 ../README.md

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <iostream>
#include <string>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double msSince(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------- SSE 解析
// 只做一件事：把字节流按 SSE 规则切成 payload，交给 handler。
// 返回 false 表示收到 [DONE]（或 handler 要求停），调用方应中断连接。
class SseParser {
public:
    using Handler = std::function<void(const std::string &)>;
    explicit SseParser(Handler h) : handler_(std::move(h)) {}

    bool feed(const char *data, size_t len) {
        buf_.append(data, len);
        size_t pos = 0;
        while ((pos = buf_.find('\n')) != std::string::npos) {
            std::string line = buf_.substr(0, pos);
            buf_.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!handleLine(line)) return false;
        }
        // 防御：服务端若一直不回换行，别让缓冲无限涨
        if (buf_.size() > (1u << 20)) buf_.clear();
        return true;
    }

    bool done() const { return done_; }

private:
    bool handleLine(const std::string &line) {
        if (line.empty()) return true;             // 事件分隔（blank line）
        if (line[0] == ':') return true;            // 注释 / keep-alive 心跳
        if (line.rfind("data:", 0) != 0) return true;  // event: / id: / retry:
        std::string payload = line.substr(5);
        if (!payload.empty() && payload.front() == ' ') payload.erase(0, 1);
        if (payload == "[DONE]") {
            done_ = true;
            return false;
        }
        if (!payload.empty()) handler_(payload);
        return true;
    }

    Handler handler_;
    std::string buf_;
    bool done_ = false;
};

// ---------------------------------------------------------------- 统计
struct Stats {
    bool streamed = true;       // 非流式时 TTFT 无意义
    double ttft_ms = -1.0;      // 首字延迟
    double total_ms = -1.0;     // 端到端
    double decode_tps = -1.0;   // 首字之后的解码速度
    long sse_events = 0;        // 收到的 data 事件数
    long pieces = 0;            // 带内容的 delta 数
    long bytes = 0;             // 正文字节数
    long tokens = -1;           // completion_tokens（usage 优先，其次 timings，最后按片数兜底）
    std::string token_src = "n/a";
    double srv_tg = -1.0;       // llama.cpp timings: predicted_per_second
};

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model = "local";
    std::string system;                 // 可选 system prompt
    std::string message;                // 有值 = 单轮模式
    bool stream = true;
    bool raw = false;                   // 打印原始 SSE payload
    bool quiet = false;
    bool include_usage = true;          // 老版本服务端不认 stream_options 时关掉
    int max_tokens = 256;
    double temperature = 0.0;
    int timeout_s = 120;
};

std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

void printStats(const Stats &st) {
    const char *tag = "[stats]";
    if (st.streamed && st.ttft_ms >= 0)
        std::fprintf(stderr, "%s ttft=%.1fms total=%.0fms ", tag, st.ttft_ms, st.total_ms);
    else
        std::fprintf(stderr, "%s ttft=n/a total=%.0fms ", tag, st.total_ms);
    if (st.decode_tps > 0)
        std::fprintf(stderr, "decode=%.1ftok/s ", st.decode_tps);
    std::fprintf(stderr, "tokens=%ld(%s) sse_events=%ld pieces=%ld bytes=%ld",
                 st.tokens, st.token_src.c_str(), st.sse_events, st.pieces, st.bytes);
    if (st.srv_tg > 0) std::fprintf(stderr, " srv_tg=%.1ftok/s", st.srv_tg);
    std::fprintf(stderr, "\n");
}

// 解析单个 SSE payload；把增量文本交给 onPiece
bool parseChunk(const std::string &payload, const Options &opt, Stats &st,
                std::string &full, std::function<void(const std::string &)> onPiece) {
    json obj;
    try {
        obj = json::parse(payload);
    } catch (const std::exception &) {
        if (opt.raw) std::fprintf(stderr, "[warn] 非 JSON payload，已跳过: %.80s\n", payload.c_str());
        return true;
    }
    if (obj.contains("error")) {
        std::fprintf(stderr, "[server-error] %s\n", obj["error"].dump().c_str());
        return false;
    }
    // 服务端 timings / usage（最后一个 chunk 才有）
    if (obj.contains("timings") && obj["timings"].is_object() && !obj["timings"].is_null()) {
        const auto &t = obj["timings"];
        if (t.contains("predicted_per_second") && t["predicted_per_second"].is_number())
            st.srv_tg = t["predicted_per_second"].get<double>();
        if (st.tokens < 0 && t.contains("predicted_n") && t["predicted_n"].is_number()) {
            st.tokens = t["predicted_n"].get<long>();
            st.token_src = "timings";
        }
    }
    if (obj.contains("usage") && obj["usage"].is_object()) {
        const auto &u = obj["usage"];
        if (u.contains("completion_tokens") && u["completion_tokens"].is_number()) {
            st.tokens = u["completion_tokens"].get<long>();
            st.token_src = "usage";
        }
    }
    if (!obj.contains("choices") || !obj["choices"].is_array() || obj["choices"].empty())
        return true;
    const auto &ch = obj["choices"][0];
    // 流式：delta.content；非流式：message.content
    const json *holder = nullptr;
    if (ch.contains("delta") && ch["delta"].is_object()) holder = &ch["delta"];
    else if (ch.contains("message") && ch["message"].is_object()) holder = &ch["message"];
    if (!holder || !holder->contains("content") || !(*holder)["content"].is_string())
        return true;

    std::string piece = (*holder)["content"].get<std::string>();
    if (piece.empty()) return true;             // 首个 chunk 常见 content=""
    st.pieces++;
    st.bytes += static_cast<long>(piece.size());
    full += piece;
    onPiece(piece);
    return true;
}

// 一次请求：返回 0 成功，非 0 失败；err 填错误描述
int doRequest(const Options &opt, const std::vector<json> &history, std::string &assistantOut,
              Stats &st, std::string &err, bool &retryable) {
    retryable = false;
    st.streamed = opt.stream;
    json req = {
        {"model", opt.model},
        {"messages", history},
        {"max_tokens", opt.max_tokens},
        {"temperature", opt.temperature},
        {"stream", opt.stream},
    };
    if (opt.stream && opt.include_usage) req["stream_options"] = {{"include_usage", true}};
    std::string body = req.dump();

    httplib::Client cli(opt.host, opt.port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(opt.timeout_s, 0);
    cli.set_write_timeout(10, 0);
    cli.set_keep_alive(false);

    Clock::time_point t0 = Clock::now();
    Clock::time_point tFirst{};
    bool gotFirst = false;
    st.ttft_ms = -1.0;
    std::string full;

    auto onPiece = [&](const std::string &piece) {
        if (!gotFirst) {
            gotFirst = true;
            tFirst = Clock::now();
            st.ttft_ms = msSince(t0, tFirst);
        }
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
    };

    bool abortedByDone = false;

    if (opt.stream) {
        SseParser parser([&](const std::string &payload) {
            st.sse_events++;
            if (opt.raw) std::fprintf(stderr, "[sse] %s\n", payload.c_str());
            if (!parseChunk(payload, opt, st, full, onPiece)) abortedByDone = true;
        });

        // 关键：content_receiver 拿到的是「到达即回调」的原始字节，而不是攒好的 body。
        // 这是 httplib 里做 POST 流式响应的正确姿势（Post(...) 的返回值会把 body 攒齐）。
        httplib::Request hreq;
        hreq.method = "POST";
        hreq.path = "/v1/chat/completions";
        hreq.set_header("Content-Type", "application/json");
        hreq.set_header("Accept", "text/event-stream");
        hreq.body = body;
        hreq.content_receiver = [&](const char *data, size_t len, uint64_t, uint64_t) {
            return parser.feed(data, len);
        };
        httplib::Response hres;
        httplib::Error herr = httplib::Error::Success;
        bool ok = cli.send(hreq, hres, herr);
        st.total_ms = msSince(t0, Clock::now());

        // 收到 [DONE] 后我们主动返回 false，httplib 会把它记成「已取消」，这不是失败
        bool success = (hres.status == 200 && (ok || parser.done() || st.pieces > 0));
        if (!success) {
            if (hres.status > 0 && hres.status != 200) {
                err = "HTTP " + std::to_string(hres.status) + " body=" + hres.body.substr(0, 200);
            } else {
                err = std::string("连接/读取失败: ") + httplib::to_string(herr);
                retryable = !gotFirst && (herr == httplib::Error::Connection ||
                                          herr == httplib::Error::Read ||
                                          herr == httplib::Error::Write);
            }
        } else if (!parser.done() && !abortedByDone && st.pieces == 0) {
            err = "流结束但没有收到任何内容（检查 stream 参数是否传到服务端）";
        }
    } else {
        auto res = cli.Post("/v1/chat/completions", body, "application/json");
        st.total_ms = msSince(t0, Clock::now());
        if (!res) {
            err = std::string("连接失败: ") + httplib::to_string(res.error());
            retryable = true;
        } else if (res->status != 200) {
            err = "HTTP " + std::to_string(res->status) + " body=" + res->body.substr(0, 200);
        } else {
            json obj;
            try {
                obj = json::parse(res->body);
            } catch (const std::exception &e) {
                err = std::string("响应不是 JSON: ") + e.what();
            }
            if (err.empty()) {
                std::string dummy;
                if (!parseChunk(res->body, opt, st, dummy, onPiece)) {
                    /* parseChunk 在 error 字段时返回 false */
                    if (err.empty()) err = "服务端返回 error 字段";
                }
            }
        }
    }

    st.total_ms = msSince(t0, Clock::now());
    if (gotFirst && st.ttft_ms < 0) st.ttft_ms = -1.0;
    if (st.tokens < 0) {
        st.tokens = st.pieces;                  // 兜底：按 delta 片数（≈token 数，非精确）
        st.token_src = "pieces~";
    } else if (st.token_src == "n/a") {
        st.token_src = "pieces~";
    }
    if (opt.stream && st.ttft_ms > 0 && st.total_ms > st.ttft_ms && st.tokens > 1) {
        double decodeMs = st.total_ms - st.ttft_ms;
        st.decode_tps = (st.tokens - 1) * 1000.0 / decodeMs;
    }
    assistantOut = trim(full);
    return err.empty() ? 0 : 1;
}

struct CliError {
    std::string msg;
};

Options parseArgs(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) throw CliError{std::string(name) + " 缺少参数值"};
            return argv[++i];
        };
        if (a == "-m" || a == "--message") o.message = need("--message");
        else if (a == "-H" || a == "--host") o.host = need("--host");
        else if (a == "-p" || a == "--port") o.port = std::stoi(need("--port"));
        else if (a == "--model") o.model = need("--model");
        else if (a == "-s" || a == "--system") o.system = need("--system");
        else if (a == "-n" || a == "--max-tokens") o.max_tokens = std::stoi(need("--max-tokens"));
        else if (a == "--temperature" || a == "-T") o.temperature = std::stod(need("--temperature"));
        else if (a == "--timeout") o.timeout_s = std::stoi(need("--timeout"));
        else if (a == "--no-stream") o.stream = false;
        else if (a == "--no-usage") o.include_usage = false;
        else if (a == "--raw") o.raw = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "-h" || a == "--help") {
            std::printf(
                "edge-client —— EdgeServe C++17 流式客户端\n"
                "用法:\n"
                "  edge-client -m \"你的问题\"          单轮\n"
                "  edge-client                         交互模式（/reset /exit）\n"
                "选项:\n"
                "  -m, --message <text>      单轮提问\n"
                "  -H, --host <ip>           默认 127.0.0.1\n"
                "  -p, --port <port>         默认 8080\n"
                "      --model <name>        默认 local\n"
                "  -s, --system <text>       system prompt\n"
                "  -n, --max-tokens <int>    默认 256\n"
                "  -T, --temperature <f>     默认 0（可复现）\n"
                "      --timeout <sec>       读超时，默认 120\n"
                "      --no-stream           一次性返回（对照组）\n"
                "      --no-usage            不发送 stream_options（老版本服务端可能不认）\n"
                "      --raw                 打印原始 SSE payload 到 stderr\n"
                "      --quiet               不打印 [stats] 行\n");
            std::exit(0);
        } else {
            throw CliError{"未知参数: " + a + "（--help 看用法）"};
        }
    }
    return o;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    try {
        opt = parseArgs(argc, argv);
    } catch (const CliError &e) {
        std::fprintf(stderr, "参数错误: %s\n", e.msg.c_str());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "参数解析失败: %s\n", e.what());
        return 2;
    }

    std::vector<json> history;
    if (!opt.system.empty()) history.push_back({{"role", "system"}, {"content", opt.system}});

    auto runOne = [&](const std::string &q) -> int {
        history.push_back({{"role", "user"}, {"content", q}});
        std::string answer;
        Stats st;
        std::string err;
        bool retryable = false;
        int rc = doRequest(opt, history, answer, st, err, retryable);

        if (rc != 0 && retryable) {
            std::fprintf(stderr, "[retry] %s —— 2 秒后重连一次\n", err.c_str());
            std::fflush(stderr);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            Stats st2;
            std::string err2;
            bool again = false;
            rc = doRequest(opt, history, answer, st2, err2, again);
            err = err2;
            if (rc == 0) st = st2;
        }

        if (!opt.stream) std::fputs("\n", stdout);
        else std::fputs("\n", stdout);

        if (rc != 0) {
            std::fprintf(stderr, "[error] %s\n", err.c_str());
            history.pop_back();
            return 1;
        }
        if (!opt.quiet) printStats(st);
        if (!answer.empty()) history.push_back({{"role", "assistant"}, {"content", answer}});
        // 只留最近 8 轮，防止 prompt 无界增长
        while (history.size() > 9) {
            auto it = history.begin();
            if (it->value("role", std::string()) == "system") ++it;
            history.erase(it);
        }
        return 0;
    };

    if (!opt.message.empty()) return runOne(opt.message);

    std::fprintf(stderr,
                 "交互模式：直接输入问题回车；/reset 清空上下文，/exit 退出。目标 %s:%d\n",
                 opt.host.c_str(), opt.port);
    std::string line;
    while (true) {
        std::fprintf(stderr, "\n> ");
        std::fflush(stderr);
        if (!std::getline(std::cin, line)) break;
        std::string q = trim(line);
        if (q.empty()) continue;
        if (q == "/exit" || q == "/quit" || q == "/q") break;
        if (q == "/reset") {
            history.clear();
            if (!opt.system.empty())
                history.push_back({{"role", "system"}, {"content", opt.system}});
            std::fprintf(stderr, "[reset] 上下文已清空\n");
            continue;
        }
        if (q == "/help") {
            std::fprintf(stderr, "/reset 清空上下文  /exit 退出\n");
            continue;
        }
        runOne(q);
    }
    return 0;
}
