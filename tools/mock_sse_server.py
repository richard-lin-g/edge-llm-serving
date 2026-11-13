#!/usr/bin/env python3
"""D5 自测用的假 llama-server（只实现 OpenAI 兼容的两条路径）。

它不是交付物，只用来在没有 GPU 的环境里验证客户端：
  - 真 chunked 传输（和 llama-server 一样），不是攒完再发；
  - 每 20ms 发一个 delta，肉眼可分辨「逐字」；
  - 故意把某些中文 token 拆到两次 write 里（半行 + 半行），逼客户端做行缓冲；
  - 结尾带 usage + timings，模拟 llama.cpp 的字段；
  - 支持 stream=false 的整包响应，用来做对照组。

用法:
  python3 tools/mock_sse_server.py --port 8099
"""
import argparse
import json
import socketserver
import time
from http.server import BaseHTTPRequestHandler

ANSWER = ("KV Cache 的作用是把已经算过的 Key/Value 存下来，解码时不必重算；"
          "它是拿显存换延迟的典型做法，上下文越长占用越大。")

def make_deltas(text, n=56):
    """把一句话切成 n 个 delta，尽量按标点/字符切。"""
    step = max(1, len(text) // n)
    out = [text[i:i + step] for i in range(0, len(text), step)]
    return out

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    chunk_delay = 0.02

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _chunk(self, data: bytes):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list",
                             "data": [{"id": "mock-3b", "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        n_out = int(req.get("max_tokens", 128))
        if not req.get("stream"):
            self._json(200, {
                "id": "mock-1", "object": "chat.completion", "model": req.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ANSWER},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 24, "completion_tokens": len(make_deltas(ANSWER)),
                          "total_tokens": 24 + len(make_deltas(ANSWER))},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        t0 = time.perf_counter()
        self._chunk(b": keep-alive\n\n")          # 注释行：客户端必须忽略
        deltas = make_deltas(ANSWER)[:max(1, n_out)]
        # 第一个 chunk 只带 role（llama-server 的真实行为），正文从第二个开始
        role = {"id": "mock-1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        self._chunk(("data: " + json.dumps(role, ensure_ascii=False) + "\n\n").encode())
        for i, d in enumerate(deltas):
            obj = {"id": "mock-1", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"content": d},
                                "finish_reason": None}]}
            line = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
            # 每 7 片故意拆成两次 write，模拟 TCP 把一行劈开
            if i % 7 == 3:
                half = len(line) // 2
                self._chunk(line[:half])
                time.sleep(0.005)
                self._chunk(line[half:])
            else:
                self._chunk(line)
            time.sleep(self.chunk_delay)

        el = (time.perf_counter() - t0) * 1000
        final = {"id": "mock-1", "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 24, "completion_tokens": len(deltas),
                           "total_tokens": 24 + len(deltas)},
                 "timings": {"predicted_n": len(deltas),
                             "predicted_ms": el,
                             "predicted_per_second": round(len(deltas) / (el / 1000), 2)}}
        self._chunk(("data: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode())
        self._chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--delay", type=float, default=0.02)
    a = ap.parse_args()
    Handler.chunk_delay = a.delay
    with Server(("127.0.0.1", a.port), Handler) as s:
        print(f"mock llama-server on http://127.0.0.1:{a.port}", flush=True)
        s.serve_forever()

if __name__ == "__main__":
    main()
