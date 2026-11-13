# client-cpp —— C++17 流式客户端（D5 交付）

一条命令就能对着 `llama-server` 对话，同时把「首字延迟 / 解码速度」按服务端口径打出来。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/edge-client -m "用三句话说明什么是 KV Cache"
```

依赖已经内置在 `third_party/`：`httplib.h`（cpp-httplib 0.18.3）、`nlohmann/json.hpp`（3.11.3）。
不需要联网、不需要 apt、不需要包管理器 —— 编译只靠 g++ 和 pthread。

## 用法

```bash
# 单轮
./build/edge-client -m "什么是量化" -n 256
# 多轮交互（上下文会保留，/reset 清空，/exit 退出）
./build/edge-client
# 换机器 / 换端口
./build/edge-client -H 192.168.1.10 -p 8080 -m "hi"
# 对照组：一次性返回（不是流式）
./build/edge-client -m "hi" --no-stream
# 排障：把原始 SSE payload 打到 stderr
./build/edge-client -m "hi" --raw
```

输出示例（stdout 是纯正文，stderr 是统计行，两条流分开，方便管道）：

```
[stats] ttft=381.2ms total=2140ms decode=37.6tok/s tokens=64(usage) sse_events=66 pieces=64 bytes=152 srv_tg=37.8tok/s
```

| 字段 | 含义 |
|---|---|
| `ttft` | 请求发出 → 第一个非空 `delta.content` 到达 |
| `total` | 请求发出 → 流结束 |
| `decode` | `(tokens-1) / (total - ttft)`，与服务端 `timings` 同口径 |
| `tokens` | 优先 `usage.completion_tokens`，其次 `timings.predicted_n`，最后按片数兜底（标 `pieces~`） |
| `srv_tg` | llama.cpp 自报的 `predicted_per_second`，用来对账 |

## 为什么这样实现（面试会被问到的四点）

1. **为什么不用 `cli.Post(...)` 直接拿 body？**
   `Post` 的返回值会把 whole body 攒齐再交给你，那已经不是流式了。
   这里手动构造 `httplib::Request`，挂 `content_receiver`，走 `cli.send()` —— 字节到达即回调。

2. **SSE 为什么要自己做行缓冲？**
   TCP 不保证「一行一个包」。一个 `data: {...}` 可能被拆成两次 write，也可能两条粘在一起。
   所以按 `'\n'` 切行、缓存半行、剥 `'\r'`、忽略 `:` 开头的注释心跳、遇 `[DONE]` 主动断开。

3. **UTF-8 会不会被切坏？**
   不会 —— 只在行边界切分，而一个 JSON 行内的多字节字符是完整的。
   反例：若按固定字节数切片打印，就会出现「半个汉字」。

4. **背压怎么处理？**
   `content_receiver` 返回 `true` 表示继续收。下游写不动（比如 UI 卡住）时可以返回 `false`，
   连接会被主动关掉，而不是让内存里堆积 buffer。

## 真机可能撞到的两个小坑

- **服务端校验 model 名**（部分版本会返回 400 `model not found`）：先用
  `curl -s localhost:8080/v1/models` 看 id，再 `--model <id>` 传进来。
- **服务端不认 `stream_options`**（老版本 llama.cpp / 其他 OpenAI 兼容实现）：加 `--no-usage`，
  代价是拿不到 `usage.completion_tokens`，tokens 会退化成按片数估算（标 `pieces~`，不要写进报告）。

## 已知限制

- 只支持 HTTP（本机服务），没编 OpenSSL；要打公网 HTTPS 需要 `-DCPPHTTPLIB_OPENSSL_SUPPORT` 并链接 OpenSSL。
- 没有做真正的「流式重连续写」：连接中途断掉只报错，不重放半截回答（重放会导致重复文本）。
- `tokens` 在服务端既不给 `usage` 也不给 `timings` 时是估算值（按 delta 片数），报告里不要用这个数。
