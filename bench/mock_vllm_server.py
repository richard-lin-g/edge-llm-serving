#!/usr/bin/env python3
"""bench/mock_vllm_server.py — 假 vLLM，只用来验证压测器与日志判定，数字全是假的。

D9 的教训：压测脚本一定要在没有 GPU 的地方先跑通一遍，否则真机上一次 20 分钟的
压测可能因为一个 SSE 解析 bug 全废。这个 mock 流的是标准 OpenAI SSE 分片。

用法：
  python3 bench/mock_vllm_server.py --port 8100 --out-tokens 64 --per-token-ms 20
  python3 bench/mock_vllm_server.py --port 8100 --fail-mode oom   # 造一条 OOM 日志
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_error(self, message):
        body = json.dumps({"error": {"message": message, "type": "mock"}}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            body = json.dumps({"object": "list", "data": [
                {"id": ARGS.model, "object": "model", "created": int(time.time())}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_error("mock: only /v1/models and /v1/completions")

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/completions":
            self._send_error("mock: only /v1/completions")
            return
        if ARGS.fail_mode == "oom":
            print("ERROR 2026-11-19 vLLM: CUDA out of memory. Tried to allocate 5.76 GiB. "
                  "GPU 0 has a total capacity of 3.77 GiB of which 1.20 GiB is free. "
                  "Model weights take 6.18GB; increase gpu_memory_utilization or use a smaller model.",
                  file=sys.stderr)
            self._send_error("CUDA out of memory")
            return
        if ARGS.fail_mode == "kvzero":
            print("ERROR 2026-11-19 ValueError: No available memory for the cache blocks. "
                  "Try increasing gpu_memory_utilization.", file=sys.stderr)
            self._send_error("No available memory for the cache blocks")
            return

        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        max_tokens = int(req.get("max_tokens") or 32)
        prompt = req.get("prompt") or ""

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload):
            data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        chunk({"id": "cmpl-mock", "object": "text_completion",
               "choices": [{"index": 0, "text": "", "finish_reason": None}]})
        for i in range(max_tokens):
            chunk({"id": "cmpl-mock", "object": "text_completion",
                   "choices": [{"index": 0, "text": f" tok{i}", "finish_reason": None}]})
            time.sleep(ARGS.per_token_ms / 1000.0)
        chunk({"id": "cmpl-mock", "object": "text_completion",
               "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
               "usage": {"prompt_tokens": len(prompt.split()), "total_tokens":
                         len(prompt.split()) + max_tokens, "completion_tokens": max_tokens}})
        data = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="qwen25-3b-awq")
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--per-token-ms", type=float, default=20.0)
    ap.add_argument("--fail-mode", choices=["none", "oom", "kvzero"], default="none")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    srv.daemon_threads = True
    print(f"mock vLLM on :{ARGS.port}  out={ARGS.out_tokens}  per_token={ARGS.per_token_ms}ms")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
