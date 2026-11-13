#!/usr/bin/env python3
"""tools/mock_llama_server.py — 假的 llama-server，只为了在没有 GPU 的机器上验证量具本身。

它不是基准测试：延迟是 sleep 出来的，数字没有物理意义。
用途只有一个：接通 metrics.py 的整条链路（SSE 解析 / 计时 / CSV / --check），
免得等真机上跑了一半才发现脚本读错了字段。

起法：
  MOCK_PP=152 MOCK_TG=40 python3 tools/mock_llama_server.py --port 8099
"""
import argparse
import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PP = float(os.environ.get("MOCK_PP", "152"))   # 假 prefill 速度 t/s
TG = float(os.environ.get("MOCK_TG", "40"))    # 假解码速度 t/s


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 安静点，别把终端刷满
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": "local", "object": "model", "owned_by": "mock"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._json({"error": "only /v1/completions and /v1/chat/completions"}, 404)
            return
        chat = self.path.endswith("chat/completions")
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req.get("prompt") or "".join(
            m.get("content") or "" for m in (req.get("messages") or [])
            if isinstance(m, dict))
        max_tokens = int(req.get("max_tokens") or req.get("n_predict") or 128)
        stream = bool(req.get("stream"))

        # 粗糙的 token 估算：中文约 2 字符/token，够用了
        prompt_n = max(1, len(prompt) // 2)
        prefill_s = prompt_n / PP

        if not stream:
            time.sleep(prefill_s + max_tokens / TG)
            pred_ms = max_tokens / TG * 1000
            choice = ({"index": 0, "message": {"role": "assistant", "content": "(mock)"},
                       "finish_reason": "length"} if chat
                      else {"index": 0, "text": "(mock)", "finish_reason": "length"})
            self._json({
                "choices": [choice],
                "usage": {"prompt_tokens": prompt_n,
                          "completion_tokens": max_tokens,
                          "total_tokens": prompt_n + max_tokens},
                "timings": {"prompt_n": prompt_n,
                            "prompt_ms": prefill_s * 1000,
                            "prompt_per_second": prompt_n / prefill_s,
                            "predicted_n": max_tokens,
                            "predicted_ms": pred_ms,
                            "predicted_per_second": max_tokens / (pred_ms / 1000)},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        t0 = time.perf_counter()
        time.sleep(prefill_s)                     # 假的 prefill
        t_prefill = time.perf_counter()

        for i in range(max_tokens):
            time.sleep(1.0 / TG + random.uniform(-0.001, 0.001))  # 假的逐 token
            if chat:
                chunk = {"choices": [{"index": 0,
                                      "delta": {"content": f" tok{i}"},
                                      "finish_reason": None}]}
            else:
                chunk = {"choices": [{"index": 0, "text": f" tok{i}",
                                      "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()

        t_end = time.perf_counter()
        pred_ms = (t_end - t_prefill) * 1000
        end_choice = ({"index": 0, "delta": {}, "finish_reason": "length"} if chat
                      else {"index": 0, "text": "", "finish_reason": "length"})
        final = {
            "choices": [end_choice],
            "usage": {"prompt_tokens": prompt_n,
                      "completion_tokens": max_tokens,
                      "total_tokens": prompt_n + max_tokens},
            "timings": {"prompt_n": prompt_n,
                        "prompt_ms": (t_prefill - t0) * 1000,
                        "prompt_per_second": prompt_n / max(t_prefill - t0, 1e-6),
                        "predicted_n": max_tokens,
                        "predicted_ms": pred_ms,
                        "predicted_per_second": max_tokens / max(pred_ms / 1000, 1e-6)},
        }
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock llama-server on :{a.port}  pp={PP} t/s  tg={TG} t/s（数字是 sleep 出来的，不是基准）")
    srv.serve_forever()


if __name__ == "__main__":
    main()
