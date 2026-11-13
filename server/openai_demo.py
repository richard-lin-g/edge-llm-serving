#!/usr/bin/env python3
"""D4 产出：OpenAI 兼容接口的最小调用示例（三条路径）。

用法:
  python server/openai_demo.py                   # 默认 http://127.0.0.1:8080
  python server/openai_demo.py http://127.0.0.1:8080 "介绍一下 GGUF 格式"

为什么不用 openai SDK: 面试里"你有没有自己解析过 SSE"是个高频追问，
这里同时给 requests 裸流与 SDK 两种写法，前者能讲清 chunk/finish_reason 的边界。

依赖: pip install requests   （SDK 那条路径需要 pip install openai）
"""
import json
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "用两句话说明什么是 KV Cache。"


def path_a_raw_stream():
    """路径 A：requests 裸流 + 自己解析 SSE（最可控，推荐）"""
    t0 = time.perf_counter()
    ttft = None
    out = []
    usage = None
    timings = None
    with requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    out.append(piece)
                    print(piece, end="", flush=True)
    total = (time.perf_counter() - t0) * 1000
    print()
    n_out = len(out)
    print(f"\n[A] TTFT={ttft:.0f} ms  total={total:.0f} ms  chunks={n_out}")
    if timings:
        print(f"[A] server timings: pp={timings.get('prompt_per_second'):.2f} t/s,"
              f" tg={timings.get('predicted_per_second'):.2f} t/s")
    if n_out > 1:
        print(f"[A] decode_tps≈{1000 * (n_out - 1) / (total - ttft):.1f}（客户端视角，"
              f"比服务端高约 10%，别拿它做锚点）")
    return "".join(out)


def path_b_openai_sdk():
    """路径 B：官方 openai SDK（证明「兼容」不是自说自话）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("\n[B] 跳过：未安装 openai（pip install openai 后可跑这条）")
        return None
    client = OpenAI(base_url=f"{BASE}/v1", api_key="not-needed")
    t0 = time.perf_counter()
    ttft, n = None, 0
    stream = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=128, temperature=0, stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content if chunk.choices else None
        if piece:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            n += 1
            print(piece, end="", flush=True)
    print(f"\n[B] SDK 流式 OK, chunks={n}, TTFT={ttft:.0f} ms")
    return n


def path_c_non_stream():
    """路径 C：非流式（对照：TTFT 概念在这里退化成一整次端到端）"""
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 128,
            "temperature": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    txt = j["choices"][0]["message"]["content"]
    print(f"\n[C] 非流式 {(time.perf_counter() - t0) * 1000:.0f} ms, "
          f"finish_reason={j['choices'][0].get('finish_reason')}")
    print(f"[C] usage={j.get('usage')}")
    return txt


if __name__ == "__main__":
    print(f"== 目标 {BASE} ==")
    print("\n--- A) 裸流 SSE ---")
    path_a_raw_stream()
    print("\n--- B) openai SDK ---")
    path_b_openai_sdk()
    print("\n--- C) 非流式 ---")
    path_c_non_stream()
