#!/usr/bin/env python3
"""bench/mem_budget.py — D10 显存账：先算清楚，再决定要不要花一小时装 vLLM。

vLLM 启动时显存只花在三处：

  1. 权重      W = 未量化参数 × 2B + 量化参数 × bits/8
  2. 运行时    C = CUDA context（~300 MiB）+ 激活峰值 + CUDA graph（enforce-eager 可砍）
  3. KV 池     K = layers × 2 × kv_heads × head_dim × kv_bytes × ctx × seqs

只有 W + C + K ≤ total × gpu_memory_utilization，服务才起得来。
三种失败长得不一样，报告里要写对是哪一种：

  W + C > 预算            → "CUDA out of memory. Tried to allocate ... model weights"
  W + C ≤ 预算，但 K 算出来 ≤ 0
                          → 不会报 OOM，报 "No available memory for the cache blocks"，
                            意思是权重放下了，可分的 block 数是 0，等于一序列都跑不了
  K > 0 但序列数 < 2      → 起得来，但没有第二个槽，"continuous batching" 无从体现

用法：
  python3 bench/mem_budget.py                          # Qwen2.5-3B + 4096 MiB 卡（默认）
  python3 bench/mem_budget.py --usable-mib 3720        # 用 vram_probe.py 的实测可用值
  python3 bench/mem_budget.py --eager                  # 关 CUDA graph 的开销假设
  python3 bench/mem_budget.py --total-mib 8192 --json  # 换卡推演，出 JSON
"""
import argparse
import json
import sys

GiB = 1024 ** 3

# Qwen2.5-3B-Instruct 的真实结构：36 层、16 个 Q 头、2 个 KV 头（GQA）、head_dim 128
PRESET = {
    "name": "Qwen2.5-3B-Instruct",
    "params_b": 3.09,        # 总参数 3.09B
    "unquant_b": 0.31,       # 词表 151936 × 2048 ≈ 0.31B 这部分（embedding/lm_head）通常不量化
    "layers": 36,
    "kv_heads": 2,
    "head_dim": 128,
    "ctx": 2048,
}

QUANTS = [
    # 名称, 量化部分位宽, 报告里怎么写
    ("fp16", 16.0, "vLLM 默认 dtype=auto → fp16/bf16，T600 有 fp16 单元"),
    ("int8", 8.0, "W8A16；精度掉得少，但 3.4 GB 权重在 4 GB 卡上仍是死刑"),
    ("awq-w4", 4.25, "AWQ/GPTQ W4A16；4.25 bit 含分组 scales/zeros，实际文件约 2.05 GB"),
]


def weights_gib(bits, preset=PRESET):
    """量化只作用在非 embedding 部分；embedding / lm_head 保持 fp16。"""
    unquant_bytes = preset["unquant_b"] * 1e9 * 2
    quant_bytes = (preset["params_b"] - preset["unquant_b"]) * 1e9 * bits / 8
    return (unquant_bytes + quant_bytes) / GiB


def kv_kib_per_token(preset=PRESET, kv_bytes=2):
    """每 token 的 KV 字节约等于 2 × layers × kv_heads × head_dim × kv_bytes。"""
    return preset["layers"] * 2 * preset["kv_heads"] * preset["head_dim"] * kv_bytes / 1024


def kv_gib(seqs, ctx, preset=PRESET, kv_bytes=2):
    return kv_kib_per_token(preset, kv_bytes) * 1024 * ctx * seqs / GiB


def evaluate(usable_mib, util, overhead_mib, preset=PRESET, kv_bytes=2):
    """对每种精度给出判定。"""
    budget_gib = usable_mib / 1024 * util
    rows = []
    for name, bits, note in QUANTS:
        w = weights_gib(bits, preset)
        c = overhead_mib / 1024
        kv_budget = budget_gib - w - c
        ctx = preset["ctx"]
        per_seq = kv_gib(1, ctx, preset, kv_bytes)
        seqs = kv_budget / per_seq if kv_budget > 0 else 0.0
        if w + c > budget_gib:
            verdict = "起不来：权重+运行时已超预算"
            code = "WEIGHT_OOM"
        elif kv_budget <= 0 or seqs < 1:
            verdict = "起不来：KV 池 0 block"
            code = "KV_ZERO"
        elif seqs < 2:
            verdict = "勉强：只有 1 个序列，无调度价值"
            code = "ONE_SEQ"
        else:
            verdict = f"可能起：最多 {int(seqs)} 序列 × {ctx} ctx"
            code = "OK"
        rows.append({
            "quant": name, "bits": bits, "weights_gib": round(w, 2),
            "overhead_gib": round(c, 2), "kv_budget_gib": round(kv_budget, 2),
            "kv_per_seq_mib": round(per_seq * 1024, 1),
            "max_seqs": round(seqs, 1), "verdict": verdict, "code": code, "note": note,
        })
    return budget_gib, rows


def print_table(usable_mib, util, overhead_mib, preset, kv_bytes, as_json=False):
    budget, rows = evaluate(usable_mib, util, overhead_mib, preset, kv_bytes)
    kv_tok = kv_kib_per_token(preset, kv_bytes)
    meta = {
        "model": preset["name"], "usable_mib": usable_mib,
        "util": util, "budget_gib": round(budget, 2),
        "overhead_mib": overhead_mib,
        "kv_kib_per_token": round(kv_tok, 1), "ctx": preset["ctx"],
        "kv_per_seq_mib": round(kv_gib(1, preset["ctx"], preset, kv_bytes) * 1024, 1),
    }
    if as_json:
        print(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2))
        return meta, rows

    print(f"\n== 显存账 · {preset['name']} · 可用 {usable_mib} MiB · "
          f"gpu_memory_utilization={util} ==")
    print(f"预算 = {usable_mib}/1024 × {util} = {budget:.2f} GiB ｜ "
          f"KV {kv_tok:.0f} KiB/token ｜ ctx {preset['ctx']} → "
          f"每序列 {kv_gib(1, preset['ctx'], preset, kv_bytes) * 1024:.0f} MiB ｜ "
          f"运行时开销 {overhead_mib} MiB\n")
    print("| 精度 | 权重 GiB | 开销 GiB | 剩给 KV | 每序列 MiB | 最多序列 | 判定 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['quant']} | {r['weights_gib']} | {r['overhead_gib']} | "
              f"{r['kv_budget_gib']} | {r['kv_per_seq_mib']} | {r['max_seqs']} | {r['verdict']} |")
    print()
    return meta, rows


def card_sweep(overhead_mib, util, preset, kv_bytes):
    """换卡推演：要多大的卡，fp16 / awq 才跑得动。"""
    print("\n== 换卡推演（同一模型，ctx 2048，单序列） ==")
    print("| 卡显存 | 预算 GiB | fp16 权重+开销 | fp16 判 | awq 权重+开销 | awq 剩余 KV | awq 最多序列 |")
    print("|---|---|---|---|---|---|---|")
    for total in (4096, 6144, 8192, 12288, 16384, 24576):
        budget = total / 1024 * util
        f_w, a_w = weights_gib(16.0, preset), weights_gib(4.25, preset)
        c = overhead_mib / 1024
        per = kv_gib(1, preset["ctx"], preset, kv_bytes)
        f_ok = "剩 " + f"{(budget - f_w - c) / per:.1f} 序列" if budget - f_w - c > 0 else "起不来"
        a_left = budget - a_w - c
        a_ok = f"{a_left / per:.1f}" if a_left > 0 else "0"
        print(f"| {total} MiB | {budget:.2f} | {f_w:.2f}+{c:.2f} | {f_ok} | "
              f"{a_w:.2f}+{c:.2f} | {a_left:.2f} | {a_ok} |")


def self_check():
    """算式的三个恒等式自检，防止哪天参数改了就悄悄错。"""
    ok = True
    w16 = weights_gib(16.0)
    if abs(w16 - PRESET["params_b"] * 1e9 * 2 / GiB) > 1e-6:
        print("FAIL: fp16 权重应等于 params × 2B")
        ok = False
    if abs(kv_kib_per_token() - 36.0) > 1e-9:
        print(f"FAIL: KV 应为 36 KiB/token，实得 {kv_kib_per_token()}")
        ok = False
    if abs(kv_gib(1, 2048) * 1024 - 72.0) > 1e-6:
        print("FAIL: 2048 ctx 单序列应为 72 MiB")
        ok = False
    # 4 GB 卡上 fp16 必须判死，awq 必须判“可能起”
    _, rows = evaluate(3720, 0.9, 900)
    by = {r["quant"]: r["code"] for r in rows}
    if by["fp16"] != "WEIGHT_OOM":
        print("FAIL: 4 GB 卡上 fp16 应判 WEIGHT_OOM")
        ok = False
    if by["awq-w4"] == "WEIGHT_OOM":
        print("FAIL: awq 在 4 GB 卡上不该判权重 OOM")
        ok = False
    print("self-check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usable-mib", type=int, default=4096,
                    help="进程可用显存 MiB；用 vram_probe.py 实测值替换")
    ap.add_argument("--util", type=float, default=0.9,
                    help="gpu_memory_utilization；老版本默认 0.9，新版本默认 1.0")
    ap.add_argument("--overhead-mib", type=int, default=None,
                    help="运行时开销 MiB；默认 900（context 300 + 激活 400 + graph 200）")
    ap.add_argument("--eager", action="store_true", help="--enforce-eager，graph 抵消，开销按 600 MiB")
    ap.add_argument("--ctx", type=int, default=None, help="max-model-len，默认 2048")
    ap.add_argument("--kv-bytes", type=int, default=2, help="KV 精度字节：fp16=2")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="附换卡推演表")
    ap.add_argument("--check", action="store_true", help="只跑算式自检")
    args = ap.parse_args()

    if args.check:
        sys.exit(self_check())

    preset = dict(PRESET)
    if args.ctx:
        preset["ctx"] = args.ctx
    overhead = args.overhead_mib if args.overhead_mib is not None else (600 if args.eager else 900)

    print_table(args.usable_mib, args.util, overhead, preset, args.kv_bytes, args.json)
    if args.sweep and not args.json:
        card_sweep(overhead, args.util, preset, args.kv_bytes)


if __name__ == "__main__":
    main()
