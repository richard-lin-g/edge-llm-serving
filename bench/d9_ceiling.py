#!/usr/bin/env python3
"""bench/d9_ceiling.py — 压测之前先把天花板算出来

为什么先算：压测只是验证。如果算出来 16 并发的理论上限只有单流的 7 倍，
那么实测涨 3 倍就不是"没调好"，而是"这台卡就这样"——面试要的是这句话，
不是"我又调了调参数好了一点"。

三个天花板：
  带宽侧   权重 2.1 GB 每 step 读一遍，160 GB/s → 13 ms/step → 单流上限约 76 tok/s
  算力侧   2 × 3.09e9 FLOP/token = 6.2 GFLOP，1.7 TFLOPS → 约 274 tok/s（FP16 翻倍就看实测）
  批上限   -ub 256 是每 step 能吃进的 token 上限，np 超过它只能排队

用法：
  python3 bench/d9_ceiling.py --model ~/models/qwen2.5-3b-instruct-q4_k_m.gguf --tg1 39.33
"""
import argparse
import os


def fmt(x, d=1):
    return f"{x:,.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="GGUF 路径，用来读权重字节数")
    ap.add_argument("--model-bytes", type=float, default=2104932768,
                    help="拿不到文件就给字节数（默认本机 Q4_K_M 实测值）")
    ap.add_argument("--bw-gbs", type=float, default=160.0, help="T600 规格 160 GB/s")
    ap.add_argument("--tflops", type=float, default=1.7, help="T600 FP32 1.7 TFLOPS")
    ap.add_argument("--params", type=float, default=3.09e9)
    ap.add_argument("--kv-kib", type=float, default=36.0, help="每 token KV：36 KiB（D8 实测）")
    ap.add_argument("--ctx-used", type=int, default=264, help="单请求实际占用的 ctx（prompt+输出）")
    ap.add_argument("--tg1", type=float, default=39.33, help="D2 单流解码锚点 tok/s")
    ap.add_argument("--np-list", default="1 4 8 16")
    args = ap.parse_args()

    weights = os.path.getsize(args.model) if args.model and os.path.exists(args.model) \
        else args.model_bytes
    wgb = weights / 1e9
    step_ms = weights / (args.bw_gbs * 1e9) * 1000
    bw_ceiling = 1000 / step_ms

    flops_per_tok = 2 * args.params
    compute_ceiling = args.tflops * 1e12 / flops_per_tok

    kv_bytes = args.kv_kib * 1024
    print(f"权重        {fmt(wgb, 3)} GB（{int(weights):,} B）")
    print(f"带宽侧      {args.bw_gbs:.0f} GB/s → 每 step 至少 {step_ms:.1f} ms "
          f"→ 单流上限 {bw_ceiling:.0f} tok/s")
    print(f"实测单流    {args.tg1:.2f} tok/s → 有效带宽 {args.tg1 * weights / 1e9 / 1.0:.0f} GB/s"
          f"（占规格 {args.tg1 / bw_ceiling * 100:.0f}%）")
    print(f"算力侧      {flops_per_tok / 1e9:.2f} GFLOP/token，{args.tflops} TFLOPS "
          f"→ 上限 {compute_ceiling:.0f} tok/s")
    print(f"KV 读取     每 step 每个请求读 ctx×36 KiB：{args.ctx_used} token × {args.kv_kib:.0f} KiB "
          f"= {args.ctx_used * kv_bytes / 1024 / 1024:.1f} MiB")

    print("\nnp   带宽侧上限(×np)   算力截断后   相对单流的倍数上限   -ub 256 是否够")
    for np in [int(x) for x in args.np_list.split()]:
        bw_side = bw_ceiling * np
        cap = min(bw_side, compute_ceiling)
        print(f"{np:2d}   {fmt(bw_side, 0):>13}   {fmt(cap, 0):>9}   "
              f"{fmt(cap / args.tg1, 1):>15}×        "
              f"{'够（np≤256）' if np <= 256 else '撞上限，需调 -ub'}")

    cap16 = min(compute_ceiling, bw_ceiling * 16)
    print(f"\n结论模板：单流 {args.tg1:.1f} tok/s，GPU 侧硬上限 "
          f"{cap16:.0f} tok/s，所以 16 并发最多约 {cap16 / args.tg1:.1f}× 单流，"
          f"而不是 16×。实测低于这个数，差额才算「没调好」，"
          f"差额要归因到 prefill 抢占或 -ub。")


if __name__ == "__main__":
    main()
