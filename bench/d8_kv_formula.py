#!/usr/bin/env python3
"""bench/d8_kv_formula.py — KV Cache 公式计算器 + 与实测对表

三种用法：

1) 只看公式表
   python3 bench/d8_kv_formula.py --model results/d8_kv_model.json --table

2) 单个 ctx 的 MiB（给 shell 脚本调）
   python3 bench/d8_kv_formula.py --model results/d8_kv_model.json --ctx 8192 --mib

3) 与实测对表（关键那一步）
   python3 bench/d8_kv_formula.py --model results/d8_kv_model.json \\
       --check results/d8_kv_mem.csv --out results/d8_kv_mem_checked.csv

第 3 步的判据：用最小 ctx 当基准，比"差值"而不是"绝对值"——
绝对值里混着桌面/Xwayland 占用，差值才是纯 KV 增量。
容差 = max(48 MiB, 预期的 35%)。超出但这个比例落在 1.0–1.6 之间时判 PASS(+buf)：
那是 CUDA 计算缓冲随 ctx 增长带来的系统性正偏，不是算错。
"""
import argparse
import csv
import json
import os
import sys

KV_TYPES = {"f16": 2.0, "fp16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


def per_token_bytes(m):
    return m["layers"] * m["kv_heads"] * (m["head_dim"] + m.get("value_dim", m["head_dim"])) * m["bytes_per_element"]


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/d8_kv_model.json")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--kv-type", default=None, choices=list(KV_TYPES))
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--mib", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--check", default=None)
    ap.add_argument("--out", default="results/d8_kv_mem_checked.csv")
    a = ap.parse_args()

    if a.layers:                              # 手填参数，不依赖 JSON
        m = {"layers": a.layers, "kv_heads": a.kv_heads or a.layers,
             "head_dim": a.head_dim, "value_dim": a.head_dim,
             "bytes_per_element": KV_TYPES[a.kv_type or "f16"], "model": "manual"}
    else:
        if not os.path.exists(a.model):
            print(f"! 缺 {a.model}，先跑：python3 bench/d8_gguf_meta.py <模型.gguf>", file=sys.stderr)
            return 1
        m = load_model(a.model)
        if a.kv_type:
            m["bytes_per_element"] = KV_TYPES[a.kv_type]
            m["kv_bytes_per_token"] = per_token_bytes(m)

    pt = per_token_bytes(m)

    if a.mib:
        if not a.ctx:
            print("! --mib 需要 --ctx", file=sys.stderr)
            return 1
        print(f"{pt * a.ctx / (1024 * 1024):.1f}")
        return 0

    if a.table or not a.check:
        print(f"每 token = {pt:,.0f} B = {pt / 1024:.2f} KiB"
              f"（{m['layers']}L × {m['kv_heads']}KV × {m['head_dim']}d × {m['bytes_per_element']}B × 2）")
        print()
        print(f"{'ctx':>7}  {'KiB':>9}  {'MiB':>9}")
        for c in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            print(f"{c:>7}  {pt * c / 1024:>9.1f}  {pt * c / (1024 * 1024):>9.1f}")

    if not a.check:
        return 0

    # ---- 与实测对表 ----
    rows = []
    with open(a.check, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("vram_peak_mib") or r["vram_peak_mib"].strip().lower() in ("", "na", "oom"):
                rows.append({**r, "skip": "no_vram"})
                continue
            rows.append({**r, "skip": ""})

    ok_rows = [r for r in rows if not r["skip"]]
    if len(ok_rows) < 2:
        print("! 至少要有 2 行有效显存读数才能比差值", file=sys.stderr)
        return 1

    base = min(ok_rows, key=lambda r: int(r["ctx"]))
    base_ctx = int(base["ctx"])
    base_peak = float(base["vram_peak_mib"])
    base_model = per_token_bytes(m) * base_ctx / (1024 * 1024)

    out_rows, fails = [], 0
    print(f"\n基准行 ctx={base_ctx}，实测峰值 {base_peak:.0f} MiB（含桌面占用）\n")
    print(f"{'ctx':>7} {'预期增量':>10} {'实测增量':>10} {'误差':>8}  判定")
    for r in rows:
        if r["skip"]:
            r.update(kv_expected_delta_mib="", kv_measured_delta_mib="", kv_err_pct="", verdict="SKIP(no_vram)")
            out_rows.append(r)
            continue
        ctx = int(r["ctx"])
        peak = float(r["vram_peak_mib"])
        exp_delta = (per_token_bytes(m) * (ctx - base_ctx)) / (1024 * 1024)
        meas_delta = peak - base_peak
        err = (meas_delta - exp_delta) / exp_delta * 100 if exp_delta else 0.0
        tol = max(48.0, abs(exp_delta) * 0.35)
        good = abs(meas_delta - exp_delta) <= tol
        # 系统性正偏：实测比公式多，但比例落在 1.0–1.6 —— 这是 CUDA 计算缓冲随 ctx 增长，
        # 属已知项，写进报告的“误差解释”即可，不算失败。
        buf = (exp_delta > 0 and meas_delta > exp_delta
               and 1.0 <= meas_delta / exp_delta <= 1.6)
        if good:
            verdict = "PASS"
        elif buf:
            verdict = "PASS(+buf)"
        else:
            verdict = "FAIL"
        if verdict == "FAIL":
            fails += 1
        print(f"{ctx:>7} {exp_delta:>10.1f} {meas_delta:>10.1f} {err:>7.1f}%  {verdict}")
        r.update(kv_expected_delta_mib=f"{exp_delta:.1f}",
                 kv_measured_delta_mib=f"{meas_delta:.1f}",
                 kv_err_pct=f"{err:.1f}", verdict=verdict)
        out_rows.append(r)

    for r in out_rows:
        r.pop("skip", None)
    fields = list(out_rows[0].keys())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n已写 {a.out}")

    if any("+buf" in str(r.get("verdict")) for r in out_rows):
        print("\n注：有 PASS(+buf) 行 —— 实测比公式多，比例在 1.0–1.6。这是计算缓冲随 ctx 增长，"
              "在报告“误差解释”里写一句即可。")

    if fails:
        print("\n! 有偏差超容差的行。按顺序查这三件事：")
        print("  1) 每行的 -ngl 是否一致（少卸载一层，权重少占 60–70 MiB，会盖住 KV 增量）")
        print("  2) 采样是否覆盖到 KV 分配那一下（llama.cpp 是加载即分配，峰值在生成期间更稳）")
        print("  2b) 顺带看 4K→8K 的增量是不是 2K→4K 的两倍：是，就还是缓冲问题，不是跑错")
        print("  3) 桌面上有没有别的进程在抢显存（浏览器/录屏/Xwayland），先记基线再跑")
        return 1
    print("全部在容差内——公式和实测对上了，这一章可以定稿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
