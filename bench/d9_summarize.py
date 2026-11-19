#!/usr/bin/env python3
"""bench/d9_summarize.py — 把逐请求 CSV 收成一张能进报告的并发表

算的东西：
  QPS        = 成功请求数 / 这一档的墙钟时间（客户端统计，含排队）
  P50/P95/P99= 端到端延迟的最近秩分位（n=100 时 P95 就是第 95 个样本，不做插值）
  TTFT P95   = 首字延迟的 95 分位（排队 + prefill 都压在这里）
  聚合吞吐   = 这一档所有成功请求的输出 token 之和 / 墙钟
  扩展效率   = QPS(np) / (np × QPS(1))，1.0 才是线性
  显存增量   = 峰值 − 这一档开跑前的基线（绝对值里混着桌面占用，只能看差值）

用法：
  python3 bench/d9_summarize.py --requests results/d9_requests.csv \
      --meta results/d9_level_meta.csv --out results/d9_summary.csv
  python3 bench/d9_summarize.py --check results/d9_summary.csv
"""
import argparse
import csv
import math
import os
import statistics
import sys

LEVELS = [1, 4, 8, 16]
TOL_ERR = 2.0        # 错误率上限 %
MIN_EFF = 0.35       # np=16 时扩展效率下限（低于它说明几乎没扩起来）


def pct(vals, p):
    """最近秩法：升序第 ceil(p/100 * n) 个，不插值。n=100 时 P95=第 95 个。"""
    if not vals:
        return None
    s = sorted(vals)
    k = max(1, math.ceil(p / 100 * len(s)))
    return s[k - 1]


def first_by(rows, key):
    for r in rows:
        if r.get(key):
            return r
    return None


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def build(req_path, meta_path):
    reqs = load(req_path)
    metas = load(meta_path)
    if not reqs:
        sys.exit(f"{req_path} 是空的：先把 d9_load.sh 跑起来")

    by_np = {}
    for r in reqs:
        by_np.setdefault(int(r["np"]), []).append(r)
    meta_by_np = {}
    for m in metas:
        meta_by_np[int(m["np"])] = m          # 同档重跑取最后一条

    out = []
    for np in sorted(by_np):
        rs = by_np[np]
        m = meta_by_np.get(np, {})
        ok = [r for r in rs if r["status"] == "ok"]
        fails = [r for r in rs if r["status"] != "ok"]
        wall = num(m.get("wall_s"))
        qps = len(ok) / wall if wall else None
        e2e = [num(r["e2e_ms"]) for r in ok if num(r["e2e_ms"]) is not None]
        ttf = [num(r["ttft_ms"]) for r in ok if num(r["ttft_ms"]) is not None]
        tpot = [num(r["tpot_ms"]) for r in ok if num(r["tpot_ms"]) is not None]
        nout = [int(r["n_out"]) for r in ok if r.get("n_out")]
        pns = [int(r["prompt_n"]) for r in ok if r.get("prompt_n")]
        base = num(m.get("vram_base_mib"))
        peak = num(m.get("vram_peak_mib"))
        out.append({
            "np": np,
            "c_ctx": m.get("c_ctx") or "",
            "ctx_slot": m.get("ctx_slot") or "",
            "req": len(rs), "ok": len(ok), "fail": len(fails),
            "err_rate": round(len(fails) / len(rs) * 100, 2) if rs else None,
            "wall_s": wall,
            "qps": round(qps, 3) if qps else None,
            "agg_tps": round(sum(nout) / wall, 1) if (wall and nout) else None,
            "p50_ms": round(pct(e2e, 50), 0) if e2e else None,
            "p95_ms": round(pct(e2e, 95), 0) if e2e else None,
            "p99_ms": round(pct(e2e, 99), 0) if e2e else None,
            "ttft_p50": round(pct(ttf, 50), 0) if ttf else None,
            "ttft_p95": round(pct(ttf, 95), 0) if ttf else None,
            "tpot_med": round(statistics.median(tpot), 2) if tpot else None,
            "n_out_med": int(statistics.median(nout)) if nout else None,
            "prompt_n_med": int(statistics.median(pns)) if pns else None,
            "vram_base": int(base) if base else None,
            "vram_peak": int(peak) if peak else None,
            "vram_delta": int(peak - base) if (base and peak) else None,
            "gpu_util_avg": num(m.get("gpu_util_avg")),
            "gpu_util_max": num(m.get("gpu_util_max")),
            "cpu_busy_avg": num(m.get("cpu_busy_avg")),
            "client_cpu_s": num(m.get("client_cpu_s")),
            "note": m.get("note", ""),
        })

    q1 = next((r["qps"] for r in out if r["np"] == 1 and r["qps"]), None)
    for r in out:
        r["scale_eff"] = round(r["qps"] / (r["np"] * q1), 2) if (q1 and r["qps"]) else None
    return out


FIELDS = ["np", "c_ctx", "ctx_slot", "req", "ok", "fail", "err_rate", "wall_s", "qps",
          "agg_tps", "p50_ms", "p95_ms", "p99_ms", "ttft_p50", "ttft_p95", "tpot_med",
          "n_out_med", "prompt_n_med", "vram_base", "vram_peak", "vram_delta",
          "gpu_util_avg", "gpu_util_max", "cpu_busy_avg", "client_cpu_s",
          "scale_eff", "note"]


def verdicts(rows):
    """把曲线形状翻译成结论候选。报告里要写的是判断依据，不是形容词。"""
    lines = []
    eff = {r["np"]: r["scale_eff"] for r in rows}
    qps = {r["np"]: r["qps"] for r in rows}
    gpu = {r["np"]: r["gpu_util_avg"] for r in rows}
    last = rows[-1]

    if qps.get(1) and qps.get(last["np"]):
        lines.append(f"QPS：1 → {last['np']}，{qps[1]:.2f} → {qps[last['np']]:.2f}，"
                     f"放大 {qps[last['np']] / qps[1]:.2f} 倍（线性应该是 {last['np']} 倍）")
    for np in sorted(eff):
        if eff[np] is not None:
            lines.append(f"  扩展效率 np={np}: {eff[np]:.2f}")
    if last.get("gpu_util_avg") is not None and last["gpu_util_avg"] >= 92:
        lines.append("判定：GPU 利用率已到 90%+ → 撞的是 GPU 侧天花板（算力或显存带宽），"
                     "不是排队问题。结论要指到具体一项：算力 6.2 GFLOP/token vs 1.7 TFLOPS，"
                     "还是权重读取 2.1 GB/step vs 160 GB/s。")
    elif last.get("cpu_busy_avg") is not None and last["cpu_busy_avg"] >= 80:
        lines.append("判定：整机 CPU 先到 80%+ → 瓶颈在主机侧（tokenize/HTTP/采样线程/客户端），"
                     "先把客户端挪开或把采样间隔放到 500 ms 再复测，否则量的是 CPU 不是 GPU。")
    elif last.get("gpu_util_avg") is not None and last["gpu_util_avg"] < 70 and \
            (eff.get(last["np"]) or 1) < 0.5:
        lines.append("判定：GPU 没吃饱但扩展效率也上不去 → 卡在批调度：每步能吃进的 token 被 "
                     "-ub 256 或槽间抢占（新请求 prefill 挤掉 decode）限制住了。"
                     "下一步测 -ub 512 对照，别急着换框架。")
    elif last.get("gpu_util_avg") is None:
        lines.append("判定：缺 GPU 采样，瓶颈只能存疑——先把 nvidia-smi 采集修好，再写结论。")
    elif (eff.get(last["np"]) or 1) >= 0.6:
        lines.append(f"判定：np={last['np']} 扩展效率仍有 {eff.get(last['np'])}，还没到硬拐点；"
                     "把并发加到 32 或把输出加到 128 token 再测一档，才能定位天花板。")
    else:
        lines.append("判定：GPU 与 CPU 都未饱和，扩展效率却明显下滑 → 优先怀疑批上限/抢占，"
                     "用 -ub 512 做一组对照再定论。")
    errs = [r for r in rows if (r["err_rate"] or 0) > TOL_ERR]
    if errs:
        lines.append(f"注意：{', '.join('np=' + str(r['np']) for r in errs)} 错误率超过 {TOL_ERR}%，"
                     "先看 results/d9_server_np*.log 的队列/超时，不要拿这张表下结论。")
    if any(r["note"] for r in rows):
        lines.append("注意：有档位带口径异常标记（" +
                     ", ".join(f"np={r['np']}:{r['note']}" for r in rows if r["note"]) +
                     "），那几档不能进报告。")
    return lines


def print_table(rows):
    head = f"{'np':>3} {'ctx槽':>5} {'QPS':>7} {'P50':>7} {'P95':>7} {'TTFTp95':>8} " \
           f"{'tok/s':>7} {'错误率':>6} {'显存峰值':>8} {'GPU':>6} {'效率':>5}"
    print(head)
    for r in rows:
        def f(v, w, d=0):
            return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(f"{r['np']:>3} {str(r['ctx_slot'] or '-'):>5} {f(r['qps'], 7, 2)} "
              f"{f(r['p50_ms'], 7, 0)} {f(r['p95_ms'], 7, 0)} {f(r['ttft_p95'], 8, 0)} "
              f"{f(r['agg_tps'], 7, 1)} {f(r['err_rate'], 5, 2)} {f(r['vram_peak'], 8, 0)} "
              f"{f(r['gpu_util_avg'], 6, 1)} {f(r['scale_eff'], 5, 2)}")


def check(path, rows):
    ok = True
    present = {r["np"] for r in rows}
    print("\n== 验收（D9）==")
    for np in LEVELS:
        if np not in present:
            print(f"  ✗ 缺 np={np} 这一档")
            ok = False
    slots = {r["np"]: r["ctx_slot"] for r in rows if str(r["ctx_slot"]) not in ("", "None")}
    if slots and len(set(map(str, slots.values()))) > 1:
        print(f"  ✗ 每槽上下文不一致：{slots} —— 各档不是同一口径，曲线不能放一张图")
        ok = False
    elif slots:
        print(f"  ✓ 每槽上下文一致：{set(map(str, slots.values()))}")

    for r in rows:
        if (r["err_rate"] or 0) > TOL_ERR:
            print(f"  ✗ np={r['np']} 错误率 {r['err_rate']}% > {TOL_ERR}%")
            ok = False
        if r["prompt_n_med"] and r["ctx_slot"] and r["n_out_med"] and \
                r["prompt_n_med"] + r["n_out_med"] > int(r["ctx_slot"] or 0):
            print(f"  ✗ np={r['np']} prompt {r['prompt_n_med']} + 输出 {r['n_out_med']} "
                  f"> 每槽 ctx {r['ctx_slot']}：prompt 被截断，这档作废")
            ok = False
        if r["note"]:
            print(f"  ✗ np={r['np']} 口径异常 {r['note']}")
            ok = False

    last = rows[-1]
    if last["scale_eff"] is not None and last["scale_eff"] < MIN_EFF:
        print(f"  ! np={last['np']} 扩展效率 {last['scale_eff']} < {MIN_EFF}："
              "曲线几乎平了，报告里要么解释清（批上限/抢占），要么换 -ub 再测一轮")
    if all(r["gpu_util_avg"] is None for r in rows):
        print("  ! 全程没采到 GPU 利用率：瓶颈判定缺一半证据（WSL 里先确认 nvidia-smi 能跑）")
        ok = False
    print("  结果：" + ("PASS" if ok else "未通过，先修口径再写报告"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="results/d9_requests.csv")
    ap.add_argument("--meta", default="results/d9_level_meta.csv")
    ap.add_argument("--out", default="results/d9_summary.csv")
    ap.add_argument("--check", nargs="?", const="__same__",
                    help="只做验收；给路径则读该汇总表，不给则先算再验")
    args = ap.parse_args()

    if args.check and args.check != "__same__":
        rows = load(args.check)
        INT_FIELDS = ("np", "req", "ok", "fail", "n_out_med", "prompt_n_med",
                      "vram_base", "vram_peak", "vram_delta", "ctx_slot", "c_ctx")
        for r in rows:
            for k, v in list(r.items()):
                if k == "note" or v in ("", None):
                    if v in ("", None) and k != "note":
                        r[k] = None
                    continue
                try:
                    r[k] = int(float(v)) if k in INT_FIELDS else float(v)
                except (TypeError, ValueError):
                    pass
        print_table(rows)
        for line in verdicts(rows):
            print(("  " if not line.startswith("  ") else "") + line)
        sys.exit(0 if check(args.check, rows) else 1)

    rows = build(args.requests, args.meta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"→ {args.out}\n")
    print_table(rows)
    print("\n== 判读 ==")
    for line in verdicts(rows):
        print(line)
    if args.check:
        sys.exit(0 if check(args.out, rows) else 1)


if __name__ == "__main__":
    main()
