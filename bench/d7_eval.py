#!/usr/bin/env python3
"""bench/d7_eval.py — D7 量化质量盲评工具箱（只用标准库，不引入新依赖）

四个子命令，按顺序用：

  collect   把 10 道题喂给当前正在跑的 llama-server，记录答案
              python3 bench/d7_eval.py collect --label q4_0 --base-url http://127.0.0.1:8081

  blind     把三档答案打乱成 A/B/C，生成盲评稿和打分表（打分时看不到档位）
              python3 bench/d7_eval.py blind

  unblind   打分表填完后解盲，算出每档均分
              python3 bench/d7_eval.py unblind

  summary   把速度（d7_quant_bench.csv）+ 显存（d7_vram.csv）+ 质量（d7_quality.csv）
            合成“量化-速度-质量”三列对照表
              python3 bench/d7_eval.py summary

  check     检查答案是否 3 档 × 10 题齐全
              python3 bench/d7_eval.py check

口径说明（写报告时必须带上）：
  temperature=0、top_p=1、seed=42、n_predict 按题给、stream=false、cache_prompt=false
  → 同一题在三档上的差异只来自模型文件，不来自采样随机性。
  盲评：每题三档答案顺序独立打乱，打分时只知道 A/B/C；先写一句话理由再给分。
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
import zlib

ANSWER_FIELDS = ["quant", "qid", "rep", "latency_ms", "n_out", "text"]
SCORE_FIELDS = ["qid", "label", "fact", "format", "useful", "note"]
DEFAULT_QUANTS = ["q4_0", "q4_k_m", "q8_0"]


# ---------- 基础 IO ----------

def load_prompts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, rows):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# 本地服务绕开系统代理：http_proxy 一开，127.0.0.1 也会被转发出去（会收到网关 404）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------- collect ----------

def call_chat(base_url, prompt, n_predict):
    payload = {
        "messages": [
            {"role": "system", "content": prompt.get("system", "")},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "n_predict": int(prompt.get("n_predict", n_predict)),
        "stream": False,
        "cache_prompt": False,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    try:
        data = post_json(url, payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        # 有些版本不认 cache_prompt / seed，退化成最小载荷再试一次
        if e.code in (400, 422, 500):
            minimal = {k: v for k, v in payload.items()
                       if k in ("messages", "temperature", "n_predict", "stream")}
            data = post_json(url, minimal)
        else:
            raise SystemExit(f"  HTTP {e.code}: {body}")
    dt = (time.time() - t0) * 1000.0
    text = data["choices"][0]["message"]["content"]
    n_out = (data.get("usage") or {}).get("completion_tokens") or len(text)
    return text.strip(), dt, n_out


def cmd_collect(a):
    prompts = load_prompts(a.prompts)
    rows = []
    print(f"== collect: {a.label} @ {a.base_url} ({len(prompts)} 题 × {a.repeat} 次) ==")
    for rep in range(1, a.repeat + 1):
        for p in prompts:
            text, dt, n_out = call_chat(a.base_url, p, a.n_predict)
            rows.append(dict(quant=a.label, qid=p["id"], rep=rep,
                             latency_ms=round(dt, 1), n_out=n_out,
                             text=" ".join(text.split())))
            print(f"  {p['id']} rep{rep}  {int(dt)} ms  {len(text)} 字  {text[:38]}...")
    append_csv(a.answers, ANSWER_FIELDS, rows)
    print(f"→ 追加 {len(rows)} 行到 {a.answers}")


# ---------- blind ----------

def cmd_blind(a):
    prompts = {p["id"]: p for p in load_prompts(a.prompts)}
    answers = read_csv(a.answers)
    if not answers:
        raise SystemExit(f"{a.answers} 不存在或为空，先跑 collect")
    quants = a.quants
    by_q = {}
    for r in answers:
        by_q.setdefault(r["qid"], {})[r["quant"]] = r["text"]
    missing = [q for q in prompts if len(by_q.get(q, {})) != len(quants)]
    if missing:
        raise SystemExit(f"这些题答案不全：{missing}（每档每题材各要一条）")

    key, score_rows, md = {}, [], []
    md.append("# D7 量化盲评稿（打分时不要翻 d7_key.json）\n")
    md.append("规则：每题三个答案顺序已独立打乱；先看 rubric，再逐条独立打分（1–5），"
              "**不要三档横向比较着打分**，也不要用“字数多少”当质量。\n")
    md.append(f"题量：{len(prompts)} 题 × 3 档 = {len(prompts)*3} 条。打分表：`results/d7_scores.csv`\n")
    md.append("维度：`fact` 事实/正确性、`format` 指令与格式服从、`useful` 现场可用性。\n")

    for qid in sorted(prompts):
        p = prompts[qid]
        order = list(quants)
        random.Random(zlib.crc32(qid.encode())).shuffle(order)
        labels = ["A", "B", "C"][:len(order)]
        key[qid] = {lbl: q for lbl, q in zip(labels, order)}
        md.append(f"\n---\n\n## {qid} · {p.get('tag','')}\n")
        md.append(f"**题目**：{p['user']}\n")
        md.append(f"**该题考查**：{p.get('intent','')}\n")
        md.append(f"**判分锚点**：{p.get('rubric','')}\n")
        for lbl in labels:
            md.append(f"\n**答案 {lbl}**\n")
            md.append("\n".join("> " + ln if ln else ">" for ln in by_q[qid][key[qid][lbl]].splitlines()))
            md.append("")
        for lbl in labels:
            score_rows.append(dict(qid=qid, label=lbl))

    os.makedirs("results", exist_ok=True)
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(a.key, "w", encoding="utf-8") as f:
        json.dump({"quants": quants, "per_question": key}, f, ensure_ascii=False, indent=2)
    write_csv(a.scores, SCORE_FIELDS, score_rows)
    print(f"→ 盲评稿 {a.out_md}（{len(score_rows)} 条待打）")
    print(f"→ 打分表 {a.scores}（fact / format / useful 填 1–5，note 写一句理由）")
    print(f"→ 对照钥匙 {a.key}（解盲前不要打开）")


# ---------- unblind ----------

def cmd_unblind(a):
    scores = read_csv(a.scores)
    key = json.load(open(a.key, encoding="utf-8"))["per_question"]
    filled, out = [], []
    for r in scores:
        vals = []
        for k in ("fact", "format", "useful"):
            v = (r.get(k) or "").strip()
            if v.isdigit() and 1 <= int(v) <= 5:
                vals.append(int(v))
        if len(vals) < 3:
            continue
        avg = sum(vals) / 3.0
        quant = key[r["qid"]][r["label"]]
        filled.append(dict(qid=r["qid"], label=r["label"], quant=quant,
                           fact=vals[0], format=vals[1], useful=vals[2],
                           avg=round(avg, 2), note=r.get("note", "")))
        out.append(filled[-1])
    if not filled:
        raise SystemExit("打分表里还没有一条填满三项的分数")
    write_csv(a.out_csv, ["qid", "label", "quant", "fact", "format", "useful", "avg", "note"], filled)
    total = len(scores)
    print(f"→ 已解盲 {len(filled)}/{total} 条，写入 {a.out_csv}")
    by_q = {}
    for r in filled:
        by_q.setdefault(r["quant"], []).append(r)
    print("\n档位     题数  fact  format useful  综合")
    for q in sorted(by_q):
        rs = by_q[q]
        print(f"{q:8s} {len(rs):>4d}  "
              f"{statistics.mean(r['fact'] for r in rs):.2f}  "
              f"{statistics.mean(r['format'] for r in rs):.2f}  "
              f"{statistics.mean(r['useful'] for r in rs):.2f}   "
              f"{statistics.mean(r['avg'] for r in rs):.2f}")
    if len(filled) < total:
        print(f"\n! 还有 {total-len(filled)} 条没打分，上面的均分只代表已打的部分")


# ---------- summary ----------

def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def cmd_summary(a):
    bench = read_csv(a.bench)
    vram = {r["quant"]: r for r in read_csv(a.vram)}
    quality = read_csv(a.quality)
    quants = a.quants

    speeds = {}
    for r in bench:
        q, t = r["quant"], _f(r.get("test_tps"))
        if q and t:
            speeds.setdefault(q, {})[r["test"]] = (t, _f(r.get("stddev"), 0.0), r.get("ngl", ""))

    qmean = {}
    for r in quality:
        qmean.setdefault(r["quant"], []).append(_f(r.get("avg"), 0.0))

    lines = ["# D7 量化三列对照：体积 / 速度 / 质量\n",
             "口径：llama-bench `-p 512 -n 128 -r 3`，固定 `-ngl`、`-t 8`，同模型同 prompt；"
             "质量分为 10 题盲评 1–5 分均值（fact/format/useful 三项）。\n",
             "| 档位 | 体积 (GiB) | pp512 t/s | tg128 t/s | 显存峰值 (MiB) | 质量均分 | 综合排序 |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for q in quants:
        sp = speeds.get(q, {})
        pp = sp.get("pp512", (None, None, ""))[0]
        tg = sp.get("tg128", (None, None, ""))[0]
        gi = _f(vram.get(q, {}).get("size_gib"))
        vr = vram.get(q, {}).get("vram_peak_mib", "")
        qm = round(statistics.mean(qmean[q]), 2) if qmean.get(q) else None
        rows.append((q, gi, pp, tg, vr, qm))
    rows.sort(key=lambda r: (r[5] is not None, r[5] or 0), reverse=True)
    for i, r in enumerate(rows, 1):
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r[0],
            f"{r[1]:.2f}" if r[1] else "—",
            f"{r[2]:.1f}" if r[2] else "—",
            f"{r[3]:.1f}" if r[3] else "—",
            r[4] or "—",
            f"{r[5]:.2f}" if r[5] is not None else "—",
            i))
    if speeds.get("q4_k_m", {}).get("tg128"):
        tg = speeds["q4_k_m"]["tg128"][0]
        lines.append(f"\n> D2 锚点复核：Q4_K_M tg128 = {tg:.1f} t/s，锚点 39.33 t/s，"
                     f"偏差 {abs(tg-39.33)/39.33*100:.1f}%（>10% 说明今天环境或参数与 D2 不同口径，先查再写报告）。")
    lines.append("\n> 结论槽（自己写，别抄）：①为什么 Q4_K_M 通常比 Q4_0 稳；②Q8_0 质量赢了但为什么不做边缘默认；③你的场景选哪档。")
    os.makedirs("results", exist_ok=True)
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {a.out_md}")
    print("\n".join(lines))


# ---------- check ----------

def cmd_check(a):
    rows = read_csv(a.answers)
    prompts = load_prompts(a.prompts)
    seen = {}
    for r in rows:
        seen.setdefault(r["quant"], set()).add(r["qid"])
    ok = True
    for q in a.quants:
        got = seen.get(q, set())
        miss = [p["id"] for p in prompts if p["id"] not in got]
        status = "OK" if not miss else f"缺 {miss}"
        print(f"  {q:8s} {len(got):>2d}/{len(prompts)}  {status}")
        ok = ok and not miss
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="bench/prompts_d7.jsonl")
    ap.add_argument("--answers", default="results/d7_answers_raw.csv")
    ap.add_argument("--quants", default=",".join(DEFAULT_QUANTS))
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--label", required=True)
    c.add_argument("--base-url", default="http://127.0.0.1:8081")
    c.add_argument("--repeat", type=int, default=1)
    c.add_argument("--n-predict", type=int, default=160)

    b = sub.add_parser("blind")
    b.add_argument("--out-md", default="results/d7_blind.md")
    b.add_argument("--key", default="results/d7_key.json")
    b.add_argument("--scores", default="results/d7_scores.csv")

    u = sub.add_parser("unblind")
    u.add_argument("--scores", default="results/d7_scores.csv")
    u.add_argument("--key", default="results/d7_key.json")
    u.add_argument("--out-csv", default="results/d7_quality.csv")

    s = sub.add_parser("summary")
    s.add_argument("--bench", default="results/d7_quant_bench.csv")
    s.add_argument("--vram", default="results/d7_vram.csv")
    s.add_argument("--quality", default="results/d7_quality.csv")
    s.add_argument("--out-md", default="results/d7_three_col.md")

    sub.add_parser("check")

    a = ap.parse_args()
    a.quants = [x.strip() for x in a.quants.split(",") if x.strip()]
    {"collect": cmd_collect, "blind": cmd_blind, "unblind": cmd_unblind,
     "summary": cmd_summary, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
