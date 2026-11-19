#!/usr/bin/env python3
"""bench/vllm_classify.py — 把 vLLM 的失败日志判成一句能写进报告的结论。

为什么需要它：D10 允许"跑不起来"，但不允许"跑不起来还说不出为什么"。
vLLM 的失败长得不一样，混为一谈，面试一问就穿：

  WEIGHT_OOM  权重要不进去（fp16 6.18 GB 撞 4 GB 卡）
  KV_ZERO     权重进去了，但 KvCacheManager 算出 0 个 block —— 不报 OOM，报
              "No available memory for the cache blocks"，最容易看错
  GRAPH_OOM   CUDA graph 捕获阶段 OOM（权重已加载，卡在 capture）
  OOM_LATE    权重和 KV 都过了，推理阶段才 OOM（batch/激活太大）
  ARCH        架构不受支持（该版本要求 compute capability > 本机）
  KERNEL      CUDA 内核不可用（轮子没编 sm75）
  DRIVER      驱动-库版本不匹配
  PORT        端口被占（上一轮服务没退干净）
  TORCH_CPU   torch 装成了 CPU 版
  BACKEND     注意力后端在该配置下不兼容

判定方法不是"看到 out of memory 就写权重 OOM"，而是先找**第一处报错**，
再看它前面最近的一个阶段标记（Loading weights / Capturing CUDA graphs / KV cache），
由"错在哪个阶段"决定结论——这是能经得起追问的写法。

用法：
  python3 bench/vllm_classify.py results/d10_server_01.log
  python3 bench/vllm_classify.py results/*.log --json
  python3 bench/vllm_classify.py log.txt --tail 60
"""
import argparse
import glob
import json
import re
import sys

UNIQUE_RULES = [
    # 这些模式一旦命中，就不该被 OOM 抢走结论
    ("KV_ZERO", r"No available memory for the cache blocks|available_kv_cache_memory=0|"
                r"GPU KV cache size:\s*0\n"),
    ("PORT", r"Address already in use|Errno 98"),
    ("TORCH_CPU", r"Torch not compiled with CUDA enabled|Found no NVIDIA driver on your system"),
    ("DRIVER", r"Driver/library version mismatch|CUDA driver version is insufficient|"
               r"CUDA_ERROR_SYSTEM_DRIVER_MISMATCH|requires? (a )?(newer|higher) (NVIDIA )?driver"),
    ("ARCH", r"requires? (compute capability|CUDA capability)|compute capability .* not supported|"
             r"is not supported on this GPU"),
    ("KERNEL", r"no kernel image is available|invalid device function|"
               r"the provided PTX was compiled with an unsupported toolchain"),
    ("BACKEND", r"No supported attention backend|attention backend.*not (supported|compatible)|"
                r"Unknown attention backend"),
]

PHASES = [
    ("graph", r"capturing (the )?cuda graph|capturing cuda graphs"),
    ("weights", r"loading model weights|loading weights took|model weights take"),
    ("kv", r"kv cache|kv_cache"),
]

ERROR_LINE = re.compile(r"ERROR|Traceback|Error:|OutOfMemoryError|RuntimeError|ValueError|"
                        r"AssertionError|CUDA error|out of memory", re.I)
OOM = re.compile(r"out of memory|OutOfMemoryError", re.I)

EXTRACT = [
    ("weights", r"Model weights take ([\d.]+)\s*(GiB|GB)"),
    ("kv_tokens", r"GPU KV cache size:\s*([\d,]+)"),
    ("kv_gib", r"Available KV cache memory:\s*([\d.]+)"),
    ("max_conc", r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"),
    ("gpu_total", r"total capacity of ([\d.]+) GiB"),
    ("gpu_free", r"of which ([\d.]+) (?:GiB|MiB) is free"),
    ("version", r"vLLM version[:\s]+([\d]+\.[\d.]+[\w.]*)|vLLM ([\d]+\.[\d.]+[\w.]*)"),
    ("engine", r"(Using V1|Using V0|VLLM_USE_V1=\d|vLLM V1 engine|vLLM V0 engine)"),
]


def facts_of(text):
    out = {}
    for key, pat in EXTRACT:
        m = re.search(pat, text, re.I)
        if m:
            vals = [g for g in m.groups() if g]
            out[key] = " ".join(vals) if vals else m.group(0)
    return out


def phase_before(lines, err_idx):
    for i in range(err_idx, -1, -1):
        low = lines[i].lower()
        for phase, pat in PHASES:
            if re.search(pat, low):
                return phase
    return None


def writeup(code, f):
    w = f"权重 {f['weights']}" if f.get("weights") else "权重"
    cap = f"本卡总容量 {f['gpu_total']} GiB、错误时仅剩 {f['gpu_free']} GiB" if f.get("gpu_total") else "本卡显存"
    kv = f"可用 KV {f['kv_gib']} GiB / {f['kv_tokens']} tokens" if f.get("kv_gib") else ""
    if code == "WEIGHT_OOM":
        return (f"模型权重加载阶段即 CUDA OOM（日志实测 {w}，{cap}）。vLLM 没有按层部分卸载的可用路径，"
                "因此本机在该精度下不具备可行性；未取得任何服务阶段数据。")
    if code == "KV_ZERO":
        return ("权重加载成功，但 KvCacheManager 未能分到任何 block（No available memory for the cache "
                "blocks），服务未进入可服务阶段" + (f"（{kv}）" if kv else "") +
                "。这说明失败点是 KV 池，不是权重——降 max-model-len / 关 CUDA graph 仍可再试一轮。")
    if code == "GRAPH_OOM":
        return ("权重与 KV 池均已就绪，失败发生在 CUDA graph 捕获：vLLM 为每个 batch size 预留执行缓冲，"
                "在小显存卡上这一步本身就会 OOM。加 --enforce-eager 可绕过，是否能起取决于剩余 KV 预算。")
    if code == "OOM_LATE":
        return ("服务已启动，OOM 发生在推理阶段（batch 激活或临时缓冲），属并发/批大小问题，"
                "应当降 --max-num-seqs 或 --max-num-batched-tokens 后复测。")
    if code == "ARCH":
        return (f"本次所用 vLLM 版本不支持本机 GPU 架构（{f.get('version', '版本见日志')}），启动即被拒，"
                "与显存无关。需换支持 sm75 的版本或用官方 docker 镜像。")
    if code == "KERNEL":
        return "预编译轮子不含 sm75 内核，运行时报 no kernel image；属构建问题，换镜像或源码编译可解。"
    if code == "DRIVER":
        return "CUDA 运行时版本高于驱动支持上限，进程初始化即失败；属环境问题，不是模型/显存问题。"
    if code == "PORT":
        return "端口冲突导致启动失败，本次日志不作为证据。"
    if code == "TORCH_CPU":
        return "环境内 torch 无 CUDA 支持，与显存无关，需重装 cu 版 torch。"
    if code == "BACKEND":
        return "所选 attention backend 在本架构/配置下不兼容，启动中止；可显式指定 TRITON_ATTN 重试。"
    return "日志未匹配已知失败模式，需人工判读；不要凭印象写结论。"


NEXT = {
    "KV_ZERO": "降 --max-model-len（2048→1024）→ 加 --enforce-eager → --max-num-seqs 1 → "
               "--gpu-memory-utilization 0.95；四步都不行定案为显存不足",
    "WEIGHT_OOM": "换 W4 量化权重（AWQ/GPTQ）；仍 OOM 则换 ≥8 GB 卡，或放弃实测改做约束分析",
    "GRAPH_OOM": "必加 --enforce-eager，再降 --max-num-seqs=1 / --max-num-batched-tokens=256",
    "OOM_LATE": "降 --max-num-seqs、降 --max-num-batched-tokens，复测同一档",
    "ARCH": "换支持 sm75 的版本（0.6.x/0.7.x 时代门槛为 7.0）或用官方 docker 镜像",
    "KERNEL": "换官方 docker 镜像（含全架构 PTX）或源码编译 TORCH_CUDA_ARCH_LIST=7.5",
    "DRIVER": "nvidia-smi 看驱动版本 → 换匹配的 cu121/cu124 轮子；WSL 下 Windows 侧驱动也要够新",
    "PORT": "pkill -f vllm 或换 --port",
    "TORCH_CPU": "重装 cu 版 torch",
    "BACKEND": "显式 --attention-backend TRITON_ATTN（或该版本支持的 XFORMERS）",
}


def classify(text, lines=None):
    f = facts_of(text)
    for code, pat in UNIQUE_RULES:
        if re.search(pat, text, re.I):
            return {"code": code, "writeup": writeup(code, f), "next": NEXT[code], "facts": f}

    lines = lines or text.splitlines()
    err_idx = next((i for i, l in enumerate(lines) if ERROR_LINE.search(l)), None)
    if err_idx is None:
        return {"code": "NO_ERROR", "writeup": "日志里没有报错行：可能启动成功，请查 /v1/models 与端口。",
                "next": "python3 bench/vllm_load.py --health", "facts": f}

    err_text = "\n".join(lines[err_idx:err_idx + 6])
    phase = phase_before(lines, err_idx)
    if OOM.search(err_text):
        if phase == "graph":
            code = "GRAPH_OOM"
        elif phase == "weights":
            code = "WEIGHT_OOM"
        else:
            # KV 已按预算分配完、或阶段不明时才 OOM → 属于启动后期/推理期
            code = "OOM_LATE" if phase == "kv" else (
                "WEIGHT_OOM" if "Loading" in err_text else "OOM_LATE")
        return {"code": code, "writeup": writeup(code, f), "next": NEXT[code], "facts": f}

    return {"code": "UNKNOWN", "writeup": writeup("UNKNOWN", f), "next": NEXT.get("UNKNOWN", ""),
            "facts": f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tail", type=int, default=0, help="只看末尾 N 行（0=全读）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = []
    for p in args.logs:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("没有匹配到日志文件", file=sys.stderr)
        sys.exit(2)

    reports = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        if args.tail:
            lines = lines[-args.tail:]
        text = "\n".join(lines)
        verdict = classify(text, lines)
        ev = [f"{i}: {l.strip()[:150]}" for i, l in enumerate(lines, 1)
              if ERROR_LINE.search(l)][-6:]
        reports.append({"file": path, "lines": len(lines), "verdict": verdict, "evidence": ev})

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for r in reports:
        v = r["verdict"]
        print(f"\n== {r['file']}（{r['lines']} 行） ==")
        print(f"判定      {v['code']}")
        if v["facts"]:
            print("抓到的数  " + " ｜ ".join(f"{k}={x}" for k, x in v["facts"].items()))
        if r["evidence"]:
            print("证据行：")
            for e in r["evidence"]:
                print("  " + e)
        print(f"\n报告落笔  {v['writeup']}")
        print(f"下一步    {v['next']}")
    print("\n提醒：WEIGHT_OOM / KV_ZERO / GRAPH_OOM / ARCH 可直接进《约束分析》；"
          "PORT / DRIVER / TORCH_CPU / BACKEND 属环境问题，修好再重试，不能当证据。")


if __name__ == "__main__":
    main()
