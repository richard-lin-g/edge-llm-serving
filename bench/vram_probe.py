#!/usr/bin/env python3
"""bench/vram_probe.py — 实测「这块 4 GB 卡到底能给我多少显存」，不装模型，30 秒。

为什么必须实测：`nvidia-smi` 的 4096 MiB 是物理规格，不是可用值。
WSL 里 Windows 桌面合成、浏览器、别的 CUDA 进程都占着同一块 VRAM；
vLLM 的 `--gpu-memory-utilization` 又是按「本进程看到的 total」算预算的。
报告里该写的是实测天花板，不是规格书上的数字。

做法：按 64 MiB 粒度往上吃，一直吃到 CUDA 抛 OOM，回退一格，
记录「峰值 + 基线 + 天花板」，然后全部释放。这个数就是 mem_budget.py 的 --usable-mib。

用法：
  source ~/venv-vllm/bin/activate       # 先有 torch，才有这个工具
  python3 bench/vram_probe.py           # 出表 + 结论
  python3 bench/vram_probe.py --json    # 只出 JSON，便于落盘
  python3 bench/vram_probe.py --no-torch  # 没装 torch：只读 nvidia-smi 的 free
"""
import argparse
import json
import subprocess
import sys
import time

MIB = 1024 ** 2


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
                            "driver_version,compute_cap",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        parts = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        keys = ["name", "total_mib", "used_mib", "free_mib", "driver", "compute_cap"]
        d = dict(zip(keys, parts))
        for k in ("total_mib", "used_mib", "free_mib"):
            d[k] = int(float(d[k]))
        return d
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def probe(step_mib=64):
    import torch
    res = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        res["error"] = "torch.cuda.is_available() = False（驱动/WSL 映射问题，先查 nvidia-smi）"
        return res
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    res.update({
        "device": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_mib": props.total_memory // MIB,
    })
    torch.cuda.empty_cache()
    base_free, base_total = torch.cuda.mem_get_info()
    res["baseline_used_mib"] = (base_total - base_free) // MIB

    blocks = []
    peak = 0
    try:
        while True:
            blocks.append(torch.empty(step_mib * MIB, dtype=torch.uint8, device="cuda"))
            peak = torch.cuda.memory_allocated() // MIB
    except RuntimeError:
        pass
    res["ceiling_mib"] = peak
    res["free_after_probe_mib"] = torch.cuda.mem_get_info()[0] // MIB
    res["note"] = "peak 为本进程实际拿到的上限；torch 缓存分配器按段申请，误差 ±64 MiB"
    del blocks
    torch.cuda.empty_cache()
    res["released_mib"] = torch.cuda.mem_get_info()[0] // MIB
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-mib", type=int, default=64)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()

    smi_info = smi()
    info = {"smi": smi_info}
    if not args.no_torch:
        try:
            info.update(probe(args.step_mib))
        except ImportError:
            info["error"] = "没有 torch：先走 scripts/d10_pip.sh，或加 --no-torch"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    print("\n== 显卡与显存实测 ==")
    s = info["smi"]
    if "error" in s:
        print(f"nvidia-smi 读不到：{s['error']}")
    else:
        print(f"型号        {s['name']}")
        print(f"规格        {s['total_mib']} MiB ｜ 已占 {s['used_mib']} MiB ｜ 空闲 {s['free_mib']} MiB")
        print(f"驱动        {s['driver']} ｜ compute capability {s['compute_cap']}")
    if "device" not in info:
        reason = info.get("error") or "未装 torch（--no-torch 模式），只读了 nvidia-smi"
        print(f"\n探针未完成：{reason}")
        print("（nvidia-smi 的 free 是含桌面占用的毛值，不等于进程可用上限；"
              "要拿到可用值必须先有 torch：bash scripts/d10_pip.sh）")
        return
    print(f"torch       {info['torch']} ｜ CUDA {info['cuda']}")
    print(f"设备        {info['device']} ｜ sm{info['capability'].replace('.', '')} ｜ "
          f"total {info['total_mib']} MiB")
    print(f"基线占用    {info['baseline_used_mib']} MiB（cuda context + 桌面，探针前）")
    print(f"实测上限    {info['ceiling_mib']} MiB ← 这就是 mem_budget.py 的 --usable-mib")
    print(f"释放后可用  {info['released_mib']} MiB\n")
    print("下一步：python3 bench/mem_budget.py --usable-mib "
          f"{info['ceiling_mib']} --sweep")


if __name__ == "__main__":
    main()
