# bench 脚本说明

| 脚本 | 用途 | 阶段 |
|---|---|---|
| `run_smoke.sh` | 单次冒烟：跑一次固定 prompt，落日志并抽取 load/eval 时间 | D1 |
| `ngl_sweep.sh` | 扫 `-ngl` 0/8/16/24/32/99，输出 `results/ngl_sweep.csv` | D2 |
| `gpu_watch.sh` | 轮询显存占用，输出峰值 | D1–D9 |
| `metrics.py` | 走 OpenAI 兼容接口，测 TTFT / TPOT / 输出 tokens/s | D3–D4 |

约定：
- 环境变量 `LLAMA_CLI`（默认 `$HOME/llama.cpp/build/bin/llama-cli`）、`MODEL`（模型路径）
- 所有原始日志写 `results/`，CSV 供报告直接引用
- 每次跑之前先确认没有别的进程占着显存（`nvidia-smi`）
