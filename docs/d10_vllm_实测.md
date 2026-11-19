# D10 · vLLM 在本机（NVIDIA T600 4 GB / sm75）的实测结论

日期：2026-11-19（D10，周四，台式机）
环境：Ubuntu 26.04.1（WSL2）、Python 3.12.16、vLLM 0.8.5.post1、torch 2.6.0+cu124、驱动 595.79、GPU 4096 MiB（实测桌面基线占用 306–653 MiB）

## 一句话结论

**能装上、能起服务、能出正确结果；但在 sm75 上 vLLM 的推理后端全面退化，单流 3.7 tok/s，比同一块卡的 llama.cpp（D9 实测 36–39 tok/s）慢约 10 倍，没有实用价值。**
判定码：**KERNEL/BACKEND**——不是显存不够，是内核与注意力后端在这个架构上没有优化路径。

## 1. 装了什么（可复现）

| 步骤 | 命令要点 | 结果/坑 |
|---|---|---|
| Python 3.12 | 系统 Python 是 3.14（PEP 668 锁死，`pip install uv` 直接报 `externally-managed-environment`）→ `python3 -m venv ~/.uvboot` + 在 venv 内装 uv（venv 不受 PEP 668 限制）→ `uv python install 3.12` | 4.28 秒装好 CPython 3.12.16，无需 sudo、不动系统包 |
| venv | `uv venv --python 3.12 --seed ~/venv-vllm` | venv 体积 7.6 GB（CUDA 轮子占大头） |
| vLLM | `uv pip install vllm==0.8.5.post1 -i 清华源` | 下载约 4.3 GB（torch 731 MB + cudnn 634 MB + vllm 311 MB + cublas/cufft/nccl…），耗时约 45 分钟 |
| **坑 1** | vLLM 0.8.5 不锁 transformers 上限，uv 装了 5.23.0 | 启动即崩：`Qwen2Tokenizer has no attribute all_special_tokens_extended` → **钉 `transformers==4.51.3`**（连带 tokenizers 0.21.4、huggingface-hub 0.36.2） |
| 权重 | `Qwen/Qwen2.5-3B-Instruct-AWQ`（魔搭） | 磁盘 **2.51 GiB**；vLLM 日志报加载权重 **1.95 GiB**（差额是 fp16 的 embed/lm_head 等未计入的部分）。原 3 个 GGUF 无法用于 vLLM |
| 对照模型 | `Qwen/Qwen2.5-0.5B-Instruct`（fp16，954 MB） | 用来区分"AWQ 内核慢"还是"整个后端慢" |

## 2. 三档配置实测

固定参数：`--gpu-memory-utilization 0.92 --dtype float16 --quantization awq --max-num-batched-tokens`（=ctx）。压测：同一提示词（约 100 token）、`max_tokens=64`、`temperature=0`、SSE 流式，每档 6 请求。

| tag | ctx | max-num-seqs | CUDA graph | 并发 | TTFT p50 (ms) | 端到端 p50 (ms) | 总输出 tok/s | 单流 tok/s | 显存峰值 (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| L1_eager | 1024 | 2 | 关（eager） | 1 | 1962 | 19076 | 3.3 | 3.7 | 3917 |
| L1_eager | 1024 | 2 | 关 | 2 | 4061 | 21282 | 5.9 | 3.7 | 3917 |
| L1_eager | 1024 | 2 | 关 | 4 | 25138 | 42366 | 5.9 | 3.6 | 3919 |
| L1b_graph | 1024 | 4 | **开** | 1 | 2150 | 19045 | 3.3 | 3.7 | 3904 |
| L1b_graph | 1024 | 4 | 开 | 2 | 4449 | 21434 | 5.9 | 3.7 | 3903 |
| L1b_graph | 1024 | 4 | 开 | 4 | 20034 | 37103 | 6.5 | 3.7 | 3905 |
| L2_ctx2048 | 2048 | 4 | 开 | 1 | 1859 | 18764 | 3.4 | 3.7 | 3897 |
| L2_ctx2048 | 2048 | 4 | 开 | 4 | 20030 | 37078 | 6.5 | 3.7 | 3895 |
| CTRL_0.5B fp16 | 1024 | 4 | 开 | 1 | 462 | 3547 | 17.6 | 20.7 | 3903 |
| CTRL_0.5B fp16 | 1024 | 4 | 开 | 2 | 912 | 5186 | 24.8 | 15.0 | 3903 |

（原始数据：`results/d10_vllm_bench.csv`；单流日志：`results/d10_server_*.log`）

## 3. 从日志里读到的事实（这才是今天的价值）

启动时 vLLM 自己就交代了三件事（`results/d10_server_L1.log`）：

1. `Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0.` —— **sm75 直接降级到旧引擎 V0**，V1 的 continuous batching/调度优化全部拿不到。
2. `Cannot use FlashAttention-2 backend for Volta and Turing GPUs. Using XFormers backend.` —— **注意力走 XFormers 回退实现**。
3. `awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.` —— **AWQ 内核在 sm75 上不是优化路径**。

三点叠加，对应实测：
- 单流 3.7 tok/s = 270 ms/token。按 1.95 GiB 权重 / 160 GB/s 带宽算，等效带宽只有 **约 7 GB/s（规格的 4.5%）**。
- 开 CUDA graph 与关掉 eager **完全没差别**（3.7 → 3.7），说明瓶颈不在发射开销，而在 kernel 本身。
- 0.5B fp16 对照 20.7 tok/s：即使不量化、模型只有 0.95 GiB，也只有理论值的 1/8 —— **后端整体就慢，AWQ 是雪上加霜**。
- 并发几乎不涨吞吐（1→4：3.3 → 6.5 tok/s），但 TTFT 从 2 s 劣化到 20 s；V0 在 4 并发时把 prefill 串成批，用户等待直接翻十倍。

## 4. 显存账（实测替换推演）

| 项 | 实测值 |
|---|---|
| 卡总量 | 4096 MiB |
| 桌面基线占用 | 306 MiB（早间 653 MiB，关掉部分 GPU 应用后会回落） |
| torch 上下文（探针内） | +144 MiB |
| vLLM 预算（util 0.92） | 3.68 GiB |
| 权重 | 1.95 GiB |
| 非 torch 内存 | 0.03 GiB |
| 激活峰值 | 0.15 GiB |
| **KV 池** | **1.55 GiB**（ctx 1024：2821 blocks，44.08× 并发；ctx 2048：2711 blocks，21.18×） |
| 服务态峰值 | 3881–3919 MiB（**余量 177 MiB**） |

**结论修正**：开工卡上"W4 只剩 0.26 GiB KV"的推演是保守的（按 2.51 GiB 文件体积算）。vLLM 实际只吃 1.95 GiB 权重，KV 池能给到 1.55 GiB，**显存不是今天的限制项，速度和内核才是**。但余量只有 177 MiB，桌面一开浏览器就可能 OOM——这条约束成立。

## 5. 与 D9（llama.cpp）并排（只做量级对照，不并表）

| 指标 | llama.cpp Q4_K_M（D9，同一块 T600） | vLLM 0.8.5 AWQ W4（D10） | 倍差 |
|---|---|---|---|
| 单流 decode | 36–39 tok/s | 3.7 tok/s | ≈ 10× |
| 8/4 并发总吞吐 | 0.58–0.60 QPS（np8） | 0.10 QPS（c4） | ≈ 5.8× |
| 显存峰值 | 2698–2976 MiB（np8） | 3895 MiB | 更高 |
| 引擎/后端 | CUDA 后端 + GGUF 量化 | V0 + XFormers + AWQ | 架构不同 |

**不可并表的原因**：模型格式（AWQ W4A16 vs GGUF Q4_K_M）、tokenizer 与采样默认值不同；两组数只有"量级"意义，不能算加速比。但"同一张卡上 vLLM 慢一个数量级"这个方向是稳的，且有 0.5B 对照佐证不是偶然。

## 6. 运维侧踩到的坑（写进 log 供后面复用）

1. **PEP 668**：Ubuntu 26.04 的系统 Python 3.14 不能直接 pip，用 venv 引导 uv 是零 sudo 的正解。
2. **transformers 版本**：vLLM 0.8.x 必须钉 4.5x，否则 tokenizer 初始化就崩（不是显存问题，别误判成 OOM）。
3. **WSL 服务超时**：压测期间出现过一次 `Wsl/Service/WSAETIMEDOUT`，新会话开不进来；`wsl --shutdown` 后恢复。同时开多个 `wsl.exe` 会话 + GPU 打满时更容易触发，压测脚本请串行跑。
4. **长任务保活**：服务与下载都用 `setsid nohup` 起，会话断开不会被带走。

## 7. 今天的判定与下一步

- **D10 判定码：KERNEL/BACKEND（非 OOM、非 DRIVER、非 PORT）。**
- **本机不建议再投时间在 vLLM 服务化**：显存能塞下，但 3.7 tok/s 没法用；调 block-size、preemption、chunked prefill 都动不了 kernel 层。
- 想继续用 vLLM 的机制价值，只剩两条路：(a) 换 ≥sm80 且显存 ≥8 GB 的卡（V1 + FlashAttention 才生效）；(b) 只做"机制对照"的文字材料，不追数字。
- **D11（Docker 化）建议以 llama.cpp 为镜像主体**，vLLM 镜像只作"能构建、能起、慢"的证据留档。
- 边缘侧最终选型：**llama.cpp（GGUF Q4_K_M）**，vLLM 留给有 sm80+ 的服务器。

## 8. 未做的（明确留白，不编数据）

- 未测 ctx 4096（KV 还够，但速度已证明不是瓶颈，无信息增量）。
- 未测 8+ 并发（KV 允许 21×，但 c4 已经证明吞吐不涨）。
- 未尝试 `VLLM_USE_V1=1` 强开（vLLM 明确拒绝 cc<8.0）。
- 未测 3B fp16（5.76 GiB 权重超出 4 GB，装不下，属已知约束）。