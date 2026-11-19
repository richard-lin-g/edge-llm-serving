# D10 · vLLM 对照：4 GB 卡上的可行性边界

> 标注规则：【算得】= 公式推算；【实测】= 本机跑出来的数。两者不混写。
> 本节结论：vLLM **装得上、起得来、答得对，但跑不快**——所以这里的边界不是"能不能装"，而是"值不值得用"。

---

## 0. 结论先行（一句话）

【实测】vLLM 0.8.5.post1 在本机（NVIDIA T600 4 GB / sm75）能装、能起、能返回正确结果：W4 权重加载 1.95 GiB、KV 池 1.55 GiB，显存装得下；但**单流只有 3.7 tok/s，比同一块卡的 llama.cpp（D9：36–39 tok/s）慢约 10 倍**。瓶颈在推理后端而非显存：sm75 上 V1 引擎不被支持（降级 V0）、FlashAttention-2 不可用（退 XFormers）、AWQ 内核未优化。判定码 **ARCH**（bench/vllm_classify.py 对四份日志一致判 ARCH，触发条件是 sm75 的 V1 不可用告警；性能根因另记作 KERNEL/BACKEND）——属架构边界，不是配置问题。

| 档位 | 参数（ctx / seqs / 其他） | 结果 | 证据文件 | 判定码 |
|---|---|---|---|---|
| L1 | 1024 / 2 / `--enforce-eager` | 起来了；单流 3.7 tok/s，c4 总吞吐 5.9 tok/s | `results/d10_server_L1.log` | ARCH |
| L1b | 1024 / 4 / CUDA graph 开 | 起来了；3.7 tok/s（与 eager **完全同速**），c4 6.5 tok/s | `results/d10_server_L1b_graph.log` | ARCH |
| L2 | 2048 / 4 / graph 开 | 起来了；c1 3.4 / c4 6.5 tok/s，KV 池 1.55 GiB，21.18× 并发 | `results/d10_server_L2_ctx2048.log` | ARCH |
| L3 | 4096 / 1 | 未测：速度已证不是瓶颈，无信息增量 | — | — |
| L4 | 512 / 1 | 未测：同上（KV 池足够，降 ctx 不会变快） | — | — |
| CTRL | 1024 / 4 / Qwen2.5-0.5B fp16（对照） | 起来了；单流 20.7 tok/s，c2 总 24.8 tok/s | `results/d10_server_CTRL05.log` | ARCH |

> 判定码一栏由 bench/vllm_classify.py 给出：四份日志全部判 ARCH。服务本身零报错，是走 V0 回退起来的：能起属服务层，ARCH 属架构层，两者不矛盾。

CTRL 这一档是今天最关键的对照：它证明"3.7 tok/s"不是我配错了，而是**整个后端在 sm75 上都慢**，AWQ 只是在此之上再叠一层。

---

## 1. 显存账

### 1.1 本机实测（`bench/vram_probe.py` + `nvidia-smi`）

| 项 | 值 |
|---|---|
| 卡型号 / 架构 | 【实测】NVIDIA T600 · sm75（compute_cap 7.5） |
| 规格显存 | 4096 MiB |
| 桌面基线占用（服务前） | 【实测】653 MiB（09:58）→ 306 MiB（关掉部分 GPU 应用后） |
| torch 自身上下文 | 【实测】+144 MiB（`baseline_used_mib` 791 − smi 647） |
| 进程可用显存（torch 视角） | 【实测】3291 → 3632 MiB（随桌面释放浮动） |
| vLLM 自报预算 | 【实测】4096 × 0.92 = **3.68 GiB** |
| 服务态峰值 | 【实测】3881–3919 MiB（**余量只剩 177–215 MiB**） |

结论：**规格 4 GB ≠ 可用显存 4 GB**，差的那 300–650 MiB 就是"桌面也在这块卡上"的代价；而服务态只剩 177 MiB 余量，意味着开个浏览器就可能 OOM——这条约束是真实的。
注：`vram_probe.py` 本次的 `ceiling_mib` 字段失真（报 11136 MiB > 4096），已记为待修项，本节以 `nvidia-smi` 读数与 vLLM 自报数为准。

### 1.2 三笔账（权重 + 运行时 + KV）

- `W = 未量化参数 0.31B × 2B + 量化参数 2.78B × bits/8`（embed_tokens / lm_head 一般不量化）
- `K = layers 36 × 2 × kv_heads 2 × head_dim 128 × 2B × ctx × seqs` = **36 KiB/token**（2048 ctx 单序列 72 MiB）
- `C = CUDA context + 激活峰值 + CUDA graph`（`--enforce-eager` 可砍掉 graph 那 200 MiB）

| 精度 | 权重（文件 / 日志） | 运行时开销 | 剩给 KV | 每序列（2048 ctx） | 最多序列 | 判定 |
|---|---|---|---|---|---|---|
| fp16 | 5.76 GiB / — | 0.88 GiB | −3.4 GiB | 72 MiB | 0 | 权重就放不下（未实测，装不下） |
| int8 | 3.17 GiB / — | 0.88 GiB | −0.8 GiB | 72 MiB | 0 | 同上 |
| **awq-w4** | **2.51 GiB 文件 / 1.95 GiB 加载** | **0.03 + 0.15 GiB** | **1.55 GiB（实测）** | 72 MiB | **21（实测 21.18× @2048）** | **起来了** |

两个权重数为什么要都留：磁盘 2.51 GiB 是文件全量（含 fp16 的 embed/lm_head），vLLM 日志的 `Model weights take 1.9536 GiB` 是它实际按量化权重装进显存的部分，口径不同，差 0.56 GiB。

### 1.3 两种 OOM 必须分清（本节最值钱的一句）

| 现象 | 日志原话 | 含义 |
|---|---|---|
| 权重阶段 OOM | `CUDA out of memory. Tried to allocate …` | 连模型都装不进，无降级空间 |
| KV 池 0 block | `No available memory for the cache blocks` | **权重已放好**，是 KvCacheManager 算不出 block；值得再降 ctx 试 |
| graph 捕获 OOM | `Capturing CUDA graphs … out of memory` | 权重+KV 都过了，卡在捕获；`--enforce-eager` 可绕 |

**本次这三类 OOM 一个都没出现。** 所以"4 GB 装不下 vLLM"这个先验判断是被实测推翻的——装得下，问题是装下之后慢。

---

## 2. 4 GB 不是硬墙，是"能装不能跑"

1. **能装**：W4 权重 1.95 GiB + KV 池 1.55 GiB，实测起服务，峰值 3919 MiB。
2. **跑不快**：启动日志三行自证（原文，不改写）：
   ```
   Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0.
   Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
   Using XFormers backend.
   awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
   ```
3. **调参救不了**：`--block-size`、`--preemption-mode`、chunked prefill 全部作用在调度与显存层；3.7 tok/s 是 kernel 层（XFormers + AWQ on sm75）的问题，参数动不到。
4. **反证**：开 CUDA graph 与用 `--enforce-eager` **完全同速**（3.7 → 3.7）。如果瓶颈是发射开销，graph 应该明显更快；没变化说明卡在 kernel 本身。
5. **数量级**：3.7 tok/s = 270 ms/token。按 1.95 GiB 权重 / 160 GB/s 带宽，等效带宽只有约 **7 GB/s（规格的 4.5%）**。
6. **对照**：0.5B fp16（0.93 GiB 权重）单流 20.7 tok/s，也只有理论值的 1/8 —— 后端整体退化，不是某一个量化档的问题。

所以本机的结论不是"vLLM 调一调就能用"，而是**"在这张卡上 vLLM 没有可用的内核路径"**：这是选型问题，不是调优问题。

---

## 3. PagedAttention vs llama.cpp 的预分配 KV

| 维度 | llama.cpp（本机 D8/D9 的用法） | vLLM |
|---|---|---|
| 分配单位 | 每 slot 一整段连续 KV：`-c = np × n_ctx_slot` | KV 池切成 block（默认 16 token/block），按需分配 |
| 浪费来源 | **预留但用不满**：每槽按满 ctx 预留 | **末端碎片**：序列最后一个不足 block 的部分（≤15 token） |
| 序列长度差异 | 差异越大浪费越大（按最长者预留） | 差异几乎不影响占用 |
| 前缀共享 | 关掉 `cache_prompt` 就不共享 | Prefix caching 可共享同一段前缀的 block |
| 上限 | 由 `-c` 与 `-np` 起服务时锁死 | 由启动算出的 block 总数决定，运行中动态分配 |

【算得】同一台机器上的差距：

| 场景 | llama.cpp 占用 | vLLM 占用 | 差额 |
|---|---|---|---|
| 4 序列，各 1024 ctx（D9 的 np4） | 4 × 1024 × 36 KiB = 144 MiB | ≈ 146 MiB（含 4×15 token 碎片） | ≈ 0 |
| 4 序列，长度 200/1800/300/1600 | 4 × 1800 × 36 KiB = 253 MiB | ≈ 139 MiB | **省 114 MiB（45%）** |

【实测】本机的实际情况：ctx1024 给了 2821 个 block（45136 token 容量），我们最多只用到 4 个序列、合计约 400 token —— **池利用率约 1%**。所以 PagedAttention 的收益在这里不是"测不出来"，是"根本用不上"：收益与"长度方差 × 并发数"成正比，而本机只跑 1–4 个短序列。

---

## 4. Continuous batching vs `-np` 静态槽

| 维度 | llama.cpp（`-np`） | vLLM（V0 调度器，本机实测跑的就是 V0） |
|---|---|---|
| 调度粒度 | 请求级：一个请求占一个 slot，生成完才释放 | 迭代级：每个 step 重组 batch，完成的当步退出、排队的当步进来 |
| 长短请求混跑 | 长请求占住 slot，别人等空槽 | 长请求不挡新请求，只占自己的 block |
| prefill 与 decode | 新请求 prefill 挤进同一 batch，拖慢在跑的 decode | chunked prefill（V1 特性；本机 V0 没有） |
| 抢占 | 无（排队等） | 显存不足时抢占，默认 recompute |

【实测】本机 3B W4 的并发曲线（每档 6 请求、64 输出 token）：

| 并发 | TTFT p50 | 单流 decode | 总吞吐 |
|---|---|---|---|
| 1 | 2.0 s | 3.7 tok/s | 3.3 tok/s |
| 2 | 4.4 s | 3.7 tok/s | 5.9 tok/s（1.8×） |
| 4 | 20.0 s | 3.7 tok/s | 6.5 tok/s（2.0×） |

单流速度一点没变（3.7 恒定），总吞吐只涨到 2 倍，但 TTFT 从 2 秒劣化到 20 秒。这与 D9 在 llama.cpp 上看到的病是同一个：**排队时间全叠到 TTFT 上**（D9 np16 时 TTFT P95 从 1131 ms 涨到 15161 ms）。差别在于 llama.cpp 是"没槽就排队"，而这里是"批太大、prefill 被串起来"。

结论：本机能观察到"并发不加速、只加延迟"，但**观察不到 continuous batching 该有的收益**——因为 V1 引擎在 sm75 上根本没启用。

---

## 5. 数据怎么用（以及怎么不能用）

| 口径项 | llama.cpp（D9 表） | vLLM（本机实测） | 能不能并排 |
|---|---|---|---|
| 量化 | GGUF Q4_K_M（2.1 GB） | AWQ W4A16（2.51 GiB 文件 / 1.95 GiB 加载） | **不能**：量化算法不同，质量与速度都不可比 |
| KV 布局 | 连续预分配 | paged block（16 token/block） | 不能：显存占用口径不同 |
| 权重格式 | GGUF（mmap） | safetensors（整块加载） | 不能：加载路径不同 |
| 引擎版本 | commit 785322d17 | **vLLM 0.8.5.post1 / torch 2.6.0+cu124 / V0 引擎 / XFormers 后端** | 必须写进表注 |
| 并发上限 | `-np 16` 实测 | `max-num-seqs 4`（KV 理论允许 21） | 不能：上限本身不同 |

允许写的只有三句：

1. 【实测】本机在 AWQ W4、ctx=1024、`max-num-seqs`=2–4 下，vLLM 能启动并返回流式响应（证据：`results/d10_server_L1*.log` + `results/d10_vllm_bench.csv`）。
2. 【实测】该配置下单流 3.7 tok/s、c4 总吞吐 6.5 tok/s、c4 TTFT p50 20.0 s。**不与 D9 表比较**，只作"最小可运行证据"。
3. 【实测】并发 1→4 总吞吐仅 1.97×，而 TTFT 放大 10 倍，因此 continuous batching 的收益在本机不可测（V1 未启用）。

反面清单（绝对不能写）：不能说"vLLM 比 llama.cpp 慢 10 倍"，只能说"两组数的量级差约 10 倍，且量化与引擎不同，不能算加速比"。

---

## 6. 要多大卡才做得成真对照（【算得】，util 0.92、桌面 0.30 GiB、2048 ctx）

| 卡显存 | 预算 | fp16（5.76 GiB）能跑几个 | awq（1.95 GiB）能跑几个 | 能否做真对照 |
|---|---|---|---|---|
| 4096 MiB | 3.38 GiB | 0 | 20（**实测 21.18×，吻合**） | 只能证明"能跑"，量不出调度收益 |
| 6144 MiB | 5.23 GiB | 0 | 46 | 勉强，单序列为主 |
| 8192 MiB | 7.05 GiB | 18 | 72 | 可以：fp16 + 多序列，能看出调度差异 |
| 12288 MiB | 10.78 GiB | 71 | 125 | 舒服，可做 16 并发对照 |
| 16384 MiB | 14.50 GiB | 124 | 178 | 可复现论文级曲线 |

结论：**vLLM 做真对照的门槛是 8 GB（3B fp16、≥8 序列、长短混杂）**；4 GB 这一档只能拿到"能跑多快"这个坏消息。

---

## 7. 面试口径（三句，中文 + 英文各一遍）

1. 中文：`我在 4 GB 的 T600 上把 vLLM 装起来并跑通了，但单流只有 3.7 tok/s，同一块卡上 llama.cpp 是 36–39 tok/s。原因写在启动日志里：sm75 不支持 V1 引擎、FlashAttention-2 不可用退到 XFormers、AWQ 内核未优化——这是内核路径缺失，不是显存不够，也不是调参能解决的。`
   English: `I did get vLLM running on a 4 GB T600, but single-stream throughput was only 3.7 tok/s versus 36–39 tok/s for llama.cpp on the same card. The startup log says why: V1 isn't supported below compute capability 8.0, FlashAttention-2 falls back to XFormers, and the AWQ kernel is unoptimized. It's a missing kernel path, not a memory or tuning problem.`
2. 中文：`显存账反而是够的：W4 权重 1.95 GiB、KV 池 1.55 GiB，2048 ctx 能开 21 个序列；所以我把结论从"装不下"改成了"装得下但不值得用"，并且用 0.5B fp16 对照证明慢的不是量化档，是整个后端。`
   English: `Memory was actually fine — 1.95 GiB of W4 weights plus a 1.55 GiB KV pool, 21 sequences at 2048 tokens. So I revised the conclusion from "doesn't fit" to "fits but isn't usable", and a 0.5B fp16 control run showed the whole backend is slow, not just the quantized path.`
3. 中文：`要看出 PagedAttention 和 continuous batching 的真实收益，得换 8 GB 以上、sm80 以上的卡，用 fp16、长短混杂的 prompt 做对照，否则调度收益和量化差异混在一起，结论不成立。`
   English: `To measure what PagedAttention and continuous batching actually buy, I need an 8 GB sm80+ card and mixed-length prompts with fp16, otherwise scheduling gains stay confounded with quantization differences.`

---

## 8. 数据声明的正确写法（本节可直接复用）

> 本节 vLLM 数据均为**本机实测**（`results/d10_vllm_bench.csv` 与 `results/d10_server_*.log`），配置为 AWQ W4A16、ctx 1024/2048、`max-num-seqs` 2–4、vLLM 0.8.5.post1（V0 引擎 + XFormers 后端）。该数据**只与自身可比**，不与 D9 的 llama.cpp 表并排：量化算法（AWQ vs GGUF Q4_K_M）、KV 布局（paged vs 连续预分配）、权重格式（safetensors vs GGUF）与引擎版本均不同。若需跨引擎比较，必须先统一架构（sm80+）、量化位宽与 ctx 长度。

---

## 9. 附：安装记录与三个坑（详细版见 `docs/d10_vllm_实测.md`）

1. **PEP 668**：Ubuntu 26.04 的系统 Python 3.14 被 `externally-managed-environment` 锁死 → 用 `python3 -m venv ~/.uvboot` 建引导环境装 uv，再由 `uv python install 3.12` 拉独立 CPython 3.12.16（4.28 秒，无需 sudo、不动系统包）。
2. **transformers 版本**：vLLM 0.8.5 不锁上限，uv 装了 5.23.0 → 启动即崩 `Qwen2Tokenizer has no attribute all_special_tokens_extended`；钉 `transformers==4.51.3` 解决（**这不是 OOM，别误判**）。
3. **WSL 服务超时**：压测时 GPU 打满并发开多个 `wsl.exe` 会话，触发 `Wsl/Service/WSAETIMEDOUT`；`wsl --shutdown` 后恢复，压测请串行单会话跑。