# EdgeServe 环境重装后的恢复包 · 2026-11-12（D4）

## 一句话结论

**你没退回 D1。** 丢的只是"环境本体"（CUDA、llama.cpp 编译产物、gguf 模型、仓库目录），
**D1–D3 的实验数据、脚本、文档全部有备份**，在这个包里。今天不需要重做任何代码，
只需要：重建编译环境（约 40 分钟）→ 下回模型（约 15 分钟）→ 解压这个包 → 直接做 D4。

## 包里有什么

| 路径 | 是什么 | 来源 |
|---|---|---|
| `README.md`、`docs/`、`bench/`、`client-cpp/`、`demo/`、`server/` | D1 立起来的完整仓库骨架 | 云端备份 |
| `docs/env.md` | D1 实测环境 + 首组数据（40.0 / 21.7 t/s，峰值 2800 MiB） | D1 实测 |
| `docs/log/d1.md` | D1 日志（含 nvcc PATH、CPU-only 静默退化两次踩坑） | D1 实测 |
| `docs/log/d2.md` | D2 六档 `-ngl` 扫描数据（0/10/20/28/32/36，tg 21.71→39.33） | D2 实测 |
| `docs/log/d3.md` | D3 量具说明 + 待重跑的真机表 | 重建 |
| `bench/metrics.py` | 量具（TTFT/TPOT/decode_tps/srv_tg/srv_pp/显存峰值 + `--check`） | D3 成品 |
| `bench/prompts.jsonl` | S/M/L 三档 prompt | D3 成品 |
| `scripts/d3_run.sh` | 起 server → 探活 → 跑测 → 关服务 | D3 成品 |
| `tools/mock_llama_server.py` | 假服务端（无 GPU 也能验链路，已补 chat 接口） | D3 成品，D4 扩 |
| `scripts/recover_env.sh` | **本次新增**：一键重建环境，幂等可重入 | 今天写 |
| `scripts/d4_sse_check.sh` | **本次新增**：D4 验收，证明 SSE 真逐字 | 今天写 |
| `server/openai_demo.py` | **本次新增**：三条调用路径（裸流 / SDK / 非流式） | 今天写 |

## 恢复步骤（三条命令）

```bash
# 1) 解压到 home（若 ~/edge-llm-serving 已存在，先改名留底再解压）
cd ~ && ls -d edge-llm-serving 2>/dev/null && mv edge-llm-serving edge-llm-serving.bak
unzip -o ~/EdgeServe_恢复包.zip -d ~ && ls ~/edge-llm-serving

# 2) 重建环境（幂等：装过的会跳过；要 sudo 密码，全程约 40–60 分钟，日志自动留）
bash ~/edge-llm-serving/scripts/recover_env.sh

# 3) 起服务 + 验 D4
~/llama.cpp/build/bin/llama-server -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    -ngl 36 -c 4096 --host 0.0.0.0 --port 8080        # 终端 A，保持前台
./scripts/d4_sse_check.sh                             # 终端 B（在 ~/edge-llm-serving 下）
```

先跑 `bash scripts/recover_env.sh --dry-run` 可以只看探测结果不动手。

## 一个关键决定：锁定 llama.cpp 版本

`recover_env.sh` 会 `git checkout 785322d17`，也就是 D1/D2 编译时的那个 commit。

**为什么**：如果编了新 HEAD，D2 那六档 `-ngl` 数据和 D3 的锚点（tg 39.33 / pp 152.14）
全部失去可比性，你得重跑一遍 6×3 遍。锁同一版本，这些数字原地生效，
今天只剩"复核一次冒烟"（30 秒）。

如果那个 commit 在远端已经取不到（脚本会明确报警），那就只能用新 HEAD，
此时 `docs/log/d2.md` 的数字**必须标注"旧版本、待重测"**，不能直接拿去讲。

## 重装后必须重新取得的东西（别漏）

| 项 | 原因 | 怎么拿 |
|---|---|---|
| 显存峰值（5 个档） | D2 表里当时就没填全 | `./bench/gpu_watch.sh 120` 另开终端跑 |
| `results/ngl_sweep.csv`、`vram_3b_q4km.csv` | 在原环境，已丢 | `./bench/ngl_sweep.sh` 重跑 |
| D3 真机 `results/d3_metrics.csv` | 已丢 | 起服务后 `bench/metrics.py --repeat 3` |
| `docs/img/` 三张截图 | 在原环境，已丢 | 重截，30 秒的事 |
| GitHub 远端仓库 | 若当时没 push，等于没有 | `git init` + `git remote add` + push |

## 时间账（今天 15:40 起算）

| 段 | 内容 | 时长 | 备注 |
|---|---|---|---|
| 1 | 工具链 + CUDA toolkit | 15–25 min | apt 装 2 GB，主要等下载 |
| 2 | llama.cpp 编译 | 25–40 min | 8 线程；和上一步串行 |
| 3 | 模型下载 2.1 GB | 0–25 min | **可与 1/2 并行**；若 Windows 侧还留着 gguf 直接拷，0 分钟 |
| 4 | 解压 + 起服务 + SSE 验证 | 20–30 min | 已用 mock 端到端验过脚本，不会卡在脚本本身 |
| 5 | 重跑 D3 量具 + 写 log + 提交 | 30 min | 顺带把 D4 和 D3 的真机数据一起补掉 |

关键路径 ≈ **1.5–2 小时**，19:00 前能收工。

## 降级预案（如果卡住）

1. **CUDA 装不上/编不过**：先不编 CUDA 版，`cmake -B build -DGGML_CUDA=OFF` 编 CPU 版，
   今天只验"API + SSE 逐字"这条链路（D4 的验收本来就是协议层），GPU 版挪到明天早上第一件。
2. **模型下不来**：用 D2 那套 `curl -C -` 断点续传，或者去 Windows 侧 `find /mnt/c/Users -iname '*qwen2.5-3b*.gguf'`。
3. **Windows 侧 4 GB 显存被桌面吃掉**：起服务前关浏览器，`-c` 降到 2048。
4. **时间到 18:30 还没起服务**：今天只交「CPU 版服务 + SSE 验证 + 一份诚实的 log（写清哪一步超时、明天第一件事）」，这不丢人，比硬撑到半夜强。

## 一句话给面试用（今天新增的）

> 「环境被我整崩过一次，重装后我用锁 commit 的方式保证历史基准数据仍然可比，
> 只重跑了必要的冒烟——因为**基准数据最怕的不是慢，是口径漂移**。」
