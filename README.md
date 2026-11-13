# EdgeServe · 工业边缘大模型推理服务平台

> 在「有 GPU 的台式机」和「只有 CPU 的笔记本」两种环境下，把一个大模型部署成**可调用、可压测、可容器化交付**的推理服务，并给出可解释的性能—精度权衡表。

本仓库只证明四件事，也是目标岗位（AI 推理部署 / AI Infra / 边缘推理）的四条硬要求：

| # | 能力 | 本仓库里的证据 |
|---|---|---|
| 1 | 会部署 | llama.cpp 在 GPU / CPU 双环境跑通，暴露 OpenAI 兼容接口（SSE 流式） |
| 2 | 会量化 | Q4_0 / Q4_K_M / Q8_0 的速度—质量对照，口径统一、可复现 |
| 3 | 会压测 | 并发 1/4/8/16 下的 QPS、P50/P95、TTFT、TPOT、显存/内存峰值 |
| 4 | 会交付 | CPU / GPU 双 Docker 镜像、C++17 客户端、部署文档、演示视频 |

**不做的事**：不训练、不微调、不搭 RAG、不搞 K8s 集群。

---

> **状态（2026-11-12）**：开发机 Ubuntu 曾整机重装，环境与原始 CSV 一度丢失；
> 仓库代码与历史基准数据已由备份恢复，恢复过程与"哪些数字要重测"见 `RECOVER.md`。

## 实验环境（D1 实测填写，不要估算）

| 项 | 实测值 |
|---|---|
| 设备 / OS | 【台式机 · Windows 11】 |
| GPU 型号 | NVIDIA T600（Turing，compute capability 7.5） |
| 显存总量 | 4096 MiB |
| Windows 驱动 / CUDA | 572.83 / 12.8 |
| WSL | Ubuntu 22.04（GPU 直通已验证） |
| CPU / 内存 | 未取得（`scripts/collect_env.sh` 可补） |
| 编译器 | gcc + CUDA 12.6（nvcc V12.6.85） |
| llama.cpp | b12033-785322d17，`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` |
| 主模型 / 量化档 | Qwen2.5-3B-Instruct Q4_K_M（2 104 932 768 B） |

> 实验边界：显存决定主模型档位，见 `docs/env.md` 的「显存档位决策」一节。

## 目录结构

```
edge-llm-serving/
├── README.md               # 本文件
├── docs/
│   ├── env.md              # 环境记录（D1 填）
│   ├── benchmark-report.md # 量化 / KV Cache / 并发 / vLLM 对照 四章
│   ├── deploy-guide.md     # CPU / GPU / 容器 三种部署方式
│   ├── img/                # 截图（nvidia-smi、终端输出）
│   └── log/                # 每日开发日志 d1.md … d18.md
├── server/                 # 启动脚本、FastAPI 代理、Dockerfile.cpu / Dockerfile.gpu
├── client-cpp/             # C++17 流式客户端（cpp-httplib + nlohmann/json）
├── bench/                  # 冒烟、卸载层扫描、显存轮询、压测脚本
├── demo/                   # 工业故障问诊 Demo（自编故障码表 + 提示词模板）
├── results/                # 原始 CSV 与压测日志（本地留存，CSV 不入库）
└── scripts/                # 环境采集等辅助脚本
```

## 快速开始

待 D4（OpenAI 兼容服务）完成后补齐；D1 阶段只用 `bench/` 下的脚本做冒烟与记录：

```bash
# 冒烟一次（GPU 全量卸载）
export LLAMA_CLI=$HOME/llama.cpp/build/bin/llama-cli
export MODEL=$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf
./bench/run_smoke.sh "$MODEL" 99

# 卸载层数扫描（D2）
LEVELS="0 8 16 24 32 99" ./bench/ngl_sweep.sh "$MODEL"
```

## 首批实测（D1，口径见 docs/log/d1.md）

| 配置 | prompt 处理 | 生成 | 显存峰值 |
|---|---|---|---|
| `-ngl 99`（GPU） | 99.3 t/s | 40.0 t/s | 2800 MiB / 4096 |
| `-ngl 0`（CPU） | 87.5 t/s | 21.7 t/s | — |

GPU 相对纯 CPU 仅 **1.84×**：3B Q4 这一档是显存带宽瓶颈（T600 ≈ 80 GB/s，理论解码上限约 40 tok/s）。后续优化的重点在批处理与并发，不在算力。

## 18 天进度

| 周 | 主题 | 日志 |
|---|---|---|
| W1 D1–D6 | 跑通 + 建度量 | `docs/log/d1.md` … |
| W2 D7–D12 | 量化 + 并发 + 容器 | |
| W3 D13–D18 | C++ 深度 + Demo + 面试素材 | |

## 已知限制

- 4 GB 显存档：3B 可全量卸载（峰值 2.8 GB），7B 只能部分卸载且尚未实测
- vLLM 在 4 GB 卡上预计无法运行，D10 只做约束分析，不外推数据
- 单请求对比只反映 batch=1 场景，并发结论待 D9
