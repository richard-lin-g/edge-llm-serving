仓库：https://github.com/richard-lin-g/edge-llm-serving

# EdgeServe · 工业边缘大模型推理服务平台

> 在「有 GPU 的台式机」和「只有 CPU 的笔记本」两种环境下，把大模型部署成可调用、可压测、可容器化交付的推理服务，并给出可解释的性能-精度权衡表。

## 实验环境（D1 实测）

| 项 | 实测值 |
|---|---|
| GPU | NVIDIA T600，compute capability 7.5 |
| 显存 | 4096 MiB（峰值实测 2800 MiB） |
| 驱动 / CUDA | 572.83 / 12.8 |
| 环境 | WSL2 + Ubuntu 22.04，GPU 直通已验证 |
| 编译 | llama.cpp b12033-785322d17，-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 |
| 主模型 | Qwen2.5-3B-Instruct Q4_K_M（2 104 932 768 B） |

## 首批实测（口径见 docs/log/d1.md）

| 配置 | prompt 处理 | 生成 | 显存峰值 |
|---|---|---|---|
| -ngl 99（GPU） | 99.3 t/s | 40.0 t/s | 2800 MiB |
| -ngl 0（CPU） | 87.5 t/s | 21.7 t/s | — |

GPU 相对纯 CPU 仅 1.84×：3B Q4 这一档是显存带宽瓶颈（理论解码上限约 40 tok/s），后续优化空间在批处理与并发。

## 目录

docs/(env、benchmark-report、deploy-guide、log) · server/ · client-cpp/ · bench/ · demo/ · results/ · scripts/

## 已知限制

- 4 GB 显存：3B 可全量卸载；7B 只能部分卸载，尚未实测
- vLLM 在 4 GB 卡上预计无法运行，D10 只做约束分析，不外推数据
