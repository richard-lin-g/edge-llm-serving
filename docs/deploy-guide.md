# 部署指南（CPU / GPU / 容器）

> D1 阶段只写占位与命令骨架，D6 / D11 / D12 补实测。

## 1. GPU 原生（WSL2）

```bash
export MODEL=$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf
$HOME/llama.cpp/build/bin/llama-server -m "$MODEL" -ngl 99 -c 4096 --host 0.0.0.0 --port 8080
```

## 2. CPU 原生（笔记本）

```bash
$HOME/llama.cpp/build/bin/llama-server -m "$MODEL" -ngl 0 -t 8 -c 4096 --host 0.0.0.0 --port 8080
```

线程数对照（4/8/12）与内存占用表：见 `docs/benchmark-report.md` 边缘章（D12 补）。

## 3. 容器

- `server/Dockerfile.cpu`：多阶段构建，运行阶段只带二进制与模型挂载点
- `server/Dockerfile.gpu`：CUDA runtime + nvidia-container-toolkit
- `docker compose up` 一条命令起服务（D11 补）

## 4. 已知约束

- 【待填】
