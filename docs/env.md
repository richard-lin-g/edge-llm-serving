# 实验环境（D1 实测）

记录时间：2026-11-09

## 硬件

| 项目 | 实测值 | 备注 |
| --- | --- | --- |
| GPU | NVIDIA T600 | 4096 MiB，Turing 架构（compute capability 7.5），功耗上限 41W |
| 显存总量 | 4096 MiB | 桌面空闲时已占约 683–766 MiB |
| 可用显存 | 约 3.2–3.4 GB | 决定了 3B 可全量卸载、7B 只能部分卸载 |
| 驱动版本 | 572.83 | Windows 侧驱动，WSL 内共用 |
| CUDA 运行时（驱动侧） | 12.8 | 来自 nvidia-smi 右上角 |

## 软件

| 项目 | 版本/路径 | 状态 |
| --- | --- | --- |
| Windows | Windows 11 + PowerShell | 开发与截图记录 |
| WSL | Ubuntu（WSL2，默认发行版） | 已启用，用户 richard |
| WSL GPU 直通 | `nvidia-smi -L` 可见 GPU 0 | 已验证通过 |
| CUDA Toolkit | 12.6.3（`/usr/local/cuda-12.6`） | nvcc 12.6.85，已验证 |
| 环境变量 | `~/.bashrc` 与 `~/.profile` 均加入 `/usr/local/cuda-12.6/bin` | 非交互 shell 也能取到 nvcc |
| 构建工具 | build-essential / cmake / git | 已装 |
| llama.cpp | 待填（编译后补 commit 号） | D1 待完成 |

## 踩坑记录

1. 装完 `cuda-toolkit-12-6` 后 `nvcc` 仍找不到：包已正确安装，只是 PATH 未包含 `/usr/local/cuda-12.6/bin`。
2. 只改 `~/.bashrc` 不够：Ubuntu 的 `.bashrc` 开头对非交互 shell 直接 `return`，`wsl -- bash -lc` 这类命令仍然取不到 nvcc，因此同时在 `~/.profile` 追加。
3. 不要执行 `sudo apt install nvidia-cuda-toolkit`（那是 Ubuntu 自带 11.x 版本，与 12.6 冲突）。

## 显存预算（据此定实验档位）

- 3B Q4_K_M 权重约 2.0 GB：全量卸载可行，建议 `-c 2048` 起步，`-c 4096` 需关闭浏览器等占显存程序。
- 3B Q8_0 权重约 3.3 GB：接近上限，与 4096 上下文同开大概率 OOM，仅作短上下文对照。
- 7B Q4_K_M 权重约 4.7 GB：超出显存，只能部分卸载（`-ngl` 逐层试），用于"显存不足时的降级方案"一节。

## 待补（D1 冒烟后回填）

- llama.cpp commit：____
- `-ngl 99` tokens/s：____
- `-ngl 0` tokens/s：____
- 显存峰值：____ MiB