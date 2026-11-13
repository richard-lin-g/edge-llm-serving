# 实验环境记录

> 规则：只写实测值。拿不到的项写 `未取得`，不估算、不凭印象。
>
> **版本说明（重要）**：D1–D3 跑在一套后来被整个卸载的 Ubuntu 上；D4（2026-11-12）环境是重装的
> **Ubuntu 26.04.1 LTS**。两份环境事实并列记录，不要混用。
> 下表中标「D1 当时」的，是当时记录；标「D4 重建」的，是今天实测。

## 0. 环境代际对照

| 项 | D1–D3（当时） | D4（今天重建） |
|---|---|---|
| 发行版 | Ubuntu 22.04（当时记录，现已不可复核） | **Ubuntu 26.04.1 LTS（codename resolute）** |
| 系统 gcc | 11.4 | **15.2.0**（另装 gcc-14 14.3.0 备用） |
| nvcc | 12.6.85 | **12.6.85 已装但不可用**（与 gcc 15 不兼容）；实际使用 **13.3.73** |
| llama.cpp | b12033-785322d17（源码自编） | b8681（apt 二进制 + 同源源码自编 CUDA 版） |
| 模型文件 | 2 104 932 768 B | **2 104 932 768 B（字节级一致）** |

## 1. 显卡与驱动（Windows 侧，D4 实测复核）

| 项 | 值 |
|---|---|
| GPU 型号 | **NVIDIA T600**（Turing，896 CUDA 核心，GDDR6） |
| 显存总量 | **4096 MiB**（可用约 4.0 GB） |
| 计算能力（compute capability） | **7.5** → 编译必须用 `-DCMAKE_CUDA_ARCHITECTURES=75` |
| 驱动版本 | **572.83**（Windows 侧） |
| Driver 显示的 CUDA 版本 | **12.8**（`nvidia-smi` 右上角） |
| WSL 内驱动版本 | 570.133.07（`nvidia-smi` 左上角，与 Windows 驱动同源，不用另装） |
| 空闲基线占用 | D1 记录 865 MiB（仅 Xwayland）；D4 重装后实测 581 MiB |

`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`：

```
NVIDIA T600, 4096 MiB, 595.79
```

## 2. WSL 与系统（D4 实测）

| 项 | 值 |
|---|---|
| WSL 版本 | **3.0.5.0** |
| 发行版 | **Ubuntu 26.04.1 LTS（resolute）**，VERSION 2（WSL2） |
| 内核 | **6.18.58.1-microsoft-standard-WSL2** |
| CPU | **12th Gen Intel Core i5-12400F**，**12 核**（`nproc`） |
| 内存 | **7 GiB**（`free -g` total=7，available≈5）→ CUDA 编译只能用 `-j4` |
| 磁盘 | `/dev/sdd` **1007 G**，已用 18 G，剩余 939 G（2%） |
| 用户 / HOME | `richard` / `/home/richard` |

WSL 内 GPU 直通：`nvidia-smi` 能看到同一张卡（4096 MiB，driver **595.79**，CUDA Version **13.2**）。

## 3. 工具链（D4 实测）

| 项 | 值 |
|---|---|
| gcc / g++ | **15.2.0**（`g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`） |
| gcc-14 / g++-14 | **14.3.0**（`/usr/bin/gcc-14`、`/usr/bin/g++-14`） |
| cmake | **4.2.3** |
| python3 | **3.14.4** |
| nvcc（12.6） | **V12.6.85**，`/usr/local/cuda-12.6/bin/nvcc` — **本机不可用** |
| nvcc（13.3） | **V13.3.73**，`/usr/local/cuda-13.3/bin/nvcc` — **实际使用** |
| cublas | `libcublas-13-3` / `libcublas-dev-13-3` |
| Docker | 未装 |

### 3.1 为什么弃用 CUDA 12.6（今天的硬阻塞，已复现）

```text
$ nvcc -c t.cu -o t.o
/usr/local/cuda-12.6/.../crt/host_config.h:143:2: error: #error -- unsupported GNU version!
  gcc versions later than 13 are not supported!
$ nvcc -allow-unsupported-compiler -c t.cu -o t.o
54 errors detected in the compilation of "t.cu".
```

- Ubuntu 26.04 的 gcc 是 15.2.0，**没有 gcc-13 包**（`apt-cache policy gcc-13` 无 Candidate）。
- 装 gcc-14 也没用：CUDA 12.6 的支持上限是 gcc **13**。
- 结论：**CUDA 12.6 在 26.04 上无法编译任何 .cu**，`-allow-unsupported-compiler` 只是绕过版本检查，libstdc++ 头依然解析失败（54 errors）。

### 3.2 CUDA 13.3 与 gcc 15 实测兼容

```text
$ /usr/local/cuda-13.3/bin/nvcc -V
Cuda compilation tools, release 13.3, V13.3.73

$ /usr/local/cuda-13.3/bin/nvcc -c t2.cu -o t2.o
（无输出，产出 t2.o 9768 B）→ 兼容
```

### 3.2.1 运行时卡在驱动版本（当晚已解决：572.83 → 595.79）

CUDA 13.3 **编译**通过，但 `llama-server --version` 启动时报：

```text
ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version
```

原因链条：

| 环节 | 版本 | 说明 |
|---|---|---|
| Windows 侧驱动 | **572.83** | 决定 kernel 驱动能力下限 |
| `nvidia-smi` 显示的 CUDA Version | **12.8** | 驱动支持的**最高 runtime 版本** |
| CUDA Toolkit（编译用） | **13.3.73** | 高于 12.8 → runtime 拒绝初始化 |

结论：**驱动 572.83 只能跑 CUDA ≤ 12.8 的 runtime**。CUDA 13.3 编出来的二进制在这台机器上起不了 GPU 后端。
要跑 13.3，需要把 Windows 侧驱动升到 **580 系列或更高**（Turing 架构仍在支持列表内）。

**当晚已解决**：用户在 Windows 更新里装了显卡可选驱动，落到 **595.79**（`DriverVersion 32.0.15.9579`，2026-03-04）。
WSL 内 `nvidia-smi` 变为 `Driver Version: 595.79  CUDA Version: 13.2`；CUDA 13.3 编译的
`bin/llama-server -ngl 99` 启动即报 `ggml_cuda_init: found 1 CUDA devices ... NVIDIA T600, compute capability 7.5`
并 `load_tensors: offloaded 37/37 layers to GPU` —— **GPU 后端跑通**。
云主机上备选的官方包是 582.08（R580 U6 企业生产分支，712.91 MB，支持列表含 NVIDIA T600），最终没用到。

补充：为什么不回退用 12.8 编译？实测 12.8 在 26.04 上另有头文件冲突：

```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(83): error: exception specification
  is incompatible with that of previous function "cospi" (declared at line 2601 of
  /usr/local/cuda-12.8/.../crt/math_functions.h)
```

glibc 2.43（26.04）给 `cospi / sinpi / rsqrt` 等加了 `noexcept(true)`，与 CUDA 12.8 的 `math_functions.h` 声明冲突。
与 `-std=` 无关（c++11 / c++14 / c++17 全部失败），换 `-ccbin g++-14` 也一样（6 errors）。

**所以 Ubuntu 26.04 上的可行组合只有一种：**

| CUDA | 编译 | 运行（驱动 572.83） |
|---|---|---|
| 12.6 | ✗ gcc 15 不兼容（54 errors） | — |
| 12.8 | ✗ glibc 2.43 头冲突（6 errors，与 gcc 版本无关） | ✓ |
| **13.3** | ✓ gcc 15 通过 | ✗ 需驱动 ≥ 580 |

### 3.3 nvcc 的 PATH（还没写进 `~/.profile`，注意）

当前 `nvcc` 仍不在默认 PATH；两条 CUDA 并存时以 `/usr/local/cuda` 软链指向的版本为准，
**编译时建议显式传 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc`**，不要依赖 PATH。

WSL 的坑：`export PATH=/usr/local/cuda-13.3/bin:$PATH` 会报 `not a valid identifier`，
因为 WSL 继承的 Windows PATH 含空格（`...\User Data\...`）。用完全替换写法：

```bash
env PATH="/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin" cmake ...
```

## 4. llama.cpp（两套并存）

### 4.1 apt 二进制（CPU，先用这个跑通协议层）

| 项 | 值 |
|---|---|
| 包 | `llama.cpp-tools`（Ubuntu 26.04 universe） |
| 版本 | **8681 (Debian)**，`8681+dfsg-1` |
| 二进制 | `/usr/bin/llama-server`、`/usr/bin/llama-cli`、`/usr/bin/llama-bench`、`/usr/bin/llama-quantize`、`/usr/bin/llama-completion` |
| 后端 | `load_backend: loaded CPU backend from /usr/lib/x86_64-linux-gnu/ggml/backends0/libggml-cpu-haswell.so` |
| 用途 | D4 的 OpenAI 兼容接口 + SSE 验收 |

### 4.2 源码自编 CUDA 版

| 项 | 值 |
|---|---|
| 源码来源 | **`apt-get source llama.cpp`**（deb-src 源，`llama.cpp_8681+dfsg.orig.tar.xz`，8.1 MB，6 秒下完） |
| 源码目录 | `/tmp/llama.cpp-b8681` |
| 版本 | **b8681** |
| 配置 | `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF` |
| CMake CUDA 探测 | `Found CUDAToolkit: ... found version "13.3.73"`、`CUDA Toolkit found`、`CMAKE_CUDA_ARCHITECTURES=75` |
| 编译 | `cmake --build build -j4 --target llama-cli llama-server`（内存只有 7 GiB，`-j12` 会 OOM） |

> **版本差异提醒**：D1–D3 的基线是 **b12033-785322d17**，这里是 **b8681**。
> 模型权重字节级一致 → 输出内容可比；但后端实现/默认参数有差异 → **性能数字不能直接并列**。
> 要与 D1–D3 严格可比，需重编 b12033（源码下载受限于 GitHub 可达性）或回退 Ubuntu 22.04。

### 4.3 与 D1 相同的历史踩坑（仍适用）

- CMake 找不到 nvcc 时**静默退化为 CPU-only、不报错** → 必须 `ldd build/bin/llama-cli | grep -E 'cuda|cublas'` 复核。
- 首版编译成功后立刻验证后端，不要直接信 CMake 输出。

## 5. 模型（D4 重新下载，已校验）

| 项 | 值 |
|---|---|
| 主模型 | Qwen2.5-3B-Instruct-GGUF `qwen2.5-3b-instruct-q4_k_m.gguf` |
| 文件大小（实测） | **2 104 932 768 字节 ≈ 2.0 GiB**（与 D1 完全一致） |
| 文件头 | `GGUF`（`head -c 4` 校验） |
| 实际路径 | `/home/richard/models/qwen2.5-3b-instruct-q4_k_m.gguf` |
| 下载源 | `https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/...` |
| 实测速度 | 稳定后 **5.05 MB/s**，6 分 36 秒下完（首次仅 120 KB/s，因 apt 抢带宽） |
| 备选（小显存档） | `qwen2.5-1.5b-instruct-q4_k_m.gguf` ≈ 1.04 GiB（未下） |
| 对照（7B） | `qwen2.5-7b-instruct-q4_k_m-00001/00002-of-00002.gguf`（未下） |

### 5.1 模型下载的坑（今天踩了）

`curl -C -` 断点续传把文件写坏：目标 2 104 932 768 B，实际涨到 2 536 585 632 → 2 647 946 048 B（超出 26%）。

- 原因：续传位置与服务器实际返回长度不匹配，内容被拼接。
- 正确做法：**删文件、不带 `-C -` 干净重下**，下完必须校验字节数。
- 校验命令：

```bash
stat -c%s ~/models/qwen2.5-3b-instruct-q4_k_m.gguf   # 必须等于 2104932768
head -c 4 ~/models/qwen2.5-3b-instruct-q4_k_m.gguf | od -c | head -1   # 必须是 G G U F
```

## 6. 显存档位决策（写死，中途不改）

| 显存 | 主模型 | 对照 | 说明 |
|---|---|---|---|
| ≥ 8 GB | 3B **Q8_0** 全量卸载 | 7B Q4_K_M 全量可试 | 精度优先 |
| 6 GB | 3B Q8_0 全量（3.37 GiB + KV） | 3B Q4_K_M | 留约 1.5 GB 余量 |
| 4 GB | 3B **Q4_K_M** 全量（2.0 GiB + KV） | 3B Q8_0 部分卸载 | 主力档 |
| 2 GB | 1.5B **Q4_K_M** 全量（1.04 GiB） | 3B 部分卸载（-ngl 8~12） | 显存极窄时的边缘真实约束 |

> 本机实测显存：**4096 MiB** → 对应第 **3** 行（4 GB 档），主模型 **Qwen2.5-3B-Instruct Q4_K_M**。
>
> D1 实测峰值 **2800 MiB / 4096 MiB**，余量约 1.3 GB。
> 若 `-ngl 99` 报分配失败：先关掉占显存的程序，再把 `-c` 降到 2048 或 `-ngl` 降到 32/28，并记进报告。

## 7. 性能基线（D1 Block 3，固定口径）

> 口径：模型 3B Q4_K_M｜prompt「你好」｜`-c 4096`｜`-n 64`｜`--no-warmup`｜各跑 1 次｜采样 `nvidia-smi -lms 500`
> **本表是 D1 在 b12033 + Ubuntu 22.04 上测的，不是今天的环境测得。**

| 配置 | 加载耗时 | prompt 处理 (t/s) | 生成 (t/s) | 显存峰值 | 是否 OOM |
|---|---|---|---|---|---|
| `-ngl 99`（GPU） | 未单独计时 | **99.3** | **40.0** | **2800 MiB** | 否 |
| `-ngl 0`（纯 CPU） | 未单独计时 | **87.5** | **21.7** | —（不占显存） | 否 |

同配置第二次 GPU 复跑：Prompt 97.2 / Generation 38.4 → 两次生成速度差 4%，在重复性要求（<10%）以内。

结论：GPU 生成速度是纯 CPU 的 **1.84 倍**。3B Q4 权重 2 GB、T600 带宽约 80 GB/s，
理论解码上限约 40 tok/s——40.0 t/s 已接近带宽上限，提升空间在内存带宽而非算力。

## 8. D4 当日实测（CPU 版，b8681，Ubuntu 26.04）

| 项 | 值 |
|---|---|
| 服务 | `llama-server -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 0 --seed 42` |
| `/health` | `{"status":"ok"}` |
| SSE 分片数 | **42** |
| prompt_per_second | **77.72 t/s**（42 tok / 540.388 ms） |
| predicted_per_second | **19.70 t/s**（41 tok / 2081.651 ms） |
| 与 D1 `-ngl 0` 对比 | 87.5 / 21.7 → 偏低约 10%，同量级，模型未损坏 |
