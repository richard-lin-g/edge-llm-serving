# 第三方依赖说明（THIRD_PARTY）

本仓库所有第三方代码均为 **vendored**（源码级内嵌，离线可编译，不依赖包管理器或网络）。
两个库都是 **MIT 许可**，允许商用与闭源分发，但要求保留版权声明与许可正文——正文附在文末。

| 组件 | 版本 | 许可 | 内嵌位置 | 上游 |
|---|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT | `client-cpp/third_party/httplib.h` | https://github.com/yhirose/cpp-httplib |
| nlohmann/json | 3.11.3 | MIT | `client-cpp/third_party/nlohmann/json.hpp` | https://github.com/nlohmann/json |

## 引入方式

两者都是 header-only，`client-cpp/CMakeLists.txt` 里只做了一件事：

```cmake
target_include_directories(edge-client PRIVATE third_party)
```

httplib 需要 pthread，因此链接 `Threads::Threads`；**未定义 `CPPHTTPLIB_OPENSSL_SUPPORT`**，
所以不引入 OpenSSL，只用明文 HTTP，与本地 `llama-server` 的部署方式一致。

## 为什么 vendored 而不是 FetchContent / vcpkg

- 面试或换机器时 `git clone && cmake -B build` 即可编译，不依赖外网与包管理器状态；
- 版本被钉死在仓库里，复现实验不会因为上游改了 API 而失效；
- 代价是仓库多出约 1.2 MB 文本。

## 升级方式

```bash
# 需要抬版本时，替换文件并同步本文档中的版本号
curl -L -o client-cpp/third_party/httplib.h \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.19.0/httplib.h
```

未内嵌的第三方：llama.cpp 通过 apt（`llama.cpp-tools` 8681）与同源源码自编 CUDA 版获得，
模型 Qwen2.5-3B-Instruct-Q4_K_M 来自 Hugging Face，均不随仓库分发。

---

## MIT License — cpp-httplib

Copyright (c) 2024 Yuji Hirose

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT License — nlohmann/json

Copyright (c) 2013-2022 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
