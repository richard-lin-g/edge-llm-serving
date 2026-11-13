cmake_minimum_required(VERSION 3.16)
project(edge-client LANGUAGES CXX)

# 生成 compile_commands.json，供 clangd / VS Code C/C++ 扩展做精确跳转
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# httplib.h / json.hpp 已放在 third_party/（离线可用，不依赖包管理器）
add_executable(edge-client src/main.cpp)
target_include_directories(edge-client PRIVATE third_party)
find_package(Threads REQUIRED)
target_link_libraries(edge-client PRIVATE Threads::Threads)

# httplib 是单头文件库，编译它本身要 10-20 秒，属正常
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(edge-client PRIVATE -O2 -Wall -Wextra)
endif()
