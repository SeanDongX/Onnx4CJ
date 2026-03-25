# Onnx4CJ

**Onnx4CJ** 是一个为 [仓颉（Cangjie）](https://cangjie-lang.cn/) 编程语言提供的 [ONNX Runtime](https://github.com/microsoft/onnxruntime) C API 封装库。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Cangjie](https://img.shields.io/badge/Language-Cangjie%200.53.x-orange)](https://cangjie-lang.cn/)

## 项目目标

- 通过 FFI（Foreign Function Interface）将 ONNX Runtime C API 绑定到仓颉语言
- 提供 RAII 安全封装，自动管理资源生命周期
- 使仓颉开发者可以方便地在本地运行 ONNX 模型进行推理计算
- 支持常见模型（ResNet、BART 等）的加载与推理

## 目录结构

```
Onnx4CJ/
├── c/
│   ├── ort_shim.h      # C shim 头文件（对 ONNX Runtime C API 的简化封装）
│   └── ort_shim.c      # C shim 实现
├── src/
│   ├── ffi/            # 仓颉 FFI 层（绑定 C shim 函数）
│   │   ├── ort_types.cj   # ONNX Runtime C 类型定义
│   │   └── ort_api.cj     # FFI 函数声明
│   ├── core/           # 高层安全封装层
│   │   ├── env.cj         # OnnxEnv — 环境与日志管理
│   │   ├── session.cj     # OnnxSession — 模型加载与推理
│   │   ├── tensor.cj      # Tensor — 输入/输出张量封装
│   │   └── error.cj       # 错误类型定义
│   ├── tests/          # 单元与集成测试
│   │   ├── test_env.cj    # OnnxEnv 测试
│   │   ├── test_session.cj# OnnxSession 测试
│   │   ├── test_tensor.cj # Tensor 测试
│   │   ├── test_inference.cj # 端到端推理测试
│   │   └── benchmark.cj   # 基准性能测试
│   ├── examples/       # 示例代码
│   │   ├── basic_inference.cj  # 基础推理示例
│   │   └── e2e_demo.cj         # 端到端演示（ResNet/BART）
│   └── models/         # 示例 ONNX 模型文件（需自行下载）
├── docs/
│   ├── api_reference.md  # API 使用说明
│   ├── benchmark.md      # 基准测试报告
│   └── getting_started.md# 快速上手指引
├── cjpm.toml           # 仓颉包管理器配置
├── LICENSE             # Apache 2.0
└── README.md
```

## 依赖

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) ≥ 1.16.0（需要预先安装）
- [仓颉（Cangjie）](https://cangjie-lang.cn/) 0.53.x 编译器与工具链

### 安装 ONNX Runtime

```bash
# Linux x86_64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
export ORT_HOME=$(pwd)/onnxruntime-linux-x64-1.16.3

# macOS arm64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -xzf onnxruntime-osx-arm64-1.16.3.tgz
export ORT_HOME=$(pwd)/onnxruntime-osx-arm64-1.16.3
```

## 快速开始

```bash
# 设置 ONNX Runtime 路径
export ORT_HOME=/path/to/onnxruntime

# 构建 C shim
cd c && make && cd ..

# 构建项目
cjpm build

# 运行测试
cjpm test

# 运行基础推理示例（需要模型文件）
cjpm run --example basic_inference
```

## 基本用法

```cangjie
import onnx4cj.core.*

main() {
    // 创建推理环境（RAII：析构时自动释放）
    let env = OnnxEnv("my_session")

    // 加载模型（支持相对/绝对路径）
    let session = OnnxSession(env, "models/resnet50.onnx")

    // 查询输入/输出信息
    println("输入数量: ${session.inputCount()}")
    println("输出数量: ${session.outputCount()}")

    // 构建输入张量（Float32，形状 [1, 3, 224, 224]）
    let inputData = Array<Float32>(1 * 3 * 224 * 224, {_ => 0.5})
    let inputTensor = Tensor<Float32>.fromArray(inputData, [1, 3, 224, 224])

    // 执行推理
    let outputs = session.run([inputTensor])
    println("推理完成，输出张量数量: ${outputs.size}")

    // 读取结果
    let result = outputs[0].toArray<Float32>()
    println("Top-1 分类结果: ${argmax(result)}")
}
```

## 测试

```bash
# 运行全部单元测试
cjpm test

# 运行特定测试
cjpm test --filter test_env
cjpm test --filter test_inference

# 运行基准性能测试
cjpm test --filter benchmark
```

## 文档

- [API 使用说明](docs/api_reference.md)
- [快速上手指引](docs/getting_started.md)
- [基准测试报告](docs/benchmark.md)

## 贡献

欢迎提交 Pull Request 或 Issue。

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。
