# Onnx4CJ

**ONNX C API for 仓颉 (Cangjie)**

Onnx4CJ 是面向[仓颉（Cangjie）](https://cangjie-lang.cn/)语言的 ONNX Runtime C API 绑定库，旨在让仓颉开发者能够方便地在仓颉程序中加载和运行 ONNX 模型进行推理。

## 项目目标

- 通过 FFI（Foreign Function Interface）封装 ONNX Runtime C API
- 提供符合仓颉语言习惯的高层接口
- 支持常见的 ONNX 模型推理场景

## 项目结构

```
Onnx4CJ/
├── src/
│   ├── ffi/       # ONNX Runtime C API 的 FFI 绑定
│   ├── core/      # 核心库逻辑
│   ├── tests/     # 测试文件
│   ├── examples/  # 示例代码
│   └── models/    # ONNX 模型文件
├── cjpm.toml      # 仓颉包管理配置
├── LICENSE        # Apache 2.0
└── README.md
```

## 依赖

- [ONNX Runtime](https://onnxruntime.ai/) C API
- 仓颉（Cangjie）编译器

## 许可证

本项目遵循 [Apache License 2.0](LICENSE)。
