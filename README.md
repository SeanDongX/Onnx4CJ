# Onnx4CJ

**Onnx4CJ** 是一个为 [仓颉（Cangjie）](https://cangjie-lang.cn/) 编程语言提供的 [ONNX](https://onnx.ai/) C API 封装库。

本项目目标是通过 FFI（Foreign Function Interface）将 ONNX Runtime 的 C API 绑定到仓颉语言，使仓颉开发者可以方便地在本地运行 ONNX 模型，进行推理计算。

## 项目目标

- 封装 ONNX Runtime C API，为仓颉语言提供原生推理能力
- 提供简洁易用的仓颉语言接口
- 支持常见的 ONNX 模型加载与推理

## 目录结构

```
Onnx4CJ/
├── src/
│   ├── ffi/        # FFI 封装层，绑定 ONNX Runtime C API
│   ├── core/       # 核心逻辑实现
│   ├── tests/      # 单元测试
│   ├── examples/   # 示例代码
│   └── models/     # 示例 ONNX 模型文件
├── cjpm.toml       # 仓颉包管理器配置
├── LICENSE         # Apache 2.0
└── README.md
```

## 依赖

- [ONNX Runtime](https://github.com/microsoft/onnxruntime)（需要预先安装）
- [仓颉（Cangjie）](https://cangjie-lang.cn/) 编译器与工具链

## 快速开始

```bash
# 构建项目
cjpm build

# 运行测试
cjpm test
```

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。
