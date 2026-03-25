# Onnx4CJ — Getting Started

## Prerequisites

| Tool | Version |
|------|---------|
| Cangjie (仓颉) toolchain | 0.53.x |
| ONNX Runtime | ≥ 1.16.0 |
| GCC / Clang | ≥ 7 |

## 1. Install ONNX Runtime

### Linux x86-64
```bash
VERSION=1.16.3
wget https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-${VERSION}.tgz
tar -xzf onnxruntime-linux-x64-${VERSION}.tgz
export ORT_HOME=$(pwd)/onnxruntime-linux-x64-${VERSION}
```

### macOS arm64 (Apple Silicon)
```bash
VERSION=1.16.3
wget https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-osx-arm64-${VERSION}.tgz
tar -xzf onnxruntime-osx-arm64-${VERSION}.tgz
export ORT_HOME=$(pwd)/onnxruntime-osx-arm64-${VERSION}
```

## 2. Clone the Repository

```bash
git clone https://github.com/SeanDongX/Onnx4CJ.git
cd Onnx4CJ
```

## 3. Build the C Shim

```bash
cd c
ORT_HOME=$ORT_HOME make
cd ..
```

This produces `c/libort_shim.so` (shared) and `c/libort_shim.a` (static).

## 4. Build the Cangjie Library

```bash
export LD_LIBRARY_PATH=$ORT_HOME/lib:$(pwd)/c:$LD_LIBRARY_PATH
cjpm build
```

## 5. Run the Tests

### Unit tests (no model required)
```bash
cjpm test
```

### Integration & inference tests (model required)
```bash
# Download a tiny test model
python3 scripts/generate_test_models.py

# Run all tests including integration
TEST_MODEL_PATH=src/models/test_add.onnx cjpm test
```

## 6. Run the Examples

### Basic inference (test_add model)
```bash
MODEL_PATH=src/models/test_add.onnx cjpm run --example basic_inference
```

### ResNet-50 image classification
```bash
# Download ResNet-50 (requires Python + onnx package)
python3 scripts/download_models.py resnet50

# Run end-to-end demo
cjpm run --example e2e_demo
```

## 7. Use in Your Own Project

Add Onnx4CJ as a dependency in your `cjpm.toml`:

```toml
[dependencies]
onnx4cj = { path = "/path/to/Onnx4CJ", version = "0.1.0" }
```

Then import:
```cangjie
import onnx4cj.core.*

main() {
    let env = OnnxEnv("my_app")
    let session = OnnxSession(env, "my_model.onnx")
    // ...
}
```

## Troubleshooting

### `ort_init() failed`
- Ensure `$ORT_HOME/lib` is in `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS).

### `Model file not found`
- Check that the path passed to `OnnxSession` is correct and the file exists.

### Build error: `onnxruntime_c_api.h not found`
- Ensure `ORT_HOME` is set and points to the ONNX Runtime installation directory.
- Check that `$ORT_HOME/include/onnxruntime_c_api.h` exists.
