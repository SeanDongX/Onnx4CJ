# Onnx4CJ API Reference

> **Onnx4CJ** — ONNX Runtime C API bindings for Cangjie (仓颉)

This document describes the public API of the `onnx4cj` library.

---

## Module Overview

| Module | Description |
|--------|-------------|
| `onnx4cj.core` | High-level, RAII-safe API (use this in your app) |
| `onnx4cj.ffi`  | Low-level FFI declarations (advanced use only) |

---

## `onnx4cj.core`

### `OnnxEnv`

Wraps `OrtEnv`. One `OnnxEnv` can be shared across multiple sessions.
Resources are released automatically on `close()` or when used in a `use` block.

```cangjie
public class OnnxEnv <: Resource {
    public init(logId: String)
    public func close(): Unit
}
```

| Member | Description |
|--------|-------------|
| `init(logId: String)` | Create a new environment. `logId` appears in runtime log messages. |
| `close()` | Release the underlying `OrtEnv`. Called automatically in `use` blocks. |

**Example:**
```cangjie
use env = OnnxEnv("my_app") {
    // ... create sessions inside this block
}
// env automatically closed here
```

---

### `SessionOptions`

Builder-style configuration for `OnnxSession`. Chain calls are supported.

```cangjie
public class SessionOptions <: Resource {
    public init()
    public func setIntraOpNumThreads(n: Int32): SessionOptions
    public func setInterOpNumThreads(n: Int32): SessionOptions
    public func close(): Unit
}
```

| Member | Description |
|--------|-------------|
| `init()` | Create options with ONNX Runtime defaults. |
| `setIntraOpNumThreads(n)` | Number of threads for intra-op parallelism (0 = auto). |
| `setInterOpNumThreads(n)` | Number of threads for inter-op parallelism (0 = auto). |
| `close()` | Release the underlying `OrtSessionOptions`. |

**Example:**
```cangjie
let opts = SessionOptions()
    .setIntraOpNumThreads(4)
    .setInterOpNumThreads(2)
defer { opts.close() }
```

---

### `OnnxSession`

Loads an ONNX model and exposes inference via `run()`.

```cangjie
public class OnnxSession <: Resource {
    public let inputCount: Int
    public let outputCount: Int
    public let inputNames: Array<String>
    public let outputNames: Array<String>

    public init(env: OnnxEnv, modelPath: String)
    public init(env: OnnxEnv, modelPath: String, opts: ?SessionOptions)
    public func run<T>(inputs: Array<Tensor<T>>): Array<Tensor<T>>
    public func close(): Unit
}
```

| Member | Description |
|--------|-------------|
| `inputCount` | Number of model input nodes. |
| `outputCount` | Number of model output nodes. |
| `inputNames` | Names of input nodes, in order. |
| `outputNames` | Names of output nodes, in order. |
| `init(env, modelPath)` | Load model from file. Throws `ModelNotFoundException` if file not found. |
| `init(env, modelPath, opts)` | Load model with custom `SessionOptions`. |
| `run<T>(inputs)` | Run inference. `inputs` must match `inputCount`. Returns output tensors. |
| `close()` | Release the underlying `OrtSession`. |

**Throws:**
- `ModelNotFoundException` — model file does not exist
- `OnnxException` — any other ONNX Runtime error

**Example:**
```cangjie
let env = OnnxEnv("app")
let session = OnnxSession(env, "models/resnet50.onnx")
defer {
    session.close()
    env.close()
}

let inputData = Array<Float32>(1 * 3 * 224 * 224, {_ => 0.5})
let input = Tensor<Float32>.fromArray(inputData, [1, 3, 224, 224])
defer { input.close() }

let outputs = session.run<Float32>([input])
defer { for o in outputs { o.close() } }
```

---

### `Tensor<T>`

Type-safe wrapper for `OrtValue` tensors.

Supported element types `T`: `Float32`, `Float64`, `Int8`, `Int16`, `Int32`,
`Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`

```cangjie
public class Tensor<T> <: Resource {
    public prop shape: Array<Int64> { get }
    public prop elementType: OrtElementType { get }
    public prop numel: Int64 { get }

    public static func fromArray(data: Array<T>, shape: Array<Int64>): Tensor<T>
    public func toArray(): Array<T>
    public func close(): Unit
}
```

| Member | Description |
|--------|-------------|
| `shape` | Dimension sizes (e.g., `[1, 3, 224, 224]`). |
| `elementType` | The `OrtElementType` enum value. |
| `numel` | Total number of elements (product of `shape`). |
| `fromArray(data, shape)` | Create a tensor from a flat array. Zero-copy where possible. |
| `toArray()` | Copy tensor data into a new `Array<T>`. |
| `close()` | Release the underlying `OrtValue`. |

**Throws:**
- `TensorException` — shape/data mismatch or unsupported element type
- `OnnxException` — underlying ORT failure

**Example:**
```cangjie
// Create a Float32 tensor of shape [2, 3]
let data = Array<Float32>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let t = Tensor<Float32>.fromArray(data, [2, 3])
defer { t.close() }

println("shape = ${t.shape}")   // [2, 3]
println("numel = ${t.numel}")   // 6

let values = t.toArray()
println("values = ${values}")   // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

---

## Exception Hierarchy

```
Exception
└── OnnxException
    ├── ModelNotFoundException
    └── TensorException
```

| Exception | Cause |
|-----------|-------|
| `OnnxException` | General ONNX Runtime error. Contains `status: OrtShimStatus` and `message: String`. |
| `ModelNotFoundException` | Model file path does not exist. |
| `TensorException` | Invalid tensor shape, size mismatch, or unsupported element type. |

---

## `onnx4cj.ffi`

The FFI layer exposes raw C bindings and is intended for advanced use cases.
Normal users should use `onnx4cj.core` instead.

Key types: `OrtEnvHandle`, `OrtSessionHandle`, `OrtValueHandle`, `OrtShimStatus`, `OrtElementType`

Key functions: mirror the C functions in `c/ort_shim.h`.

---

## Thread Safety

- `OnnxEnv` initialization is thread-safe (protected by an internal mutex).
- Multiple `OnnxSession` instances can be created from the same `OnnxEnv` concurrently.
- A single `OnnxSession` can be called from multiple goroutines concurrently
  (ONNX Runtime guarantees this for `Run()`).
- `Tensor` objects are **not** thread-safe; do not share a single tensor across threads.
