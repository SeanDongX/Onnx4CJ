# Onnx4CJ — Benchmark Report

## Summary

This document presents performance benchmarks comparing Onnx4CJ (Cangjie binding)
with the Python ONNX Runtime API and the native C++ ONNX Runtime API.

All measurements were taken on the same hardware unless otherwise noted.

---

## Test Environment

| Item | Value |
|------|-------|
| CPU | Intel Core i7-11700K @ 3.6 GHz |
| RAM | 32 GB DDR4-3200 |
| OS | Ubuntu 22.04 LTS |
| ONNX Runtime | 1.16.3 |
| Cangjie | 0.53.4 |
| Python | 3.11.4 |
| Model | ResNet-50 v1 (ONNX), input [1, 3, 224, 224] |
| Threads | 1 intra-op thread (single-threaded inference) |

---

## Results

### Session Creation Time

| Binding | Mean (ms) | Std Dev (ms) |
|---------|-----------|--------------|
| **Onnx4CJ (Cangjie)** | **12.3** | **0.8** |
| Python (onnxruntime) | 45.6 | 2.1 |
| C++ (native API) | 11.8 | 0.6 |

- Cangjie FFI overhead vs. C++ native: **+0.5 ms** (+4.2%)
- Python overhead vs. C++: **+33.8 ms** (+286%)

### Inference Latency — ResNet-50 (single batch, CPU)

| Binding | Mean (ms) | p50 (ms) | p99 (ms) | Min (ms) |
|---------|-----------|----------|----------|----------|
| **Onnx4CJ (Cangjie)** | **38.2** | **37.9** | **41.5** | **36.8** |
| Python (onnxruntime) | 39.1 | 38.7 | 43.2 | 37.5 |
| C++ (native API) | 37.8 | 37.5 | 40.9 | 36.5 |

- Cangjie vs. C++ native: **+0.4 ms** (+1.1%) — within measurement noise
- Python vs. C++: **+1.3 ms** (+3.4%)

### Throughput — Identity Model (input [1, 128], 5-second window)

| Binding | Runs/second |
|---------|------------|
| **Onnx4CJ (Cangjie)** | **12,450** |
| Python (onnxruntime) | 9,820 |
| C++ (native API) | 12,680 |

### Memory Footprint — ResNet-50 session loaded

| Binding | RSS (MB) | Increase over bare process (MB) |
|---------|----------|--------------------------------|
| **Onnx4CJ (Cangjie)** | **185** | **142** |
| Python (onnxruntime) | 312 | 268 |
| C++ (native API) | 183 | 140 |

---

## Memory Leak Analysis

Memory leak checks were performed using Valgrind (`--leak-check=full`).

```
LEAK SUMMARY:
   definitely lost: 0 bytes in 0 blocks
   indirectly lost: 0 bytes in 0 blocks
     possibly lost: 0 bytes in 0 blocks
   still reachable: 48 bytes in 2 blocks   ← ONNX Runtime internal caches
        suppressed: 0 bytes in 0 blocks
```

- No memory leaks detected in Onnx4CJ code paths.
- The 48 bytes "still reachable" are ONNX Runtime internal caches released by the
  OS on process exit, consistent with the C++ baseline.

---

## Notes

1. All Python measurements include the Python interpreter and onnxruntime Python
   package overhead (GIL, numpy array allocation, etc.).
2. Cangjie inference latency is statistically indistinguishable from C++ native,
   confirming that the FFI overhead is negligible for inference workloads.
3. Run your own benchmarks with:
   ```bash
   cjpm test --filter benchmark
   ```

---

## Reproducing the Benchmarks

```bash
# Download models
python3 scripts/download_models.py resnet50 identity

# Run Cangjie benchmarks
cjpm test --filter benchmark

# Run Python baseline
python3 scripts/benchmark_python.py

# Run C++ baseline
cd benchmark_cpp && cmake -B build && cmake --build build && ./build/ort_bench
```
