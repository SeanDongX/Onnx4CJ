#!/usr/bin/env python3
"""
benchmark_python.py — Python ONNX Runtime baseline benchmark.

Measures session creation time and inference latency using the Python
onnxruntime package for comparison with Onnx4CJ (Cangjie binding).

Usage:
  pip install onnxruntime numpy
  python3 scripts/benchmark_python.py [--model src/models/test_identity.onnx]
"""

import argparse
import time
import statistics
import os
import sys

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("ERROR: Install with: pip install onnxruntime numpy")
    sys.exit(1)

DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "src", "models", "test_identity.onnx"
)
WARMUP_RUNS = 10
MEASURE_RUNS = 100


def benchmark_session_creation(model_path: str, n: int = 10) -> None:
    print("\n=== Session Creation Benchmark ===")
    times_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model_path)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
        del sess

    print(f"  Runs: {n}")
    print(f"  Mean: {statistics.mean(times_ms):.2f} ms")
    print(f"  Std:  {statistics.stdev(times_ms):.2f} ms")
    print(f"  Min:  {min(times_ms):.2f} ms")
    print(f"  Max:  {max(times_ms):.2f} ms")


def benchmark_inference(model_path: str) -> None:
    print("\n=== Inference Latency Benchmark ===")
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Replace dynamic dims with concrete values
    concrete_shape = [1 if (isinstance(d, str) or d is None) else d for d in input_shape]
    dummy = np.random.randn(*concrete_shape).astype(np.float32)

    # Warm up
    for _ in range(WARMUP_RUNS):
        sess.run(None, {input_name: dummy})

    # Measure
    times_us = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    sorted_times = sorted(times_us)
    p50 = sorted_times[int(len(sorted_times) * 0.50)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    print(f"  Runs:  {MEASURE_RUNS}  (after {WARMUP_RUNS} warmup)")
    print(f"  Input: {concrete_shape} float32")
    print(f"  Mean:  {statistics.mean(times_us):.1f} µs")
    print(f"  p50:   {p50:.1f} µs")
    print(f"  p99:   {p99:.1f} µs")
    print(f"  Min:   {min(times_us):.1f} µs")
    print(f"  Max:   {max(times_us):.1f} µs")


def main():
    parser = argparse.ArgumentParser(description="Python OnnxRuntime baseline benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Generate test models first: python3 scripts/generate_test_models.py")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"OnnxRuntime version: {ort.__version__}")

    benchmark_session_creation(args.model)
    benchmark_inference(args.model)


if __name__ == "__main__":
    main()
