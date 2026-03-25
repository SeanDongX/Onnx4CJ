/**
 * benchmark.cj — Performance benchmarks for Onnx4CJ
 *
 * Measures:
 *   - Session creation time
 *   - Single-run inference latency (warm/cold)
 *   - Throughput (runs/second)
 *   - Memory usage (baseline comparison)
 *
 * Run: cjpm test --filter benchmark
 *
 * Results are printed to stdout. See docs/benchmark.md for reference
 * numbers compared to Python/C++ baselines.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.tests

import onnx4cj.core.*
import std.unittest.*
import std.time.*
import std.env.*
import std.math.*

private let BENCH_MODEL_PATH = "src/models/test_identity.onnx"
private let BENCH_WARMUP_RUNS = 10
private let BENCH_MEASURE_RUNS = 100

@TestCase
class BenchmarkTests {

    /**
     * Benchmark: measure session creation time.
     * Requires: src/models/test_identity.onnx
     */
    @Test
    @ConditionalIgnore(condition: { !benchModelAvailable() })
    func benchmarkSessionCreation() {
        let env = OnnxEnv("bench_session_creation")
        defer { env.close() }

        let times = Array<Int64>(10) { _ =>
            let t0 = SystemClock.now().toMilliseconds()
            let session = OnnxSession(env, BENCH_MODEL_PATH)
            let t1 = SystemClock.now().toMilliseconds()
            session.close()
            t1 - t0
        }

        let avgMs = mean(times)
        println("[Benchmark] Session creation: avg=${avgMs} ms over 10 runs")
        // No hard assertion; this is a measurement test
        assertTrue(avgMs >= 0)
    }

    /**
     * Benchmark: measure per-inference latency.
     * Requires: src/models/test_identity.onnx
     */
    @Test
    @ConditionalIgnore(condition: { !benchModelAvailable() })
    func benchmarkInferenceLatency() {
        let env = OnnxEnv("bench_inference")
        defer { env.close() }

        let session = OnnxSession(env, BENCH_MODEL_PATH)
        defer { session.close() }

        let inputData = Array<Float32>(224 * 224 * 3, { i => Float32(i) / Float32(224 * 224 * 3) })
        let input = Tensor<Float32>.fromArray(inputData, [1, 3, 224, 224])
        defer { input.close() }

        // Warm up
        for _ in 0..BENCH_WARMUP_RUNS {
            let outputs = session.run<Float32>([input])
            for o in outputs { o.close() }
        }

        // Measure
        let times = Array<Int64>(BENCH_MEASURE_RUNS) { _ =>
            let t0 = SystemClock.now().toMicroseconds()
            let outputs = session.run<Float32>([input])
            let t1 = SystemClock.now().toMicroseconds()
            for o in outputs { o.close() }
            t1 - t0
        }

        let avgUs = mean(times)
        let minUs = times.min()
        let maxUs = times.max()
        let p50Us = percentile(times, 50)
        let p99Us = percentile(times, 99)

        println("[Benchmark] Inference latency (${BENCH_MEASURE_RUNS} runs):")
        println("  avg = ${avgUs} µs")
        println("  min = ${minUs} µs")
        println("  max = ${maxUs} µs")
        println("  p50 = ${p50Us} µs")
        println("  p99 = ${p99Us} µs")

        assertTrue(avgUs > 0)
    }

    /**
     * Benchmark: measure inference throughput (runs/second).
     */
    @Test
    @ConditionalIgnore(condition: { !benchModelAvailable() })
    func benchmarkThroughput() {
        let env = OnnxEnv("bench_throughput")
        defer { env.close() }

        let session = OnnxSession(env, BENCH_MODEL_PATH)
        defer { session.close() }

        let inputData = Array<Float32>(128, { i => Float32(i) })
        let input = Tensor<Float32>.fromArray(inputData, [1, 128])
        defer { input.close() }

        let durationSeconds: Int64 = 5
        let endTime = SystemClock.now().toMilliseconds() + durationSeconds * 1000
        var runCount: Int64 = 0

        while SystemClock.now().toMilliseconds() < endTime {
            let outputs = session.run<Float32>([input])
            for o in outputs { o.close() }
            runCount += 1
        }

        let throughput = runCount / durationSeconds
        println("[Benchmark] Throughput: ${throughput} runs/second over ${durationSeconds}s")
        assertTrue(throughput > 0)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

private func benchModelAvailable(): Bool {
    import std.fs.*
    File.exists(BENCH_MODEL_PATH)
}

private func mean(values: Array<Int64>): Float64 {
    if values.isEmpty() { return 0.0 }
    var sum: Int64 = 0
    for v in values { sum += v }
    Float64(sum) / Float64(values.size)
}

private func percentile(values: Array<Int64>, p: Int): Int64 {
    if values.isEmpty() { return 0 }
    let sorted = values.sorted()
    let idx = Int(Float64(sorted.size - 1) * Float64(p) / 100.0)
    sorted[idx]
}
