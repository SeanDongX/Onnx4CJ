/**
 * basic_inference.cj — Basic Onnx4CJ inference example
 *
 * Demonstrates the minimal workflow:
 *   1. Create OnnxEnv
 *   2. Load a model via OnnxSession
 *   3. Build input tensors
 *   4. Call session.run()
 *   5. Read output tensors
 *
 * Usage:
 *   ORT_HOME=/path/to/onnxruntime MODEL_PATH=model.onnx cjpm run --example basic_inference
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.examples

import onnx4cj.core.*
import std.env.*
import std.math.*

main(): Int64 {
    let modelPath = getEnvOrDefault("MODEL_PATH", "src/models/test_add.onnx")

    println("Onnx4CJ — Basic Inference Example")
    println("Model: ${modelPath}")
    println()

    // Step 1: Create environment
    // OnnxEnv is RAII — close() is called when `use` block exits
    use env = OnnxEnv("basic_inference") {

        // Step 2: Load model
        let session: OnnxSession
        do {
            session = OnnxSession(env, modelPath)
        } catch (e: ModelNotFoundException) {
            println("ERROR: ${e.message}")
            println("Tip: Set MODEL_PATH env variable to an existing .onnx file.")
            return 1
        }
        defer { session.close() }

        println("Model loaded successfully.")
        println("  Input count : ${session.inputCount}")
        println("  Output count: ${session.outputCount}")
        println("  Input names : ${session.inputNames}")
        println("  Output names: ${session.outputNames}")
        println()

        // Step 3: Build input tensors
        // For the default test_add model: A + B = C
        let inputA = Tensor<Float32>.fromArray([Float32(3.0)], [1])
        let inputB = Tensor<Float32>.fromArray([Float32(4.0)], [1])
        defer {
            inputA.close()
            inputB.close()
        }

        println("Running inference with A=${inputA.toArray()[0]}, B=${inputB.toArray()[0]} ...")

        // Step 4: Run inference
        let outputs: Array<Tensor<Float32>>
        do {
            outputs = session.run<Float32>([inputA, inputB])
        } catch (e: OnnxException) {
            println("ERROR during inference: ${e.message}")
            return 1
        }
        defer { for o in outputs { o.close() } }

        // Step 5: Read results
        println()
        println("Results:")
        for (i, output) in outputs.enumerated() {
            let data = output.toArray()
            println("  Output[${i}] shape=${output.shape} values=${data}")
        }
    }

    return 0
}

private func getEnvOrDefault(key: String, default_: String): String {
    let val = getEnv(key)
    if val.isEmpty() { default_ } else { val }
}
