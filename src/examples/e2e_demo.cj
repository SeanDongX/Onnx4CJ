/**
 * e2e_demo.cj — End-to-end demo with ResNet-50 image classification
 *
 * This example demonstrates running ResNet-50 for image classification.
 * It loads a pre-processed image tensor (or generates a random one) and
 * prints the top-5 predicted class indices.
 *
 * Prerequisites:
 *   - ONNX Runtime installed and ORT_HOME set
 *   - ResNet-50 ONNX model: src/models/resnet50.onnx
 *     Download: https://github.com/onnx/models/tree/main/vision/classification/resnet
 *
 * Usage:
 *   ORT_HOME=/path/to/onnxruntime cjpm run --example e2e_demo
 *   # or with a custom model:
 *   MODEL_PATH=path/to/your.onnx cjpm run --example e2e_demo
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.examples

import onnx4cj.core.*
import std.env.*
import std.math.*
import std.time.*

// ImageNet class names (abbreviated; first 10 shown in demo)
private let IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich"
    // ... 1000 classes total
]

main(): Int64 {
    let modelPath = getEnvOrDefault("MODEL_PATH", "src/models/resnet50.onnx")

    println("=== Onnx4CJ End-to-End Demo: ResNet-50 Image Classification ===")
    println("Model: ${modelPath}")
    println()

    use env = OnnxEnv("e2e_demo") {

        // Load model
        let session: OnnxSession
        do {
            session = OnnxSession(env, modelPath)
        } catch (e: ModelNotFoundException) {
            println("ERROR: Model not found at '${modelPath}'")
            println()
            println("To download ResNet-50:")
            println("  python3 scripts/download_models.py resnet50")
            return 1
        }
        defer { session.close() }

        println("Model info:")
        println("  Inputs : ${session.inputCount} → ${session.inputNames}")
        println("  Outputs: ${session.outputCount} → ${session.outputNames}")
        println()

        // Build a synthetic input image: shape [1, 3, 224, 224], values in [0, 1]
        // In a real application you would load and pre-process an actual image here.
        println("Generating random input image [1, 3, 224, 224] ...")
        let batchSize = 1
        let channels  = 3
        let height    = 224
        let width     = 224
        let numElements = batchSize * channels * height * width

        // Simulate ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        let imagenetMean = [Float32(0.485), Float32(0.456), Float32(0.406)]
        let imagenetStd  = [Float32(0.229), Float32(0.224), Float32(0.225)]

        let inputData = Array<Float32>(numElements) { idx =>
            let c = (idx / (height * width)) % channels
            // LCG (Linear Congruential Generator) constants from Knuth TAOCP
            // to produce a deterministic pseudo-random pattern in [0, 255]
            let LCG_A: Int64 = 6364136223846793005
            let LCG_C: Int64 = 1442695040888963407
            let rawVal = Float32((idx * LCG_A + LCG_C) % 256) / 255.0
            (rawVal - imagenetMean[c]) / imagenetStd[c]
        }
        let inputTensor = Tensor<Float32>.fromArray(inputData, [1, 3, 224, 224])
        defer { inputTensor.close() }

        // Run inference and measure latency
        println("Running inference ...")
        let t0 = SystemClock.now().toMilliseconds()
        let outputs: Array<Tensor<Float32>>
        do {
            outputs = session.run<Float32>([inputTensor])
        } catch (e: OnnxException) {
            println("ERROR: ${e.message}")
            return 1
        }
        let t1 = SystemClock.now().toMilliseconds()
        defer { for o in outputs { o.close() } }

        println("Inference completed in ${t1 - t0} ms")
        println()

        // Interpret output: logits of shape [1, 1000]
        let logits = outputs[0].toArray()
        println("Output shape: ${outputs[0].shape}")
        println()

        // Compute softmax probabilities
        let probs = softmax(logits)

        // Find top-5 predictions
        let top5 = topK(probs, k: 5)
        println("Top-5 predictions:")
        for (rank, pair) in top5.enumerated() {
            let (classIdx, prob) = pair
            let className = if classIdx < IMAGENET_CLASSES.size {
                IMAGENET_CLASSES[classIdx]
            } else {
                "class_${classIdx}"
            }
            println("  ${rank + 1}. ${className} (idx=${classIdx}) — ${formatProb(prob)}")
        }
    }

    return 0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

private func getEnvOrDefault(key: String, default_: String): String {
    let val = getEnv(key)
    if val.isEmpty() { default_ } else { val }
}

private func softmax(logits: Array<Float32>): Array<Float32> {
    let maxVal = logits.max() ?? Float32(0.0)
    var sum = Float32(0.0)
    let exps = Array<Float32>(logits.size) { i =>
        let e = exp(logits[i] - maxVal)
        sum += e
        e
    }
    Array<Float32>(exps.size) { i => exps[i] / sum }
}

private func topK(probs: Array<Float32>, k: Int): Array<(Int, Float32)> {
    var indexed = Array<(Int, Float32)>(probs.size) { i => (i, probs[i]) }
    // Partial sort: find k largest
    indexed.sort { a, b => b.1 > a.1 }
    Array<(Int, Float32)>(min(k, indexed.size)) { i => indexed[i] }
}

private func formatProb(p: Float32): String {
    let pct = p * 100.0
    "${pct:.2f}%"
}
