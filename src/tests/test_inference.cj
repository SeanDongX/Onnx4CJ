/**
 * test_inference.cj — End-to-end inference tests
 *
 * Tests:
 *   - Run inference on a minimal "add" model (y = a + b)
 *   - Run inference with wrong input count throws OnnxException
 *   - Output tensor shape matches expected
 *
 * The tiny test model (test_add.onnx) performs float32 addition:
 *   inputs:  A[1], B[1]
 *   output:  C[1]  (C = A + B)
 *
 * To generate the test model:
 *   python3 scripts/generate_test_models.py
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.tests

import onnx4cj.core.*
import std.unittest.*
import std.env.*

// Path to the tiny "add" test model
private let ADD_MODEL_PATH = "src/models/test_add.onnx"
// Path to the tiny "identity" test model (output = input)
private let IDENTITY_MODEL_PATH = "src/models/test_identity.onnx"

@TestCase
class InferenceTests {

    /**
     * End-to-end test: load add model, run inference, verify result.
     * Requires: src/models/test_add.onnx
     */
    @Test
    @ConditionalIgnore(condition: { !fileExists(ADD_MODEL_PATH) })
    func testAddModelInference() {
        let env = OnnxEnv("inference_test")
        defer { env.close() }

        let session = OnnxSession(env, ADD_MODEL_PATH)
        defer { session.close() }

        // Inputs: A = [3.0], B = [4.0]
        let inputA = Tensor<Float32>.fromArray([Float32(3.0)], [1])
        let inputB = Tensor<Float32>.fromArray([Float32(4.0)], [1])
        defer {
            inputA.close()
            inputB.close()
        }

        let outputs = session.run<Float32>([inputA, inputB])
        defer { for o in outputs { o.close() } }

        assertEquals(outputs.size, 1)
        let result = outputs[0].toArray()
        assertEquals(result.size, 1)
        // Expected: 3.0 + 4.0 = 7.0
        assertApproxEqual(result[0], Float32(7.0), delta: Float32(1e-5))
    }

    /**
     * End-to-end test: identity model returns same values as input.
     * Requires: src/models/test_identity.onnx
     */
    @Test
    @ConditionalIgnore(condition: { !fileExists(IDENTITY_MODEL_PATH) })
    func testIdentityModelInference() {
        let env = OnnxEnv("identity_test")
        defer { env.close() }

        let session = OnnxSession(env, IDENTITY_MODEL_PATH)
        defer { session.close() }

        let inputData = Array<Float32>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let input = Tensor<Float32>.fromArray(inputData, [2, 3])
        defer { input.close() }

        let outputs = session.run<Float32>([input])
        defer { for o in outputs { o.close() } }

        assertEquals(outputs.size, 1)
        let result = outputs[0].toArray()
        assertEquals(result.size, inputData.size)
        for i in 0..inputData.size {
            assertApproxEqual(result[i], inputData[i], delta: Float32(1e-6))
        }
    }

    /**
     * Passing wrong number of inputs should throw OnnxException.
     * Requires: src/models/test_add.onnx (expects 2 inputs)
     */
    @Test
    @ConditionalIgnore(condition: { !fileExists(ADD_MODEL_PATH) })
    func testWrongInputCountThrows() {
        let env = OnnxEnv("wrong_input_test")
        defer { env.close() }

        let session = OnnxSession(env, ADD_MODEL_PATH)
        defer { session.close() }

        let singleInput = Tensor<Float32>.fromArray([Float32(1.0)], [1])
        defer { singleInput.close() }

        assertThrows<OnnxException> {
            let _ = session.run<Float32>([singleInput])
        }
    }
}

private func fileExists(path: String): Bool {
    import std.fs.*
    File.exists(path)
}
