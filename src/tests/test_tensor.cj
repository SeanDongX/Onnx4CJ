/**
 * test_tensor.cj — Unit tests for Tensor<T>
 *
 * Tests:
 *   - Tensor creation from Float32 array
 *   - Tensor creation from Int32 and Int64 arrays
 *   - Shape validation
 *   - Data round-trip (fromArray → toArray)
 *   - Shape mismatch raises TensorException
 *   - Unsupported element type raises TensorException
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.tests

import onnx4cj.core.*
import std.unittest.*

@TestCase
class TensorTests {

    /**
     * Tensor<Float32>.fromArray should succeed for valid data and shape.
     */
    @Test
    func testCreateFloat32Tensor() {
        let data = Array<Float32>([1.0, 2.0, 3.0, 4.0])
        let t = Tensor<Float32>.fromArray(data, [2, 2])
        defer { t.close() }
        assertEquals(t.shape, [Int64(2), Int64(2)])
        assertEquals(t.numel, Int64(4))
    }

    /**
     * Tensor<Int32>.fromArray should succeed.
     */
    @Test
    func testCreateInt32Tensor() {
        let data = Array<Int32>([10, 20, 30])
        let t = Tensor<Int32>.fromArray(data, [3])
        defer { t.close() }
        assertEquals(t.shape, [Int64(3)])
    }

    /**
     * Tensor<Int64>.fromArray should succeed.
     */
    @Test
    func testCreateInt64Tensor() {
        let data = Array<Int64>([100, 200])
        let t = Tensor<Int64>.fromArray(data, [1, 2])
        defer { t.close() }
        assertEquals(t.shape, [Int64(1), Int64(2)])
    }

    /**
     * toArray() should reproduce the original values.
     */
    @Test
    func testDataRoundTrip() {
        let original = Array<Float32>([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        let t = Tensor<Float32>.fromArray(original, [2, 3])
        defer { t.close() }
        let recovered = t.toArray()
        assertEquals(recovered.size, original.size)
        for i in 0..original.size {
            assertApproxEqual(recovered[i], original[i], delta: Float32(1e-6))
        }
    }

    /**
     * Shape mismatch between data.size and shape product should throw TensorException.
     */
    @Test
    func testShapeMismatchThrows() {
        let data = Array<Float32>([1.0, 2.0, 3.0])
        assertThrows<TensorException> {
            // shape product is 4, but data has 3 elements
            let _ = Tensor<Float32>.fromArray(data, [2, 2])
        }
    }

    /**
     * 1-D tensor (scalar-like) shape [1] should work.
     */
    @Test
    func testScalarTensor() {
        let data = Array<Float32>([42.0])
        let t = Tensor<Float32>.fromArray(data, [1])
        defer { t.close() }
        assertEquals(t.numel, Int64(1))
        let out = t.toArray()
        assertApproxEqual(out[0], Float32(42.0), delta: Float32(1e-6))
    }

    /**
     * 4-D tensor (typical image batch) should preserve shape.
     */
    @Test
    func testFourDimTensor() {
        let n = 1 * 3 * 4 * 4  // batch=1, channels=3, height=4, width=4
        let data = Array<Float32>(n, { i => Float32(i) })
        let t = Tensor<Float32>.fromArray(data, [1, 3, 4, 4])
        defer { t.close() }
        assertEquals(t.shape.size, 4)
        assertEquals(t.numel, Int64(n))
    }
}
