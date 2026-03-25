/**
 * tensor.cj — Tensor: type-safe wrapper for OrtValue (tensor)
 *
 * Supports Float32, Float64, Int32, Int64, UInt8 element types.
 * Data can be created from a Cangjie Array (zero-copy where possible).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.core

import onnx4cj.ffi.*
import std.core.*
import std.math.*

/**
 * Tensor<T> — a typed tensor value backed by an OrtValue handle.
 *
 * Lifecycle:
 *   - Created from a Cangjie Array with `Tensor<T>.fromArray(data, shape)`.
 *   - The data array must outlive the Tensor object.
 *   - Released automatically when the Tensor is closed or garbage collected.
 */
public class Tensor<T> <: Resource {
    // Package-internal so OnnxSession can access the raw handle
    internal let handle: OrtValueHandle

    // Shape stored for quick access without re-querying C
    private let _shape: Array<Int64>
    private let _elementType: OrtElementType

    // We hold a reference to the backing data array to prevent GC
    private let _dataRef: Array<T>

    private init(handle: OrtValueHandle,
                 shape: Array<Int64>,
                 elementType: OrtElementType,
                 dataRef: Array<T>) {
        this.handle = handle
        this._shape = shape
        this._elementType = elementType
        this._dataRef = dataRef
    }

    /**
     * Create a Tensor wrapper from an existing OrtValue handle (e.g., an inference output).
     * The shape is queried from the handle via C.
     *
     * @param handle  An OrtValueHandle obtained from ort_run().
     * @throws OnnxException  If the shape query fails.
     */
    internal static func fromHandle(handle: OrtValueHandle): Tensor<T> {
        // Query shape from the handle
        var shapePtr: CPointer<Int64> = CPointer<Int64>.null()
        var rank: UIntNative = 0
        let code = unsafe {
            ort_get_tensor_shape(handle, CPointer.addressOf(&shapePtr), CPointer.addressOf(&rank))
        }
        requireOk(OrtShimStatus.fromInt(code), "Tensor.fromHandle: get shape")

        let shape = Array<Int64>(Int(rank)) { i => unsafe { shapePtr[i] } }
        unsafe { ort_free_shape(shapePtr) }

        let elementType = resolveElementType<T>()
        // Output tensors returned by ORT own their memory internally;
        // we pass an empty array for dataRef because we are not providing
        // backing storage — ORT is.  The data pointer remains valid as long
        // as the OrtValue handle (and thus this Tensor) is alive.
        return Tensor<T>(handle, shape, elementType, Array<T>(0, {_ => zeroValue<T>()}))
    }

    /**
     * Create a tensor from a flat Array<T>.
     *
     * @param data   Flat array of elements in row-major order.
     * @param shape  Dimension sizes (e.g., [1, 3, 224, 224]).
     * @throws TensorException  If element type T is not supported.
     * @throws OnnxException    If OrtValue creation fails.
     */
    public static func fromArray(data: Array<T>, shape: Array<Int64>): Tensor<T> {
        let elementType = resolveElementType<T>()

        // Validate shape product matches data length
        var total: Int64 = 1
        for dim in shape {
            total = total * dim
        }
        if Int64(data.size) != total {
            throw TensorException(
                "Tensor data size ${data.size} does not match shape product ${total}"
            )
        }

        let dataLen = UIntNative(data.size) * UIntNative(sizeOf<T>())
        let shapeArr = shape

        var valueHandle: OrtValueHandle = CPointer<Unit>.null()
        let code = unsafe {
            ort_create_tensor_with_data(
                CPointer<Unit>.fromArray(data),
                dataLen,
                CPointer<Int64>.fromArray(shapeArr),
                UIntNative(shapeArr.size),
                elementType.toInt(),
                CPointer.addressOf(&valueHandle)
            )
        }
        requireOk(OrtShimStatus.fromInt(code), "Tensor.fromArray")
        return Tensor<T>(valueHandle, shape, elementType, data)
    }

    /**
     * Copy the tensor data into a new Cangjie Array<T>.
     *
     * @throws OnnxException If data extraction fails.
     */
    public func toArray(): Array<T> {
        var dataPtr: CPointer<Unit> = CPointer<Unit>.null()
        let code = unsafe {
            ort_get_tensor_data(handle, CPointer.addressOf(&dataPtr))
        }
        requireOk(OrtShimStatus.fromInt(code), "Tensor.toArray")

        var total: Int64 = 1
        for dim in _shape { total = total * dim }
        let count = Int(total)

        let result = Array<T>(count, {_ => zeroValue<T>()})
        unsafe {
            let src = CPointer<T>.fromVoidPtr(dataPtr)
            for i in 0..count {
                result[i] = src[i]
            }
        }
        return result
    }

    /** The shape of this tensor. */
    public prop shape: Array<Int64> { get { _shape } }

    /** The element type of this tensor. */
    public prop elementType: OrtElementType { get { _elementType } }

    /** Total number of elements. */
    public prop numel: Int64 {
        get {
            var n: Int64 = 1
            for d in _shape { n = n * d }
            n
        }
    }

    /** Release the underlying OrtValue. */
    public func close(): Unit {
        unsafe { ort_release_value(handle) }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

private func resolveElementType<T>(): OrtElementType {
    // Use type identity to map Cangjie primitive types to ORT element types
    if T is Float32  { return OrtElementType.Float32  }
    if T is Float64  { return OrtElementType.Float64  }
    if T is Int32    { return OrtElementType.Int32    }
    if T is Int64    { return OrtElementType.Int64    }
    if T is UInt8    { return OrtElementType.UInt8    }
    if T is UInt16   { return OrtElementType.UInt16   }
    if T is Int8     { return OrtElementType.Int8     }
    if T is Int16    { return OrtElementType.Int16    }
    if T is UInt32   { return OrtElementType.UInt32   }
    if T is UInt64   { return OrtElementType.UInt64   }
    throw TensorException("Unsupported tensor element type")
}

private func zeroValue<T>(): T {
    if T is Float32  { return Float32(0.0) as T }
    if T is Float64  { return Float64(0.0) as T }
    if T is Int32    { return Int32(0) as T     }
    if T is Int64    { return Int64(0) as T     }
    if T is UInt8    { return UInt8(0) as T     }
    if T is UInt16   { return UInt16(0) as T    }
    if T is Int8     { return Int8(0) as T      }
    if T is Int16    { return Int16(0) as T     }
    if T is UInt32   { return UInt32(0) as T    }
    if T is UInt64   { return UInt64(0) as T    }
    throw TensorException("Unsupported tensor element type")
}
