/**
 * error.cj — Error types for Onnx4CJ
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.core

import onnx4cj.ffi.*

/**
 * Exception thrown when an ONNX Runtime operation fails.
 */
public class OnnxException <: Exception {
    public let status: OrtShimStatus
    public let message: String

    public init(status: OrtShimStatus, message: String) {
        super(message)
        this.status = status
        this.message = message
    }

    public init(message: String) {
        super(message)
        this.status = OrtShimStatus.Fail
        this.message = message
    }
}

/**
 * Exception thrown when a model file cannot be found.
 */
public class ModelNotFoundException <: OnnxException {
    public init(modelPath: String) {
        super(OrtShimStatus.NoSuchFile, "Model file not found: ${modelPath}")
    }
}

/**
 * Exception thrown when an invalid tensor shape or type is provided.
 */
public class TensorException <: OnnxException {
    public init(message: String) {
        super(OrtShimStatus.InvalidArgument, message)
    }
}

/**
 * Internal helper: raise OnnxException if status != Ok.
 */
public func requireOk(status: OrtShimStatus, context: String): Unit {
    if status != OrtShimStatus.Ok {
        let detail = getLastOrtError()
        throw OnnxException(status, "${context}: ${detail}")
    }
}
