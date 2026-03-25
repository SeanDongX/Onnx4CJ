/**
 * session.cj — OnnxSession: model loading and inference (RAII)
 *
 * OnnxSession wraps OrtSession and provides:
 *   - Model loading from a file path
 *   - Input/output name and count queries
 *   - Type-safe run() for inference
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.core

import onnx4cj.ffi.*
import std.core.*

/**
 * SessionOptions — builder-style configuration for OnnxSession.
 */
public class SessionOptions <: Resource {
    internal let handle: OrtSessionOptsHandle

    /**
     * Create a new SessionOptions with default settings.
     * @throws OnnxException  If creation fails.
     */
    public init() {
        var optsHandle: OrtSessionOptsHandle = CPointer<Unit>.null()
        let code = unsafe {
            ort_create_session_options(CPointer.addressOf(&optsHandle))
        }
        requireOk(OrtShimStatus.fromInt(code), "SessionOptions: create")
        this.handle = optsHandle
    }

    /**
     * Set the number of threads for intra-op parallelism.
     * @param n  Number of threads (0 = use all available).
     */
    public func setIntraOpNumThreads(n: Int32): SessionOptions {
        let code = unsafe { ort_set_intra_op_num_threads(handle, n) }
        requireOk(OrtShimStatus.fromInt(code), "SessionOptions.setIntraOpNumThreads")
        return this
    }

    /**
     * Set the number of threads for inter-op parallelism.
     * @param n  Number of threads (0 = use all available).
     */
    public func setInterOpNumThreads(n: Int32): SessionOptions {
        let code = unsafe { ort_set_inter_op_num_threads(handle, n) }
        requireOk(OrtShimStatus.fromInt(code), "SessionOptions.setInterOpNumThreads")
        return this
    }

    public func close(): Unit {
        unsafe { ort_release_session_options(handle) }
    }
}

/**
 * OnnxSession — loads an ONNX model and runs inference.
 *
 * Example:
 *   let env = OnnxEnv("app")
 *   let opts = SessionOptions().setIntraOpNumThreads(4)
 *   let session = OnnxSession(env, "model.onnx", opts)
 *   let outputs = session.run([inputTensor])
 */
public class OnnxSession <: Resource {
    private let handle: OrtSessionHandle

    /** Number of model input nodes. */
    public let inputCount: Int

    /** Number of model output nodes. */
    public let outputCount: Int

    /** Names of input nodes. */
    public let inputNames: Array<String>

    /** Names of output nodes. */
    public let outputNames: Array<String>

    /**
     * Load a model from `modelPath`.
     *
     * @param env        The OnnxEnv to use.
     * @param modelPath  Path to the .onnx model file.
     * @param opts       Optional SessionOptions (defaults used if not provided).
     * @throws ModelNotFoundException  If the file does not exist.
     * @throws OnnxException           If session creation fails.
     */
    public init(env: OnnxEnv, modelPath: String, opts: ?SessionOptions) {
        let optsHandle: OrtSessionOptsHandle = match opts {
            case Some(o) => o.handle
            case None    => CPointer<Unit>.null()
        }

        var sessionHandle: OrtSessionHandle = CPointer<Unit>.null()
        let code = unsafe {
            ort_create_session(
                env.handle,
                modelPath.toCString(),
                optsHandle,
                CPointer.addressOf(&sessionHandle)
            )
        }
        let status = OrtShimStatus.fromInt(code)
        if status == OrtShimStatus.NoSuchFile {
            throw ModelNotFoundException(modelPath)
        }
        requireOk(status, "OnnxSession: load '${modelPath}'")
        this.handle = sessionHandle

        // Query input count
        var inCount: UIntNative = 0
        let icCode = unsafe {
            ort_session_get_input_count(sessionHandle, CPointer.addressOf(&inCount))
        }
        requireOk(OrtShimStatus.fromInt(icCode), "OnnxSession: get input count")
        this.inputCount = Int(inCount)

        // Query output count
        var outCount: UIntNative = 0
        let ocCode = unsafe {
            ort_session_get_output_count(sessionHandle, CPointer.addressOf(&outCount))
        }
        requireOk(OrtShimStatus.fromInt(ocCode), "OnnxSession: get output count")
        this.outputCount = Int(outCount)

        // Fetch input names
        this.inputNames = Array<String>(this.inputCount) { i =>
            var nameCStr: CString = CString.null()
            let nc = unsafe {
                ort_session_get_input_name(
                    sessionHandle, UIntNative(i), CPointer.addressOf(&nameCStr))
            }
            requireOk(OrtShimStatus.fromInt(nc), "OnnxSession: get input name ${i}")
            let name = String.fromCString(nameCStr)
            unsafe { ort_free_string(nameCStr) }
            name
        }

        // Fetch output names
        this.outputNames = Array<String>(this.outputCount) { i =>
            var nameCStr: CString = CString.null()
            let nc = unsafe {
                ort_session_get_output_name(
                    sessionHandle, UIntNative(i), CPointer.addressOf(&nameCStr))
            }
            requireOk(OrtShimStatus.fromInt(nc), "OnnxSession: get output name ${i}")
            let name = String.fromCString(nameCStr)
            unsafe { ort_free_string(nameCStr) }
            name
        }
    }

    /**
     * Convenience constructor using default SessionOptions.
     */
    public init(env: OnnxEnv, modelPath: String) {
        this(env, modelPath, None)
    }

    /**
     * Run inference.
     *
     * @param inputs  Array of input tensors in the same order as `inputNames`.
     * @return        Array of output tensors (caller is responsible for releasing).
     * @throws OnnxException  If input count mismatches or inference fails.
     */
    public func run<T>(inputs: Array<Tensor<T>>): Array<Tensor<T>> {
        if inputs.size != inputCount {
            throw OnnxException(
                OrtShimStatus.InvalidArgument,
                "Expected ${inputCount} inputs, got ${inputs.size}"
            )
        }

        // Build C-compatible arrays of names and handles
        let inNames  = Array<CString>(inputCount) { i => inputNames[i].toCString() }
        let outNames = Array<CString>(outputCount) { i => outputNames[i].toCString() }
        let inHandles = Array<OrtValueHandle>(inputCount) { i => inputs[i].handle }
        var outHandles = Array<OrtValueHandle>(outputCount) { _ => CPointer<Unit>.null() }

        let code = unsafe {
            ort_run(
                handle,
                CPointer<CString>.fromArray(inNames),
                CPointer<OrtValueHandle>.fromArray(inHandles),
                UIntNative(inputCount),
                CPointer<CString>.fromArray(outNames),
                UIntNative(outputCount),
                CPointer<OrtValueHandle>.fromArray(outHandles)
            )
        }
        requireOk(OrtShimStatus.fromInt(code), "OnnxSession.run")

        // Wrap output handles into Tensor objects
        // Note: shape queried lazily from the handle; we use empty shape placeholder
        Array<Tensor<T>>(outputCount) { i =>
            Tensor<T>.fromHandle(outHandles[i])
        }
    }

    public func close(): Unit {
        unsafe { ort_release_session(handle) }
    }
}
