/**
 * env.cj — OnnxEnv: ONNX Runtime environment management (RAII)
 *
 * OnnxEnv wraps OrtEnv and ensures resources are released on destruction.
 * It also calls ort_init() exactly once to initialize the global OrtApi.
 *
 * Usage:
 *   let env = OnnxEnv("my_app")
 *   // ... use env ...
 *   // env released automatically when it goes out of scope
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.core

import onnx4cj.ffi.*
import std.core.*
import std.sync.*

/**
 * OnnxEnv — wraps OrtEnv with RAII lifecycle management.
 *
 * One OnnxEnv instance is typically shared across multiple sessions.
 * Thread-safe initialization is handled internally.
 */
public class OnnxEnv <: Resource {
    private static var initialized: Bool = false
    private static let initLock: Mutex = Mutex()

    // Raw C handle — package-internal so OnnxSession can access it
    internal let handle: OrtEnvHandle

    /**
     * Create a new OnnxEnv.
     *
     * @param logId  Human-readable identifier shown in log output.
     * @throws OnnxException  If ONNX Runtime initialization or env creation fails.
     */
    public init(logId: String) {
        OnnxEnv.ensureInitialized()

        var envHandle: OrtEnvHandle = CPointer<Unit>.null()
        let code = unsafe {
            ort_create_env(logId.toCString(), CPointer.addressOf(&envHandle))
        }
        requireOk(OrtShimStatus.fromInt(code), "OnnxEnv: create env '${logId}'")
        this.handle = envHandle
    }

    /**
     * Release the underlying OrtEnv.
     * Called automatically when the object is finalized or used with `use`.
     */
    public func close(): Unit {
        unsafe { ort_release_env(handle) }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private static func ensureInitialized(): Unit {
        initLock.lock()
        defer { initLock.unlock() }
        if !initialized {
            let code = unsafe { ort_init() }
            requireOk(OrtShimStatus.fromInt(code), "OnnxEnv: ort_init")
            initialized = true
        }
    }
}
