/**
 * ort_api.cj — Cangjie FFI declarations for ort_shim
 *
 * Each `foreign func` declaration corresponds to a C function in
 * c/ort_shim.c.  The names must match exactly.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package onnx4cj.ffi

import std.core.*

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

@C
foreign func ort_init(): Int32

@C
foreign func ort_get_last_error(): CString

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

@C
foreign func ort_create_env(
    log_id: CString,
    out_env: CPointer<OrtEnvHandle>
): Int32

@C
foreign func ort_release_env(env: OrtEnvHandle): Unit

// ---------------------------------------------------------------------------
// Session Options
// ---------------------------------------------------------------------------

@C
foreign func ort_create_session_options(
    out_opts: CPointer<OrtSessionOptsHandle>
): Int32

@C
foreign func ort_release_session_options(opts: OrtSessionOptsHandle): Unit

@C
foreign func ort_set_intra_op_num_threads(
    opts: OrtSessionOptsHandle,
    num_threads: Int32
): Int32

@C
foreign func ort_set_inter_op_num_threads(
    opts: OrtSessionOptsHandle,
    num_threads: Int32
): Int32

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

@C
foreign func ort_create_session(
    env: OrtEnvHandle,
    model_path: CString,
    opts: OrtSessionOptsHandle,
    out_session: CPointer<OrtSessionHandle>
): Int32

@C
foreign func ort_release_session(session: OrtSessionHandle): Unit

@C
foreign func ort_session_get_input_count(
    session: OrtSessionHandle,
    out_count: CPointer<UIntNative>
): Int32

@C
foreign func ort_session_get_output_count(
    session: OrtSessionHandle,
    out_count: CPointer<UIntNative>
): Int32

@C
foreign func ort_session_get_input_name(
    session: OrtSessionHandle,
    index: UIntNative,
    out_name: CPointer<CString>
): Int32

@C
foreign func ort_session_get_output_name(
    session: OrtSessionHandle,
    index: UIntNative,
    out_name: CPointer<CString>
): Int32

@C
foreign func ort_free_string(str: CString): Unit

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

@C
foreign func ort_create_tensor_with_data(
    data: CPointer<Unit>,
    data_len: UIntNative,
    shape: CPointer<Int64>,
    shape_len: UIntNative,
    element_type: Int32,
    out_tensor: CPointer<OrtValueHandle>
): Int32

@C
foreign func ort_release_value(value: OrtValueHandle): Unit

@C
foreign func ort_get_tensor_data(
    tensor: OrtValueHandle,
    out_data: CPointer<CPointer<Unit>>
): Int32

@C
foreign func ort_get_tensor_shape(
    tensor: OrtValueHandle,
    out_shape: CPointer<CPointer<Int64>>,
    out_rank: CPointer<UIntNative>
): Int32

@C
foreign func ort_free_shape(shape: CPointer<Int64>): Unit

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

@C
foreign func ort_run(
    session: OrtSessionHandle,
    input_names: CPointer<CString>,
    inputs: CPointer<OrtValueHandle>,
    input_count: UIntNative,
    output_names: CPointer<CString>,
    output_count: UIntNative,
    out_outputs: CPointer<OrtValueHandle>
): Int32

// ---------------------------------------------------------------------------
// Convenience wrapper: check status and get error message
// ---------------------------------------------------------------------------

public func checkOrtStatus(code: Int32): OrtShimStatus {
    OrtShimStatus.fromInt(code)
}

public func getLastOrtError(): String {
    let cstr = unsafe { ort_get_last_error() }
    String.fromCString(cstr)
}
