/*
 * ort_shim.h — ONNX Runtime C shim layer for Cangjie FFI
 *
 * This header provides a simplified interface over the ONNX Runtime C API,
 * designed to be called from Cangjie via FFI. It avoids the complexity of
 * the raw OrtApi dispatch table and provides straightforward C functions.
 *
 * Usage:
 *   #include "ort_shim.h"
 *   ORT_SHIM_STATUS status = ort_create_env("session_name", &env_ptr);
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ORT_SHIM_H
#define ORT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef void* OrtEnvHandle;
typedef void* OrtSessionHandle;
typedef void* OrtSessionOptionsHandle;
typedef void* OrtValueHandle;
typedef void* OrtMemoryInfoHandle;
typedef void* OrtAllocatorHandle;
typedef void* OrtRunOptionsHandle;
typedef void* OrtTypeInfoHandle;
typedef void* OrtTensorTypeAndShapeInfoHandle;

/* Status codes */
typedef enum {
    ORT_SHIM_OK = 0,
    ORT_SHIM_FAIL = 1,
    ORT_SHIM_INVALID_ARGUMENT = 2,
    ORT_SHIM_NO_SUCHFILE = 3,
    ORT_SHIM_NO_MODEL = 4,
    ORT_SHIM_ENGINE_ERROR = 5,
    ORT_SHIM_RUNTIME_EXCEPTION = 6,
    ORT_SHIM_INVALID_PROTOBUF = 7,
    ORT_SHIM_MODEL_LOADED = 8,
    ORT_SHIM_NOT_IMPLEMENTED = 9,
    ORT_SHIM_INVALID_GRAPH = 10,
    ORT_SHIM_EP_FAIL = 11
} ORT_SHIM_STATUS;

/* Element types (mirrors OrtElementType) */
typedef enum {
    ORT_TYPE_UNDEFINED = 0,
    ORT_TYPE_FLOAT = 1,
    ORT_TYPE_UINT8 = 2,
    ORT_TYPE_INT8 = 3,
    ORT_TYPE_UINT16 = 4,
    ORT_TYPE_INT16 = 5,
    ORT_TYPE_INT32 = 6,
    ORT_TYPE_INT64 = 7,
    ORT_TYPE_STRING = 8,
    ORT_TYPE_BOOL = 9,
    ORT_TYPE_FLOAT16 = 10,
    ORT_TYPE_DOUBLE = 11,
    ORT_TYPE_UINT32 = 12,
    ORT_TYPE_UINT64 = 13,
    ORT_TYPE_COMPLEX64 = 14,
    ORT_TYPE_COMPLEX128 = 15,
    ORT_TYPE_BFLOAT16 = 16
} ORT_ELEMENT_TYPE;

/* ------------------------------------------------------------------
 * Initialization
 * ------------------------------------------------------------------ */

/** Initialize the OrtApi. Must be called once before any other function. */
ORT_SHIM_STATUS ort_init(void);

/** Get the last error message string (valid until next error). */
const char* ort_get_last_error(void);

/* ------------------------------------------------------------------
 * Environment
 * ------------------------------------------------------------------ */

/**
 * Create an OrtEnv with WARNING log level.
 * @param log_id    Human-readable identifier for log messages.
 * @param out_env   Output handle; caller must call ort_release_env().
 */
ORT_SHIM_STATUS ort_create_env(const char* log_id, OrtEnvHandle* out_env);

/** Release an OrtEnv created by ort_create_env(). */
void ort_release_env(OrtEnvHandle env);

/* ------------------------------------------------------------------
 * Session Options
 * ------------------------------------------------------------------ */

ORT_SHIM_STATUS ort_create_session_options(OrtSessionOptionsHandle* out_opts);
void ort_release_session_options(OrtSessionOptionsHandle opts);
ORT_SHIM_STATUS ort_set_intra_op_num_threads(OrtSessionOptionsHandle opts, int num_threads);
ORT_SHIM_STATUS ort_set_inter_op_num_threads(OrtSessionOptionsHandle opts, int num_threads);

/* ------------------------------------------------------------------
 * Session
 * ------------------------------------------------------------------ */

/**
 * Load an ONNX model from a file path.
 * @param env         Environment handle.
 * @param model_path  Null-terminated path to the .onnx file.
 * @param opts        Session options handle (may be NULL for defaults).
 * @param out_session Output handle; caller must call ort_release_session().
 */
ORT_SHIM_STATUS ort_create_session(OrtEnvHandle env,
                                    const char* model_path,
                                    OrtSessionOptionsHandle opts,
                                    OrtSessionHandle* out_session);

/** Release an OrtSession created by ort_create_session(). */
void ort_release_session(OrtSessionHandle session);

/** Get the number of model input nodes. */
ORT_SHIM_STATUS ort_session_get_input_count(OrtSessionHandle session, size_t* out_count);

/** Get the number of model output nodes. */
ORT_SHIM_STATUS ort_session_get_output_count(OrtSessionHandle session, size_t* out_count);

/** Get the name of the i-th input node (caller must free with ort_free_string). */
ORT_SHIM_STATUS ort_session_get_input_name(OrtSessionHandle session,
                                            size_t index,
                                            char** out_name);

/** Get the name of the i-th output node (caller must free with ort_free_string). */
ORT_SHIM_STATUS ort_session_get_output_name(OrtSessionHandle session,
                                             size_t index,
                                             char** out_name);

/** Free a string allocated by ort_session_get_input/output_name. */
void ort_free_string(char* str);

/* ------------------------------------------------------------------
 * Tensor Creation
 * ------------------------------------------------------------------ */

/**
 * Create a tensor backed by existing data (zero-copy where possible).
 * @param data          Pointer to element data buffer.
 * @param data_len      Size of data in bytes.
 * @param shape         Pointer to array of dimension sizes.
 * @param shape_len     Number of dimensions.
 * @param element_type  ORT_ELEMENT_TYPE value.
 * @param out_tensor    Output value handle; caller must call ort_release_value().
 */
ORT_SHIM_STATUS ort_create_tensor_with_data(void* data,
                                              size_t data_len,
                                              const int64_t* shape,
                                              size_t shape_len,
                                              ORT_ELEMENT_TYPE element_type,
                                              OrtValueHandle* out_tensor);

/** Release an OrtValue created by ort_create_tensor_with_data(). */
void ort_release_value(OrtValueHandle value);

/** Get a pointer to the tensor's data buffer (zero-copy). */
ORT_SHIM_STATUS ort_get_tensor_data(OrtValueHandle tensor, void** out_data);

/** Get the shape of a tensor as an array of int64. out_shape must be freed by caller. */
ORT_SHIM_STATUS ort_get_tensor_shape(OrtValueHandle tensor,
                                      int64_t** out_shape,
                                      size_t* out_rank);

/** Free a shape array returned by ort_get_tensor_shape(). */
void ort_free_shape(int64_t* shape);

/* ------------------------------------------------------------------
 * Inference
 * ------------------------------------------------------------------ */

/**
 * Run inference.
 * @param session        Session handle.
 * @param input_names    Array of C-string input node names.
 * @param inputs         Array of input OrtValue handles.
 * @param input_count    Number of inputs.
 * @param output_names   Array of C-string output node names.
 * @param output_count   Number of outputs.
 * @param out_outputs    Output array of OrtValue handles (caller must release each).
 */
ORT_SHIM_STATUS ort_run(OrtSessionHandle session,
                         const char* const* input_names,
                         const OrtValueHandle* inputs,
                         size_t input_count,
                         const char* const* output_names,
                         size_t output_count,
                         OrtValueHandle* out_outputs);

#ifdef __cplusplus
}
#endif

#endif /* ORT_SHIM_H */
