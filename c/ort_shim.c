/*
 * ort_shim.c — ONNX Runtime C shim implementation for Cangjie FFI
 *
 * Wraps the ONNX Runtime C API dispatch table into plain C functions
 * that are straightforward to call from Cangjie via FFI.
 *
 * Build:
 *   gcc -shared -fPIC -o libort_shim.so ort_shim.c \
 *       -I${ORT_HOME}/include \
 *       -L${ORT_HOME}/lib -lonnxruntime
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ort_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ONNX Runtime public C API header */
#include <onnxruntime_c_api.h>

/* --------------------------------------------------------------------------
 * Globals
 * -------------------------------------------------------------------------- */

static const OrtApi* g_ort = NULL;
static char g_last_error[4096] = {0};

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static ORT_SHIM_STATUS ort_check_status(OrtStatus* status) {
    if (status == NULL) {
        return ORT_SHIM_OK;
    }
    OrtErrorCode code = g_ort->GetErrorCode(status);
    const char* msg = g_ort->GetErrorMessage(status);
    snprintf(g_last_error, sizeof(g_last_error), "[OrtError %d] %s", (int)code, msg);
    g_ort->ReleaseStatus(status);
    return (ORT_SHIM_STATUS)code;
}

/* --------------------------------------------------------------------------
 * Initialization
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_init(void) {
    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == NULL) {
        snprintf(g_last_error, sizeof(g_last_error), "OrtGetApiBase() returned NULL");
        return ORT_SHIM_FAIL;
    }
    g_ort = api_base->GetApi(ORT_API_VERSION);
    if (g_ort == NULL) {
        snprintf(g_last_error, sizeof(g_last_error),
                 "OrtApiBase->GetApi(%d) returned NULL", ORT_API_VERSION);
        return ORT_SHIM_FAIL;
    }
    return ORT_SHIM_OK;
}

const char* ort_get_last_error(void) {
    return g_last_error;
}

/* --------------------------------------------------------------------------
 * Environment
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_create_env(const char* log_id, OrtEnvHandle* out_env) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, log_id, &env);
    ORT_SHIM_STATUS rc = ort_check_status(status);
    if (rc == ORT_SHIM_OK) {
        *out_env = (OrtEnvHandle)env;
    }
    return rc;
}

void ort_release_env(OrtEnvHandle env) {
    if (g_ort != NULL && env != NULL) {
        g_ort->ReleaseEnv((OrtEnv*)env);
    }
}

/* --------------------------------------------------------------------------
 * Session Options
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_create_session_options(OrtSessionOptionsHandle* out_opts) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtSessionOptions* opts = NULL;
    OrtStatus* status = g_ort->CreateSessionOptions(&opts);
    ORT_SHIM_STATUS rc = ort_check_status(status);
    if (rc == ORT_SHIM_OK) {
        *out_opts = (OrtSessionOptionsHandle)opts;
    }
    return rc;
}

void ort_release_session_options(OrtSessionOptionsHandle opts) {
    if (g_ort != NULL && opts != NULL) {
        g_ort->ReleaseSessionOptions((OrtSessionOptions*)opts);
    }
}

ORT_SHIM_STATUS ort_set_intra_op_num_threads(OrtSessionOptionsHandle opts, int num_threads) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtStatus* status = g_ort->SetIntraOpNumThreads((OrtSessionOptions*)opts, num_threads);
    return ort_check_status(status);
}

ORT_SHIM_STATUS ort_set_inter_op_num_threads(OrtSessionOptionsHandle opts, int num_threads) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtStatus* status = g_ort->SetInterOpNumThreads((OrtSessionOptions*)opts, num_threads);
    return ort_check_status(status);
}

/* --------------------------------------------------------------------------
 * Session
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_create_session(OrtEnvHandle env,
                                    const char* model_path,
                                    OrtSessionOptionsHandle opts,
                                    OrtSessionHandle* out_session) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtSession* session = NULL;
    OrtStatus* status = g_ort->CreateSession(
        (OrtEnv*)env,
        model_path,
        (OrtSessionOptions*)opts,
        &session
    );
    ORT_SHIM_STATUS rc = ort_check_status(status);
    if (rc == ORT_SHIM_OK) {
        *out_session = (OrtSessionHandle)session;
    }
    return rc;
}

void ort_release_session(OrtSessionHandle session) {
    if (g_ort != NULL && session != NULL) {
        g_ort->ReleaseSession((OrtSession*)session);
    }
}

ORT_SHIM_STATUS ort_session_get_input_count(OrtSessionHandle session, size_t* out_count) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtStatus* status = g_ort->SessionGetInputCount((OrtSession*)session, out_count);
    return ort_check_status(status);
}

ORT_SHIM_STATUS ort_session_get_output_count(OrtSessionHandle session, size_t* out_count) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtStatus* status = g_ort->SessionGetOutputCount((OrtSession*)session, out_count);
    return ort_check_status(status);
}

ORT_SHIM_STATUS ort_session_get_input_name(OrtSessionHandle session,
                                            size_t index,
                                            char** out_name) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    OrtStatus* status = g_ort->SessionGetInputName(
        (OrtSession*)session, index, allocator, out_name);
    return ort_check_status(status);
}

ORT_SHIM_STATUS ort_session_get_output_name(OrtSessionHandle session,
                                             size_t index,
                                             char** out_name) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    OrtStatus* status = g_ort->SessionGetOutputName(
        (OrtSession*)session, index, allocator, out_name);
    return ort_check_status(status);
}

void ort_free_string(char* str) {
    if (g_ort != NULL && str != NULL) {
        OrtAllocator* allocator = NULL;
        g_ort->GetAllocatorWithDefaultOptions(&allocator);
        allocator->Free(allocator, str);
    }
}

/* --------------------------------------------------------------------------
 * Tensor Creation
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_create_tensor_with_data(void* data,
                                              size_t data_len,
                                              const int64_t* shape,
                                              size_t shape_len,
                                              ORT_ELEMENT_TYPE element_type,
                                              OrtValueHandle* out_tensor) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;

    OrtMemoryInfo* mem_info = NULL;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    ORT_SHIM_STATUS rc = ort_check_status(status);
    if (rc != ORT_SHIM_OK) return rc;

    OrtValue* tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        mem_info,
        data,
        data_len,
        shape,
        shape_len,
        (ONNXTensorElementDataType)element_type,
        &tensor
    );
    g_ort->ReleaseMemoryInfo(mem_info);

    rc = ort_check_status(status);
    if (rc == ORT_SHIM_OK) {
        *out_tensor = (OrtValueHandle)tensor;
    }
    return rc;
}

void ort_release_value(OrtValueHandle value) {
    if (g_ort != NULL && value != NULL) {
        g_ort->ReleaseValue((OrtValue*)value);
    }
}

ORT_SHIM_STATUS ort_get_tensor_data(OrtValueHandle tensor, void** out_data) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;
    OrtStatus* status = g_ort->GetTensorMutableData((OrtValue*)tensor, out_data);
    return ort_check_status(status);
}

ORT_SHIM_STATUS ort_get_tensor_shape(OrtValueHandle tensor,
                                      int64_t** out_shape,
                                      size_t* out_rank) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;

    OrtTensorTypeAndShapeInfo* info = NULL;
    OrtStatus* status = g_ort->GetTensorTypeAndShape((OrtValue*)tensor, &info);
    ORT_SHIM_STATUS rc = ort_check_status(status);
    if (rc != ORT_SHIM_OK) return rc;

    status = g_ort->GetDimensionsCount(info, out_rank);
    rc = ort_check_status(status);
    if (rc != ORT_SHIM_OK) {
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        return rc;
    }

    *out_shape = (int64_t*)malloc(sizeof(int64_t) * (*out_rank));
    if (*out_shape == NULL) {
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        return ORT_SHIM_FAIL;
    }

    status = g_ort->GetDimensions(info, *out_shape, *out_rank);
    g_ort->ReleaseTensorTypeAndShapeInfo(info);
    rc = ort_check_status(status);
    if (rc != ORT_SHIM_OK) {
        free(*out_shape);
        *out_shape = NULL;
    }
    return rc;
}

void ort_free_shape(int64_t* shape) {
    free(shape);
}

/* --------------------------------------------------------------------------
 * Inference
 * -------------------------------------------------------------------------- */

ORT_SHIM_STATUS ort_run(OrtSessionHandle session,
                         const char* const* input_names,
                         const OrtValueHandle* inputs,
                         size_t input_count,
                         const char* const* output_names,
                         size_t output_count,
                         OrtValueHandle* out_outputs) {
    if (g_ort == NULL) return ORT_SHIM_FAIL;

    OrtStatus* status = g_ort->Run(
        (OrtSession*)session,
        NULL,                           /* RunOptions (NULL = defaults) */
        input_names,
        (const OrtValue* const*)inputs,
        input_count,
        output_names,
        output_count,
        (OrtValue**)out_outputs
    );
    return ort_check_status(status);
}
