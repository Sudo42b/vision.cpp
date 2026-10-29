#ifndef GGML_GTX_KERNEL_ARGS_H
#define GGML_GTX_KERNEL_ARGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel-private payloads are shared by the host packer and the uploaded
 * kernel only. Firmware transports these bytes without interpreting them. */
#define GGML_GTX_MUL_MAT_ARGS_VERSION 1u
#define GGML_GTX_ADD_ARGS_VERSION 1u
#define GGML_GTX_MUL_ARGS_VERSION 1u

typedef struct ggml_gtx_mul_mat_args {
    uint32_t version;
    uint32_t size;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t reserved[3];
} ggml_gtx_mul_mat_args_t;

typedef struct ggml_gtx_add_args {
    uint32_t version;
    uint32_t size;
    uint32_t scalar_count;
    uint32_t reserved;
    uint16_t scalars[1];
} ggml_gtx_add_args_t;

#define GGML_GTX_ADD_ARGS_HEADER_SIZE offsetof(ggml_gtx_add_args_t, scalars)

typedef struct ggml_gtx_mul_args {
    uint32_t version;
    uint32_t size;
    uint32_t scalar_count;
    uint32_t reserved;
    uint16_t scalars[1];
} ggml_gtx_mul_args_t;

#define GGML_GTX_MUL_ARGS_HEADER_SIZE offsetof(ggml_gtx_mul_args_t, scalars)

#ifdef __cplusplus
}
#endif

#endif
