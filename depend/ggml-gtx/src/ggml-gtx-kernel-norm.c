//==================================================================
// GGML GTX kernel: NORM
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_NORM_L2_A             0x000000u
#define GGML_GTX_NORM_L2_RESULT        0x020000u

#define GGML_GTX_NORM_BANK_A           0x00000u
#define GGML_GTX_NORM_BANK_B           0x20000u
#define GGML_GTX_NORM_BANK_C           0x30000u
#define GGML_GTX_NORM_BANK_R           0x50000u

#define GGML_GTX_NORM_MAX_DMA_BYTES    65535u
#define GGML_GTX_NORM_NOT_USE          0xBEEFu

static inline uint32_t ggml_gtx_norm_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ggml_gtx_norm_rows_for_tid(uint32_t total_rows, uint8_t tid) {
    const uint32_t q = total_rows / GGML_GTX_SPU_NUM_PER_NEST;
    const uint32_t r = total_rows % GGML_GTX_SPU_NUM_PER_NEST;
    return tid < r ? q + 1u : q;
}

static inline uint32_t ggml_gtx_norm_row_start_for_tid(uint32_t total_rows, uint8_t tid) {
    const uint32_t q = total_rows / GGML_GTX_SPU_NUM_PER_NEST;
    const uint32_t r = total_rows % GGML_GTX_SPU_NUM_PER_NEST;

    if (tid < r) {
        return (uint32_t) tid * (q + 1u);
    }

    return r * (q + 1u) + ((uint32_t) tid - r) * q;
}

static inline uint16_t ggml_gtx_norm_active_tid_mask(uint32_t total_rows) {
    const uint32_t active_tid_count =
        ggml_gtx_norm_min_u32(total_rows, GGML_GTX_SPU_NUM_PER_NEST);

    if (active_tid_count >= GGML_GTX_SPU_NUM_PER_NEST) {
        return 0xFFFFu;
    }

    return (uint16_t) ((1u << active_tid_count) - 1u);
}

static inline float ggml_gtx_norm_f32_from_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    return v.f;
}

static inline uint32_t ggml_gtx_norm_f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    return v.u;
}

static inline float ggml_gtx_norm_fp16_to_f32(uint16_t h) {
    return ggml_gtx_norm_f32_from_bits(ggml_gtx_fp16_to_fp32_bits(h));
}

static inline uint16_t ggml_gtx_norm_f32_to_fp16(float f) {
    return ggml_gtx_fp32_bits_to_fp16(ggml_gtx_norm_f32_to_bits(f));
}

static inline float ggml_gtx_norm_rsqrtf(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }

    const float xhalf = 0.5f * x;
    union {
        float f;
        uint32_t i;
    } v;
    v.f = x;
    v.i = 0x5f3759dfu - (v.i >> 1);
    v.f = v.f * (1.5f - xhalf * v.f * v.f);
    v.f = v.f * (1.5f - xhalf * v.f * v.f);
    return v.f;
}

void ggml_gtx_compute_norm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t rows,
    uint16_t cols,
    uint16_t epsilon);

int ggml_gtx_kernel_norm_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_NORM_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_NORM_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint64_t rows64 = (uint64_t) desc->ne[1] *
                            (uint64_t) desc->ne[2] *
                            (uint64_t) desc->ne[3];
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->ne[0] == 0 ||
        rows64 == 0 ||
        desc->ne[0] > UINT16_MAX ||
        rows64 > UINT16_MAX ||
        desc->src_ne[0][0] != desc->ne[0] ||
        desc->src_ne[0][1] != desc->ne[1] ||
        desc->src_ne[0][2] != desc->ne[2] ||
        desc->src_ne[0][3] != desc->ne[3] ||
        desc->nb[0] != GGML_GTX_DTYPE_SIZE ||
        desc->src_nb[0][0] != GGML_GTX_DTYPE_SIZE ||
        desc->nb[1] != desc->ne[0] * GGML_GTX_DTYPE_SIZE ||
        desc->src_nb[0][1] != desc->src_ne[0][0] * GGML_GTX_DTYPE_SIZE) {
        return -2;
    }

    ggml_gtx_compute_norm(desc->src_addr[0],
                          desc->dst_addr,
                          (uint16_t) rows64,
                          (uint16_t) desc->ne[0],
                          ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[0]));
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_norm_f16(arg);
}

void ggml_gtx_compute_norm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t rows,
    uint16_t cols,
    uint16_t epsilon)
{
#ifdef GGML_GTX_USE_GTX
    if (rows == 0 || cols == 0) {
        return;
    }

    volatile const uint16_t * src = (volatile const uint16_t *) (uintptr_t) ddr_input;
    volatile uint16_t * dst = (volatile uint16_t *) (uintptr_t) ddr_result;
    const float eps = ggml_gtx_norm_fp16_to_f32(epsilon);

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t row_base = (uint32_t) row * cols;
        float mean = 0.0f;
        for (uint16_t col = 0; col < cols; ++col) {
            mean += ggml_gtx_norm_fp16_to_f32(src[row_base + col]);
        }
        mean /= (float) cols;

        float variance = 0.0f;
        for (uint16_t col = 0; col < cols; ++col) {
            const float d = ggml_gtx_norm_fp16_to_f32(src[row_base + col]) - mean;
            variance += d * d;
        }
        variance /= (float) cols;

        const float scale = ggml_gtx_norm_rsqrtf(variance + eps);
        for (uint16_t col = 0; col < cols; ++col) {
            const float v = (ggml_gtx_norm_fp16_to_f32(src[row_base + col]) - mean) * scale;
            dst[row_base + col] = ggml_gtx_norm_f32_to_fp16(v);
        }
    }
#else
    (void) ddr_input; (void) ddr_result; (void) rows; (void) cols; (void) epsilon;
#endif
}
