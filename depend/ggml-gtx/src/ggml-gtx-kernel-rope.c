//==================================================================
// GGML Baremetal - RoPE (Rotary Position Embedding) Kernel
//
// GTX path now follows the CPU rope semantics directly on the original
// FP16 tensor layout instead of adapting the small packed-row reference
// micro-kernel contract.
//==================================================================

#include <stdint.h>

#include "ggml-gtx-compute.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#ifdef GGML_GTX_USE_GTX
#include "ggml-gtx/intrinsics/intrin.h"
#endif

static int ggml_gtx_compute_rope_l2(const ggml_gtx_kernel_launch_desc_t * desc);

int ggml_gtx_kernel_rope_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_ROPE_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_ROPE_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode == GGML_GTX_MEMORY_MODE_L2_STASHED_N1) {
        return ggml_gtx_compute_rope_l2(desc);
    }

    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_DDR_DIRECT ||
        desc->dst_addr == 0 || desc->src_addr[0] == 0 || desc->src_addr[1] == 0) {
        return -2;
    }

    ggml_gtx_compute_rope(desc->src_addr[0],
                          desc->src_addr[1],
                          desc->dst_addr,
                          (uint16_t) desc->src_ne[0][0],
                          (uint16_t) desc->src_ne[0][1],
                          (uint16_t) desc->src_ne[0][2],
                          (uint16_t) desc->src_ne[0][3],
                          (uint16_t) desc->op_params[1],
                          (uint8_t) desc->op_params[2],
                          (uint32_t) desc->op_params[5],
                          (uint32_t) desc->op_params[6],
                          (uint32_t) desc->op_params[8],
                          GGML_GTX_NEST_NUM,
                          GGML_GTX_SPU_NUM_PER_NEST);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_rope_f16(arg);
}

static float ggml_gtx_bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value_bits;
    value_bits.u = bits;
    return value_bits.f;
}

static uint32_t ggml_gtx_float_to_bits(float value) {
    union { uint32_t u; float f; } value_bits;
    value_bits.f = value;
    return value_bits.u;
}

static float ggml_gtx_expf(float arg) {
    if (arg > 88.0f) {
        arg = 88.0f;
    }
    if (arg < -88.0f) {
        arg = -88.0f;
    }

    const float ln2 = 0.69314718056f;
    int exponent = (int)(arg / ln2);
    float reduced = arg - ((float) exponent * ln2);
    float reduced_sq = reduced * reduced;
    float poly = 1.0f + reduced + reduced_sq * 0.5f + reduced * reduced_sq * (1.0f / 6.0f) + reduced_sq * reduced_sq * (1.0f / 24.0f);

    if (exponent >= 0) {
        while (exponent-- > 0) {
            poly *= 2.0f;
        }
    } else {
        while (exponent++ < 0) {
            poly *= 0.5f;
        }
    }

    return poly;
}

static float ggml_gtx_logf(float arg) {
    if (arg <= 0.0f) {
        return -3.402823466e+38f;
    }

    int exponent = 0;
    while (arg >= 2.0f) {
        arg *= 0.5f;
        exponent++;
    }
    while (arg < 1.0f) {
        arg *= 2.0f;
        exponent--;
    }

    const float y = (arg - 1.0f) / (arg + 1.0f);
    const float y_sq = y * y;
    float acc = y;
    float term = y;

    term *= y_sq;
    acc += term / 3.0f;
    term *= y_sq;
    acc += term / 5.0f;
    term *= y_sq;
    acc += term / 7.0f;
    term *= y_sq;
    acc += term / 9.0f;

    const float result = 2.0f * acc + ((float) exponent * 0.69314718056f);
    return result;
}

static int ggml_gtx_floor_to_i32(float arg) {
    int truncated = (int)arg;
    if ((float)truncated > arg) {
        truncated--;
    }
    return truncated;
}

static float ggml_gtx_sinf(float arg) {
    const float half_pi = 1.57079632679f;
    const float inv_half_pi = 0.63661977236f;
    int quadrant = ggml_gtx_floor_to_i32(arg * inv_half_pi + (arg >= 0.0f ? 0.5f : -0.5f));
    float reduced = arg - (float)quadrant * half_pi;
    float sign = 1.0f;

    switch (quadrant & 3) {
        case 0:
            break;
        case 1: {
            float tmp = reduced;
            reduced = half_pi - tmp;
        } break;
        case 2:
            sign = -1.0f;
            break;
        default: {
            float tmp = reduced;
            reduced = half_pi - tmp;
            sign = -1.0f;
        } break;
    }

    const float x2 = reduced * reduced;
    const float poly = (((((-1.0f / 39916800.0f) * x2 + (1.0f / 362880.0f)) * x2
                       - (1.0f / 5040.0f)) * x2 + (1.0f / 120.0f)) * x2
                       - (1.0f / 6.0f)) * x2 + 1.0f;

    return sign * reduced * poly;
}

static float ggml_gtx_cosf(float arg) {
    return ggml_gtx_sinf(arg + 1.57079632679f);
}

#ifdef GGML_GTX_USE_GTX
#define GGML_GTX_ROPE_VEC_ELEMS      4u
#define GGML_GTX_ROPE_VEC_BYTES      (GGML_GTX_ROPE_VEC_ELEMS * 2u)
#define GGML_GTX_ROPE_SVR_BYTES      32u
#define GGML_GTX_ROPE_FP16_ONE       0x3C00u
#define GGML_GTX_ROPE_FP16_NEG_HALF  0xB800u
#define GGML_GTX_ROPE_FP16_INV24     0x2955u
#define GGML_GTX_ROPE_FP16_NEG_INV6  0xB155u
#define GGML_GTX_ROPE_FP16_INV120    0x2044u
#define GGML_GTX_ROPE_FP16_INV_2PI   0x3118u
#define GGML_GTX_ROPE_FP16_2PI       0x4648u
#define GGML_GTX_ROPE_SIGN_MASK0     0x3C00BC003C00BC00ULL
#define GGML_GTX_SINCOS_VEC_ELEMS    16u
#define GGML_GTX_SINCOS_VEC_BYTES    (GGML_GTX_SINCOS_VEC_ELEMS * 2u)

static void ggml_gtx_sincos_vec_inline(
    uint64_t src,
    uint64_t dst,
    uint32_t scratch_base,
    uint8_t nest_id,
    uint8_t thread_id,
    uint8_t sincos_sel,
    uint8_t l2_io)
{
    const uint32_t work_a = scratch_base;
    const uint32_t work_b = scratch_base + GGML_GTX_ROPE_SCRATCH_WORK_B_OFFSET;
    __start_plan(nest_id);
        __start_shared();
            if (l2_io) {
                __copy((uint32_t)src, work_a,
                       GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                       1, GGML_GTX_SINCOS_VEC_BYTES);
            } else {
                __load(src, work_a,
                       GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                       1, GGML_GTX_SINCOS_VEC_BYTES);
            }
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(work_a, GGML_GTX_L1_BANK_A_OFFSET,
                   GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                   1, GGML_GTX_SINCOS_VEC_BYTES);

            __load_svr(GGML_GTX_L1_BANK_A_OFFSET, 0);
            __mul_is(0, GGML_GTX_ROPE_FP16_INV_2PI, 0, 0);
            __rne_i(0, 0);

            __wrspr(0x800 + 4, 0, 0x012BC248C2493C00ULL, 0);
            __store_svr(GGML_GTX_L1_BANK_A_OFFSET + GGML_GTX_SINCOS_VEC_BYTES, 0);
            __store_svr(GGML_GTX_L1_BANK_B_OFFSET, 1);

            __store(GGML_GTX_L1_BANK_A_OFFSET, work_a,
                    GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                    1, GGML_GTX_SINCOS_VEC_BYTES);
            for (uint8_t i = 0; i < 3; ++i) {
                __store(GGML_GTX_L1_BANK_A_OFFSET + GGML_GTX_SINCOS_VEC_BYTES,
                        work_a + (uint32_t)(i + 1) * GGML_GTX_SINCOS_VEC_BYTES,
                        GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                        1, GGML_GTX_SINCOS_VEC_BYTES);
            }
        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            __tpose(work_a, work_b, 16, 4, 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(work_b, GGML_GTX_L1_BANK_C_OFFSET,
                   GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                   4, GGML_GTX_SINCOS_VEC_BYTES);

            __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_A_OFFSET,
                           GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_C_OFFSET);
            __mm(16, 4, 1);

            __load_svr(GGML_GTX_L1_BANK_R_OFFSET, 0);
            __mul_is(0, 0x3518, 0, 0);
            __rne_i(0, 0);

            __wrspr(0x800 + 4, 0, 0x0095BE48BE493C00ULL, 0);
            __store_svr(GGML_GTX_L1_BANK_R_OFFSET + GGML_GTX_SINCOS_VEC_BYTES, 0);
            __store_svr(GGML_GTX_L1_BANK_B_OFFSET, 1);

            __store(GGML_GTX_L1_BANK_R_OFFSET, work_a,
                    GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                    1, GGML_GTX_SINCOS_VEC_BYTES);
            for (uint8_t i = 0; i < 3; ++i) {
                __store(GGML_GTX_L1_BANK_R_OFFSET + GGML_GTX_SINCOS_VEC_BYTES,
                        work_a + (uint32_t)(i + 1) * GGML_GTX_SINCOS_VEC_BYTES,
                        GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                        1, GGML_GTX_SINCOS_VEC_BYTES);
            }
        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            __tpose(work_a, work_b, 16, 4, 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(work_b, GGML_GTX_L1_BANK_C_OFFSET,
                   GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                   4, GGML_GTX_SINCOS_VEC_BYTES);

            __mm(16, 4, 1);

            __mul_is(0, 0x3800, 1, 0);
            __ceil_i(1, 1);
            __mul_is(1, 0x4000, 1, 0);
            __sub_ii(0, 1, 1);
            __add_is(1, 0x3C00, 2, 0);
            __add_ii(1, 2, 0);
            __store_svr(GGML_GTX_L1_BANK_C_OFFSET, 0);

            __load_svr(GGML_GTX_L1_BANK_R_OFFSET, 1);
            __mvsvr(1, 2, 0);
            __mul_ii(1, 2, 2);

            if (sincos_sel) {
                __wrspr(0x800 + 4, 0, 0x3C003C003C003C00ULL, 0);
                __cpsvr(1, 3);
                __store_svr(GGML_GTX_L1_BANK_R_OFFSET, 1);
            }

            for (uint8_t i = 0; i < (uint8_t)(3 + sincos_sel); ++i) {
                __mul_ii(1, 2, 1);
                __store_svr(GGML_GTX_L1_BANK_R_OFFSET + (uint32_t)(i + 1) * GGML_GTX_SINCOS_VEC_BYTES, 1);
            }

            __store(GGML_GTX_L1_BANK_R_OFFSET, work_a,
                    GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                    (uint8_t)(4 + sincos_sel), GGML_GTX_SINCOS_VEC_BYTES);
        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            __tpose(work_a, work_b, 16, (uint8_t)(4 + sincos_sel), 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(work_b, GGML_GTX_L1_BANK_A_OFFSET,
                   GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                   (uint8_t)(4 + sincos_sel), GGML_GTX_SINCOS_VEC_BYTES);

            if (sincos_sel) {
                __copy(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_B_OFFSET, 2, 2, 1, 2);
                __wrspr(0x800 + 4, 0, 0x01A095B02955B800ULL, 0);
                __store_svr(GGML_GTX_L1_BANK_B_OFFSET + GGML_GTX_SINCOS_VEC_BYTES, 1);
                __copy(GGML_GTX_L1_BANK_B_OFFSET + GGML_GTX_SINCOS_VEC_BYTES,
                       GGML_GTX_L1_BANK_B_OFFSET + 2,
                       8, 8, 1, 2);
            } else {
                __wrspr(0x800 + 4, 0, 0x8A802044B1553C00ULL, 0);
                __store_svr(GGML_GTX_L1_BANK_B_OFFSET, 1);
            }

            __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                           GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);
            __mm(16, (uint8_t)(4 + sincos_sel), 1);

            __set_spm_addr(GGML_GTX_L1_BANK_A_OFFSET, GGML_GTX_L1_BANK_B_OFFSET,
                           GGML_GTX_L1_BANK_C_OFFSET, GGML_GTX_L1_BANK_R_OFFSET);
            __mul_vv(16);

            __store(GGML_GTX_L1_BANK_A_OFFSET, work_a,
                    GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                    1, GGML_GTX_SINCOS_VEC_BYTES);
        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            if (l2_io) {
                __copy(work_a, (uint32_t)dst,
                       GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                       1, GGML_GTX_SINCOS_VEC_BYTES);
            } else {
                __store(work_a, dst,
                        GGML_GTX_SINCOS_VEC_BYTES, GGML_GTX_SINCOS_VEC_BYTES,
                        1, GGML_GTX_SINCOS_VEC_BYTES);
            }
        __end_shared();
    __end_plan(nest_id);
}

static void ggml_gtx_rope_range_reduce_to_A(uint32_t n) {
    const uint32_t n_bytes = n * 2;

    __copy(GGML_GTX_L1_BANK_A_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);

    __mul_vs(n, GGML_GTX_ROPE_FP16_INV_2PI, 0);
    __copy(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_A_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);
    __rne_v(n);
    __copy(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_A_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);
    __mul_vs(n, GGML_GTX_ROPE_FP16_2PI, 0);
    __copy(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_B_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);
    __copy(GGML_GTX_L1_BANK_C_OFFSET, GGML_GTX_L1_BANK_A_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);
    __sub_vv(n);
    __copy(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_A_OFFSET,
           n_bytes, n_bytes, 1, n_bytes);
}
#endif

static void ggml_gtx_compute_rope_cpu(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t n_dims,
    uint8_t  mode,
    uint32_t freq_base_b,
    uint32_t freq_scale_b,
    uint32_t attn_factor_b,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
    (void)nest_num;
    (void)spu_num_per_nest;

    volatile uint16_t * src0_fp16 = (volatile uint16_t *)(uintptr_t)ddr_src0;
    volatile int32_t  * pos       = (volatile int32_t *)(uintptr_t)ddr_src1;
    volatile uint16_t * dst_fp16  = (volatile uint16_t *)(uintptr_t)ddr_dst;

    const float freq_base = ggml_gtx_bits_to_float(freq_base_b);
    const float freq_scale = ggml_gtx_bits_to_float(freq_scale_b);
    const float attn_factor = ggml_gtx_bits_to_float(attn_factor_b);
    const uint16_t half_dims = n_dims / 2;
    const float theta_scale = ggml_gtx_expf((-2.0f / (float)n_dims) * ggml_gtx_logf(freq_base));

    for (uint16_t i3 = 0; i3 < ne3; ++i3) {
        for (uint16_t i2 = 0; i2 < ne2; ++i2) {
            const float p = (float)pos[i2] * freq_scale;

            for (uint16_t i1 = 0; i1 < ne1; ++i1) {
                const uint64_t base_idx = (uint64_t)i3 * ne2 * ne1 * ne0 +
                                          (uint64_t)i2 * ne1 * ne0 +
                                          (uint64_t)i1 * ne0;

                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    dst_fp16[base_idx + i0] = src0_fp16[base_idx + i0];
                }

                if (mode == 2) {
                    float theta = p * attn_factor;
                    for (uint16_t k = 0; k < half_dims; ++k) {
                        const float c = ggml_gtx_cosf(theta);
                        const float s = ggml_gtx_sinf(theta);
                        const uint16_t i_a = k;
                        const uint16_t i_b = (uint16_t)(k + half_dims);

                        const float x0 = ggml_gtx_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src0_fp16[base_idx + i_a]));
                        const float x1 = ggml_gtx_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src0_fp16[base_idx + i_b]));

                        dst_fp16[base_idx + i_a] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(x0 * c - x1 * s));
                        dst_fp16[base_idx + i_b] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(x0 * s + x1 * c));
                        theta *= theta_scale;
                    }
                } else {
                    float theta = p * attn_factor;
                    for (uint16_t k = 0; k < half_dims; ++k) {
                        const float c = ggml_gtx_cosf(theta);
                        const float s = ggml_gtx_sinf(theta);
                        const uint16_t i_a = (uint16_t)(2 * k);
                        const uint16_t i_b = (uint16_t)(2 * k + 1);

                        const float x0 = ggml_gtx_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src0_fp16[base_idx + i_a]));
                        const float x1 = ggml_gtx_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src0_fp16[base_idx + i_b]));

                        dst_fp16[base_idx + i_a] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(x0 * c - x1 * s));
                        dst_fp16[base_idx + i_b] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(x0 * s + x1 * c));
                        theta *= theta_scale;
                    }
                }
            }
        }
    }
}

static int ggml_gtx_compute_rope_l2(const ggml_gtx_kernel_launch_desc_t * desc) {
#ifdef GGML_GTX_USE_GTX
    const uint32_t ne0 = desc->ne[0];
    const uint32_t ne1 = desc->ne[1];
    const uint32_t ne2 = desc->ne[2];
    const uint32_t ne3 = desc->ne[3];
    const uint32_t n_dims = (uint32_t)desc->op_params[1];
    const uint32_t mode = (uint32_t)desc->op_params[2];
    const uint32_t required = ggml_gtx_rope_scratch_size(ne2, n_dims);
    const uint32_t pair_count = n_dims / 2u;
    const uint32_t pairs_padded = (pair_count + 15u) & ~15u;
    const uint32_t table_bytes = ne2 * pairs_padded * sizeof(uint16_t);
    const uint32_t scratch = desc->l2_scratch_addr;
    const uint32_t cos_l2 = scratch;
    const uint32_t sin_l2 = cos_l2 + table_bytes;
    const uint32_t row_bytes = ne0 * sizeof(uint16_t);
    const uint32_t rows_total = ne1 * ne2 * ne3;

    if (mode != 0u || desc->nest_id != 0u ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->src_type[1] != GGML_GTX_TYPE_I32 ||
        desc->elem_size != sizeof(uint16_t) ||
        desc->l2_src_addr[0] == 0u || desc->l2_src_addr[1] == 0u ||
        desc->src_addr[1] == 0u ||
        desc->l2_dst_addr == 0u || required == 0u ||
        required > desc->l2_scratch_capacity ||
        n_dims > ne0 || ne0 > UINT16_MAX || ne1 > UINT16_MAX ||
        ne2 > UINT16_MAX || ne3 > UINT16_MAX || row_bytes > UINT16_MAX ||
        table_bytes > 64u * 1024u || rows_total == 0u || rows_total > UINT16_MAX) {
        return -2;
    }

    const float freq_base = ggml_gtx_bits_to_float((uint32_t)desc->op_params[5]);
    const float freq_scale = ggml_gtx_bits_to_float((uint32_t)desc->op_params[6]);
    const float attn_factor = ggml_gtx_bits_to_float((uint32_t)desc->op_params[8]);
    const float theta_scale = ggml_gtx_expf((-2.0f / (float)n_dims) * ggml_gtx_logf(freq_base));
    volatile const int32_t * pos = (volatile const int32_t *)(uintptr_t)desc->src_addr[1];

    /* The tensor stays L2 resident. Position ids are bounded scalar metadata;
     * the control core expands them into FP16 cos/sin tables in dynamic L2
     * scratch, avoiding a tensor-sized DDR workspace or writeback. */
    __split();
    for (uint32_t i2 = 0; i2 < ne2; ++i2) {
        float theta = (float)pos[i2] * freq_scale * attn_factor;
        for (uint32_t pair_base = 0; pair_base < pairs_padded; pair_base += 16u) {
            uint64_t cos_words[4] = { 0u, 0u, 0u, 0u };
            uint64_t sin_words[4] = { 0u, 0u, 0u, 0u };
            for (uint32_t lane = 0; lane < 16u; ++lane) {
                const uint32_t pair = pair_base + lane;
                uint16_t cos_h = GGML_GTX_ROPE_FP16_ONE;
                uint16_t sin_h = 0u;
                if (pair < pair_count) {
                    cos_h = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(ggml_gtx_cosf(theta)));
                    sin_h = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(ggml_gtx_sinf(theta)));
                    theta *= theta_scale;
                }
                cos_words[lane / 4u] |= (uint64_t)cos_h << ((lane % 4u) * 16u);
                sin_words[lane / 4u] |= (uint64_t)sin_h << ((lane % 4u) * 16u);
            }

            const uint32_t table_off = (i2 * pairs_padded + pair_base) * sizeof(uint16_t);
            __start_plan(0);
                __start_thread(0);
                    for (uint32_t word = 0; word < 4u; ++word) {
                        __wrspr(0x800u + word, 0, cos_words[word], 0);
                        __wrspr(0x800u + 4u + word, 0, sin_words[word], 0);
                    }
                    __store_svr(GGML_GTX_L1_BANK_A_OFFSET, 0);
                    __store(GGML_GTX_L1_BANK_A_OFFSET, cos_l2 + table_off,
                            32u, 32u, 1, 32u);
                    __store_svr(GGML_GTX_L1_BANK_A_OFFSET, 1);
                    __store(GGML_GTX_L1_BANK_A_OFFSET, sin_l2 + table_off,
                            32u, 32u, 1, 32u);
                __end_thread(0);
            __end_plan(0);
        }
    }
    __join();

    __start_plan(0);
        __start_shared();
            __copy((uint32_t)desc->l2_src_addr[0], (uint32_t)desc->l2_dst_addr,
                   row_bytes, (uint16_t)row_bytes, (uint16_t)rows_total, (uint16_t)row_bytes);
        __end_shared();
    __end_plan(0);

    __split();
    __start_plan(0);
        for (uint32_t pair_base = 0; pair_base < pair_count; pair_base += GGML_GTX_SPU_NUM_PER_NEST) {
            const uint32_t eff_spu = pair_base + GGML_GTX_SPU_NUM_PER_NEST <= pair_count ?
                                     GGML_GTX_SPU_NUM_PER_NEST : pair_count - pair_base;
            for (uint32_t tid = 0; tid < eff_spu; ++tid) {
                __start_thread((uint8_t)tid);
                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);
                    const uint32_t pair = pair_base + tid;
                    const uint32_t pair_off = pair * 2u * sizeof(uint16_t);
                    for (uint32_t i3 = 0; i3 < ne3; ++i3) {
                        for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                            const uint32_t row0 = (i3 * ne2 + i2) * ne1;
                            const uint32_t cache_off = i2 * pairs_padded * sizeof(uint16_t) +
                                                       pair * sizeof(uint16_t);
                            __load((uint32_t)desc->l2_src_addr[0] + row0 * row_bytes + pair_off,
                                   GGML_GTX_L1_BANK_A_OFFSET,
                                   row_bytes, 2u * sizeof(uint16_t), (uint16_t)ne1,
                                   2u * sizeof(uint16_t));
                            __load(cos_l2 + cache_off, GGML_GTX_L1_BANK_B_OFFSET,
                                   table_bytes, sizeof(uint16_t), 2, sizeof(uint16_t));
                            __copy(GGML_GTX_L1_BANK_B_OFFSET + sizeof(uint16_t),
                                   GGML_GTX_L1_BANK_B_OFFSET + 2u * sizeof(uint16_t),
                                   sizeof(uint16_t), sizeof(uint16_t), 1, sizeof(uint16_t));
                            __copy(GGML_GTX_L1_BANK_B_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET + 3u * sizeof(uint16_t),
                                   sizeof(uint16_t), sizeof(uint16_t), 1, sizeof(uint16_t));
                            __set_spm_addr_A(GGML_GTX_L1_BANK_B_OFFSET + 2u * sizeof(uint16_t));
                            __set_spm_addr_R(GGML_GTX_L1_BANK_B_OFFSET + sizeof(uint16_t));
                            __neg_v(1);
                            __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                           GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);
                            __mm(ne1, 2, 2);
                            __store(GGML_GTX_L1_BANK_R_OFFSET,
                                    (uint32_t)desc->l2_dst_addr + row0 * row_bytes + pair_off,
                                    row_bytes, 2u * sizeof(uint16_t), (uint16_t)ne1,
                                    2u * sizeof(uint16_t));
                        }
                    }
                __end_thread((uint8_t)tid);
            }
        }
    __end_plan(0);
    __join();
    return 0;
#else
    (void)desc;
    return -2;
#endif
}

void ggml_gtx_compute_rope(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t n_dims,
    uint8_t  mode,
    uint32_t freq_base_b,
    uint32_t freq_scale_b,
    uint32_t attn_factor_b,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    if (mode != 0 || n_dims < 2 || n_dims > ne0 || (n_dims & 1u) != 0 || (n_dims % GGML_GTX_ROPE_VEC_ELEMS) != 0) {
        ggml_gtx_compute_rope_cpu(
            ddr_src0, ddr_src1, ddr_dst,
            ne0, ne1, ne2, ne3, n_dims, mode,
            freq_base_b, freq_scale_b, attn_factor_b,
            nest_num, spu_num_per_nest);
        return;
    }
    volatile uint16_t * src0_fp16 = (volatile uint16_t *)(uintptr_t)ddr_src0;
    volatile int32_t  * pos       = (volatile int32_t *)(uintptr_t)ddr_src1;
    volatile uint16_t * dst_fp16  = (volatile uint16_t *)(uintptr_t)ddr_dst;

    const float freq_base = ggml_gtx_bits_to_float(freq_base_b);
    const float freq_scale = ggml_gtx_bits_to_float(freq_scale_b);
    const float attn_factor = ggml_gtx_bits_to_float(attn_factor_b);
    const float freq_log = ggml_gtx_logf(freq_base);
    const float n_dims_f = (float)n_dims;
    const float exp_factor = -2.0f / n_dims_f;
    const float exp_arg = exp_factor * freq_log;
    const float theta_scale = ggml_gtx_expf(exp_arg);
    const uint16_t pair_count = n_dims / 2;
    const uint16_t row_bytes = ne0 * 2;
    const uint32_t rows_total = (uint32_t)ne1 * ne2 * ne3;
    const uint32_t row_stride_l2 = row_bytes;
    const uint32_t cache_row_bytes = (uint32_t)pair_count * 2;

    volatile uint16_t * cache_ddr = (volatile uint16_t *)(uintptr_t)GGML_GTX_DDR_TEMP_0;
    volatile uint16_t * theta_ddr = (volatile uint16_t *)(uintptr_t)GGML_GTX_DDR_TEMP_1;

    for (uint16_t i3 = 0; i3 < ne3; ++i3) {
        for (uint16_t i2 = 0; i2 < ne2; ++i2) {
            for (uint16_t i1 = 0; i1 < ne1; ++i1) {
                const uint64_t base_idx = (uint64_t)i3 * ne2 * ne1 * ne0 +
                                          (uint64_t)i2 * ne1 * ne0 +
                                          (uint64_t)i1 * ne0;
                volatile uint16_t * dst_row = dst_fp16 + base_idx;
                const volatile uint16_t * src_row = src0_fp16 + base_idx;

                for (uint16_t i0 = n_dims; i0 < ne0; ++i0) {
                    dst_row[i0] = src_row[i0];
                }
            }
        }
    }

    for (uint16_t i2 = 0; i2 < ne2; ++i2) {
        float theta = (float)pos[i2] * freq_scale * attn_factor;
        for (uint16_t pair = 0; pair < pair_count; ++pair) {
            const uint32_t theta_off = (uint32_t)i2 * cache_row_bytes + (uint32_t)pair * 2;
            theta_ddr[theta_off / 2] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(theta));
            theta *= theta_scale;
        }
    }

    {
        const uint32_t theta_elems = (uint32_t)ne2 * pair_count;
        const uint32_t full_vec_elems = theta_elems & ~(GGML_GTX_SINCOS_VEC_ELEMS - 1u);

        for (uint32_t elem = 0; elem < full_vec_elems; elem += GGML_GTX_SINCOS_VEC_ELEMS) {
            const uint64_t theta_addr = GGML_GTX_DDR_TEMP_1 + (uint64_t)elem * 2;
            ggml_gtx_sincos_vec_inline(theta_addr,
                                       GGML_GTX_DDR_TEMP_0 + (uint64_t)ne2 * cache_row_bytes + (uint64_t)elem * 2,
                                       0u,
                                       0,
                                       0,
                                       0,
                                       0);
            ggml_gtx_sincos_vec_inline(theta_addr,
                                       GGML_GTX_DDR_TEMP_0 + (uint64_t)elem * 2,
                                       0u,
                                       0,
                                       0,
                                       1,
                                       0);
        }

        if (full_vec_elems < theta_elems) {
            for (uint32_t elem = full_vec_elems; elem < theta_elems; ++elem) {
                const float theta = ggml_gtx_bits_to_float(ggml_gtx_fp16_to_fp32_bits(theta_ddr[elem]));
                cache_ddr[elem] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(ggml_gtx_cosf(theta)));
                cache_ddr[(uint32_t)ne2 * pair_count + elem] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_float_to_bits(ggml_gtx_sinf(theta)));
            }
        }
    }

    {
        const uint8_t nest_id = 0;

        __start_plan(nest_id);
            __start_shared();
                __load(
                    ddr_src0,
                    GGML_GTX_L2_A,
                    row_bytes,
                    row_bytes,
                    rows_total,
                    row_bytes
                );

                __copy(
                    GGML_GTX_L2_A,
                    GGML_GTX_L2_RESULT,
                    row_bytes,
                    row_bytes,
                    rows_total,
                    row_bytes
                );

                __load(
                    GGML_GTX_DDR_TEMP_0,
                    GGML_GTX_L2_B,
                    cache_row_bytes,
                    cache_row_bytes,
                    ne2 * 2,
                    cache_row_bytes
                );
            __end_shared();
        __end_plan(nest_id);
    }

    __split();

    {
        const uint8_t nest_id = 0;

        __start_plan(nest_id);
            for (uint16_t pair_base = 0; pair_base < pair_count; pair_base += spu_num_per_nest) {
                const uint16_t eff_spu = (uint16_t)((pair_base + spu_num_per_nest <= pair_count) ? spu_num_per_nest : (pair_count - pair_base));
                for (uint16_t thread_id = 0; thread_id < eff_spu; ++thread_id) {
                    __start_thread(thread_id);
                        __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                       GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);

                        const uint16_t pair_idx = (uint16_t)(pair_base + thread_id);
                        const uint32_t pair_off_bytes = (uint32_t)pair_idx * 2 * 2;

                        for (uint16_t i3 = 0; i3 < ne3; ++i3) {
                            for (uint16_t i2 = 0; i2 < ne2; ++i2) {
                                const uint32_t seq_pair_off = (uint32_t)i2 * cache_row_bytes + (uint32_t)pair_idx * 2;
                                const uint32_t row0_idx = ((uint32_t)i3 * ne2 + i2) * ne1;

                                __load(
                                    GGML_GTX_L2_A + row0_idx * row_stride_l2 + pair_off_bytes,
                                    GGML_GTX_L1_BANK_A_OFFSET,
                                    row_bytes,
                                    2 * 2,
                                    ne1,
                                    2 * 2
                                );

                                __load(
                                    GGML_GTX_L2_B + seq_pair_off,
                                    GGML_GTX_L1_BANK_B_OFFSET,
                                    ne2 * cache_row_bytes,
                                    2,
                                    2,
                                    2
                                );

                                __copy(GGML_GTX_L1_BANK_B_OFFSET + 2,
                                       GGML_GTX_L1_BANK_B_OFFSET + 4,
                                       2,
                                       2,
                                       1,
                                       2);
                                __copy(GGML_GTX_L1_BANK_B_OFFSET,
                                       GGML_GTX_L1_BANK_B_OFFSET + 6,
                                       2,
                                       2,
                                       1,
                                       2);
                                __set_spm_addr_A(GGML_GTX_L1_BANK_B_OFFSET + 4);
                                __set_spm_addr_R(GGML_GTX_L1_BANK_B_OFFSET + 2);
                                __neg_v(1);
                                __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                               GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);

                                __mm(ne1, 2, 2);

                                __store(
                                    GGML_GTX_L1_BANK_R_OFFSET,
                                    GGML_GTX_L2_RESULT + row0_idx * row_stride_l2 + pair_off_bytes,
                                    row_bytes,
                                    2 * 2,
                                    ne1,
                                    2 * 2
                                );
                            }
                        }
                    __end_thread(thread_id);
                }
            }
        __end_plan(nest_id);
    }

    __join();

    __start_plan(0);
        __start_shared();
            __store(
                GGML_GTX_L2_RESULT,
                ddr_dst,
                row_bytes,
                row_bytes,
                rows_total,
                row_bytes
            );
        __end_shared();
    __end_plan(0);
#else
    ggml_gtx_compute_rope_cpu(
        ddr_src0, ddr_src1, ddr_dst,
        ne0, ne1, ne2, ne3, n_dims, mode,
        freq_base_b, freq_scale_b, attn_factor_b,
        nest_num, spu_num_per_nest);
#endif
}
