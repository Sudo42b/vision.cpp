//==================================================================
// GGML GTX kernel: UPSCALE
//
// Descriptor wrapper around the reference nearest-neighbor 2x kernel
// structure.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_SCALE_MODE_NEAREST  0
#define GGML_GTX_SCALE_MODE_BILINEAR 1
#define GGML_GTX_SCALE_MODE_BICUBIC  2
#define GGML_GTX_SCALE_FLAG_ALIGN_CORNERS (1 << 8)
#define GGML_GTX_SCALE_FLAG_ANTIALIAS     (1 << 9)

#define DTYPE 2
#define SCALE_FACTOR 2
#define SPU_NUM_PER_NEST GGML_GTX_SPU_NUM_PER_NEST
#define L2_A 0x000000
#define L2_RESULT 0x002000
#define BANK_A GGML_GTX_L1_BANK_A_OFFSET
#define BANK_B GGML_GTX_L1_BANK_B_OFFSET
#define BANK_C GGML_GTX_L1_BANK_C_OFFSET
#define BANK_R GGML_GTX_L1_BANK_R_OFFSET
#define ACTIVE_TID_MASK ((1u << SPU_NUM_PER_NEST) - 1u)

static float ggml_gtx_upscale_bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value_bits;
    value_bits.u = bits;
    return value_bits.f;
}

static uint32_t ggml_gtx_upscale_float_to_bits(float value) {
    union { uint32_t u; float f; } value_bits;
    value_bits.f = value;
    return value_bits.u;
}

static float ggml_gtx_upscale_load_f16(uint64_t addr) {
    const uint16_t h = *(volatile const uint16_t *)(uintptr_t)addr;
    return ggml_gtx_upscale_bits_to_float(ggml_gtx_fp16_to_fp32_bits(h));
}

static void ggml_gtx_upscale_store_f16(uint64_t addr, float value) {
    *(volatile uint16_t *)(uintptr_t)addr =
        ggml_gtx_fp32_bits_to_fp16(ggml_gtx_upscale_float_to_bits(value));
}

static int32_t ggml_gtx_upscale_floor_f32(float value) {
    const int32_t iv = (int32_t)value;
    return ((float)iv > value) ? iv - 1 : iv;
}

static uint32_t ggml_gtx_upscale_clamp_i32(int32_t value, uint32_t hi) {
    if (value < 0) {
        return 0;
    }
    if ((uint32_t)value > hi) {
        return hi;
    }
    return (uint32_t)value;
}

static uint32_t ggml_gtx_upscale_map_nearest(uint32_t dst_idx, float scale, uint32_t hi) {
    uint32_t mapped = (uint32_t)((float)dst_idx / scale);
    if (mapped > hi) {
        mapped = hi;
    }
    return mapped;
}

static float ggml_gtx_upscale_weight1(float x) {
    const float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

static float ggml_gtx_upscale_weight2(float x) {
    const float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

static float ggml_gtx_upscale_bicubic(float p0, float p1, float p2, float p3, float x) {
    const float w0 = ggml_gtx_upscale_weight2(x + 1.0f);
    const float w1 = ggml_gtx_upscale_weight1(x);
    const float w2 = ggml_gtx_upscale_weight1(1.0f - x);
    const float w3 = ggml_gtx_upscale_weight2(2.0f - x);
    return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

static float ggml_gtx_upscale_load_src(const ggml_gtx_kernel_launch_desc_t * desc,
                                       uint32_t x,
                                       uint32_t y,
                                       uint32_t z,
                                       uint32_t w) {
    const uint64_t addr = desc->src_addr[0] +
                          (uint64_t)x * desc->src_nb[0][0] +
                          (uint64_t)y * desc->src_nb[0][1] +
                          (uint64_t)z * desc->src_nb[0][2] +
                          (uint64_t)w * desc->src_nb[0][3];
    return ggml_gtx_upscale_load_f16(addr);
}

static int ggml_gtx_kernel_upscale_generic_f16(const ggml_gtx_kernel_launch_desc_t * desc) {
    const uint32_t mode_flags = (uint32_t)desc->op_params[0];
    const uint32_t mode = mode_flags & 0xffu;
    const int align_corners = (mode_flags & GGML_GTX_SCALE_FLAG_ALIGN_CORNERS) != 0;

    if ((mode_flags & GGML_GTX_SCALE_FLAG_ANTIALIAS) != 0) {
        return -5;
    }
    if (mode != GGML_GTX_SCALE_MODE_NEAREST &&
        mode != GGML_GTX_SCALE_MODE_BILINEAR &&
        mode != GGML_GTX_SCALE_MODE_BICUBIC) {
        return -6;
    }

    const uint32_t src_width = desc->src_ne[0][0];
    const uint32_t src_height = desc->src_ne[0][1];
    const uint32_t src_depth = desc->src_ne[0][2];
    const uint32_t src_batches = desc->src_ne[0][3];
    const uint32_t dst_width = desc->ne[0];
    const uint32_t dst_height = desc->ne[1];
    const uint32_t dst_depth = desc->ne[2];
    const uint32_t dst_batches = desc->ne[3];

    if (src_width == 0 || src_height == 0 || src_depth == 0 || src_batches == 0 ||
        dst_width == 0 || dst_height == 0 || dst_depth == 0 || dst_batches == 0 ||
        src_depth != dst_depth || src_batches != dst_batches) {
        return -7;
    }

    float scale_x = (float)dst_width / (float)src_width;
    float scale_y = (float)dst_height / (float)src_height;
    if (align_corners) {
        if (src_width > 1 && dst_width > 1) {
            scale_x = (float)(dst_width - 1) / (float)(src_width - 1);
        }
        if (src_height > 1 && dst_height > 1) {
            scale_y = (float)(dst_height - 1) / (float)(src_height - 1);
        }
    }

    const float pixel_offset = align_corners ? 0.0f : 0.5f;
    const uint32_t max_src_x = src_width - 1;
    const uint32_t max_src_y = src_height - 1;

    for (uint32_t b = 0; b < dst_batches; ++b) {
        for (uint32_t z = 0; z < dst_depth; ++z) {
            for (uint32_t y = 0; y < dst_height; ++y) {
                for (uint32_t x = 0; x < dst_width; ++x) {
                    float value = 0.0f;

                    if (mode == GGML_GTX_SCALE_MODE_NEAREST) {
                        const uint32_t sx = ggml_gtx_upscale_map_nearest(x, scale_x, max_src_x);
                        const uint32_t sy = ggml_gtx_upscale_map_nearest(y, scale_y, max_src_y);
                        value = ggml_gtx_upscale_load_src(desc, sx, sy, z, b);
                    } else if (mode == GGML_GTX_SCALE_MODE_BILINEAR) {
                        const float fx = ((float)x + pixel_offset) / scale_x - pixel_offset;
                        const float fy = ((float)y + pixel_offset) / scale_y - pixel_offset;
                        const int32_t x0_i = ggml_gtx_upscale_floor_f32(fx);
                        const int32_t y0_i = ggml_gtx_upscale_floor_f32(fy);
                        const uint32_t x0 = ggml_gtx_upscale_clamp_i32(x0_i, max_src_x);
                        const uint32_t y0 = ggml_gtx_upscale_clamp_i32(y0_i, max_src_y);
                        const uint32_t x1 = ggml_gtx_upscale_clamp_i32(x0_i + 1, max_src_x);
                        const uint32_t y1 = ggml_gtx_upscale_clamp_i32(y0_i + 1, max_src_y);
                        float dx = fx - (float)x0_i;
                        float dy = fy - (float)y0_i;
                        if (dx < 0.0f) dx = 0.0f;
                        if (dx > 1.0f) dx = 1.0f;
                        if (dy < 0.0f) dy = 0.0f;
                        if (dy > 1.0f) dy = 1.0f;

                        const float a = ggml_gtx_upscale_load_src(desc, x0, y0, z, b);
                        const float c = ggml_gtx_upscale_load_src(desc, x0, y1, z, b);
                        const float b0 = ggml_gtx_upscale_load_src(desc, x1, y0, z, b);
                        const float d = ggml_gtx_upscale_load_src(desc, x1, y1, z, b);
                        value = a * (1.0f - dx) * (1.0f - dy) +
                                b0 * dx * (1.0f - dy) +
                                c * (1.0f - dx) * dy +
                                d * dx * dy;
                    } else {
                        const float fx = ((float)x + pixel_offset) / scale_x - pixel_offset;
                        const float fy = ((float)y + pixel_offset) / scale_y - pixel_offset;
                        const int32_t x1_i = ggml_gtx_upscale_floor_f32(fx);
                        const int32_t y1_i = ggml_gtx_upscale_floor_f32(fy);
                        const float dx = fx - (float)x1_i;
                        const float dy = fy - (float)y1_i;

                        float rows[4];
                        for (int32_t yy = -1; yy <= 2; ++yy) {
                            const uint32_t sy = ggml_gtx_upscale_clamp_i32(y1_i + yy, max_src_y);
                            const float p0 = ggml_gtx_upscale_load_src(desc, ggml_gtx_upscale_clamp_i32(x1_i - 1, max_src_x), sy, z, b);
                            const float p1 = ggml_gtx_upscale_load_src(desc, ggml_gtx_upscale_clamp_i32(x1_i,     max_src_x), sy, z, b);
                            const float p2 = ggml_gtx_upscale_load_src(desc, ggml_gtx_upscale_clamp_i32(x1_i + 1, max_src_x), sy, z, b);
                            const float p3 = ggml_gtx_upscale_load_src(desc, ggml_gtx_upscale_clamp_i32(x1_i + 2, max_src_x), sy, z, b);
                            rows[yy + 1] = ggml_gtx_upscale_bicubic(p0, p1, p2, p3, dx);
                        }
                        value = ggml_gtx_upscale_bicubic(rows[0], rows[1], rows[2], rows[3], dy);
                    }

                    const uint64_t dst_addr = desc->dst_addr +
                                              (uint64_t)x * desc->nb[0] +
                                              (uint64_t)y * desc->nb[1] +
                                              (uint64_t)z * desc->nb[2] +
                                              (uint64_t)b * desc->nb[3];
                    ggml_gtx_upscale_store_f16(dst_addr, value);
                }
            }
        }
    }

    return 0;
}

int ggml_gtx_kernel_upscale_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_UPSCALE_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_UPSCALE_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16) {
        return -2;
    }

    const uint32_t src_width = desc->src_ne[0][0];
    const uint32_t src_rows = desc->src_ne[0][1];
    const uint32_t planes = desc->src_ne[0][2] * desc->src_ne[0][3];
    const uint32_t dst_width = desc->ne[0];
    const uint32_t dst_rows = desc->ne[1];

    if (src_width == 0 || src_rows == 0 || planes == 0 ||
        dst_width == 0 || dst_rows == 0 ||
        desc->ne[2] != desc->src_ne[0][2] ||
        desc->ne[3] != desc->src_ne[0][3]) {
        return -3;
    }

    const uint32_t mode = (uint32_t)desc->op_params[0] & 0xffu;
    if (mode != GGML_GTX_SCALE_MODE_NEAREST ||
        dst_width != src_width * SCALE_FACTOR ||
        dst_rows != src_rows * SCALE_FACTOR ||
        desc->src_ne[0][3] != 1 ||
        desc->ne[3] != 1) {
        return ggml_gtx_kernel_upscale_generic_f16(desc);
    }

    const uint32_t src_row_bytes = src_width * DTYPE;
    const uint32_t dst_row_bytes = dst_width * DTYPE;
    const uint32_t src_tile_bytes = src_rows * src_row_bytes;
    const uint32_t dst_tile_bytes = dst_rows * dst_row_bytes;
    const uint32_t src_rows_per_spu = (src_rows + SPU_NUM_PER_NEST - 1) / SPU_NUM_PER_NEST;

    if (src_tile_bytes > 0x2000 || dst_tile_bytes > 0x20000 ||
        src_row_bytes > UINT16_MAX || dst_row_bytes > UINT16_MAX ||
        src_tile_bytes > UINT16_MAX || dst_tile_bytes > UINT16_MAX) {
        return ggml_gtx_kernel_upscale_generic_f16(desc);
    }

    for (uint32_t plane = 0; plane < planes; ++plane) {
        const uint64_t src_plane = desc->src_addr[0] + plane * desc->src_nb[0][2];
        const uint64_t dst_plane = desc->dst_addr + plane * desc->nb[2];

        __split();

        __start_plan(0);
            __start_shared();
                __load_cr(src_plane, L2_A,
                    src_tile_bytes,
                    (uint16_t)src_tile_bytes,
                    1, (uint16_t)src_tile_bytes,
                    1, ACTIVE_TID_MASK, 0xBEEF);

                __credit_chk(ACTIVE_TID_MASK);

                __store_cr(L2_RESULT, dst_plane,
                    dst_tile_bytes,
                    (uint16_t)dst_tile_bytes,
                    1, (uint16_t)dst_tile_bytes,
                    1, ACTIVE_TID_MASK);
            __end_shared();

            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                const uint16_t tid_mask = (uint16_t)(0x1u << tid);
                const uint32_t src_row_start = (uint32_t)tid * src_rows_per_spu;
                const uint32_t src_row_end = src_row_start + src_rows_per_spu;

                __start_thread(tid);
                    __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                    __credit_chk(0xBEEF);

                    for (uint32_t src_row = src_row_start; src_row < src_row_end && src_row < src_rows; src_row++) {
                        const uint32_t src_off = src_row * src_row_bytes;
                        const uint32_t dst_off0 = (src_row * SCALE_FACTOR) * dst_row_bytes;
                        const uint32_t dst_off1 = dst_off0 + dst_row_bytes;
                        const int last_row = (src_row + 1 == src_row_end) || (src_row + 1 == src_rows);

                        if (last_row) {
                            __load_cr(L2_A + src_off, BANK_A,
                                src_row_bytes, (uint16_t)src_row_bytes,
                                1, (uint16_t)src_row_bytes,
                                1, tid_mask, 0);
                        } else {
                            __load(L2_A + src_off, BANK_A,
                                src_row_bytes, (uint16_t)src_row_bytes,
                                1, (uint16_t)src_row_bytes);
                        }

                        for (uint32_t c = 0; c < src_width; c++) {
                            const uint32_t src_elem_off = c * DTYPE;
                            const uint32_t dst_elem_off = c * SCALE_FACTOR * DTYPE;

                            __copy(BANK_A + src_elem_off,
                                   BANK_R + dst_elem_off,
                                   DTYPE, DTYPE, 1, DTYPE);
                            __copy(BANK_A + src_elem_off,
                                   BANK_R + dst_elem_off + DTYPE,
                                   DTYPE, DTYPE, 1, DTYPE);
                        }
                        __store(BANK_R, L2_RESULT + dst_off0,
                            dst_row_bytes, (uint16_t)dst_row_bytes,
                            1, (uint16_t)dst_row_bytes);

                        if (last_row) {
                            __store_cr(BANK_R, L2_RESULT + dst_off1,
                                dst_row_bytes, (uint16_t)dst_row_bytes,
                                1, (uint16_t)dst_row_bytes,
                                1, tid_mask);
                        } else {
                            __store(BANK_R, L2_RESULT + dst_off1,
                                dst_row_bytes, (uint16_t)dst_row_bytes,
                                1, (uint16_t)dst_row_bytes);
                        }
                    }
                __end_thread(tid);
            }

        __end_plan(0);
        __join();
    }

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_upscale_f16(arg);
}
