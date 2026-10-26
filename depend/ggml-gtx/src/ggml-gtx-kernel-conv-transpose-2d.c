//==================================================================
// GGML GTX kernel: CONV_TRANSPOSE_2D
//
// Generalized descriptor wrapper around reference/conv_transpose_2d.c.
// The reference structure is preserved: each output element packs its
// contributing input/kernel values and computes one dot product.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_CT2D_BANK_A    0x00000u
#define GGML_GTX_CT2D_BANK_B    0x20000u
#define GGML_GTX_CT2D_BANK_C    0x30000u
#define GGML_GTX_CT2D_BANK_R    0x50000u

#define GGML_GTX_CT2D_PACK_INPUT  GGML_GTX_CT2D_BANK_C
#define GGML_GTX_CT2D_PACK_KERNEL (GGML_GTX_CT2D_BANK_C + 0x10000u)
#define GGML_GTX_CT2D_SLOT_ADDR   (GGML_GTX_CT2D_BANK_C + 0x1f000u)
#define GGML_GTX_CT2D_FP16_B      2u
#define GGML_GTX_CT2D_SPUS        GGML_GTX_SPU_NUM_PER_NEST
#define GGML_GTX_CT2D_MAX_CONTRIB 32768u
#define GGML_GTX_CT2D_L1_TILE_BYTES 0x10000u
#define GGML_GTX_CT2D_MM_T_MAX_M 256u

static inline uint32_t ggml_gtx_ct2d_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ggml_gtx_ct2d_outputs_for_tid(uint32_t total, uint8_t tid) {
    const uint32_t q = total / GGML_GTX_CT2D_SPUS;
    const uint32_t r = total % GGML_GTX_CT2D_SPUS;
    return tid < r ? q + 1u : q;
}

static inline uint32_t ggml_gtx_ct2d_start_for_tid(uint32_t total, uint8_t tid) {
    const uint32_t q = total / GGML_GTX_CT2D_SPUS;
    const uint32_t r = total % GGML_GTX_CT2D_SPUS;
    if (tid < r) {
        return (uint32_t) tid * (q + 1u);
    }
    return r * (q + 1u) + ((uint32_t) tid - r) * q;
}

/* Reference-style preload/pack path for the common upsampling case.  With a
 * 2x2 kernel and stride 2, each (ky,kx) maps an input pixel to a disjoint
 * output sub-lattice.  Pack spatial rows as MxIC and the corresponding
 * weights as OCxIC, then use the same __mm_t contract as MUL_MAT. */
static int ggml_gtx_ct2d_stride2_k2_block(const ggml_gtx_kernel_launch_desc_t * desc,
                                          uint32_t iw, uint32_t ih,
                                          uint32_t ic, uint32_t oc,
                                          uint32_t ow, uint32_t oh) {
    const uint32_t input_plane_bytes = iw * ih * GGML_GTX_CT2D_FP16_B;
    const uint32_t kernel_ic_stride = 2u * 2u * oc * GGML_GTX_CT2D_FP16_B;
    const uint32_t k_bytes = ic * GGML_GTX_CT2D_FP16_B;
    if (k_bytes == 0u || k_bytes > UINT16_MAX || iw == 0u || iw > UINT16_MAX) {
        return -1;
    }

    uint32_t rows_per_block = GGML_GTX_CT2D_L1_TILE_BYTES / (iw * k_bytes);
    const uint32_t rows_by_result = GGML_GTX_CT2D_L1_TILE_BYTES /
                                    (oc * iw * GGML_GTX_CT2D_FP16_B);
    if (rows_per_block > rows_by_result) rows_per_block = rows_by_result;
    /* The FPGA mm.t path rejects the otherwise L1-valid M=512 tile used by
     * the 128x128 -> 256x256 MobileSAM upsample.  Keep spatial M within the
     * verified hardware range and split additional input rows instead. */
    const uint32_t rows_by_mm_m = GGML_GTX_CT2D_MM_T_MAX_M / iw;
    if (rows_per_block > rows_by_mm_m) rows_per_block = rows_by_mm_m;
    if (rows_per_block > ih) rows_per_block = ih;
    if (rows_per_block == 0u) return -1;

    for (uint32_t ky = 0u; ky < 2u; ++ky) {
        for (uint32_t kx = 0u; kx < 2u; ++kx) {
            for (uint32_t y0 = 0u; y0 < ih; y0 += rows_per_block) {
                const uint32_t cur_rows = (ih - y0) < rows_per_block ? ih - y0 : rows_per_block;
                const uint32_t m = cur_rows * iw;
                if (m == 0u || m > UINT16_MAX) return -1;

                __split();
                    __start_plan(0u);
                        __start_thread(0u);
                            __set_spm_addr(GGML_GTX_CT2D_BANK_R, GGML_GTX_CT2D_BANK_C,
                                           GGML_GTX_CT2D_BANK_B, GGML_GTX_CT2D_BANK_A);

                            for (uint32_t p = 0u; p < m; ++p) {
                                const uint32_t input_pixel = y0 * iw + p;
                                __load((uint32_t) desc->l2_src_addr[1] +
                                           input_pixel * GGML_GTX_CT2D_FP16_B,
                                       GGML_GTX_CT2D_BANK_A + p * k_bytes,
                                       input_plane_bytes,
                                       GGML_GTX_CT2D_FP16_B,
                                       (uint16_t) ic,
                                       GGML_GTX_CT2D_FP16_B);
                            }

                            for (uint32_t out_ch = 0u; out_ch < oc; ++out_ch) {
                                const uint32_t kernel_elem =
                                    ((out_ch * 2u + ky) * 2u + kx);
                                __load((uint32_t) desc->l2_src_addr[0] +
                                           kernel_elem * GGML_GTX_CT2D_FP16_B,
                                       GGML_GTX_CT2D_BANK_B + out_ch * k_bytes,
                                       kernel_ic_stride,
                                       GGML_GTX_CT2D_FP16_B,
                                       (uint16_t) ic,
                                       GGML_GTX_CT2D_FP16_B);
                            }

                            __mm_t((uint16_t) m, (uint16_t) ic, (uint16_t) oc);

                            for (uint32_t out_ch = 0u; out_ch < oc; ++out_ch) {
                                for (uint32_t ry = 0u; ry < cur_rows; ++ry) {
                                    const uint32_t out_y = (y0 + ry) * 2u + ky;
                                    const uint32_t dst_elem =
                                        (out_ch * oh + out_y) * ow + kx;
                                    __store(GGML_GTX_CT2D_BANK_R +
                                                (out_ch * m + ry * iw) * GGML_GTX_CT2D_FP16_B,
                                            (uint32_t) desc->l2_dst_addr +
                                                dst_elem * GGML_GTX_CT2D_FP16_B,
                                            2u * GGML_GTX_CT2D_FP16_B,
                                            GGML_GTX_CT2D_FP16_B,
                                            (uint16_t) iw,
                                            GGML_GTX_CT2D_FP16_B);
                                }
                            }
                        __end_thread(0u);
                    __end_plan(0u);
                __join();
            }
        }
    }
    return 0;
}

int ggml_gtx_kernel_conv_transpose_2d_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->l2_src_addr[1] == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->src_type[1] != GGML_GTX_TYPE_F16) {
        return -2;
    }

    const uint32_t kw = desc->src_ne[0][0];
    const uint32_t kh = desc->src_ne[0][1];
    const uint32_t oc = desc->src_ne[0][2];
    const uint32_t ic = desc->src_ne[0][3];
    const uint32_t iw = desc->src_ne[1][0];
    const uint32_t ih = desc->src_ne[1][1];
    const uint32_t in_c = desc->src_ne[1][2];
    const uint32_t batch = desc->src_ne[1][3];
    const uint32_t ow = desc->ne[0];
    const uint32_t oh = desc->ne[1];
    const uint32_t out_c = desc->ne[2];
    const uint32_t out_b = desc->ne[3];
    const int32_t stride = desc->op_params[0];

    if (kw == 0 || kh == 0 || oc == 0 || ic == 0 ||
        iw == 0 || ih == 0 || batch == 0 ||
        in_c != ic || out_c != oc || out_b != batch ||
        stride <= 0 ||
        ow != (iw - 1u) * (uint32_t) stride + kw ||
        oh != (ih - 1u) * (uint32_t) stride + kh ||
        kw * kh * ic > GGML_GTX_CT2D_MAX_CONTRIB) {
        return -3;
    }

    const uint64_t input_bytes = (uint64_t) iw * ih * ic * batch * GGML_GTX_CT2D_FP16_B;
    const uint64_t kernel_bytes = (uint64_t) kw * kh * oc * ic * GGML_GTX_CT2D_FP16_B;
    const uint64_t output_bytes = (uint64_t) ow * oh * oc * batch * GGML_GTX_CT2D_FP16_B;
    if (input_bytes == 0 || kernel_bytes == 0 || output_bytes == 0 ||
        iw * GGML_GTX_CT2D_FP16_B > UINT16_MAX ||
        kw * GGML_GTX_CT2D_FP16_B > UINT16_MAX ||
        ow * GGML_GTX_CT2D_FP16_B > UINT16_MAX ||
        ih * ic * batch > UINT16_MAX ||
        kh * oc * ic > UINT16_MAX ||
        oh * oc * batch > UINT16_MAX) {
        return -4;
    }

    if (kw == 2u && kh == 2u && stride == 2 && batch == 1u) {
        return ggml_gtx_ct2d_stride2_k2_block(desc, iw, ih, ic, oc, ow, oh);
    }

    const uint32_t total_outputs = ow * oh * oc * batch;
    __split();
    __start_plan(0);
        for (uint8_t tid = 0; tid < GGML_GTX_CT2D_SPUS; ++tid) {
            const uint32_t thread_outputs = ggml_gtx_ct2d_outputs_for_tid(total_outputs, tid);
            const uint32_t base_output = ggml_gtx_ct2d_start_for_tid(total_outputs, tid);
            __start_thread(tid);
                if (thread_outputs > 0u) {
                    __set_spm_addr(GGML_GTX_CT2D_BANK_R, GGML_GTX_CT2D_BANK_C,
                                   GGML_GTX_CT2D_BANK_B, GGML_GTX_CT2D_BANK_A);
                    for (uint32_t out_local = 0; out_local < thread_outputs; ++out_local) {
                        const uint32_t out_idx = base_output + out_local;
                        const uint32_t out_x = out_idx % ow;
                        const uint32_t out_y = (out_idx / ow) % oh;
                        const uint32_t out_ch = (out_idx / (ow * oh)) % oc;
                        const uint32_t out_batch = out_idx / (ow * oh * oc);
                        uint32_t contrib_count = 0;

                        for (uint32_t ky = 0; ky < kh; ++ky) {
                            if (out_y < ky) {
                                continue;
                            }
                            const uint32_t dy = out_y - ky;
                            if ((dy % (uint32_t) stride) != 0u) {
                                continue;
                            }
                            const uint32_t in_y = dy / (uint32_t) stride;
                            if (in_y >= ih) {
                                continue;
                            }

                            for (uint32_t kx = 0; kx < kw; ++kx) {
                                if (out_x < kx) {
                                    continue;
                                }
                                const uint32_t dx = out_x - kx;
                                if ((dx % (uint32_t) stride) != 0u) {
                                    continue;
                                }
                                const uint32_t in_x = dx / (uint32_t) stride;
                                if (in_x >= iw) {
                                    continue;
                                }

                                for (uint32_t in_ch = 0; in_ch < ic; ++in_ch) {
                                    const uint32_t input_off =
                                        (((out_batch * ic + in_ch) * ih + in_y) * iw + in_x) *
                                        GGML_GTX_CT2D_FP16_B;
                                    const uint32_t kernel_off =
                                        (((in_ch * oc + out_ch) * kh + ky) * kw + kx) *
                                        GGML_GTX_CT2D_FP16_B;
                                    const uint32_t pack_off = contrib_count * GGML_GTX_CT2D_FP16_B;

                                    __load((uint32_t)desc->l2_src_addr[1] + input_off,
                                           GGML_GTX_CT2D_PACK_INPUT + pack_off,
                                           GGML_GTX_CT2D_FP16_B,
                                           GGML_GTX_CT2D_FP16_B,
                                           1, GGML_GTX_CT2D_FP16_B);
                                    __load((uint32_t)desc->l2_src_addr[0] + kernel_off,
                                           GGML_GTX_CT2D_PACK_KERNEL + pack_off,
                                           GGML_GTX_CT2D_FP16_B,
                                           GGML_GTX_CT2D_FP16_B,
                                           1, GGML_GTX_CT2D_FP16_B);
                                    ++contrib_count;
                                }
                            }
                        }

                        __set_spm_addr_A(GGML_GTX_CT2D_PACK_INPUT);
                        __set_spm_addr_B(GGML_GTX_CT2D_PACK_KERNEL);
                        __dot_product(contrib_count, 0);
                        __set_spm_addr_A(GGML_GTX_CT2D_BANK_A);
                        __set_spm_addr_B(GGML_GTX_CT2D_BANK_B);
                        __store_svr(GGML_GTX_CT2D_SLOT_ADDR, 0);

                        __store(GGML_GTX_CT2D_SLOT_ADDR,
                                (uint32_t)desc->l2_dst_addr +
                                    out_idx * GGML_GTX_CT2D_FP16_B,
                                GGML_GTX_CT2D_FP16_B,
                                GGML_GTX_CT2D_FP16_B,
                                1, GGML_GTX_CT2D_FP16_B);
                    }
                }
            __end_thread(tid);
        }
    __end_plan(0);
    __join();

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_conv_transpose_2d_f16(arg);
}
