//==================================================================
// GGML GTX kernel: CONV_2D_DW
//
// Descriptor wrapper around the host-owned GTX depthwise-convolution core.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#include "ggml-gtx-kernel-conv-2d-dw-core.h"

static void ggml_gtx_dwconv_rebase_scratch(L2Info * l2i, uint32_t scratch_base) {
    l2i->input_addr += scratch_base;
    l2i->kernel_addr += scratch_base;
    l2i->bias_addr += scratch_base;
    l2i->output_addr += scratch_base;
    l2i->bias_ones_addr += scratch_base;
}

static bool ggml_gtx_dwconv_l2_slice_fits(const ConvParams * p, const L2Info * l2i) {
    const uint64_t output_bytes = (uint64_t) p->cout * (uint64_t) p->cout_b;
    return (uint64_t) l2i->output_addr + output_bytes <= (uint64_t) l2i->bias_ones_addr;
}

static bool ggml_gtx_dwconv_planar_layout(const ggml_gtx_kernel_launch_desc_t * desc,
                                          const ConvParams * p) {
    const uint32_t input_row_bytes = (uint32_t) p->win * FP16_B;
    const uint32_t input_ch_bytes = (uint32_t) p->hin * input_row_bytes;
    const uint32_t kernel_row_bytes = (uint32_t) p->k_size * FP16_B;
    const uint32_t kernel_ch_bytes = (uint32_t) p->k_size * kernel_row_bytes;
    const uint32_t output_row_bytes = (uint32_t) p->wout * FP16_B;
    const uint32_t output_ch_bytes = (uint32_t) p->hout * output_row_bytes;

    return desc->src_nb[1][0] == FP16_B &&
           desc->src_nb[1][1] == input_row_bytes &&
           desc->src_nb[1][2] == input_ch_bytes &&
           desc->src_nb[0][0] == FP16_B &&
           desc->src_nb[0][1] == kernel_row_bytes &&
           desc->src_nb[0][3] == kernel_ch_bytes &&
           desc->nb[0] == FP16_B &&
           desc->nb[1] == output_row_bytes &&
           desc->nb[2] == output_ch_bytes;
}

static void ggml_gtx_dwconv_copy_strided_elems(uint32_t src,
                                                uint32_t dst,
                                                uint32_t src_stride,
                                                uint32_t dst_stride,
                                                uint32_t elems) {
    while (elems > 0u) {
        const uint16_t chunk = (uint16_t) (elems > UINT16_MAX ? UINT16_MAX : elems);
        __copy(src, dst, src_stride, FP16_B, chunk, dst_stride);
        src += (uint32_t) chunk * src_stride;
        dst += (uint32_t) chunk * dst_stride;
        elems -= chunk;
    }
}

static void ggml_gtx_dwconv_load_l2_strided(
    int nest_id,
    const ConvParams * p,
    const L2Info * l2i,
    uint64_t input_base,
    uint64_t kernel_base,
    uint64_t input_x_stride,
    uint64_t input_y_stride,
    uint64_t input_c_stride,
    uint64_t kernel_x_stride,
    uint64_t kernel_y_stride,
    uint64_t kernel_c_stride) {

    const uint32_t l2_pad_offset = (uint32_t) p->pad_h * (uint32_t) p->win_pad_b +
                                   (uint32_t) p->pad_w * FP16_B;
    const uint64_t bias_fill_pattern = p->bias ? 0x3C003C003C003C00ull : 0x0000ull;
    __start_plan(nest_id);
        __start_shared();
            __fill(l2i->input_addr,
                   (uint32_t) p->win_pad_b,
                   (uint16_t) p->win_pad_b,
                   (uint16_t) ((uint32_t) p->hin_pad * (uint32_t) p->cin),
                   0x00,
                   0);

            for (int cid = 0; cid < p->cin; ++cid) {
                const uint64_t input_ch = input_base + (uint64_t) cid * input_c_stride;
                const uint32_t l2_ch = l2i->input_addr + (uint32_t) cid * (uint32_t) p->cin_pad_b;
                if (p->pad_w == 0 && p->win_pad == p->win &&
                    input_y_stride == (uint64_t) p->win * input_x_stride) {
                    ggml_gtx_dwconv_copy_strided_elems((uint32_t) input_ch,
                                                       l2_ch + l2_pad_offset,
                                                       (uint32_t) input_x_stride,
                                                       FP16_B,
                                                       (uint32_t) p->hin * (uint32_t) p->win);
                } else {
                    for (int y = 0; y < p->hin; ++y) {
                        ggml_gtx_dwconv_copy_strided_elems(
                            (uint32_t) input_ch + (uint32_t) y * (uint32_t) input_y_stride,
                            l2_ch + l2_pad_offset + (uint32_t) y * (uint32_t) p->win_pad_b,
                            (uint32_t) input_x_stride,
                            FP16_B,
                            (uint32_t) p->win);
                    }
                }

                const uint64_t kernel_ch = kernel_base + (uint64_t) cid * kernel_c_stride;
                const uint32_t l2_kernel_ch = l2i->kernel_addr + (uint32_t) cid * (uint32_t) p->knl_ch_b;
                if (kernel_y_stride == (uint64_t) p->k_size * kernel_x_stride) {
                    ggml_gtx_dwconv_copy_strided_elems((uint32_t) kernel_ch,
                                                       l2_kernel_ch,
                                                       (uint32_t) kernel_x_stride,
                                                       FP16_B,
                                                       (uint32_t) p->k_size * (uint32_t) p->k_size);
                } else {
                    for (int ky = 0; ky < p->k_size; ++ky) {
                        ggml_gtx_dwconv_copy_strided_elems(
                            (uint32_t) kernel_ch + (uint32_t) ky * (uint32_t) kernel_y_stride,
                            l2_kernel_ch + (uint32_t) ky * (uint32_t) p->k_size * FP16_B,
                            (uint32_t) kernel_x_stride,
                            FP16_B,
                            (uint32_t) p->k_size);
                    }
                }
            }

            __fill(l2i->bias_addr,
                   (uint32_t) p->cin * FP16_B,
                   (uint16_t) ((uint32_t) p->cin * FP16_B),
                   1,
                   0x00,
                   0);
            __fill(l2i->bias_ones_addr,
                   BIAS_ONES_LENGTH,
                   BIAS_ONES_LENGTH,
                   1,
                   bias_fill_pattern,
                   0);
        __end_shared();
    __end_plan(nest_id);
}

static void ggml_gtx_dwconv_store_l2_strided(
    int nest_id,
    const ConvParams * p,
    uint32_t l2_out_addr,
    uint64_t ddr_output,
    uint64_t output_x_stride,
    uint64_t output_y_stride,
    uint64_t output_c_stride) {

    __start_plan(nest_id);
        __start_shared();
            for (int cid = 0; cid < p->cout; ++cid) {
                const uint32_t l2_ch = l2_out_addr + (uint32_t) cid * (uint32_t) p->cout_b;
                const uint64_t ddr_ch = ddr_output + (uint64_t) cid * output_c_stride;
                if (output_y_stride == (uint64_t) p->wout * output_x_stride) {
                    ggml_gtx_dwconv_copy_strided_elems(l2_ch,
                                                       (uint32_t) ddr_ch,
                                                       FP16_B,
                                                       (uint32_t) output_x_stride,
                                                       (uint32_t) p->hout * (uint32_t) p->wout);
                } else {
                    for (int y = 0; y < p->hout; ++y) {
                        ggml_gtx_dwconv_copy_strided_elems(
                            l2_ch + (uint32_t) y * (uint32_t) p->wout_b,
                            (uint32_t) ddr_ch + (uint32_t) y * (uint32_t) output_y_stride,
                            FP16_B,
                            (uint32_t) output_x_stride,
                            (uint32_t) p->wout);
                    }
                }
            }
        __end_shared();
    __end_plan(nest_id);
}

static void ggml_gtx_dwconv_2d_tile_compute_only(
    int nest_id,
    ConvParams * p,
    SPUTileMap * stm,
    TileExecPlan * t,
    L2Info * l2i) {

    int use_spu = stm->use_spu;

    int start_cidx[use_spu];
    int cur_iter = 0;
    for (int spu_id = 0; spu_id < use_spu; spu_id++) {
        start_cidx[spu_id] = cur_iter;
        cur_iter += (spu_id < stm->cout_rem_cnt) ? stm->cout_len : stm->cout_len - 1;
    }

    for (int cidx = 0; cidx < stm->cout_len; cidx++) {
        int last_spu = (cidx < stm->cout_len - 1) ? use_spu : stm->cout_rem_cnt;

        for (int hid = 0; hid < t->h_div; hid++) {
            int hout_idx = hid * t->hout_len;
            int cur_hout = (hid == t->h_div - 1) ? ceil_rem(stm->hout_len, t->hout_len) : t->hout_len;
            int cur_hin  = FIND_INPUT_LENGTH(cur_hout, p->k_eff, p->stride);

            uint32_t h_off_out = (uint32_t) hout_idx * (uint32_t) p->wout_b;
            uint32_t h_off_in  = (uint32_t) hout_idx * (uint32_t) p->stride * (uint32_t) p->win_pad_b;

            for (int wid = 0; wid < t->w_div; wid++) {
                int wout_idx = wid * t->wout_len;
                int cur_wout = (wid == t->w_div - 1) ? ceil_rem(stm->wout_len, t->wout_len) : t->wout_len;
                int cur_win  = FIND_INPUT_LENGTH(cur_wout, p->k_eff, p->stride);

                uint32_t w_off_out = (uint32_t) wout_idx * FP16_B;
                uint32_t w_off_in  = (uint32_t) wout_idx * (uint32_t) p->stride * FP16_B;

                int ohow = cur_hout * cur_wout;
                uint32_t ones_height = (uint32_t) ceil_div(ohow * FP16_B, BIAS_ONES_LENGTH);
                __split();
                __start_plan(nest_id);
                    for (int spu_id = 0; spu_id < last_spu; spu_id++) {
                        int abs_c = start_cidx[spu_id] + cidx;
                        uint32_t in_addr  = l2i->input_addr  + (uint32_t) abs_c * (uint32_t) p->cin_pad_b +
                                            h_off_in + w_off_in;
                        uint32_t out_addr = l2i->output_addr + (uint32_t) abs_c * (uint32_t) p->cout_b +
                                            h_off_out + w_off_out;
                        uint32_t knl_addr = l2i->kernel_addr + (uint32_t) abs_c * (uint32_t) p->knl_ch_b;
                        uint32_t bias_ptr = l2i->bias_addr   + (uint32_t) abs_c * FP16_B;

                        __start_thread(spu_id);
                            __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);

                            __load(l2i->bias_ones_addr, BASE_L1_BANK_A, 0, BIAS_ONES_LENGTH,
                                   (uint16_t) ones_height, BIAS_ONES_LENGTH);
                            __load(bias_ptr, BASE_L1_BANK_B, FP16_B, FP16_B, 1, FP16_B);
                            __mm_s(ohow, 1, 1);

                            __load(in_addr, BASE_L1_BANK_R, (uint32_t) p->win_pad_b,
                                   (uint16_t) (cur_win * FP16_B), (uint16_t) cur_hin,
                                   (uint32_t) (cur_win * FP16_B));
                            __im2col_n(cur_hin, cur_win, p->k_size, p->dilation, p->stride, 1);
                            __load(knl_addr, BASE_L1_BANK_B, (uint32_t) p->knl_ch_b,
                                   (uint16_t) p->knl_ch_b, 1, (uint32_t) p->knl_ch_b);
                            __mmc_t(ohow, p->k_n, 1);

                            __store(BASE_L1_BANK_R, out_addr, (uint32_t) p->wout_b,
                                    (uint16_t) (cur_wout * FP16_B), (uint16_t) cur_hout,
                                    (uint32_t) (cur_wout * FP16_B));
                        __end_thread(spu_id);
                    }
                __end_plan(nest_id);
                __join();
            }
        }
    }
}

static void ggml_gtx_dwconv_2d_spu_tile_compute_only(
    int nest_id,
    ConvParams * p,
    SPUTileMap * stm,
    TileExecPlan * t,
    L2Info * l2i) {

    const int use_spu = stm->use_spu;

    int start_c[use_spu];
    int hout_assigned[use_spu];
    int wout_assigned[use_spu];
    uint32_t in_base_spu[use_spu];
    uint32_t out_base_spu[use_spu];

    {
        int spu_id = 0;
        for (int cid = 0; cid < stm->cout_div; cid++) {
            int start_hout = 0;
            for (int hid = 0; hid < stm->h_div; hid++) {
                int start_wout = 0;
                int cur_h = (hid < stm->h_rem_cnt) ? stm->hout_len : stm->hout_len - 1;
                for (int wid = 0; wid < stm->w_div; wid++) {
                    int cur_w = (wid < stm->w_rem_cnt) ? stm->wout_len : stm->wout_len - 1;

                    start_c[spu_id] = cid;
                    hout_assigned[spu_id] = cur_h;
                    wout_assigned[spu_id] = cur_w;

                    in_base_spu[spu_id] = l2i->input_addr + (uint32_t) cid * (uint32_t) p->cin_pad_b +
                                          (uint32_t) start_hout * (uint32_t) p->stride * (uint32_t) p->win_pad_b +
                                          (uint32_t) start_wout * (uint32_t) p->stride * FP16_B;
                    out_base_spu[spu_id] = l2i->output_addr + (uint32_t) cid * (uint32_t) p->cout_b +
                                           (uint32_t) start_hout * (uint32_t) p->wout_b +
                                           (uint32_t) start_wout * FP16_B;

                    start_wout += cur_w;
                    spu_id++;
                }
                start_hout += cur_h;
            }
        }
    }

    for (int hid = 0; hid < t->h_div; hid++) {
        const bool is_h_rem = (hid == t->h_div - 1);
        const int hout_idx = hid * t->hout_len;
        const uint32_t hout_off = (uint32_t) hout_idx * (uint32_t) p->wout_b;
        const uint32_t hin_off = (uint32_t) hout_idx * (uint32_t) p->stride * (uint32_t) p->win_pad_b;

        for (int wid = 0; wid < t->w_div; wid++) {
            const bool is_w_rem = (wid == t->w_div - 1);
            const int wout_idx = wid * t->wout_len;
            const uint32_t wout_off = (uint32_t) wout_idx * FP16_B;
            const uint32_t win_off = (uint32_t) wout_idx * (uint32_t) p->stride * FP16_B;
            __split();
            __start_plan(nest_id);
                for (int spu_id = 0; spu_id < use_spu; spu_id++) {
                    int cur_hout = is_h_rem ? ceil_rem(hout_assigned[spu_id], t->hout_len) : t->hout_len;
                    int cur_wout = is_w_rem ? ceil_rem(wout_assigned[spu_id], t->wout_len) : t->wout_len;
                    int cur_hin  = FIND_INPUT_LENGTH(cur_hout, p->k_eff, p->stride);
                    int cur_win  = FIND_INPUT_LENGTH(cur_wout, p->k_eff, p->stride);
                    int ohow     = cur_hout * cur_wout;

                    uint32_t in_addr   = in_base_spu[spu_id] + hin_off + win_off;
                    uint32_t out_addr  = out_base_spu[spu_id] + hout_off + wout_off;
                    uint32_t knl_addr  = l2i->kernel_addr + (uint32_t) start_c[spu_id] * (uint32_t) p->knl_ch_b;
                    uint32_t bias_ptr  = l2i->bias_addr + (uint32_t) start_c[spu_id] * FP16_B;
                    uint32_t ones_height = (uint32_t) ceil_div(ohow * FP16_B, BIAS_ONES_LENGTH);

                    __start_thread(spu_id);
                        __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);

                        __load(l2i->bias_ones_addr, BASE_L1_BANK_A, 0, BIAS_ONES_LENGTH,
                               (uint16_t) ones_height, BIAS_ONES_LENGTH);
                        __load(bias_ptr, BASE_L1_BANK_B, FP16_B, FP16_B, 1, FP16_B);
                        __mm_s(ohow, 1, 1);

                        __load(in_addr, BASE_L1_BANK_R, (uint32_t) p->win_pad_b,
                               (uint16_t) (cur_win * FP16_B), (uint16_t) cur_hin,
                               (uint32_t) (cur_win * FP16_B));

                        __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
                        __im2col_n(cur_hin, cur_win, p->k_size, p->dilation, p->stride, 1);

                        __load(knl_addr, BASE_L1_BANK_B, (uint32_t) p->knl_ch_b,
                               (uint16_t) p->knl_ch_b, 1, (uint32_t) p->knl_ch_b);
                        __mmc_t(ohow, p->k_n, 1);

                        __store(BASE_L1_BANK_R, out_addr, (uint32_t) p->wout_b,
                                (uint16_t) (cur_wout * FP16_B), (uint16_t) cur_hout,
                                (uint32_t) (cur_wout * FP16_B));
                    __end_thread(spu_id);
                }
            __end_plan(nest_id);
            __join();
        }
    }
}

int ggml_gtx_kernel_conv_2d_dw_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CONV_2D_DW_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CONV_2D_DW_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->l2_src_addr[1] == 0 ||
        desc->l2_scratch_capacity == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->src_type[1] != GGML_GTX_TYPE_F16) {
        return -2;
    }

    const bool c_fast = desc->op_params[GGML_GTX_CONV_LAYOUT_PARAM] == GGML_GTX_CONV_LAYOUT_C_FAST;
    const uint32_t kw = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_KW_PARAM] : desc->src_ne[0][0];
    const uint32_t kh = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_KH_PARAM] : desc->src_ne[0][1];
    const uint32_t c  = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_C_PARAM] : desc->src_ne[1][2];
    const uint32_t iw = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_IW_PARAM] : desc->src_ne[1][0];
    const uint32_t ih = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_IH_PARAM] : desc->src_ne[1][1];
    const uint32_t ow = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_OW_PARAM] : desc->ne[0];
    const uint32_t oh = c_fast ? (uint32_t)desc->op_params[GGML_GTX_CONV_LOGICAL_OH_PARAM] : desc->ne[1];
    const uint32_t n  = c_fast ? 1u : desc->src_ne[1][3];
    const int32_t s0 = desc->op_params[0];
    const int32_t s1 = desc->op_params[1];
    const int32_t p0 = desc->op_params[2];
    const int32_t p1 = desc->op_params[3];
    const int32_t d0 = desc->op_params[4];
    const int32_t d1 = desc->op_params[5];

    const bool raw_shape_ok = c_fast ?
        (desc->src_ne[0][0] == c && desc->src_ne[0][1] == kw && desc->src_ne[0][2] == kh &&
         desc->src_ne[0][3] == 1 && desc->src_ne[1][0] == c && desc->src_ne[1][1] == iw &&
         desc->src_ne[1][2] == ih && desc->src_ne[1][3] == 1 && desc->ne[0] == c &&
         desc->ne[1] == ow && desc->ne[2] == oh && desc->ne[3] == 1) :
        (desc->src_ne[0][2] == 1 && desc->src_ne[0][3] == c && desc->ne[2] == c &&
         desc->ne[3] == 1);
    if (kw == 0 || kh == 0 || c == 0 || iw == 0 || ih == 0 || ow == 0 || oh == 0 ||
        kw != kh || !raw_shape_ok || n != 1 ||
        s0 <= 0 || s1 <= 0 || s0 != s1 ||
        d0 <= 0 || d1 <= 0 || d0 != d1 ||
        p0 < 0 || p1 < 0) {
        return -3;
    }

    ConvParams full = {
        .cin = (int) c,
        .hin = (int) ih,
        .win = (int) iw,
        .cout = (int) c,
        .k_size = (int) kw,
        .pad_h = 0,
        .pad_w = (int) p0,
        .stride = (int) s0,
        .dilation = (int) d0,
        .bias = false,
    };
    derive_convParam(&full);

    const int64_t required_hin_pad = ((int64_t)oh - 1) * s1 + full.k_eff;
    const int64_t pad_total = required_hin_pad - ih;
    if (pad_total < p1 || pad_total - p1 > UINT16_MAX) {
        return -4;
    }
    full.pad_h = (int)p1;
    full.hin_pad = (int)required_hin_pad;
    full.hout = (int)oh;
    full.cin_pad_b = full.hin_pad * full.win_pad_b;
    full.cout_b = full.hout * full.wout_b;

    if ((uint32_t)full.wout != ow) {
        return -4;
    }

    const uint32_t nest_count = 1u;
    const uint64_t input_x_stride = desc->src_nb[1][c_fast ? 1 : 0];
    const uint64_t input_y_stride = desc->src_nb[1][c_fast ? 2 : 1];
    const uint64_t input_ch_stride = desc->src_nb[1][c_fast ? 0 : 2];
    const uint64_t kernel_x_stride = desc->src_nb[0][c_fast ? 1 : 0];
    const uint64_t kernel_y_stride = desc->src_nb[0][c_fast ? 2 : 1];
    const uint64_t kernel_ch_stride = desc->src_nb[0][c_fast ? 0 : 3];
    const uint64_t output_x_stride = desc->nb[c_fast ? 1 : 0];
    const uint64_t output_y_stride = desc->nb[c_fast ? 2 : 1];
    const uint64_t output_ch_stride = desc->nb[c_fast ? 0 : 2];
    const uint32_t scratch_base = desc->l2_scratch_addr;
    uint32_t ch_start = 0;

    while (ch_start < c) {
        for (uint32_t nest_id = 0; nest_id < nest_count && ch_start < c; ++nest_id) {
            const uint32_t remaining_ch = c - ch_start;
            const uint32_t remaining_nests = nest_count - nest_id;
            uint32_t ch_count = (remaining_ch + remaining_nests - 1u) / remaining_nests;

            ConvParams p;
            L2Info l2i;
            bool found_slice = false;
            while (ch_count > 0) {
                p = full;
                p.cin = (int) ch_count;
                p.cout = (int) ch_count;
                derive_convParam(&p);
                p.hin_pad = full.hin_pad;
                p.hout = full.hout;
                p.cin_pad_b = p.hin_pad * p.win_pad_b;
                p.cout_b = p.hout * p.wout_b;
                if ((uint64_t) p.hin_pad * (uint64_t) p.cin <= UINT16_MAX &&
                    calculate_l2_usage_bias32_dwconv(&p, &l2i, 0u, desc->l2_scratch_capacity) &&
                    ggml_gtx_dwconv_l2_slice_fits(&p, &l2i)) {
                    ggml_gtx_dwconv_rebase_scratch(&l2i, scratch_base);
                    found_slice = true;
                    break;
                }
                --ch_count;
            }

            if (!found_slice) {
                return -5;
            }

            DDRInfo ddri = {
                .input_addr = desc->l2_src_addr[1] + (uint64_t) ch_start * input_ch_stride,
                .kernel_addr = desc->l2_src_addr[0] + (uint64_t) ch_start * kernel_ch_stride,
                .bias_addr = desc->l2_src_addr[0] + (uint64_t) ch_start * kernel_ch_stride,
                .output_addr = desc->l2_dst_addr + (uint64_t) ch_start * output_ch_stride,
            };

            SPUTileMap stm;
            TileExecPlan t;
            tile_dwconv_workload(&p, &stm, &t, GGML_GTX_SPU_NUM_PER_NEST);

            ggml_gtx_dwconv_load_l2_strided((int) nest_id,
                                            &p,
                                            &l2i,
                                            ddri.input_addr,
                                            ddri.kernel_addr,
                                            input_x_stride,
                                            input_y_stride,
                                            input_ch_stride,
                                            kernel_x_stride,
                                            kernel_y_stride,
                                            kernel_ch_stride);
            if (stm.h_div * stm.w_div > 1) {
                ggml_gtx_dwconv_2d_spu_tile_compute_only((int) nest_id, &p, &stm, &t, &l2i);
            } else {
                ggml_gtx_dwconv_2d_tile_compute_only((int) nest_id, &p, &stm, &t, &l2i);
            }
            ggml_gtx_dwconv_store_l2_strided((int) nest_id,
                                             &p,
                                             l2i.output_addr,
                                             ddri.output_addr,
                                             output_x_stride,
                                             output_y_stride,
                                             output_ch_stride);
            ch_start += ch_count;
        }
    }

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_conv_2d_dw_f16(arg);
}
