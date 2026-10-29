//==================================================================
// GGML GTX kernel: CONV_2D
//
// Descriptor wrapper around the host-owned GTX convolution core.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#include "ggml-gtx-kernel-conv-2d-core.h"

static void ggml_gtx_conv_rebase_scratch(L2Info * l2i, uint32_t scratch_base) {
    l2i->input_addr += scratch_base;
    l2i->kernel_addr += scratch_base;
    l2i->bias_addr += scratch_base;
    l2i->output_addr += scratch_base;
    l2i->bias_ones_addr += scratch_base;
}

static void set_single_conv_l2_mapped_input(
    int nest_id, ConvParams * p,
    uint32_t l2_offset, uint32_t l2_cap, L2Info * l2i,
    uint32_t l2_input, uint32_t l2_kernel,
    uint32_t input_channel_stride,
    uint32_t pad_h_top,
    uint32_t pad_h_bottom)
{
    (void) l2_offset;
    (void) l2_cap;

    const uint32_t input_row_bytes = p->w_in * FP16_B;
    const uint32_t padded_row_bytes = p->win_pad * FP16_B;
    const uint32_t padded_channel_bytes = p->hin_pad * padded_row_bytes;
    const uint32_t knl_read_length = p->c_in * p->k_n * FP16_B;

    __start_plan(nest_id);
        __start_shared();
            __fill(l2i->input_addr, padded_row_bytes, (uint16_t)padded_row_bytes,
                   (uint16_t)((uint32_t)p->hin_pad * p->c_in), 0u, 0);
            for (int c = 0; c < p->c_in; ++c) {
                __copy(l2_input + (uint32_t)c * input_channel_stride,
                       l2i->input_addr + (uint32_t)c * padded_channel_bytes +
                           pad_h_top * padded_row_bytes + (uint32_t)p->pad_w * FP16_B,
                       input_row_bytes, (uint16_t)input_row_bytes,
                       (uint16_t)p->h_in, padded_row_bytes);
            }

            __copy(l2_kernel, l2i->kernel_addr,
                   knl_read_length, (uint16_t)knl_read_length,
                   (uint16_t)p->c_out, knl_read_length);

            const uint64_t bias_fill_pattern = (p->bias > 0) ? 0x3C003C003C003C00ull : 0x0000ull;
            const uint32_t bias_length = p->c_out * FP16_B;
            __fill(l2i->bias_addr, bias_length, (uint16_t)bias_length, 1, 0u, 0);
            __fill(l2i->bias_ones_addr, 0x1000, 0x0fff, 0x10, bias_fill_pattern, 0);
        __end_shared();
    __end_plan(nest_id);
}

static void copy_conv_l2_h_slice(
    int nest_id,
    const ConvParams * p,
    uint32_t l2_out_addr,
    uint32_t l2_output,
    uint32_t output_oc_stride)
{
    const uint32_t row_bytes = (uint32_t) p->w_out * FP16_B;
    const uint32_t channel_bytes = (uint32_t) p->h_out * row_bytes;

    __start_plan(nest_id);
        __start_shared();
            for (int oc_idx = 0; oc_idx < p->c_out; ++oc_idx) {
                __copy(l2_out_addr + (uint32_t) oc_idx * channel_bytes,
                       l2_output + (uint32_t) oc_idx * output_oc_stride,
                       row_bytes, (uint16_t)row_bytes,
                       (uint16_t)p->h_out, row_bytes);
            }
        __end_shared();
    __end_plan(nest_id);
}

int ggml_gtx_kernel_conv_2d_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CONV_2D_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CONV_2D_F16) {
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

    const uint32_t kw = desc->src_ne[0][0];
    const uint32_t kh = desc->src_ne[0][1];
    const uint32_t ic = desc->src_ne[0][2];
    const uint32_t oc = desc->src_ne[0][3];
    const uint32_t iw = desc->src_ne[1][0];
    const uint32_t ih = desc->src_ne[1][1];
    const uint32_t n  = desc->src_ne[1][3];
    const int32_t s0 = desc->op_params[0];
    const int32_t s1 = desc->op_params[1];
    const int32_t p0 = desc->op_params[2];
    const int32_t p1 = desc->op_params[3];
    const int32_t d0 = desc->op_params[4];
    const int32_t d1 = desc->op_params[5];

    if (kw == 0 || kh == 0 || ic == 0 || oc == 0 || iw == 0 || ih == 0 ||
        kw != kh || desc->src_ne[1][2] != ic || desc->ne[2] != oc ||
        n != 1 || desc->ne[3] != 1 ||
        s0 <= 0 || s1 <= 0 || s0 != s1 ||
        d0 <= 0 || d1 <= 0 || d0 != d1 ||
        p0 < 0 || p1 < 0) {
        return -3;
    }

    ConvParams full = {
        .c_in = (int) ic,
        .h_in = (int) ih,
        .w_in = (int) iw,
        .c_out = (int) oc,
        .k_size = (int) kw,
        .pad_h = 0,
        .pad_w = (int) p0,
        .stride = (int) s0,
        .dilation = (int) d0,
        .bias = false,
    };
    derive_convParam(&full);

    const int64_t required_hin_pad = ((int64_t)desc->ne[1] - 1) * s1 + full.k_eff;
    const int64_t pad_total = required_hin_pad - ih;
    if (pad_total < p1 || pad_total - p1 > UINT16_MAX) {
        return -4;
    }
    const uint32_t local_pad_bottom = (uint32_t)(pad_total - p1);
    full.hin_pad = (int)required_hin_pad;
    full.h_out = (int)desc->ne[1];

    if ((uint32_t)full.w_out != desc->ne[0]) {
        return -4;
    }

    const uint32_t nest_count = 1u;
    const uint32_t dst_oc_stride = desc->nb[2];
    const uint32_t scratch_base = desc->l2_scratch_addr;
    L2Info full_l2i;
    if (calculate_l2_usage_bias(&full, &full_l2i, 0u, desc->l2_scratch_capacity)) {
        ggml_gtx_conv_rebase_scratch(&full_l2i, scratch_base);
        SPUTileMap stm;
        TileExecPlan t;
        tile_nmconv_general(&full, &stm, &t, GGML_GTX_SPU_NUM_PER_NEST);

        set_single_conv_l2_mapped_input(0, &full, scratch_base, desc->l2_scratch_capacity, &full_l2i,
                                        (uint32_t)desc->l2_src_addr[1],
                                        (uint32_t)desc->l2_src_addr[0],
                                        desc->src_nb[1][2], (uint32_t)p1, local_pad_bottom);
        conv_tile_dim4_with_bias(0, &full, &stm, &t, &full_l2i);
        copy_conv_l2_h_slice(0, &full, full_l2i.output_addr,
                             (uint32_t)desc->l2_dst_addr, dst_oc_stride);
        return 0;
    }

    uint32_t out_h_start = 0;

    while (out_h_start < desc->ne[1]) {
        for (uint32_t nest_id = 0; nest_id < nest_count && out_h_start < desc->ne[1]; ++nest_id) {
            const uint32_t remaining_h = desc->ne[1] - out_h_start;
            const uint32_t remaining_nests = nest_count - nest_id;
            uint32_t out_h_count = (remaining_h + remaining_nests - 1u) / remaining_nests;

            ConvParams p;
            L2Info l2i;
            int32_t clip_h_start = 0;
            uint32_t local_pad_top = 0;
            uint32_t local_pad_bottom = 0;
            bool found_slice = false;

            while (out_h_count > 0) {
                const int32_t raw_h_start = (int32_t) out_h_start * s1 - p1;
                const int32_t raw_h_end =
                    (int32_t) (out_h_start + out_h_count - 1u) * s1 - p1 + full.k_eff;
                clip_h_start = raw_h_start < 0 ? 0 : raw_h_start;
                const int32_t clip_h_end = raw_h_end > (int32_t) ih ? (int32_t) ih : raw_h_end;
                local_pad_top = (uint32_t) (raw_h_start < 0 ? -raw_h_start : 0);
                local_pad_bottom = (uint32_t) (raw_h_end > (int32_t) ih ? raw_h_end - (int32_t) ih : 0);
                const uint32_t local_h_in =
                    clip_h_end > clip_h_start ? (uint32_t) (clip_h_end - clip_h_start) : 0u;
                if (local_h_in == 0) {
                    --out_h_count;
                    continue;
                }

                p = full;
                p.h_in = (int) local_h_in;
                p.pad_h = 0;
                derive_convParam(&p);
                p.hin_pad = (int) (local_h_in + local_pad_top + local_pad_bottom);
                p.h_out = (p.hin_pad - p.k_eff) / p.stride + 1;
                if ((uint32_t) p.h_out != out_h_count) {
                    --out_h_count;
                    continue;
                }
                if (!calculate_l2_usage_bias(&p, &l2i, 0u, desc->l2_scratch_capacity)) {
                    --out_h_count;
                    continue;
                }
                ggml_gtx_conv_rebase_scratch(&l2i, scratch_base);
                found_slice = true;
                break;
            }

            if (!found_slice) {
                return -7;
            }

            SPUTileMap stm;
            TileExecPlan t;
            tile_nmconv_general(&p, &stm, &t, GGML_GTX_SPU_NUM_PER_NEST);

            const uint32_t input_addr = (uint32_t)desc->l2_src_addr[1] +
                (uint32_t)clip_h_start * desc->src_nb[1][1];
            const uint32_t dst_addr = (uint32_t)desc->l2_dst_addr +
                out_h_start * desc->nb[1];

            set_single_conv_l2_mapped_input((int)nest_id, &p, scratch_base,
                                            desc->l2_scratch_capacity, &l2i,
                                            input_addr, (uint32_t)desc->l2_src_addr[0],
                                            desc->src_nb[1][2], local_pad_top, local_pad_bottom);
            conv_tile_dim4_with_bias((int) nest_id, &p, &stm, &t, &l2i);
            copy_conv_l2_h_slice((int)nest_id, &p, l2i.output_addr, dst_addr, dst_oc_stride);
            out_h_start += out_h_count;
        }
    }

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_conv_2d_f16(arg);
}
