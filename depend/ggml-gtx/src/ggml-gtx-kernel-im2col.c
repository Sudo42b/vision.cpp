//==================================================================
// GGML GTX kernel: IM2COL
//==================================================================

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#ifdef GGML_GTX_USE_GTX
#include "ggml-gtx-plan.h"
#include "ggml-gtx-config.h"
#include "ggml-gtx/intrinsics/intrin.h"
#endif

int ggml_gtx_kernel_im2col_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_IM2COL_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_IM2COL_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->src_addr[1] == 0) {
        return -2;
    }

    const bool is_2d = desc->op_params[6] == 1;
    const uint16_t iw = (uint16_t) desc->src_ne[1][0];
    const uint16_t ih = is_2d ? (uint16_t) desc->src_ne[1][1] : 1;
    const uint16_t ic = is_2d ? (uint16_t) desc->src_ne[1][2] : (uint16_t) desc->src_ne[1][1];
    const uint16_t n  = is_2d ? (uint16_t) desc->src_ne[1][3] : (uint16_t) desc->src_ne[1][2];
    const uint16_t kw = (uint16_t) desc->src_ne[0][0];
    const uint16_t kh = is_2d ? (uint16_t) desc->src_ne[0][1] : 1;

    ggml_gtx_compute_im2col(desc->src_addr[1],
                            desc->dst_addr,
                            iw,
                            ih,
                            ic,
                            n,
                            kw,
                            kh,
                            (int16_t) desc->op_params[0],
                            (int16_t) desc->op_params[1],
                            (int16_t) desc->op_params[2],
                            (int16_t) desc->op_params[3],
                            (int16_t) desc->op_params[4],
                            (int16_t) desc->op_params[5],
                            is_2d);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_im2col_f16(arg);
}

//==================================================================

#ifdef GGML_GTX_USE_GTX
#define GTX_IM2COL_FP16_B           2
#define GTX_IM2COL_L1_4K            0x01000
#define GTX_IM2COL_L1_END           0x60000
#define GTX_IM2COL_SPU_PER_NEST     GGML_GTX_SPU_NUM_PER_NEST
#define GTX_IM2COL_HALO_RATIO       4

#define GTX_IM2COL_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define GTX_IM2COL_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define GTX_IM2COL_FIND_INPUT_LENGTH(l_out, k_eff, stride) (((l_out) - 1) * (stride) + (k_eff))

typedef struct {
    int c_in;
    int h_in;
    int w_in;
    int h_out;
    int w_out;
    int h_in_pad;
    int w_in_pad;
    int k_size;
    int k_n;
    int k_eff;
    int pad_h;
    int pad_w;
    int stride;
    int dilation;
} gtx_im2col_params_t;

typedef struct {
    int num_w;
    int num_h;
    int num_c_in;
    int len_w_out;
    int len_h_out;
    int len_w_in;
    int len_h_in;
    int len_c_in;
} gtx_im2col_tile_plan_t;

typedef struct {
    uint32_t input;
    uint32_t result;
} gtx_im2col_l1_banks_t;

static inline int gtx_im2col_ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline int gtx_im2col_ceil_rem(int a, int b) {
    return (a + b - 1) % b + 1;
}

static void gtx_im2col_derive_params(gtx_im2col_params_t * p) {
    p->h_in_pad = p->h_in + p->pad_h * 2;
    p->w_in_pad = p->w_in + p->pad_w * 2;
    p->k_n = p->k_size * p->k_size;
    p->k_eff = (p->k_size - 1) * p->dilation + 1;
    p->h_out = (p->h_in_pad - p->k_eff) / p->stride + 1;
    p->w_out = (p->w_in_pad - p->k_eff) / p->stride + 1;
}

static void gtx_im2col_init_tile(gtx_im2col_tile_plan_t * t) {
    t->num_c_in = 1;
    t->num_w = 1;
    t->num_h = 1;
}

static void gtx_im2col_update_w_tile(int num_w, const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t) {
    t->num_w = num_w;
    t->len_w_out = gtx_im2col_ceil_div(p->w_out, t->num_w);
    t->len_w_in = GTX_IM2COL_FIND_INPUT_LENGTH(t->len_w_out, p->k_eff, p->stride);
}

static void gtx_im2col_update_h_tile(int num_h, const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t) {
    t->num_h = num_h;
    t->len_h_out = gtx_im2col_ceil_div(p->h_out, num_h);
    t->len_h_in = GTX_IM2COL_FIND_INPUT_LENGTH(t->len_h_out, p->k_eff, p->stride);
}

static void gtx_im2col_update_cin_tile(int num_cin, const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t) {
    t->num_c_in = num_cin;
    t->len_c_in  = gtx_im2col_ceil_div(p->c_in, t->num_c_in);
}

static void gtx_im2col_update_tile(const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t) {
    t->len_w_out = gtx_im2col_ceil_div(p->w_out, t->num_w);
    t->len_h_out = gtx_im2col_ceil_div(p->h_out, t->num_h);
    t->len_c_in  = gtx_im2col_ceil_div(p->c_in, t->num_c_in);
    t->len_w_in = GTX_IM2COL_FIND_INPUT_LENGTH(t->len_w_out, p->k_eff, p->stride);
    t->len_h_in = GTX_IM2COL_FIND_INPUT_LENGTH(t->len_h_out, p->k_eff, p->stride);
}

static bool gtx_im2col_calculate_l1_usage(const gtx_im2col_params_t * p, const gtx_im2col_tile_plan_t * t, gtx_im2col_l1_banks_t * b) {
    uint32_t input_bytes = (uint32_t)t->len_c_in * t->len_w_in * t->len_h_in * GTX_IM2COL_FP16_B;
    uint64_t im2col_bytes = (uint64_t)t->len_c_in * p->k_n * t->len_h_out * t->len_w_out * GTX_IM2COL_FP16_B;
    uint32_t ceil_input_bytes = (uint32_t)gtx_im2col_ceil_div((int)input_bytes, GTX_IM2COL_L1_4K) * GTX_IM2COL_L1_4K;
    uint64_t ceil_im2col_bytes = (uint64_t)gtx_im2col_ceil_div((int)im2col_bytes, GTX_IM2COL_L1_4K) * GTX_IM2COL_L1_4K;
    b->input = ceil_input_bytes;
    b->result = (uint32_t)ceil_im2col_bytes;
    return ceil_input_bytes + ceil_im2col_bytes < GTX_IM2COL_L1_END;
}

static bool gtx_im2col_tile_cin(const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t, gtx_im2col_l1_banks_t * b) {
    gtx_im2col_init_tile(t);
    gtx_im2col_update_tile(p, t);

    if (p->c_in < GTX_IM2COL_SPU_PER_NEST) {
        gtx_im2col_update_cin_tile(p->c_in, p, t);
        return gtx_im2col_calculate_l1_usage(p, t, b);
    }

    int div_cin = GTX_IM2COL_SPU_PER_NEST;
    while (t->len_c_in > 1) {
        gtx_im2col_update_cin_tile(div_cin, p, t);
        if (gtx_im2col_calculate_l1_usage(p, t, b)) return true;
        div_cin = GTX_IM2COL_MIN(div_cin + GTX_IM2COL_SPU_PER_NEST, p->c_in);
    }

    return false;
}

static bool gtx_im2col_tile_wh(const gtx_im2col_params_t * p, gtx_im2col_tile_plan_t * t, gtx_im2col_l1_banks_t * b) {
    uint64_t block_bytes = (uint64_t)b->input + b->result;
    int div_estimate = gtx_im2col_ceil_div((int)block_bytes, GTX_IM2COL_L1_END);
    int div_h = div_estimate > 8 ? 8 : div_estimate;
    int halo = p->k_eff - p->stride;
    if (halo <= 0) halo = 1;

    int max_h_div = GTX_IM2COL_MAX(p->h_in / p->k_eff, p->h_in / (halo * GTX_IM2COL_HALO_RATIO));
    while (div_h < max_h_div) {
        gtx_im2col_update_h_tile(div_h, p, t);
        if (gtx_im2col_calculate_l1_usage(p, t, b)) return true;
        div_h++;
    }

    int div_w = 2;
    int max_w_div = GTX_IM2COL_MAX(p->w_in / p->k_eff, p->w_in / (halo * GTX_IM2COL_HALO_RATIO));
    while (div_w < max_w_div) {
        gtx_im2col_update_w_tile(div_w, p, t);
        if (gtx_im2col_calculate_l1_usage(p, t, b)) return true;
        div_w++;
    }

    return false;
}

static uint64_t gtx_im2col_kernel_tile_cin(uint8_t nest_id, uint64_t ddr_input, uint64_t ddr_result,
                                           const gtx_im2col_params_t * p,
                                           const gtx_im2col_tile_plan_t * t,
                                           const gtx_im2col_l1_banks_t * b) {
    uint32_t l2_input   = 0x0c00000;
    uint32_t l2_result  = 0x0;
    int l2_input_cap    = 0x500000;
    int l2_result_cap   = 0x1100000;

    uint32_t bank_a = GGML_GTX_L1_BANK_A_OFFSET;
    uint32_t bank_r = GTX_IM2COL_L1_END - b->input;

    __start_plan(nest_id);
        for (int spu_id = 0; spu_id < GTX_IM2COL_SPU_PER_NEST; spu_id++) {
            __start_thread(spu_id);
                __set_spm_addr(bank_r, 0x0, 0x0, bank_a);
            __end_thread(spu_id);
        }
    __end_plan(nest_id);

    int len_c_in_extra = gtx_im2col_ceil_rem(p->c_in, t->len_c_in);

    uint32_t input_bytes_row = (uint32_t)p->w_in * GTX_IM2COL_FP16_B;
    uint32_t input_bytes_channel = (uint32_t)p->h_in * input_bytes_row;
    uint32_t input_ddr_stride = (uint32_t)t->len_c_in * input_bytes_channel;
    uint32_t input_ddr_offset = 0;

    uint32_t input_bytes_row_pad = (uint32_t)p->w_in_pad * GTX_IM2COL_FP16_B;
    uint32_t input_l2_stride = (uint32_t)t->len_c_in * p->h_in_pad * p->w_in_pad * GTX_IM2COL_FP16_B;
    uint32_t input_l2_offset = 0;

    uint32_t output_height = (uint32_t)p->h_out * p->w_out;
    uint32_t output_length = (uint32_t)t->len_c_in * p->k_n * GTX_IM2COL_FP16_B;
    uint32_t output_stride = output_height * output_length;
    uint32_t output_l2_offset = 0;
    uint32_t output_ddr_offset = 0;

    for (int tile_id = 0; tile_id < t->num_c_in; tile_id++) {
        int spu_id = tile_id % GTX_IM2COL_SPU_PER_NEST;
        uint64_t spu_mask = (uint64_t)0x1 << spu_id;

        int len_c_in = (tile_id == t->num_c_in - 1) ? len_c_in_extra : t->len_c_in;
        uint32_t input_l1_read_height = (uint32_t)len_c_in * p->h_in_pad;
        uint32_t output_row = (uint32_t)len_c_in * p->k_n;
        uint32_t output_bytes_row = output_row * GTX_IM2COL_FP16_B;

        __start_plan(nest_id);
            __start_shared();
                __pad(ddr_input + input_ddr_offset, l2_input + input_l2_offset,
                      input_bytes_row, input_bytes_row, p->h_in,
                      (uint8_t)p->pad_h, (uint8_t)p->pad_h, (uint8_t)p->pad_w, (uint8_t)p->pad_w,
                      0x0, input_bytes_channel, len_c_in, 0b10, 0x1);
                __credit_ld(spu_mask, 0);

                __credit_chk(spu_mask);
                __tpose(l2_result + output_l2_offset, ddr_result + output_ddr_offset, output_row,
                        p->w_out, p->h_out, 1, 0, 2, 0b01, 1);
                __credit_st(spu_mask);
            __end_shared();

            __start_thread(spu_id);
                __credit_chk(0xDEAD);
                __load(l2_input + input_l2_offset, bank_r,
                       input_bytes_row_pad, input_bytes_row_pad, input_l1_read_height, input_bytes_row_pad);
                __credit_ld(spu_mask, 0);
                __im2col_n(p->h_in_pad, p->w_in_pad, p->k_size, p->dilation, p->stride, len_c_in);
                __store(bank_a, l2_result + output_l2_offset,
                        output_bytes_row, output_bytes_row, output_height, output_bytes_row);
                __credit_st(spu_mask);
            __end_thread(spu_id);
        __end_plan(nest_id);

        input_ddr_offset += input_ddr_stride;
        output_ddr_offset += output_stride;
        input_l2_offset = input_l2_offset < (uint32_t)l2_input_cap ? input_l2_offset + input_l2_stride : 0;
        output_l2_offset = output_l2_offset < (uint32_t)l2_result_cap ? output_l2_offset + output_stride : 0;
    }

    return 0;
}

static uint64_t gtx_im2col_kernel_tile_h(uint8_t nest_id, uint64_t ddr_input, uint64_t ddr_result,
                                         const gtx_im2col_params_t * p,
                                         const gtx_im2col_tile_plan_t * t,
                                         const gtx_im2col_l1_banks_t * b) {
    uint32_t l2_result      = 0x0;
    uint32_t l2_tpose       = 0x0600000;
    uint32_t l2_input       = 0x0c00000;
    uint32_t l2_input_cap   = 0x04A0000;
    uint32_t l2_result_cap  = 0x08A0000;

    uint32_t bank_a = GGML_GTX_L1_BANK_A_OFFSET;
    uint32_t bank_r = GTX_IM2COL_L1_END - b->input;

    int len_c_in_extra  = gtx_im2col_ceil_rem(p->c_in, t->len_c_in);
    int len_h_out_extra = gtx_im2col_ceil_rem(p->h_out, t->len_h_out);
    int len_h_in_extra  = GTX_IM2COL_FIND_INPUT_LENGTH(len_h_out_extra, p->k_eff, p->stride);

    uint32_t input_ddr_row = (uint32_t)p->w_in * GTX_IM2COL_FP16_B;
    uint32_t input_ddr_height_stride = (uint32_t)t->len_h_out * p->stride * input_ddr_row;
    uint32_t input_ddr_channel = (uint32_t)p->h_in * input_ddr_row;
    uint32_t input_ddr_channel_stride = (uint32_t)t->len_c_in * input_ddr_channel;

    uint32_t input_l2_row = (uint32_t)p->w_in_pad * GTX_IM2COL_FP16_B;
    uint32_t input_l2_height_stride = (uint32_t)t->len_h_in * input_l2_row;
    uint32_t input_l2_channel_stride = (uint32_t)t->len_c_in * p->h_in_pad * p->w_in_pad * GTX_IM2COL_FP16_B;

    uint32_t output_l2_width_stride = (uint32_t)t->len_c_in * p->k_n * GTX_IM2COL_FP16_B;
    uint32_t output_l2_height_stride = (uint32_t)t->len_h_out * p->w_out * output_l2_width_stride;

    uint32_t output_ddr_width = (uint32_t)p->w_out * GTX_IM2COL_FP16_B;
    uint32_t output_ddr_row = (uint32_t)p->h_out * output_ddr_width;
    uint32_t output_ddr_height_stride = (uint32_t)t->len_h_out * output_ddr_width;
    uint64_t output_ddr_channel_stride = (uint64_t)t->len_c_in * p->k_n * output_ddr_row;

    for (int c_id = 0; c_id < t->num_c_in; c_id++) {
        int len_c_in = (c_id < t->num_c_in - 1) ? t->len_c_in : len_c_in_extra;
        int t_cin_kn = len_c_in * p->k_n;

        for (int h_id = 0; h_id < t->num_h; h_id++) {
            int tile_id = c_id * t->num_h + h_id;
            int spu_id = tile_id % GTX_IM2COL_SPU_PER_NEST;
            uint16_t spu_mask = (uint16_t)(0x1 << spu_id);

            int len_h_in = t->len_h_in;
            int len_h_out = t->len_h_out;
            int len_h_l2_load = len_h_in;
            int pad_h_ddr_offset = p->pad_h * (int)input_ddr_row;
            int pad_u = 0;
            int pad_d = 0;

            if (h_id == 0) {
                pad_u = p->pad_h;
                len_h_l2_load = len_h_in - pad_u;
                pad_h_ddr_offset = 0;
            } else if (h_id == t->num_h - 1) {
                len_h_out = len_h_out_extra;
                len_h_in = len_h_in_extra;
                pad_d = p->pad_h;
                len_h_l2_load = len_h_in - pad_d;
            }

            uint64_t input_ddr_offset = (uint64_t)c_id * input_ddr_channel_stride +
                                        (uint64_t)h_id * input_ddr_height_stride -
                                        (uint64_t)pad_h_ddr_offset;
            uint32_t input_l2_offset = (c_id * input_l2_channel_stride + h_id * input_l2_height_stride) % l2_input_cap;
            uint32_t output_l2_offset = (h_id * output_l2_height_stride + c_id * output_l2_width_stride) % l2_result_cap;
            uint64_t output_ddr_offset = (uint64_t)c_id * output_ddr_channel_stride + (uint64_t)h_id * output_ddr_height_stride;
            uint32_t output_ddr_length = output_ddr_width * (uint32_t)len_h_out;

            uint32_t output_l2_width = (uint32_t)t_cin_kn * GTX_IM2COL_FP16_B;
            uint32_t output_l2_height = (uint32_t)len_h_out * p->w_out;
            uint32_t input_l1_height = (uint32_t)len_c_in * len_h_in;

            __split();
            __start_plan(nest_id);
                __start_shared();
                    __pad(ddr_input + input_ddr_offset, l2_input + input_l2_offset,
                          input_ddr_row, input_ddr_row, len_h_l2_load,
                          (uint8_t)pad_u, (uint8_t)pad_d, (uint8_t)p->pad_w, (uint8_t)p->pad_w,
                          0x0, input_ddr_channel, len_c_in, 0b10, 0x1);
                    __credit_ld(spu_mask, 0);

                    __credit_chk(spu_mask);
                    __tpose(l2_result + output_l2_offset, l2_tpose + output_l2_offset,
                            t_cin_kn, p->w_out, len_h_out, 1, 0, 2, 0b00, 1);
                    __store(l2_tpose + output_l2_offset, ddr_result + output_ddr_offset,
                            output_ddr_row, output_ddr_length, t_cin_kn, output_ddr_length);
                    __credit_st(spu_mask);
                __end_shared();

                __start_thread(spu_id);
                    __set_spm_addr(bank_r, 0x0, 0x0, bank_a);
                    __credit_chk(0xDEAD);
                    __load(l2_input + input_l2_offset, bank_r,
                           input_l2_row, input_l2_row, input_l1_height, input_l2_row);
                    __credit_ld(spu_mask, 0);
                    __im2col_n(len_h_in, p->w_in_pad, p->k_size, p->dilation, p->stride, len_c_in);
                    __store(bank_a, l2_result + output_l2_offset,
                            output_l2_width, output_l2_width, output_l2_height, output_l2_width);
                    __credit_st(spu_mask);
                __end_thread(spu_id);
            __end_plan(nest_id);
            __join();
        }
    }

    return 0;
}

static void gtx_im2col_kernel_1d(uint8_t nest_id, uint64_t ddr_input, uint64_t ddr_result,
                                 uint16_t ch, uint16_t w_in, uint16_t pad_w,
                                 uint16_t k_size, uint16_t stride,
                                 uint16_t dilation, uint16_t n_seq) {
    const uint32_t l2_result = 0x0;
    const uint32_t l2_input  = 0x800000;

    const int w_in_pad = (int)w_in + (int)pad_w * 2;
    const int k_eff = ((int)k_size - 1) * (int)dilation + 1;
    if (w_in_pad < k_eff || stride == 0) {
        return;
    }

    const uint32_t pad_offset = (uint32_t)pad_w * GTX_IM2COL_FP16_B;
    const uint32_t in_length = (uint32_t)w_in * GTX_IM2COL_FP16_B;
    const uint32_t padded_length = (uint32_t)w_in_pad * GTX_IM2COL_FP16_B;
    const uint32_t w_out = (uint32_t)((w_in_pad - k_eff) / (int)stride + 1);

    const uint32_t stride_bytes = (uint32_t)stride * GTX_IM2COL_FP16_B;
    const uint32_t dilation_bytes = (uint32_t)dilation * GTX_IM2COL_FP16_B;
    const uint32_t kernel_bytes = (uint32_t)k_size * GTX_IM2COL_FP16_B;
    const uint32_t channel_kernel_bytes = (uint32_t)ch * kernel_bytes;
    const uint32_t input_seq_bytes = in_length * (uint32_t)ch;
    const uint32_t padded_seq_bytes = padded_length * (uint32_t)ch;
    const uint32_t output_seq_bytes = w_out * channel_kernel_bytes;

    __split();
    __start_plan(nest_id);
        __start_shared();
            __fill(l2_input, padded_length, padded_length, (uint32_t)ch * n_seq, 0x0, 0b0);
            __load(ddr_input, l2_input + pad_offset,
                   in_length, in_length, (uint32_t)ch * n_seq, padded_length);

            for (uint16_t sid = 0; sid < n_seq; sid++) {
                uint32_t base_src = l2_input + (uint32_t)sid * padded_seq_bytes;
                uint32_t base_dst = l2_result + (uint32_t)sid * output_seq_bytes;
                for (uint32_t ow = 0; ow < w_out; ow++) {
                    uint32_t base_src_w = base_src + ow * stride_bytes;
                    uint32_t base_dst_w = base_dst + ow * channel_kernel_bytes;
                    for (uint16_t cid = 0; cid < ch; cid++) {
                        uint32_t addr_src = base_src_w + (uint32_t)cid * padded_length;
                        uint32_t addr_dst = base_dst_w + (uint32_t)cid * kernel_bytes;
                        __copy(addr_src, addr_dst,
                               dilation_bytes, GTX_IM2COL_FP16_B, k_size, GTX_IM2COL_FP16_B);
                    }
                }
            }

            __store(l2_result, ddr_result,
                    channel_kernel_bytes, channel_kernel_bytes,
                    w_out * (uint32_t)n_seq, channel_kernel_bytes);
        __end_shared();
    __end_plan(nest_id);
    __join();
}
#endif

void ggml_gtx_compute_im2col(
    uint64_t ddr_src,       // input data (F32)
    uint64_t ddr_dst,       // output (FP16)
    uint16_t IW,
    uint16_t IH,
    uint16_t IC,
    uint16_t N,             // batch size
    uint16_t KW,
    uint16_t KH,
    int16_t  s0,            // stride W
    int16_t  s1,            // stride H
    int16_t  p0,            // padding W
    int16_t  p1,            // padding H
    int16_t  d0,            // dilation W
    int16_t  d1,            // dilation H
    bool     is_2D)
{
#ifdef GGML_GTX_USE_GTX
    if (s0 <= 0 || p0 < 0 || d0 <= 0) {
        return;
    }

    if (is_2D && (KH != KW || s1 != s0 || p1 != p0 || d1 != d0)) {
        return;
    }

    if (!is_2D) {
        gtx_im2col_kernel_1d(0, ddr_src, ddr_dst,
                             IC, IW, (uint16_t)p0, KW, (uint16_t)s0, (uint16_t)d0, N);
        return;
    }

    const uint16_t OW = (uint16_t)((IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1);
    const uint16_t OH = (uint16_t)((IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1);
    volatile const uint16_t * src = (volatile const uint16_t *)(uintptr_t)ddr_src;
    volatile uint16_t * dst = (volatile uint16_t *)(uintptr_t)ddr_dst;
    const uint32_t dst_ne0 = (uint32_t)IC * KH * KW;

    for (uint16_t in = 0; in < N; in++) {
        for (uint16_t ioh = 0; ioh < OH; ioh++) {
            for (uint16_t iow = 0; iow < OW; iow++) {
                const uint32_t dst_col = (uint32_t)in * OH * OW + (uint32_t)ioh * OW + iow;

                for (uint16_t iic = 0; iic < IC; iic++) {
                    for (uint16_t ikh = 0; ikh < KH; ikh++) {
                        for (uint16_t ikw = 0; ikw < KW; ikw++) {
                            const int32_t iiw = (int32_t)iow * s0 + (int32_t)ikw * d0 - p0;
                            const int32_t iih = (int32_t)ioh * s1 + (int32_t)ikh * d1 - p1;

                            uint16_t fp16_val = 0;
                            if (iiw >= 0 && iiw < IW && iih >= 0 && iih < IH) {
                                const uint64_t src_idx = (uint64_t)in * IC * IH * IW
                                                       + (uint64_t)iic * IH * IW
                                                       + (uint64_t)iih * IW
                                                       + (uint64_t)iiw;
                                fp16_val = src[src_idx];
                            }

                            const uint32_t dst_row = (uint32_t)iic * KH * KW + (uint32_t)ikh * KW + ikw;
                            dst[dst_col * dst_ne0 + dst_row] = fp16_val;
                        }
                    }
                }
            }
        }
    }
    return;

    gtx_im2col_params_t p = {
        .c_in = IC,
        .h_in = IH,
        .w_in = IW,
        .pad_h = p1,
        .pad_w = p0,
        .k_size = KW,
        .stride = s0,
        .dilation = d0,
    };
    gtx_im2col_derive_params(&p);
    if (p.h_out <= 0 || p.w_out <= 0) {
        return;
    }

    gtx_im2col_tile_plan_t t;
    gtx_im2col_l1_banks_t b;
    bool use_tile_cin = gtx_im2col_tile_cin(&p, &t, &b);
    bool use_tile_h = false;

    if (!use_tile_cin) {
        use_tile_h = gtx_im2col_tile_wh(&p, &t, &b);
    }

    if (!use_tile_cin && !use_tile_h) {
        return;
    }

    uint64_t src_batch_bytes = (uint64_t)p.c_in * p.h_in * p.w_in * GTX_IM2COL_FP16_B;
    uint64_t dst_batch_bytes = (uint64_t)p.c_in * p.k_n * p.h_out * p.w_out * GTX_IM2COL_FP16_B;

    if (use_tile_cin) {
        __split();
    }

    for (uint16_t batch = 0; batch < N; ++batch) {
        uint64_t src_batch = ddr_src + (uint64_t)batch * src_batch_bytes;
        uint64_t dst_batch = ddr_dst + (uint64_t)batch * dst_batch_bytes;

        if (use_tile_cin) {
            gtx_im2col_kernel_tile_cin(0, src_batch, dst_batch, &p, &t, &b);
        } else {
            gtx_im2col_kernel_tile_h(0, src_batch, dst_batch, &p, &t, &b);
        }
    }

    if (use_tile_cin) {
        __join();
    }
#else
    // Output dimensions
    uint16_t OW = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    uint16_t OH = is_2D ? (IH + 2 * p1 - d1 * (KH - 1) - 1) / s1 + 1 : 1;

    volatile uint32_t * src = (volatile uint32_t *)(uintptr_t)ddr_src;
    volatile uint16_t * dst = (volatile uint16_t *)(uintptr_t)ddr_dst;

    // Output layout: [IC*KH*KW, OW*OH, N]
    // dst index: [batch_in * OH * OW + ioh * OW + iow] * (IC*KH*KW) + (iic*KH*KW + ikh*KW + ikw)
    uint32_t dst_ne0 = (uint32_t)IC * KH * KW;

    for (uint16_t in = 0; in < N; in++) {
        for (uint16_t ioh = 0; ioh < OH; ioh++) {
            for (uint16_t iow = 0; iow < OW; iow++) {
                uint32_t dst_col = (uint32_t)in * OH * OW + ioh * OW + iow;

                for (uint16_t iic = 0; iic < IC; iic++) {
                    for (uint16_t ikh = 0; ikh < KH; ikh++) {
                        for (uint16_t ikw = 0; ikw < KW; ikw++) {
                            int32_t iiw = (int32_t)iow * s0 + (int32_t)ikw * d0 - p0;
                            int32_t iih = is_2D ? ((int32_t)ioh * s1 + (int32_t)ikh * d1 - p1) : 0;

                            uint16_t fp16_val = 0;  // zero-padding
                            if (iiw >= 0 && iiw < IW && iih >= 0 && iih < IH) {
                                // src index: [in][iic][iih][iiw] — layout [IW, IH, IC, N]
                                uint64_t src_idx;
                                if (is_2D) {
                                    src_idx = (uint64_t)in * IC * IH * IW
                                            + (uint64_t)iic * IH * IW
                                            + (uint64_t)iih * IW
                                            + iiw;
                                } else {
                                    src_idx = (uint64_t)in * IC * IW
                                            + (uint64_t)iic * IW
                                            + iiw;
                                }
                                // F32 → FP16 conversion
                                fp16_val = ggml_gtx_fp32_bits_to_fp16(src[src_idx]);
                            }

                            uint32_t dst_row = (uint32_t)iic * KH * KW + ikh * KW + ikw;
                            dst[dst_col * dst_ne0 + dst_row] = fp16_val;
                        }
                    }
                }
            }
        }
    }
#endif
}

//==================================================================
// POOL_2D: 2D max or average pooling
//
// Input: F32 [W, H, C, N]
// Output: F32 [OW, OH, C, N]
//
// pool_type: 0 = MAX, 1 = AVG
//==================================================================

// Minimal F32 bit representation of -FLT_MAX
// IEEE 754: 0xFF7FFFFF = -3.4028235e+38
#define F32_NEG_MAX_BITS 0xFF7FFFFFu
