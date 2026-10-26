#include "gtx/kernel_params.h"
#include "gtx/print.h"
#include "gtx/intrin.h"

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>


#define FP16_B          2
#define FP32_B          4

#define MAX_SPU_NUM     16

#define L1_CAP          0x60000
#define L1_64K          0x10000
#define L1_128K         0x20000
#define L1_ALIGNMENT    0x100
#define L2_ALIGNMENT    0x1000
#define L2_CAP          0x1000000

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))


static const int MIN_OUT_T_LEN = 8;

static inline int ceil_div(int a, int b){ return (a + b - 1u) / b; }
static inline int ceil_rem(int a, int b){ return (a + b - 1u) % b + 1; }
static inline int FIND_INPUT_LENGTH(int l_out, int k_eff, int stride) {
    return (l_out - 1) * stride + k_eff;
}
static inline uint64_t align_up(uint64_t size, uint64_t align) {
    return ((size) + (align) - 1) & ~((align) - 1);
}

static inline uint64_t isqrt(uint64_t n) {
    if (n < 2) return n;
    uint64_t low = 0;
    uint64_t high = (n / 2) + 1; 
    uint64_t res = 0;
    while (low <= high) {
        uint64_t mid = low + (high - low) / 2;
        if (mid == 0) { low = mid + 1; continue; }
        else if (mid <= n / mid) { low = mid + 1; res = mid; }
        else { high = mid - 1; }
    }
    return res;
}
//=================== common ======================

typedef struct {
    int c_in, h_in, w_in;
    int c_out;
    int k_size;
    int pad_h, pad_w;
    int stride, dilation;
    bool bias;

    int hin_pad, win_pad;
    int k_eff, k_n;
    int h_out, w_out;
} ConvParams;

typedef struct {
    uint32_t input_addr;
    uint32_t kernel_addr;
    uint32_t bias_addr;
    uint32_t bias_ones_addr;
    uint32_t output_addr;
    uint32_t total_bytes;
} L2Info;

typedef struct {
    // ==== 분할 계획 (Global) ====
    int cout_div, cout_rem_cnt, cout_len;
    int h_div, h_rem_cnt, hout_len;
    int w_div, w_rem_cnt, wout_len;
    int use_spu;

    // ==== SPU별 할당 정보 (Per-SPU) - 추가된 부분 ====
    int start_cout[MAX_SPU_NUM];
    int start_hout[MAX_SPU_NUM];
    int start_wout[MAX_SPU_NUM];
    
    int cout_assigned[MAX_SPU_NUM];
    int hout_assigned[MAX_SPU_NUM];
    int wout_assigned[MAX_SPU_NUM];

} SPUTileMap;

typedef struct {
    int cin_div;
    int cout_div;
    int w_div, h_div;

    int cin_len, cout_len;
    int wout_len, hout_len;

    int work_per_spu;
} TileExecPlan;

void print_convParam(ConvParams* p) {
    gtx_printf("\n============= ConvParams ==============\n");
    gtx_printf("input      = %d x %d x %d\n", p->c_in, p->h_in, p->w_in);
    gtx_printf("knl        = %d x %d x %d x %d\n",p->c_out, p->c_in, p->k_size, p->k_size);
    gtx_printf("out(h x w) = %d x %d\n", p->h_out, p->w_out);
    gtx_printf("dilation   = %d\n", p->dilation);
    gtx_printf("stride     = %d\n", p->stride);
    gtx_printf("pad (h, w) = %d, %d\n", p->pad_h, p->pad_w);
    gtx_printf("bias       = %s\n", p->bias? "true" : "false");
}


void print_l2Info(L2Info* l2i) {
    gtx_printf("\n============= l2 Info ==============\n");
    gtx_printf("input_addr:  %8x\n", l2i->input_addr); 
    gtx_printf("kernel_addr: %8x\n", l2i->kernel_addr);
    gtx_printf("bias_addr:   %8x\n", l2i->bias_addr);
    gtx_printf("bias_ones:   %8x\n", l2i->bias_ones_addr);
    gtx_printf("output_addr: %8x\n", l2i->output_addr); 
    gtx_printf("total_bytes: %8x\n", l2i->total_bytes); 
}

void print_sputilemap(SPUTileMap* stm){
    gtx_printf("\n==== SPUTileMap ====\n");
    gtx_printf("div: OC x H x W   = %d x %d x %d\n", stm->cout_div, stm->h_div, stm->w_div);
    gtx_printf("len: OC x H x W   = %d x %d x %d\n", stm->cout_len, stm->hout_len, stm->wout_len);
    gtx_printf("rem_cnt: OC, H, W = %d, %d, %d\n", stm->cout_rem_cnt, stm->h_rem_cnt, stm->w_rem_cnt);
    gtx_printf("use_spu           = %d\n", stm->use_spu);
    gtx_printf("====================\n");
}


void print_tileinfo(TileExecPlan* t) {
    gtx_printf("\nTile Info\n");
    gtx_printf("==== tile Number of each dim ====\n");
    gtx_printf("cin | OC x H x W : %3d | %3d x %3d x %3d  \n", t->cin_div, t->cout_div, t->h_div, t->w_div);
    gtx_printf("\n==== tile length of each dim ====\n");
    gtx_printf("cin | OC x H x W : %3d | %3d x %3d x %3d \n\n", t->cin_len, t->cout_len, t->hout_len, t->wout_len);
}

void derive_convParam(ConvParams* p) {
    p->win_pad   = p->w_in + p->pad_w*2;
    p->hin_pad   = p->h_in + p->pad_h*2;
    p->k_n        = p->k_size * p->k_size;
    p->k_eff      = (p->k_size-1) * p->dilation + 1;
    p->h_out = (p->hin_pad - p->k_eff) / p->stride +1;
    p->w_out = (p->win_pad - p->k_eff) / p->stride +1;
}

bool calculate_l2_usage_bias(ConvParams* p, L2Info * l2i, uint32_t l2_offset, uint32_t l2_cap) {
    uint64_t input_size = p->c_in * p->hin_pad * p->win_pad *FP16_B;
             input_size = align_up(input_size, L2_ALIGNMENT);
    uint64_t kernel_size = p->c_out * p->k_n*p->c_in*FP16_B;
    uint64_t output_size = p->c_out * p->h_out * p->w_out * FP32_B;
    
    uint64_t bias_size  = align_up(p->c_out*FP16_B, L1_ALIGNMENT);
    uint64_t bias_ones_size = L1_64K;
    kernel_size = align_up(kernel_size + bias_size, L2_ALIGNMENT);
    output_size = align_up(output_size + bias_ones_size, L2_ALIGNMENT);
    
    uint64_t total_l2_usage = input_size + kernel_size + output_size;

    if(total_l2_usage > l2_cap) return false;

    l2i->input_addr  = l2_offset;
    l2i->kernel_addr = l2_offset + input_size;
    l2i->output_addr = l2_offset + input_size + kernel_size;
    l2i->total_bytes = total_l2_usage;
    l2i->bias_addr = l2i->output_addr - bias_size;
    l2i->bias_ones_addr = l2i->total_bytes - bias_ones_size;
    
    return true;
}


void set_single_conv_l2(
    int nest_id, ConvParams* p,
    uint32_t l2_offset, uint32_t l2_cap, L2Info* l2i,
    uint64_t ddrinput, uint64_t ddr_kernel, uint64_t ddr_bias
) {
    uint32_t input_row_bytes = p->w_in * FP16_B;
    uint32_t input_cin_bytes = p->h_in * input_row_bytes;
    uint32_t knl_read_length = p->c_in*p->k_n*FP16_B;
    gtx_printf("\nset_single_conv_l2 start plan\n");
    __start_plan(nest_id);
        __start_shared();
            // input load with padding
            __pad(ddrinput, l2i->input_addr, p->w_in*FP16_B, p->w_in*FP16_B, p->h_in,
                    p->pad_h, p->pad_h, p->pad_w, p->pad_w, 0x0, input_cin_bytes, p->c_in, 0b10, 1);
                    
            // kernel load
            __load(ddr_kernel, l2i->kernel_addr, knl_read_length, knl_read_length, p->c_out, knl_read_length);
                
            // bias load and set ones(or zeros)
            uint64_t bias_fill_pattern = (p->bias>0)? 0x3C003C003C003C00 : 0x0000;
            uint32_t bias_length = p->c_out * FP16_B;
            __load(ddr_bias, l2i->bias_addr, bias_length, bias_length, 1, bias_length);
            __fill(l2i->bias_ones_addr, 0x1000, 0x0fff, 0x10, bias_fill_pattern, 0);

            // __credit_ld(0xFFFF, 0);

        __end_shared();
    __end_plan(nest_id);
    gtx_printf("set_single_conv_l2 end plan\n\n");
}


void store_conv_l2(
    int nest_id, ConvParams* p,
    uint32_t l2_out_addr,
    uint64_t ddr_output){

    uint32_t length = p->w_out * FP16_B;
    uint32_t height = p->c_out * p->h_out;

    __start_plan(nest_id);
        __start_shared();
            __store(l2_out_addr, ddr_output, length, length, height, length);
        __end_shared();
    __end_plan(nest_id);
}


//===================nmconv core=====================


void tile_nmconv_general(ConvParams* p, SPUTileMap* stm, TileExecPlan* t, int spu_num) {

    const int HALO_RATIO = 3;
    // 1. spu workload division ==============================
    const int min_out_tile_len = MAX((p->k_size - p->stride) * HALO_RATIO, MIN_OUT_T_LEN);
    const int cout_target = L1_64K / (min_out_tile_len * min_out_tile_len * min_out_tile_len);

    int block_num = 1;
    int h_div    = 1;
    int w_div    = 1;
    int cout_div = 1;
    int hout_len = p->h_out;
    int cout_len = p->c_out;
    int wout_len = p->w_out;

    while(block_num < spu_num) {
        int ohow = hout_len * wout_len;
        int cin_kn = p->c_in * p->k_n;
        int out_bytes = cout_len * ohow * FP32_B;
        int knl_bytes = cout_len * cin_kn * FP16_B;
        int im2col_bytes = cin_kn * ohow * FP16_B;
        // if(out_bytes <= L1_64K && im2col_bytes <= L1_64K && knl_bytes <= L1_64K) break;
        
        if(out_bytes > L1_64K || knl_bytes * 2 > im2col_bytes) {
            cout_div *= 2;
            cout_len = ceil_div(p->c_out, cout_div);
        } else if(hout_len > min_out_tile_len) {
            h_div *= 2;
            hout_len = ceil_div(p->h_out, h_div);
        } else if(wout_len > min_out_tile_len) {
            w_div *= 2;
            wout_len = ceil_div(p->w_out, w_div);
        } else break;
        block_num *=2;
    }

    int cout_rem_cnt = ceil_rem(p->c_out, cout_div);
    int h_rem_cnt = ceil_rem(p->h_out, h_div);
    int w_rem_cnt = ceil_rem(p->w_out, w_div);

    stm->cout_div = cout_div;
    stm->h_div = h_div;
    stm->w_div = w_div;
    stm->cout_len = cout_len;
    stm->hout_len = hout_len;
    stm->wout_len = wout_len;
    stm->cout_rem_cnt = cout_rem_cnt;
    stm->h_rem_cnt = h_rem_cnt;
    stm->w_rem_cnt = w_rem_cnt;
    stm->use_spu = block_num;

    int spu_idx = 0;
    int curr_c = 0;
    for (int cd = 0; cd < stm->cout_div; cd++) {
        int curr_h = 0;

        int assigned_c = (cd < stm->cout_rem_cnt) ? stm->cout_len : stm->cout_len - 1;
        
        for (int hd = 0; hd < stm->h_div; hd++) {
            int curr_w = 0;
            int assigned_h = (hd < stm->h_rem_cnt) ? stm->hout_len : stm->hout_len - 1;
            
            for (int wd = 0; wd < stm->w_div; wd++) {
                int assigned_w = (wd < stm->w_rem_cnt) ? stm->wout_len : stm->wout_len - 1;

                if (spu_idx < spu_num) {
                    stm->start_cout[spu_idx] = curr_c;
                    stm->start_hout[spu_idx] = curr_h;
                    stm->start_wout[spu_idx] = curr_w;
                    
                    stm->cout_assigned[spu_idx] = assigned_c;
                    stm->hout_assigned[spu_idx] = assigned_h;
                    stm->wout_assigned[spu_idx] = assigned_w;
                    spu_idx++;
                }
                curr_w += assigned_w;
            }
            curr_h += assigned_h;
        }
        curr_c += assigned_c;
    }


    // 2. Define the tile size for a single SPU workload.
    int t_num_cin  = 1,  t_len_cin  = p->c_in;
    int t_num_cout = 1,  t_len_cout = stm->cout_len;
    int t_num_h    = 1,  t_len_hout = stm->hout_len;
    int t_num_w    = 1,  t_len_wout = stm->wout_len;

    if(t_len_cout > cout_target) {
        t_num_cout = ceil_div(t_len_cout, cout_target);
        t_len_cout = ceil_div(t_len_cout, t_num_cout);
    }

    int knl_bytes = p->k_n * FP16_B; // im2col bytes per pixel
    int mmc_bytes_per_pixel = t_len_cout * FP32_B;
    int max_img_pixel = L1_128K / MAX(mmc_bytes_per_pixel, knl_bytes);
    max_img_pixel = MIN(max_img_pixel, L1_64K / FP16_B / (p->stride*p->stride));
    /* The GTX im2col unit does not support a 1x1 kernel.  That path packs A
     * with ordinary loads below, one command per output position; keep its
     * per-plan command count bounded during bring-up. */
    if (p->k_size == 1) {
        max_img_pixel = MIN(max_img_pixel, 64);
    }

    // set tile img length
    if(max_img_pixel < t_len_hout * t_len_wout) {
        int ideal_img_len = (int)isqrt((uint64_t) max_img_pixel);
        t_num_h = ceil_div(t_len_hout, ideal_img_len);
        t_num_w = ceil_div(t_len_wout, ideal_img_len);
        t_len_hout = ceil_div(stm->hout_len, t_num_h);
        t_len_wout = ceil_div(stm->wout_len, t_num_w);

        while (t_len_hout * t_len_wout > max_img_pixel) {
            if (t_len_hout >= t_len_wout && t_len_hout > 1) t_num_h++;
            else t_num_w++;

            t_len_hout = ceil_div(stm->hout_len, t_num_h);
            t_len_wout = ceil_div(stm->wout_len, t_num_w);
        }
    }

    // set cin length
    const float in_im2col_ratio = (float)p->k_n / (float)(p->stride * p->stride);
    if(in_im2col_ratio >= 2.0f) {
        int im2col_bytes_per_cin = t_len_hout * t_len_wout * knl_bytes;
        t_len_cin = MIN(L1_128K / im2col_bytes_per_cin, p->c_in);
        t_num_cin = ceil_div(p->c_in, t_len_cin);
    } else {
        int input_bytes_per_cin = t_len_hout * t_len_wout * FP16_B;
        t_len_cin = MIN(L1_64K / input_bytes_per_cin, p->c_in);
        t_num_cin = ceil_div(p->c_in, t_len_cin);
    }

    // set cout length (fix)
    int new_t_len_cout = L1_64K / (t_len_cin * knl_bytes);
    if(new_t_len_cout < t_len_cout) {
        t_len_cout = new_t_len_cout;
        t_num_cout = ceil_div(stm->cout_len, t_len_cout);
    }

    t->cin_div    = t_num_cin;
    t->cout_div   = t_num_cout;
    t->w_div      = t_num_w;
    t->h_div      = t_num_h;

    t->cin_len    = t_len_cin;
    t->cout_len   = t_len_cout;
    t->wout_len   = t_len_wout;
    t->hout_len   = t_len_hout;

    t->work_per_spu = t_num_cout * t_num_h * t_num_w;
}


void conv_tile_dim4_with_bias(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i) {
    gtx_printf("\n\n============ conv accumulation with fp32 bias (Refactored) ==============\n");

    const int use_spu = stm->use_spu;

    const uint32_t input_row_bytes   = p->win_pad * FP16_B;
    const uint32_t input_cin_bytes   = p->hin_pad * input_row_bytes;
    const uint32_t output_row_bytes  = p->w_out * FP16_B;
    const uint32_t output_cout_bytes = p->h_out * output_row_bytes;
    
    const uint32_t knl_img_bytes     = p->k_n * FP16_B;
    const uint32_t knl_cout_bytes    = p->c_in * knl_img_bytes;
    const int len_cin_rem            = ceil_rem(p->c_in, t->cin_len);

    for(int ocid = 0; ocid < t->cout_div; ocid++) {
        const bool is_c_rem = (ocid == t->cout_div - 1);
        const int oc_idx_offset = ocid * t->cout_len;

        for(int hid = 0; hid < t->h_div; hid++) {
            const bool is_h_rem = (hid == t->h_div - 1);
            const int hout_idx_offset = hid * t->hout_len;

            for(int wid = 0; wid < t->w_div; wid++) {
                const bool is_w_rem = (wid == t->w_div - 1);
                const int wout_idx_offset = wid * t->wout_len;

                __split();
                __start_plan(nest_id);
                for(int spu_id = 0; spu_id < use_spu; spu_id++) {
                    
                    const int cur_cout = is_c_rem ? ceil_rem(stm->cout_assigned[spu_id], t->cout_len) : t->cout_len;
                    const int cur_hout = is_h_rem ? ceil_rem(stm->hout_assigned[spu_id], t->hout_len) : t->hout_len;
                    const int cur_wout = is_w_rem ? ceil_rem(stm->wout_assigned[spu_id], t->wout_len) : t->wout_len;
                    
                    const int cur_hin  = FIND_INPUT_LENGTH(cur_hout, p->k_eff, p->stride);
                    const int cur_win  = FIND_INPUT_LENGTH(cur_wout, p->k_eff, p->stride);

                    const int cout_idx = stm->start_cout[spu_id] + oc_idx_offset;
                    const int hout_idx = stm->start_hout[spu_id] + hout_idx_offset;
                    const int wout_idx = stm->start_wout[spu_id] + wout_idx_offset;

                    const uint32_t bias_addr   = l2i->bias_addr + (cout_idx * FP16_B);
                    const uint32_t knl_base    = l2i->kernel_addr + (cout_idx * knl_cout_bytes);
                    const uint32_t out_addr    = l2i->output_addr + (cout_idx * output_cout_bytes) + (hout_idx * output_row_bytes) + (wout_idx * FP16_B);
                    const uint32_t in_tile_off = (hout_idx * p->stride * input_row_bytes) + (wout_idx * p->stride * FP16_B);

                    __start_thread(spu_id);
                        __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);

                        const int bias_len = cur_cout * FP16_B;
                        const int ones_len = cur_wout * FP16_B;
                        
                        __load(l2i->bias_ones_addr, BASE_L1_BANK_A, ones_len, ones_len, cur_hout, ones_len);
                        __load(bias_addr, BASE_L1_BANK_B, bias_len, bias_len, 1, bias_len);
                        __mm_s(cur_hout * cur_wout, 1, cur_cout);
   
                        for(int icid = 0; icid < t->cin_div; icid++) {
                            const bool is_cin_rem = (icid == t->cin_div - 1);
                            const int cur_cin     = is_cin_rem ? len_cin_rem : t->cin_len;
                            const int ic_idx      = icid * t->cin_len;

                            const uint32_t in_addr  = l2i->input_addr + (ic_idx * input_cin_bytes) + in_tile_off;
                            const uint32_t knl_ptr  = knl_base + (ic_idx * knl_img_bytes);
                            const uint32_t knl_len  = cur_cin * knl_img_bytes;

                            if (p->k_size == 1) {
                                /* Direct 1x1 lowering.  L2 input is planar
                                 * [cin][hin][win], while MM consumes A as
                                 * [hout*wout][cin].  Gather one FP16 value
                                 * from every channel for each output point. */
                                const uint16_t cin_bytes =
                                    (uint16_t)((uint32_t)cur_cin * FP16_B);
                                for (int out_y = 0; out_y < cur_hout; ++out_y) {
                                    for (int out_x = 0; out_x < cur_wout; ++out_x) {
                                        const uint32_t out_pixel =
                                            (uint32_t)out_y * (uint32_t)cur_wout +
                                            (uint32_t)out_x;
                                        const uint32_t input_pixel_off =
                                            (uint32_t)out_y * (uint32_t)p->stride * input_row_bytes +
                                            (uint32_t)out_x * (uint32_t)p->stride * FP16_B;
                                        __load(in_addr + input_pixel_off,
                                               BASE_L1_BANK_A + out_pixel * cin_bytes,
                                               input_cin_bytes,
                                               FP16_B,
                                               (uint16_t)cur_cin,
                                               FP16_B);
                                    }
                                }
                            } else {
                                /* Padding is materialized in L2; pack the
                                 * channel planes in L1 before im2col. */
                                const uint32_t l1_input_channel_bytes =
                                    (uint32_t)cur_hin * (uint32_t)cur_win * FP16_B;
                                for (int cin_idx = 0; cin_idx < cur_cin; ++cin_idx) {
                                    __load(in_addr + (uint32_t)cin_idx * input_cin_bytes,
                                           BASE_L1_BANK_R + (uint32_t)cin_idx * l1_input_channel_bytes,
                                           input_row_bytes,
                                           (uint16_t)((uint32_t)cur_win * FP16_B),
                                           (uint16_t)cur_hin,
                                           (uint32_t)cur_win * FP16_B);
                                }
                                __im2col_n(cur_hin, cur_win, p->k_size,
                                           p->dilation, p->stride, cur_cin);
                            }
                            __load(knl_ptr, BASE_L1_BANK_B, knl_cout_bytes, knl_len, cur_cout, knl_len);

                            const int M = cur_hout * cur_wout;
                            const int K = cur_cin * p->k_n;

                            if(!is_cin_rem) {
                                __mmc_s(M, K, cur_cout);
                            } else {
                                __mmc_t(M, K, cur_cout);
                                /* MMC_T leaves a dense tile in L1 as
                                 * [cout][cur_hout][cur_wout].  L2, however,
                                 * keeps the full [cout][h_out][w_out] planes.
                                 * Store each channel separately so both the
                                 * compact L1 row/channel strides and the full
                                 * L2 row/channel strides are respected. */
                                const uint32_t l1_channel_bytes =
                                    (uint32_t)cur_hout * (uint32_t)cur_wout * FP16_B;
                                for (uint32_t stored_channel = 0;
                                     stored_channel < (uint32_t)cur_cout;
                                     ++stored_channel) {
                                    __store(BASE_L1_BANK_R +
                                                stored_channel * l1_channel_bytes,
                                            out_addr +
                                                stored_channel * output_cout_bytes,
                                            output_row_bytes,
                                            (uint16_t)(cur_wout * FP16_B),
                                            (uint16_t)cur_hout,
                                            (uint32_t)cur_wout * FP16_B);
                                }
                            }
                        }
                    __end_thread(spu_id);
                }
                __end_plan(nest_id);
                __join();
            }
        }
    }
}
