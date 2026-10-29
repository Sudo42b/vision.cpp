#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "gtx/kernel_params.h"
#include "gtx/intrin.h"
#include "gtx/print.h"


#define FP16_B          2
#define FP32_B          4

#define L1_CAP          0x60000
#define L1_64K          0x10000
#define L1_128K         0x20000
#define L1_ALIGNMENT    0x100
#define L2_ALIGNMENT    0x1000
#define L2_CAP          0x1000000

#define ALIGN_B         0x20

#define BIAS_ONES_LENGTH 0x100

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define FIND_INPUT_LENGTH(l_out, k_eff, stride) ((l_out - 1) * stride + k_eff)

static const int MIN_OUT_T_LEN = 8;

static inline int ceil_div(int a, int b){ return (a + b - 1u) / b; }
static inline int ceil_rem(int a, int b){ return (a + b - 1u) % b + 1; }
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
        if (mid == 0) {
             low = mid + 1;
             continue;
        }
        else if (mid <= n / mid) {
            low = mid + 1;
            res = mid;
        } else {
            high = mid - 1;
        }
    }
    return res;
}

//=========================//common//===============================

typedef struct {
    int cin, hin, win;
    int cout;
    int k_size;
    int pad_h, pad_w;
    int stride, dilation;
    bool bias;

    int hin_pad, win_pad;
    int k_eff, k_n;
    int hout, wout;
    int win_b;
    int cin_b;
    int win_pad_b;
    int cin_pad_b;
    int wout_b;
    int cout_b;
    int knl_ch_b;

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
    uint64_t input_addr;
    uint64_t kernel_addr;
    uint64_t bias_addr;
    uint64_t output_addr;
} DDRInfo;

typedef struct {
    // ==== per spu ====
    int cout_div, cout_rem_cnt, cout_len;
    int h_div, h_rem_cnt, hout_len;
    int w_div, w_rem_cnt, wout_len;
    int use_spu;

}SPUTileMap;

typedef struct {
    int cin_div;
    int cout_div;
    int w_div, h_div;

    int cin_len, cout_len;
    int wout_len, hout_len;

    int work_per_spu;
} TileExecPlan;


static void derive_convParam(ConvParams* p) {
    p->win_pad   = p->win + p->pad_w*FP16_B;
    p->hin_pad   = p->hin + p->pad_h*FP16_B;
    p->k_n        = p->k_size * p->k_size;
    p->k_eff      = (p->k_size-1) * p->dilation + 1;
    p->hout = (p->hin_pad - p->k_eff) / p->stride +1;
    p->wout = (p->win_pad - p->k_eff) / p->stride +1;

    p->win_b = p->win * FP16_B;
    p->cin_b = p->win_b * p->hin;
    p->win_pad_b = p->win_pad * FP16_B;
    p->cin_pad_b = p->hin_pad * p->win_pad_b;
    p->wout_b = p->wout * FP16_B;
    p->cout_b = p->hout * p->wout_b;
    p->knl_ch_b = p->k_n * FP16_B;
}

static void store_conv_l2(
    int nest_id, ConvParams* p,
    uint32_t l2_out_addr,
    uint64_t ddr_output){

    uint32_t length = p->wout * FP16_B;
    uint32_t height = p->cout * p->hout;

    __start_plan(nest_id);
        __start_shared();
            __store(l2_out_addr, ddr_output, length, length, height, length);
        __end_shared();
    __end_plan(nest_id);
}


//=====================dwconv only==============================

static bool calculate_l2_usage_bias32_dwconv(ConvParams* p, L2Info * l2i, uint32_t l2_offset, uint32_t l2_cap) {
    uint64_t input_size = p->cin * align_up(p->cin_pad_b, ALIGN_B);
             input_size = align_up(input_size, L2_ALIGNMENT);
    uint64_t kernel_size = p->cout * align_up(p->knl_ch_b, ALIGN_B);
             kernel_size = align_up(kernel_size, L2_ALIGNMENT);
    uint64_t output_size = align_up(p->cout_b, L2_ALIGNMENT);

    uint64_t bias_size  = align_up(p->cout*FP16_B, L1_ALIGNMENT);
    uint64_t bias_ones_size = BIAS_ONES_LENGTH;
    
    uint64_t total_l2_usage = input_size + kernel_size + bias_size + bias_ones_size + output_size;

    if(total_l2_usage > l2_cap) return false;

    l2i->input_addr  = l2_offset;
    l2i->kernel_addr = l2_offset + input_size;
    l2i->bias_addr = l2i->kernel_addr + kernel_size;
    l2i->output_addr = l2i->bias_addr + bias_size;
    l2i->bias_ones_addr = l2_cap - bias_ones_size;
    l2i->total_bytes = total_l2_usage;
    
    return true;
}



static void tile_dwconv_workload(ConvParams* p, SPUTileMap* stm, TileExecPlan* t, int spu_num) {

    int c_div    = MIN(p->cin, spu_num);
    int len_c    = ceil_div(p->cin, c_div);
    int h_div    = 1;
    int w_div    = 1;
    int len_hout = p->hout;
    int len_wout = p->wout;
    int block_num = c_div;

    const int wh_ratio = 8;

    while(block_num*2 <= spu_num) {
        // len_c == 1
        int im2col_bytes = len_hout * len_wout * p->knl_ch_b;
        if(im2col_bytes <= L1_64K) break;

        if(len_hout * wh_ratio > len_wout) {
            h_div *=2;
            len_hout = ceil_div(p->hout, h_div);
        }else {
            w_div *=2;
            len_wout = ceil_div(p->wout, w_div);
        }
        block_num *=2;
    }

    int cout_rem_cnt = ceil_rem(p->cout, c_div);
    int h_rem_cnt = ceil_rem(p->hout, h_div);
    int w_rem_cnt = ceil_rem(p->wout, w_div);

    stm->cout_div = c_div;
    stm->h_div = h_div;
    stm->w_div = w_div;
    stm->cout_len = len_c;
    stm->hout_len = len_hout;
    stm->wout_len = len_wout;
    stm->cout_rem_cnt = cout_rem_cnt;
    stm->h_rem_cnt = h_rem_cnt;
    stm->w_rem_cnt = w_rem_cnt;
    stm->use_spu = block_num;

    int t_num_cout = 1,  t_len_cout = stm->cout_len;
    int t_num_h    = 1,  t_len_hout = stm->hout_len;
    int t_num_w    = 1,  t_len_wout = stm->wout_len;


    /////////////////////////////
    int ohow = t_len_hout * t_len_wout;
    int im2col_bytes = ohow * p->knl_ch_b;
    int input_bytes = p->hin_pad * p->win_pad * FP16_B;
    int output_bytes = ohow * FP32_B;


    //k_n < ohow assure
    int spu_max_c = L1_128K / MAX(im2col_bytes, output_bytes);
    spu_max_c = MIN(spu_max_c, L1_64K / input_bytes);

    if(spu_max_c < 1) {
        t_len_cout = 1;
        t_num_cout = ceil_div(stm->cout_len, t_len_cout);
        int max_img_pixel = L1_128K / MAX(FP32_B, p->knl_ch_b);
        max_img_pixel = MIN(max_img_pixel, L1_64K / FP16_B / (p->stride * p->stride));
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
    else if(spu_max_c < t_len_cout) {
        t_len_cout = spu_max_c;
        t_num_cout = ceil_div(stm->cout_len, spu_max_c);
    }

    t->cin_div    = t_num_cout;
    t->cout_div   = t_num_cout;
    t->w_div      = t_num_w;
    t->h_div      = t_num_h;

    t->cin_len    = t_len_cout;
    t->cout_len   = t_len_cout;
    t->wout_len   = t_len_wout;
    t->hout_len   = t_len_hout;

    t->work_per_spu = t_num_cout * t_num_h * t_num_w;

}

static void dwconv_2d_tile(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i, DDRInfo* ddri)
{
    int use_spu = stm->use_spu;

    // 1. Pre-calculate SPU-specific base offsets
    // SPU별로 할당된 시작 채널 인덱스를 미리 계산하여 루프 내부 계산 최소화
    int start_cidx[use_spu];
    int cur_iter = 0;
    for (int spu_id = 0; spu_id < use_spu; spu_id++) {
        start_cidx[spu_id] = cur_iter;
        cur_iter += (spu_id < stm->cout_rem_cnt) ? stm->cout_len : stm->cout_len - 1;
    }

    {
        const uint64_t bias_fill_pattern = (p->bias>0)? 0x3C003C003C003C00 : 0x0000;

        uint32_t ddr_cin_offset = 0;
        uint32_t l2_pad_offset = p->pad_h * p->win_pad_b + p->pad_w * FP16_B;
        
        __start_plan(nest_id);
        __start_shared();
            __fill(l2i->input_addr, p->win_pad_b, p->win_pad_b, p->hin_pad * p->cin, 0x00, 0b0);
            for(int cid=0; cid<p->cin; cid++) {
                __load(ddri->input_addr + ddr_cin_offset, l2i->input_addr + l2_pad_offset, 
                        p->win*FP16_B, p->win*FP16_B, p->hin, p->win_pad_b);
                
                    ddr_cin_offset += p->hin * p->win * FP16_B;
                    l2_pad_offset += p->cin_pad_b;
            }
            __load(ddri->kernel_addr, l2i->kernel_addr, p->knl_ch_b, p->knl_ch_b, p->cin, p->knl_ch_b);
            __load(ddri->bias_addr, l2i->bias_addr, p->cin*FP16_B, p->cin*FP16_B, 1, p->cin*FP16_B);
            __fill(l2i->bias_ones_addr, BIAS_ONES_LENGTH, BIAS_ONES_LENGTH, 1, bias_fill_pattern, 0b0);
        __end_shared();
        __end_plan(nest_id);
    }


    /* --- 2D Tiled Workload Case --- */
    for (int cidx = 0; cidx < t->cout_len; cidx++) {
        int last_spu = (cidx < stm->cout_len-1)? use_spu : stm->cout_rem_cnt;
        
        for (int hid = 0; hid < t->h_div; hid++) {
            // H 관련 오프셋 계산 (SPU 공통)
            int hout_idx = hid * t->hout_len;
            int cur_hout = (hid == t->h_div - 1) ? ceil_rem(stm->hout_len, t->hout_len) : t->hout_len;
            int cur_hin  = FIND_INPUT_LENGTH(cur_hout, p->k_eff, p->stride);
            
            uint32_t h_off_out = hout_idx * p->wout_b;
            uint32_t h_off_in  = hout_idx * p->stride * p->win_pad_b;

            for (int wid = 0; wid < t->w_div; wid++) {
                // W 관련 오프셋 계산 (SPU 공통)
                int wout_idx = wid * t->wout_len;
                int cur_wout = (wid == t->w_div - 1) ? ceil_rem(stm->wout_len, t->wout_len) : t->wout_len;
                int cur_win  = FIND_INPUT_LENGTH(cur_wout, p->k_eff, p->stride);

                uint32_t w_off_out = wout_idx * FP16_B;
                uint32_t w_off_in  = wout_idx * p->stride * FP16_B;
                
                int ohow = cur_hout * cur_wout;
                uint32_t ones_height = ceil_div(ohow * FP16_B, BIAS_ONES_LENGTH);
                __split();
                __start_plan(nest_id);
                for (int spu_id = 0; spu_id < last_spu; spu_id++) {
                    // SPU별 주소를 하드웨어 명령 직전에 계산 (컴파일러가 레지스터에 바로 올리도록 유도)
                    int abs_c = start_cidx[spu_id] + cidx;
                    uint32_t in_addr  = l2i->input_addr  + (abs_c * p->cin_pad_b)  + h_off_in  + w_off_in;
                    uint32_t out_addr = l2i->output_addr + (abs_c * p->cout_b) + h_off_out + w_off_out;
                    uint32_t knl_addr = l2i->kernel_addr + (abs_c * p->knl_ch_b);
                    uint32_t bias_ptr = l2i->bias_addr   + (abs_c * FP16_B);

                    __start_thread(spu_id);
                        __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
                        
                        // Bias
                        __load(l2i->bias_ones_addr, BASE_L1_BANK_A, 0, BIAS_ONES_LENGTH, ones_height, BIAS_ONES_LENGTH);
                        __load(bias_ptr, BASE_L1_BANK_B, FP16_B, FP16_B, 1, FP16_B);
                        __mm_s(ohow, 1, 1);

                        // Conv
                        __load(in_addr, BASE_L1_BANK_R, p->win_pad_b, cur_win*FP16_B, cur_hin, cur_win*FP16_B);
                        __im2col_n(cur_hin, cur_win, p->k_size, p->dilation, p->stride, 1);
                        __load(knl_addr, BASE_L1_BANK_B, p->knl_ch_b, p->knl_ch_b, 1, p->knl_ch_b);
                        __mmc_t(ohow, p->k_n, 1);
                        
                        __store(BASE_L1_BANK_R, out_addr, p->wout_b, cur_wout*FP16_B, cur_hout, cur_wout*FP16_B);
                    __end_thread(spu_id);
                }
                __end_plan(nest_id);
                __join();
            }
        }
    }
}

static void dwconv_2d_spu_tile(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i, DDRInfo* ddri) {
    const int use_spu = stm->use_spu;

    // 1. SPU별 시작 좌표 및 채널 정보 선계산
    int start_c[use_spu];
    int hout_assigned[use_spu];
    int wout_assigned[use_spu];
    uint32_t in_base_spu[use_spu];
    uint32_t out_base_spu[use_spu];

    // 공통 바이트 상수
    {
        const uint64_t bias_fill_pattern = (p->bias>0)? 0x3C003C003C003C00 : 0x0000;

        uint32_t ddr_cin_offset = 0;
        uint32_t l2_pad_offset = p->pad_h * p->win_pad_b + p->pad_w * FP16_B;
        
        __start_plan(nest_id);
        __start_shared();
        __fill(l2i->input_addr, p->win_pad_b, p->win_pad_b, p->hin_pad * p->cin, 0x00, 0b0);
        for(int cid=0; cid<p->cin; cid++) {
            __load(ddri->input_addr + ddr_cin_offset, l2i->input_addr + l2_pad_offset, 
                    p->win*FP16_B, p->win*FP16_B, p->hin, p->win_pad_b);
            
                ddr_cin_offset += p->hin * p->win * FP16_B;
                l2_pad_offset += p->cin_pad_b;
        }
        __load(ddri->kernel_addr, l2i->kernel_addr, p->knl_ch_b, p->knl_ch_b, p->cin, p->knl_ch_b);
        __load(ddri->bias_addr, l2i->bias_addr, p->cin*FP16_B, p->cin*FP16_B, 1, p->cin*FP16_B);
        __fill(l2i->bias_ones_addr, BIAS_ONES_LENGTH, BIAS_ONES_LENGTH, 1, bias_fill_pattern, 0b0);
        __end_shared();
        __end_plan(nest_id);
    }

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

                    // L2 내 SPU별 시작 주소 확정
                    in_base_spu[spu_id] = l2i->input_addr + (cid * p->cin_pad_b) + 
                                     (start_hout * p->stride * p->win_pad_b) + (start_wout * p->stride * FP16_B);
                    out_base_spu[spu_id] = l2i->output_addr + (cid * p->cout_b) + 
                                      (start_hout * p->wout_b) + (start_wout * FP16_B);

                    start_wout += cur_w;
                    spu_id++;
                }
                start_hout += cur_h;
            }
        }
    }

    /* --- 2D Tiled Workload Logic --- */
    for (int hid = 0; hid < t->h_div; hid++) {
        const bool is_h_rem = (hid == t->h_div - 1);
        const int hout_idx = hid * t->hout_len;
        const uint32_t hout_off = hout_idx * p->wout_b;
        const uint32_t hin_off = hout_idx * p->stride * p->win_pad_b;

        for (int wid = 0; wid < t->w_div; wid++) {
            const bool is_w_rem = (wid == t->w_div - 1);
            const int wout_idx = wid * t->wout_len;
            const uint32_t wout_off = wout_idx * FP16_B;
            const uint32_t win_off = wout_idx * p->stride * FP16_B;
            __split();
            __start_plan(nest_id);
            for (int spu_id = 0; spu_id < use_spu; spu_id++) {
                // 현재 타일의 가변 길이 계산
                int cur_hout = is_h_rem ? ceil_rem(hout_assigned[spu_id], t->hout_len) : t->hout_len;
                int cur_wout = is_w_rem ? ceil_rem(wout_assigned[spu_id], t->wout_len) : t->wout_len;
                int cur_hin  = FIND_INPUT_LENGTH(cur_hout, p->k_eff, p->stride);
                int cur_win  = FIND_INPUT_LENGTH(cur_wout, p->k_eff, p->stride);
                int ohow     = cur_hout * cur_wout;

                // SPU별 L2 주소 계산
                uint32_t in_addr   = in_base_spu[spu_id] + hin_off + win_off;
                uint32_t out_addr  = out_base_spu[spu_id] + hout_off + wout_off;
                uint32_t knl_addr  = l2i->kernel_addr + (start_c[spu_id] * p->knl_ch_b);
                uint32_t bias_ptr  = l2i->bias_addr + (start_c[spu_id] * FP16_B);
                uint32_t ones_height = ceil_div(ohow * FP16_B, BIAS_ONES_LENGTH);

                __start_thread(spu_id);
                    __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);

                    // 1. Bias 처리 (BANK_A: Ones, BANK_B: Bias Value)
                    __load(l2i->bias_ones_addr, BASE_L1_BANK_A, 0, BIAS_ONES_LENGTH, ones_height, BIAS_ONES_LENGTH);
                    __load(bias_ptr, BASE_L1_BANK_B, FP16_B, FP16_B, 1, FP16_B);
                    __mm_s(ohow, 1, 1);

                    // 2. Input Load & Im2col (BANK_R: Input, BANK_A: Im2col Result)
                    __load(in_addr, BASE_L1_BANK_R, p->win_pad_b, cur_win * FP16_B, cur_hin, cur_win * FP16_B);
                    
                    // [중요] im2col 결과 주소를 32-byte aligned 된 BANK_A로 명시적 설정
                    __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
                    __im2col_n(cur_hin, cur_win, p->k_size, p->dilation, p->stride, 1);

                    // 3. Kernel Load (BANK_B)
                    __load(knl_addr, BASE_L1_BANK_B, p->knl_ch_b, p->knl_ch_b, 1, p->knl_ch_b);

                    // 4. MXE Convolution 연산
                    __mmc_t(ohow, p->k_n, 1);

                    // 5. 결과 저장 (BANK_R -> L2)
                    __store(BASE_L1_BANK_R, out_addr, p->wout_b, cur_wout * FP16_B, cur_hout, cur_wout * FP16_B);
                __end_thread(spu_id);
            }
            __end_plan(nest_id);
            __join();
        }
    }
}


void print_convParams(ConvParams* p) {
    gtx_printf("\n[ ConvParams ]\n");
    gtx_printf(" - Input  (C x H x W)         : %d x %d x %d\n", p->cin, p->hin_pad, p->win_pad);
    gtx_printf(" - Kernel (OC x IC x KH x KW) : %d x %d x %d x %d\n", p->cout, p->cin, p->k_size, p->k_size);
    // gtx_printf(" - Effective (Kn, Keff)       : %d, %d\n", p->k_n, p->k_eff);
    gtx_printf(" - Stride / Dilation          : %d / %d\n", p->stride, p->dilation);
    gtx_printf(" - Output (H x W)             : %d x %d\n", p->hout, p->wout);
    gtx_printf(" - Bias / Activation          : %s \n", p->bias? "true":"false");
    gtx_printf("--------------------------------------\n");
}


void print_l2Info(L2Info* l2i, const char* name) {
    gtx_printf("\n[ L2 Info: %s ]\n", name);
    gtx_printf(" Input  : 0x%08x\n", l2i->input_addr);
    gtx_printf(" Kernel : 0x%08x\n", l2i->kernel_addr);
    gtx_printf(" Bias   : 0x%08x\n", l2i->bias_addr);
    gtx_printf(" Ones   : 0x%08x\n", l2i->bias_ones_addr);
    gtx_printf(" Output : 0x%08x\n", l2i->output_addr);
    gtx_printf(" Total  : %d Bytes\n", l2i->total_bytes);
    gtx_printf("--------------------------------------\n");
}


void print_sputilemap(SPUTileMap* stm) {
    gtx_printf("\n[ SPUTileMap ]\n");
    gtx_printf(" - SPU Div (C x H x W) : %d x %d x %d (Total SPU: %d)\n", 
              stm->cout_div, stm->h_div, stm->w_div, stm->use_spu);
    gtx_printf(" - Base Len (C, H, W)  : %d, %d, %d\n", stm->cout_len, stm->hout_len, stm->wout_len);
    gtx_printf(" - Rem Cnt  (C, H, W)  : %d, %d, %d\n", stm->cout_rem_cnt, stm->h_rem_cnt, stm->w_rem_cnt);
    gtx_printf("--------------------\n");
}

void print_tileinfo(TileExecPlan* t, const char* name) {
    gtx_printf("\n[ Tile Info: %s ]\n", name);
    gtx_printf(" - Tile Div (Cin, Cout, H, W) : %d, %d, %d, %d\n", 
              t->cin_div, t->cout_div, t->h_div, t->w_div);
    gtx_printf(" - Tile Len (Cin, Cout, H, W) : %d, %d, %d, %d\n", 
              t->cin_len, t->cout_len, t->hout_len, t->wout_len);
    gtx_printf(" - Work per SPU : %d\n", t->work_per_spu);
    gtx_printf("--------------------------------------\n");
}




// static void dwconv_ch_tile(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i, DDRInfo* ddri) {
//     const int use_spu = stm->use_spu;
    
//     int start_cidx[use_spu];
//     int cur_iter = 0;
//     for (int spu_id = 0; spu_id < use_spu; spu_id++) {
//         start_cidx[spu_id] = cur_iter;
//         cur_iter += (spu_id <= stm->cout_rem_cnt) ? stm->cout_len : stm->cout_len - 1;
//     }

//     {
//         const uint64_t bias_fill_pattern = (p->bias>0)? 0x3C003C003C003C00 : 0x0000;
//         uint32_t l2_cin_offset = 0;
//         uint32_t ddr_cin_offset = 0;
//         uint32_t l2_pad_offset = p->pad_h * p->win_pad_b + p->pad_w * FP16_B;

//         gtx_printf("\nload ddr to l2\n");
//         __start_plan(nest_id);
//         __start_shared();

//         for(int cid=0; cid<p->cin; cid++) {
//             __fill(l2i->input_addr + l2_cin_offset, p->win_pad_b, p->win_pad_b, p->hin_pad, 0x00, 0b0);
//             __load(ddri->input_addr + ddr_cin_offset, l2i->input_addr + l2_cin_offset + l2_pad_offset, 
//                     p->win*FP16_B, p->win*FP16_B, p->hin, p->win_pad_b);
            
//                 ddr_cin_offset += p->hin*p->win*FP16_B;
//                 l2_cin_offset += p->cin_pad_b;
//         }
//         __load(ddri->kernel_addr, l2i->kernel_addr, p->knl_ch_b, p->knl_ch_b, p->cin, p->knl_ch_b);
//         __load(ddri->bias_addr, l2i->bias_addr, p->cin*FP16_B, p->cin*FP16_B, 1, p->cin*FP16_B);
//         __fill(l2i->bias_addr, BIAS_ONES_LENGTH, BIAS_ONES_LENGTH, 1, bias_fill_pattern, 0b0);
//         __end_shared();
//         __end_plan(nest_id);
//     }

//     /* --- Channel Chunk Workload Case --- */
//     int ohow = t->hout_len * t->wout_len;
//     int ohow_aligned = align_up(ohow, ALIGN_B / FP32_B);
//     uint32_t ones_height = ceil_div(ohow_aligned * FP16_B, BIAS_ONES_LENGTH);

//     for (int cid = 0; cid < t->cout_div; cid++) {
//         const int is_c_rem = (cid == t->cout_div - 1);
//         const int cidx_base = cid * t->cout_len;
//         gtx_printf("ch tile id: %d\n", cid);
//         __start_plan(nest_id);
//         for (int spu_id = 0; spu_id < use_spu; spu_id++) {
//             // SPU별 처리량 및 주소 확정
//             int ch_assigned = (spu_id <= stm->cout_rem_cnt) ? stm->cout_len : stm->cout_len - 1;
//             int cur_c_len = is_c_rem ? ceil_rem(ch_assigned, t->cout_len) : t->cout_len;
//             int abs_c_start = start_cidx[spu_id] + cidx_base;
           

//             __start_thread(spu_id);
//                 __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
                
//                 // Batch Load (Input, Bias, Kernel)
//                 __load(l2i->bias_ones_addr, BASE_L1_BANK_B, 0, BIAS_ONES_LENGTH, ones_height, BIAS_ONES_LENGTH);
//                 __load(l2i->bias_addr + abs_c_start*FP16_B, BASE_L1_BANK_A, cur_c_len*FP16_B, cur_c_len*FP16_B, 1, cur_c_len*FP16_B);
//                 __mm_s(cur_c_len, 1, ohow_aligned);

//                 // 연산 집약 구간 (Inner Loop)
//                 for (int co = 0; co < cur_c_len; co++) {
//                     __load(l2i->input_addr + abs_c_start * p->cin_pad_b, BASE_L1_BANK_R, p->win_pad_b, p->win_pad_b, p->hin_pad, p->win_pad_b);
//                     __im2col_d(p->hin_pad, p->win_pad, p->k_size, p->dilation, p->stride, 1);
                    
//                     // __set_spm_addr(BASE_L1_BANK_R + co * p->cout_b,
//                     //                 BASE_L1_BANK_C + co * p->cout_b * 2,
//                     //                 BASE_L1_BANK_B + co * p->knl_ch_b,
//                     //                 BASE_L1_BANK_A + co * im2col_c_b_aligned);
                    
//                     __mmc_t(ohow, p->k_n, 1);
//                 }

//                 __store(BASE_L1_BANK_R, l2i->output_addr + abs_c_start * p->cout_b, p->wout_b, p->wout_b, p->hout * cur_c_len, p->wout_b);
//             __end_thread(spu_id);
//         }
//         __end_plan(nest_id);
//     }
// }



// //stm->cout_len >= 1
// static void dwconv_tile_ch(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i) {
//     const int use_spu = stm->use_spu;

//     int c_assigned[use_spu];
//     int start_cidx[use_spu];

//     {
//         int spu_id = 0;
//         int start_c_iter = 0;

//         for(int cid=0; cid<stm->cout_div; cid ++) {
//             c_assigned[spu_id] = cid <= stm->cout_rem_cnt? stm->cout_len : stm->cout_len -1;
//             start_cidx[spu_id] = start_c_iter;
            
//             start_c_iter += c_assigned[spu_id];
//             spu_id++;
//         }
//     }

//     uint32_t input_row_bytes = p->win_pad * FP16_B;
//     uint32_t input_cin_bytes = p->hin_pad * input_row_bytes;
//     uint32_t output_row_bytes = p->wout * FP16_B;
//     uint32_t output_cout_bytes = p->hout * output_row_bytes;
//     uint32_t knl_ch_bytes = p->k_n * FP16_B;
    
//     int t_hout_len_rem = ceil_rem(stm->hout_len, t->hout_len);
//     int t_hin_len = FIND_INPUT_LENGTH(t->hout_len, p->k_eff, p->stride);
//     int t_hin_len_rem = FIND_INPUT_LENGTH(t_hout_len_rem, p->k_eff, p->stride);

//     int t_wout_len_rem = ceil_rem(stm->wout_len, t->wout_len);
//     int t_win_len = FIND_INPUT_LENGTH(t->wout_len, p->k_eff, p->stride);
//     int t_win_len_rem = FIND_INPUT_LENGTH(t_wout_len_rem, p->k_eff, p->stride);

    
//     uint32_t t_win_bytes = t_win_len * FP16_B;
//     uint32_t t_wout_bytes = t->wout_len * FP16_B;

//     if(t->h_div * t->w_div > 1) {//2D tiled workload
//         gtx_printf("2D tiled workload \n");
//         for(int cidx=0; cidx<t->cout_div; cidx++) {
//             // gtx_printf("cidx: %d\n", cidx);
//             uint32_t knl_base[use_spu];
//             uint32_t input_base[use_spu];
//             uint32_t output_base[use_spu];
//             uint32_t bias_base[use_spu];
//             for(int spu_id=0; spu_id<use_spu; spu_id++) {
//                 knl_base[spu_id] = l2i->kernel_addr + knl_ch_bytes * (start_cidx[spu_id] + cidx);
//                 input_base[spu_id] = l2i->input_addr + input_cin_bytes * (start_cidx[spu_id] + cidx);
//                 output_base[spu_id] = l2i->output_addr + output_cout_bytes * (start_cidx[spu_id] + cidx);
//                 bias_base[spu_id] = l2i->bias_addr + FP16_B * (start_cidx[spu_id] + cidx);
//             }
            
//             for(int hid=0; hid<t->h_div; hid++) {
//                 int hout_idx = hid * t->hout_len;
//                 int hout_offset = hout_idx * output_row_bytes;
//                 int hin_offset = hout_idx * p->stride * input_row_bytes;
//                 bool is_h_rem = (hid == t->h_div -1);
//                 int cur_len_hout = is_h_rem? t_hout_len_rem : t->hout_len;
//                 int cur_len_hin = is_h_rem? t_hin_len_rem : t_hin_len;

//                 for(int wid=0; wid<t->w_div; wid++) {
//                     // gtx_printf("hid, wid :  %d, %d\n", hid, wid);
//                     int wout_idx = wid * t->wout_len;
//                     int wout_offset = wout_idx * FP16_B;
//                     int win_offset = wout_idx * p->stride * FP16_B;
//                     bool is_w_rem = (wid == t->w_div -1);
//                     int cur_len_wout = is_w_rem? t_wout_len_rem : t->wout_len;
//                     int cur_len_win = is_w_rem? t_win_len_rem : t_win_len;
//                     uint32_t cur_length_in = cur_len_win * FP16_B;
//                     uint32_t cur_length_out = cur_len_wout * FP16_B;
//                     int bias_ones_len = cur_len_wout * FP16_B;
//                     int ohow = cur_len_hout * cur_len_wout;
                    
//                     // __split();
//                     __start_plan(nest_id);
//                     for(int spu_id=0; spu_id<use_spu; spu_id++) {
//                         uint32_t input_addr = input_base[spu_id] + hin_offset + win_offset;
//                         uint32_t output_addr = output_base[spu_id] + hout_offset + wout_offset;
//                         __start_thread(spu_id);
//                             __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
//                             //bias set
//                             __load(l2i->bias_ones_addr, BASE_L1_BANK_A,
//                                 bias_ones_len, bias_ones_len, cur_len_hout, bias_ones_len);
//                             __load(bias_base[spu_id], BASE_L1_BANK_B, FP16_B, FP16_B, 1, FP16_B);
//                             __mm_s(cur_len_hout * cur_len_wout, 1, 1);
//                             // accumulate once (dwconv)
//                             __load(input_addr , BASE_L1_BANK_R,
//                                 input_row_bytes, cur_length_in, cur_len_hin, cur_length_in);
//                             __im2col_n(cur_len_hin, cur_len_win, p->k_size, p->dilation, p->stride, 1);
//                             __load(knl_base[spu_id], BASE_L1_BANK_B,
//                                 knl_ch_bytes, knl_ch_bytes, 1, knl_ch_bytes);
//                             __mmc_t(ohow, p->k_n, 1);
//                             __store(BASE_L1_BANK_R, output_addr, output_row_bytes, cur_length_out, cur_len_hout, cur_length_out);
//                         __end_thread(spu_id);
//                     }
//                     __end_plan(nest_id);
//                     // __join();
//                 }
//             }
//         }
//     }
    
//     else {// channel chunk workload
//         gtx_printf("channel chunk workload\n");
//         int ohow = t->hout_len * t->wout_len;
//         int bias_ones_len = t->wout_len * FP16_B;
//         uint32_t im2col_c_bytes = output_cout_bytes * p->k_n;
        
//         // gtx_printf("cid: ");
//         for(int cid=0; cid<t->cout_div; cid++) {
//             // gtx_printf("%d, ",cid);
//             int is_c_rem = (cid == t->cout_div - 1);
//             int cidx = cid * t->cout_len;

//             uint32_t knl_base[use_spu];
//             uint32_t input_base[use_spu];
//             uint32_t output_base[use_spu];
//             uint32_t bias_base[use_spu];
//             int cur_c_len[use_spu];
//             for(int spu_id=0; spu_id<use_spu; spu_id++) {
//                 knl_base[spu_id] = l2i->kernel_addr + knl_ch_bytes * (start_cidx[spu_id] + cidx);
//                 input_base[spu_id] = l2i->input_addr + input_cin_bytes * (start_cidx[spu_id] + cidx);
//                 output_base[spu_id] = l2i->output_addr + output_cout_bytes * (start_cidx[spu_id] + cidx);
//                 bias_base[spu_id] = l2i->bias_addr + FP16_B * (start_cidx[spu_id] + cidx);
//                 cur_c_len[spu_id] = is_c_rem? ceil_rem(c_assigned[spu_id], t->cout_len) : t->cout_len;
//             }
           
//             // bias fp32 setting : first GEMM
//             // __split();
//             __start_plan(nest_id);
//             for(int spu_id=0; spu_id<use_spu; spu_id++) {
//                 uint32_t bias_length = cur_c_len[spu_id]* FP16_B;
//                 uint32_t input_height = p->hin_pad * cur_c_len[spu_id];
//                 uint32_t out_height = p->hout * cur_c_len[spu_id];
//                 __start_thread(spu_id);

//                     __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
//                     // bias ones
//                     __load(l2i->bias_ones_addr, BASE_L1_BANK_B,
//                         bias_ones_len, bias_ones_len, p->hout, bias_ones_len);
//                     __load(bias_base[spu_id], BASE_L1_BANK_A, bias_length, bias_length, 1, bias_length);
//                     __mm_s(cur_c_len[spu_id], 1, ohow);
//                     __load(input_base[spu_id], BASE_L1_BANK_R,
//                         input_row_bytes, input_row_bytes, input_height, input_row_bytes);
//                     __im2col_d(p->hin_pad, p->win_pad, p->k_size, p->dilation, p->stride, cur_c_len[spu_id]);
                    
//                     __load(knl_base[spu_id], BASE_L1_BANK_B,
//                         knl_ch_bytes, knl_ch_bytes, cur_c_len[spu_id], knl_ch_bytes);

//                     for(int c_off=0; c_off<cur_c_len[spu_id]; c_off++){
//                         __set_spm_addr( BASE_L1_BANK_R + c_off * output_cout_bytes,
//                                         BASE_L1_BANK_C + c_off * output_cout_bytes * 2,
//                                         BASE_L1_BANK_B + c_off * knl_ch_bytes,
//                                         BASE_L1_BANK_A + c_off * im2col_c_bytes);
//                         __mmc_t(ohow, p->k_n, 1);
//                     }
                    
                    
//                     __store(BASE_L1_BANK_R, output_base[spu_id], output_row_bytes, output_row_bytes, out_height, output_row_bytes);

//                 __end_thread(spu_id);
//             }
//             __end_plan(nest_id);
//             // __join();
//         }
//     }
// }


// // stm->h_div * stm->w_div > 1
// static void dwconv_tile_img(int nest_id, ConvParams* p, SPUTileMap* stm, TileExecPlan* t, L2Info* l2i) {
//     int use_spu = stm->use_spu;

//     uint32_t input_row_bytes = p->win_pad * FP16_B;
//     uint32_t input_cin_bytes = p->hin_pad * input_row_bytes;
//     uint32_t output_row_bytes = p->wout * FP16_B;
//     uint32_t output_cout_bytes = p->hout * output_row_bytes;
//     uint32_t knl_ch_bytes = p->k_n * FP16_B;
//     int bias_length = FP16_B;

//     int hout_assigned[use_spu];
//     int wout_assigned[use_spu];
//     int start_c[use_spu];
//     uint32_t input_base[use_spu];
//     uint32_t knl_base[use_spu];
//     uint32_t out_base[use_spu];

//     {
//         int spu_id = 0;

//         for(int cid=0; cid<stm->cout_div; cid++) {
//             int start_hout=0;
            
//             for(int hid=0; hid<stm->h_div; hid++) {
//                 int start_wout=0;
//                 int cur_len_hout = (hid < stm->h_rem_cnt)? stm->hout_len : stm->hout_len-1;
                
//                 for(int wid=0; wid<stm->w_div; wid++) {
//                     int cur_len_wout = wid < stm->w_rem_cnt? stm->wout_len : stm->wout_len-1;
//                     start_c[spu_id] = cid;
//                     int start_hin = start_hout * p->stride;
//                     int start_win = start_wout * p->stride;

//                     hout_assigned[spu_id] = cur_len_hout;
//                     wout_assigned[spu_id] = cur_len_wout;

//                     input_base[spu_id] = l2i->input_addr + cid * input_cin_bytes + start_hin * input_row_bytes + start_win * FP16_B;
//                     knl_base[spu_id] = l2i->kernel_addr + cid * knl_ch_bytes;
//                     out_base[spu_id] = l2i->output_addr + cid * output_cout_bytes + start_hout * output_row_bytes + start_wout * FP16_B;

//                     start_wout += cur_len_wout;                
//                     spu_id++;
//                 }
//                 start_hout += cur_len_hout;
//             } 
//         }
//     }

//     // tile iteration under SPU work
//     for(int hid=0; hid<t->h_div; hid++){
//         bool is_h_rem = (hid == t->h_div-1);
//         int hout_idx = hid * t->hout_len;
//         uint32_t hout_offset = hout_idx * output_row_bytes;
//         uint32_t hin_offset = hout_idx * p->stride * input_row_bytes;
//         int cur_hout_len[use_spu];
//         int cur_hin_len[use_spu];
//         for(int spu_id=0; spu_id<use_spu; spu_id++){
//             cur_hout_len[spu_id] = is_h_rem? ceil_rem(hout_assigned[spu_id], t->hout_len) : t->hout_len;
//             cur_hin_len[spu_id] = FIND_INPUT_LENGTH(cur_hout_len[spu_id], p->k_eff, p->stride);
//         }
//         for(int wid=0; wid<t->w_div; wid++){
//             gtx_printf("\nhid, wid:  %d, %d  ||", hid, wid);
//             bool is_w_rem = (wid == t->w_div-1);
//             int wout_idx = wid * t->wout_len;
//             uint32_t wout_offset = wout_idx * FP16_B;
//             uint32_t win_offset = wout_idx * p->stride * FP16_B;
//             gtx_printf("- bias - ");
//             // bias fp32 setting : first GEMM
//             // __split();
//             __start_plan(nest_id);
//             for(int spu_id=0; spu_id<use_spu; spu_id++) {
//                 int cur_wout_len = is_w_rem? ceil_rem(wout_assigned[spu_id], t->wout_len) : t->wout_len;
//                 int cur_win_len = FIND_INPUT_LENGTH(cur_wout_len, p->k_eff, p->stride);
//                 int ohow = cur_wout_len * cur_hout_len[spu_id];

//                 uint32_t bias_addr = l2i->bias_addr + start_c[spu_id] * FP16_B;
//                 int ones_length = cur_wout_len * FP16_B;
//                 uint32_t input_addr = input_base[spu_id] + hin_offset + win_offset;
//                 uint32_t output_addr = out_base[spu_id] + hout_offset + wout_offset;
//                 // gtx_printf("mm_S : M, K, N : %d, %d, %d \n", cur_hout[spu_id]*cur_wout[spu_id], 1, cur_cout[spu_id]);
//                 __start_thread(spu_id);
//                     __set_spm_addr(BASE_L1_BANK_R, BASE_L1_BANK_C, BASE_L1_BANK_B, BASE_L1_BANK_A);
//                     __load(l2i->bias_ones_addr, BASE_L1_BANK_A, ones_length, ones_length, cur_hout_len[spu_id], ones_length);
//                     __load(bias_addr, BASE_L1_BANK_B, bias_length, bias_length, 1, bias_length);
//                     __mm_s(cur_hout_len[spu_id]*cur_wout_len, 1, 1);

//                     // accumulate once (dwconv)
//                     __load(input_addr , BASE_L1_BANK_R,
//                         input_row_bytes, cur_win_len*FP16_B, cur_hin_len[spu_id], cur_win_len*FP16_B);
//                     __im2col_n(cur_hin_len[spu_id], cur_win_len, p->k_size, p->dilation, p->stride, 1);
//                     __load(knl_base[spu_id], BASE_L1_BANK_B,
//                         knl_ch_bytes, knl_ch_bytes, 1, knl_ch_bytes);
//                     __mmc_t(ohow, p->k_n, 1);
//                     __store(BASE_L1_BANK_R, output_addr, output_row_bytes, cur_wout_len*FP16_B, cur_hout_len[spu_id], cur_wout_len*FP16_B);
//                 __end_thread(spu_id);
//             }
//             __end_plan(nest_id);
//             // __join();
//         }
//     }
// }
