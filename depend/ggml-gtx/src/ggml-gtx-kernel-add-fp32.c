//==================================================================
// GGML GTX kernel: ADD F16 + F32 vector -> F16
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_ADD_FP32_L1_A       0x00000u
#define GGML_GTX_ADD_FP32_L1_B       0x20000u
#define GGML_GTX_ADD_FP32_L1_C       0x30000u
#define GGML_GTX_ADD_FP32_L1_R       0x50000u
#define GGML_GTX_ADD_FP32_CONST_ONE  0x3c00u

void ggml_gtx_compute_add_f16_f32_to_f16_tv(
    uint64_t ddr_a_f16,
    uint64_t ddr_b_f32,
    uint64_t ddr_result_f16,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest);

int ggml_gtx_kernel_add_f16_f32(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_ADD_F16_F32_TO_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_ADD_F16_F32_TO_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint64_t rows64 = (uint64_t) desc->ne[1] *
                            (uint64_t) desc->ne[2] *
                            (uint64_t) desc->ne[3];
    const uint64_t row_bytes_f16 = (uint64_t) desc->ne[0] * 2u;
    const uint64_t row_bytes_f32 = (uint64_t) desc->ne[0] * 4u;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->src_addr[1] == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->src_type[1] != GGML_GTX_TYPE_F32 ||
        desc->ne[0] == 0 ||
        rows64 == 0 ||
        desc->ne[0] > UINT16_MAX ||
        desc->src_ne[0][0] != desc->ne[0] ||
        desc->src_ne[1][0] != desc->ne[0] ||
        desc->src_ne[0][1] != desc->ne[1] ||
        desc->src_ne[0][2] != desc->ne[2] ||
        desc->src_ne[0][3] != desc->ne[3] ||
        desc->src_ne[1][1] != 1 ||
        desc->src_ne[1][2] != 1 ||
        desc->src_ne[1][3] != 1 ||
        desc->nb[1] != row_bytes_f16 ||
        desc->src_nb[0][1] != row_bytes_f16 ||
        desc->src_nb[1][1] != row_bytes_f32 ||
        row_bytes_f16 > UINT16_MAX ||
        row_bytes_f16 > GGML_GTX_L1_BANK_SIZE ||
        row_bytes_f32 > UINT16_MAX ||
        row_bytes_f32 > GGML_GTX_L1_BANK_SIZE) {
        return -2;
    }

    const uint16_t cols = (uint16_t) desc->ne[0];
    const uint64_t max_rows_per_call = UINT16_MAX;
    for (uint64_t row = 0; row < rows64; row += max_rows_per_call) {
        const uint64_t remaining = rows64 - row;
        const uint16_t rows_chunk =
            (uint16_t) (remaining < max_rows_per_call ? remaining : max_rows_per_call);
        const uint64_t offset = row * desc->nb[1];
        ggml_gtx_compute_add_f16_f32_to_f16_tv(desc->src_addr[0] + offset,
                                               desc->src_addr[1],
                                               desc->dst_addr + offset,
                                               rows_chunk,
                                               cols,
                                               GGML_GTX_NEST_NUM,
                                               GGML_GTX_SPU_NUM_PER_NEST);
    }
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_add_f16_f32(arg);
}

void ggml_gtx_printf(const char * fmt, ...) {
    (void) fmt;
}

void ggml_gtx_compute_add_f16_f32_to_f16_tv(
    uint64_t ddr_a_f16,
    uint64_t ddr_b_f32,
    uint64_t ddr_result_f16,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    if (i_row == 0 || i_col == 0 || spu_num_per_nest == 0) {
        return;
    }

    const uint8_t nest_id = 0;
    const uint16_t not_use = 0xBEEF;
    const uint32_t row_bytes_f16 = (uint32_t) i_col * 2u;
    const uint32_t row_bytes_f32 = (uint32_t) i_col * 4u;
    const uint16_t eff_thread = 1;
    const uint16_t eff_thread_hex = GGML_GTX_EFF_MASK_16(eff_thread);
    const uint16_t cal_quot = (uint16_t) (i_row / eff_thread);
    const uint16_t cal_mod = (uint16_t) (i_row % eff_thread);
    const uint32_t cal_max =
        row_bytes_f16 != 0 ? (GGML_GTX_L1_BANK_SIZE / row_bytes_f16) : 0;

    (void) nest_num;

    if (row_bytes_f16 == 0 ||
        row_bytes_f32 == 0 ||
        row_bytes_f16 > UINT16_MAX ||
        row_bytes_f32 > UINT16_MAX ||
        row_bytes_f32 > GGML_GTX_L1_BANK_SIZE ||
        cal_max == 0) {
        return;
    }

    __mcast_g2s(ddr_b_f32,
                GGML_GTX_L2_B,
                row_bytes_f32,
                (uint16_t) row_bytes_f32,
                1,
                (uint64_t) 0x1u << nest_id);

    __mcast_s2l(GGML_GTX_L2_B,
                GGML_GTX_ADD_FP32_L1_C,
                row_bytes_f32,
                (uint16_t) row_bytes_f32,
                1,
                eff_thread_hex);

    uint32_t bank_offset_shared = 0;
    uint32_t bank_offset_thread = 0;

    __split();

        __start_plani(nest_id);

            __set_spm_addr(GGML_GTX_ADD_FP32_L1_R,
                           GGML_GTX_ADD_FP32_L1_C,
                           GGML_GTX_ADD_FP32_L1_B,
                           GGML_GTX_ADD_FP32_L1_A);
            __wrspr(0x800, 0x0, GGML_GTX_ADD_FP32_CONST_ONE, 0x0);
            __store_svr(GGML_GTX_ADD_FP32_L1_B, 0x0);

            __start_shared();

                for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                    const uint16_t cal_num_per_thread =
                        (thread_id < cal_mod) ? (cal_quot + 1) : cal_quot;
                    const uint32_t offset_thread =
                        (((thread_id < cal_mod) ?
                          (uint32_t) thread_id * (cal_quot + 1) :
                          (uint32_t) cal_mod * (cal_quot + 1) +
                          (uint32_t) (thread_id - cal_mod) * cal_quot) *
                         row_bytes_f16);

                    __load(ddr_a_f16 + offset_thread,
                           GGML_GTX_L2_A + bank_offset_shared,
                           row_bytes_f16,
                           (uint16_t) row_bytes_f16,
                           cal_num_per_thread,
                           row_bytes_f16);

                    __credit_ld((uint16_t) (0x1u << thread_id), not_use);
                    bank_offset_shared += GGML_GTX_L2_BANK_OFFSET;
                }

                bank_offset_shared = 0;

                for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                    const uint16_t target = (uint16_t) (0x1u << thread_id);
                    const uint16_t cal_num_per_thread =
                        (thread_id < cal_mod) ? (cal_quot + 1) : cal_quot;
                    const uint32_t offset_thread =
                        (((thread_id < cal_mod) ?
                          (uint32_t) thread_id * (cal_quot + 1) :
                          (uint32_t) cal_mod * (cal_quot + 1) +
                          (uint32_t) (thread_id - cal_mod) * cal_quot) *
                         row_bytes_f16);

                    __credit_chk(target);
                    __store(GGML_GTX_L2_RESULT + bank_offset_shared,
                            ddr_result_f16 + offset_thread,
                            row_bytes_f16,
                            (uint16_t) row_bytes_f16,
                            cal_num_per_thread,
                            row_bytes_f16);
                    __credit_st(target);
                    bank_offset_shared += GGML_GTX_L2_BANK_OFFSET;
                }

            __end_shared();

            for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                __start_thread(thread_id);

                    const uint16_t cal_num_per_thread =
                        (thread_id < cal_mod) ? (cal_quot + 1) : cal_quot;
                    const uint16_t iteration =
                        (uint16_t) ((cal_num_per_thread + cal_max - 1) / cal_max);
                    const uint32_t l2_a_thread = GGML_GTX_L2_A + bank_offset_thread;
                    const uint32_t l2_res_thread = GGML_GTX_L2_RESULT + bank_offset_thread;
                    uint32_t offset_cal = 0;

                    __credit_chk(not_use);

                    for (uint16_t iter = 0; iter < iteration; iter++) {
                        const bool last = (iter + 1) == iteration;
                        const uint16_t cal_tile =
                            last ?
                            (uint16_t) (cal_num_per_thread - (uint32_t) iter * cal_max) :
                            (uint16_t) cal_max;
                        uint32_t l1_a_addr = GGML_GTX_ADD_FP32_L1_A;
                        uint32_t l1_r_addr = GGML_GTX_ADD_FP32_L1_R;

                        __load(l2_a_thread + offset_cal,
                               l1_a_addr,
                               row_bytes_f16,
                               (uint16_t) row_bytes_f16,
                               cal_tile,
                               not_use);
                        __credit_ld(not_use, not_use);

                        for (uint16_t row = 0; row < cal_tile; row++) {
                            __set_spm_addr_A(l1_a_addr);
                            __set_spm_addr_B(GGML_GTX_ADD_FP32_L1_B);
                            __set_spm_addr_C(GGML_GTX_ADD_FP32_L1_C);
                            __set_spm_addr_R(l1_r_addr);

                            __mmc(i_col, 1, 1);

                            l1_a_addr += row_bytes_f16;
                            l1_r_addr += row_bytes_f16;
                        }

                        __set_spm_addr_A(GGML_GTX_ADD_FP32_L1_A);
                        __set_spm_addr_B(GGML_GTX_ADD_FP32_L1_B);
                        __set_spm_addr_C(GGML_GTX_ADD_FP32_L1_C);
                        __set_spm_addr_R(GGML_GTX_ADD_FP32_L1_R);

                        __store(GGML_GTX_ADD_FP32_L1_R,
                                l2_res_thread + offset_cal,
                                row_bytes_f16,
                                (uint16_t) row_bytes_f16,
                                cal_tile,
                                not_use);

                        offset_cal += row_bytes_f16 * cal_tile;
                    }

                    __credit_st(not_use);
                    bank_offset_thread += GGML_GTX_L2_BANK_OFFSET;

                __end_thread(thread_id);
            }

        __end_plani(nest_id);

    __join();
#else
    (void) ddr_a_f16;
    (void) ddr_b_f32;
    (void) ddr_result_f16;
    (void) i_row;
    (void) i_col;
    (void) nest_num;
    (void) spu_num_per_nest;
#endif
}
