//==================================================================
// GGML GTX kernel: RMS_NORM
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

int ggml_gtx_kernel_rms_norm_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_RMS_NORM_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_RMS_NORM_F16) {
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
        desc->ne[0] > UINT16_MAX) {
        return -2;
    }

    if (desc->memory_mode == GGML_GTX_MEMORY_MODE_L2_STASHED_N1) {
        const uint64_t row_bytes = (uint64_t) desc->ne[0] * sizeof(uint16_t);
        if (desc->nest_id >= GGML_GTX_NEST_NUM ||
            desc->dtype != GGML_GTX_TYPE_F16 ||
            desc->src_type[0] != GGML_GTX_TYPE_F16 ||
            desc->elem_size != GGML_GTX_DTYPE_SIZE ||
            desc->l2_src_addr[0] == 0 ||
            desc->l2_dst_addr == 0 ||
            row_bytes == 0 ||
            row_bytes > GGML_GTX_L1_BANK_SIZE ||
            rows64 > UINT16_MAX) {
            return -3;
        }

        ggml_gtx_compute_rmsnorm_l2((uint32_t) desc->l2_src_addr[0],
                                    (uint32_t) desc->l2_dst_addr,
                                    (uint16_t) rows64,
                                    (uint16_t) desc->ne[0],
                                    ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[0]),
                                    1u,
                                    GGML_GTX_SPU_NUM_PER_NEST);
        return 0;
    }

    if (rows64 > UINT16_MAX) {
        return -4;
    }

    ggml_gtx_compute_rmsnorm(desc->src_addr[0],
                             desc->dst_addr,
                             (uint16_t) rows64,
                             (uint16_t) desc->ne[0],
                             ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[0]),
                             GGML_GTX_NEST_NUM,
                             GGML_GTX_SPU_NUM_PER_NEST);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_rms_norm_f16(arg);
}

void ggml_gtx_compute_rmsnorm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    ggml_gtx_debug_dump_fp16_head("RMS_A", ddr_input);
    const uint16_t not_use = 0xBEEF;
    const uint32_t cal_unit = i_col;
    const uint32_t cal_unit_byte = (uint32_t)i_col * GGML_GTX_DTYPE_SIZE;
    const uint32_t cal_max = cal_unit_byte != 0 ? (GGML_GTX_L1_BANK_SIZE / cal_unit_byte) : 0;
    const uint32_t cal_num = i_row;
    const uint8_t eff_thread = cal_num == 0 ? 1 : (cal_num / spu_num_per_nest ? spu_num_per_nest : (uint8_t)cal_num);
    const uint32_t cal_quot = eff_thread != 0 ? (cal_num / eff_thread) : 0;
    const uint32_t cal_mod = eff_thread != 0 ? (cal_num % eff_thread) : 0;

    uint32_t bank_offset_shared = 0;
    uint32_t bank_offset_thread = 0;
    uint32_t cal_offset = 0;
    uint16_t exe_cnt = 0;

    GGML_GTX_UNUSED(nest_num);

    if (cal_unit == 0 || cal_unit_byte == 0 || cal_num == 0 || cal_max == 0 || eff_thread == 0) {
        return;
    }

    __split();
        __start_plani(0);
            __start_shared();
                for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? (cal_quot + 1) : cal_quot;
                    const uint32_t offset_thread =
                        ((GGML_GTX_MIN(thread_id, cal_mod) * (cal_quot + 1)) +
                         (GGML_GTX_MAX(0, (int32_t)thread_id - (int32_t)cal_mod) * cal_quot)) * cal_unit_byte;

                    __load(ddr_input + offset_thread,
                           GGML_GTX_L2_INPUT + bank_offset_shared,
                           cal_unit_byte, cal_unit_byte,
                           cal_num_per_thread, cal_unit_byte);
                    __credit_ld(0x1 << thread_id, not_use);
                    bank_offset_shared += GGML_GTX_L2_BANK_OFFSET;
                }
            __end_shared();

            for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                const uint32_t cal_num_per_thread = thread_id < cal_mod ? (cal_quot + 1) : cal_quot;
                const uint32_t cal_last = (cal_num_per_thread % cal_max) ? (cal_num_per_thread % cal_max) : cal_max;

                __start_thread(thread_id);
                    __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                    __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);
                    __credit_chk(not_use);
                    __load(GGML_GTX_L2_INPUT + bank_offset_thread,
                           GGML_GTX_L1_BANK_A_OFFSET,
                           cal_unit_byte, cal_unit_byte,
                           cal_last, not_use);
                    __credit_ld(not_use, not_use);
                    __rmsnorm(cal_unit,
                              GGML_GTX_L1_BANK_A_OFFSET,
                              0,
                              GGML_GTX_L1_BANK_R_OFFSET,
                              epsilon);
                    bank_offset_thread += GGML_GTX_L2_BANK_OFFSET;
                __end_thread(thread_id);
            }
        __end_plani(0);
    __join();

    exe_cnt++;
    bank_offset_shared = 0;
    bank_offset_thread = 0;

    for (uint32_t iter = 0; iter + 1 < cal_quot; iter++) {
        cal_offset += cal_unit_byte;

        __split();
            __start_plani(0);
                for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                    __start_thread(thread_id);
                        const uint32_t l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET + cal_offset;
                        const uint32_t l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET + cal_offset;
                        __set_spm_addr_A(l1_A_addr);
                        __set_spm_addr_R(l1_R_addr);
                        __rmsnorm(cal_unit, l1_A_addr, 0, l1_R_addr, epsilon);
                    __end_thread(thread_id);
                }
            __end_plani(0);
        __join();

        exe_cnt++;
    }

    cal_offset += cal_unit_byte;

    __split();
        __start_plani(0);
            __start_shared();
                for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? (cal_quot + 1) : cal_quot;
                    const uint32_t offset_thread =
                        ((GGML_GTX_MIN(thread_id, cal_mod) * (cal_quot + 1)) +
                         (GGML_GTX_MAX(0, (int32_t)thread_id - (int32_t)cal_mod) * cal_quot)) * cal_unit_byte;
                    const uint16_t target = (uint16_t)(0x1U << thread_id);

                    __credit_chk(target);
                    __store(GGML_GTX_L2_RESULT + bank_offset_shared,
                            ddr_result + offset_thread,
                            cal_unit_byte, cal_unit_byte,
                            cal_num_per_thread, cal_unit_byte);
                    __credit_st(target);
                    bank_offset_shared += GGML_GTX_L2_BANK_OFFSET;
                }
            __end_shared();

            for (uint8_t thread_id = 0; thread_id < eff_thread; thread_id++) {
                const uint32_t cal_num_per_thread = thread_id < cal_mod ? (cal_quot + 1) : cal_quot;
                const uint32_t cal_last = (cal_num_per_thread % cal_max) ? (cal_num_per_thread % cal_max) : cal_max;

                __start_thread(thread_id);
                    if (exe_cnt < cal_num_per_thread) {
                        const uint32_t l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET + cal_offset;
                        const uint32_t l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET + cal_offset;
                        __set_spm_addr_A(l1_A_addr);
                        __set_spm_addr_R(l1_R_addr);
                        __rmsnorm(cal_unit, l1_A_addr, 0, l1_R_addr, epsilon);
                    }

                    __store(GGML_GTX_L1_BANK_R_OFFSET,
                            GGML_GTX_L2_RESULT + bank_offset_thread,
                            cal_unit_byte, cal_unit_byte,
                            cal_last, not_use);
                    __credit_st(not_use);
                    bank_offset_thread += GGML_GTX_L2_BANK_OFFSET;
                __end_thread(thread_id);
            }
        __end_plani(0);
    __join();

    ggml_gtx_debug_dump_fp16_head("RMS_R", ddr_result);
#else
    (void)ddr_input; (void)ddr_result; (void)i_row; (void)i_col;
    (void)epsilon; (void)nest_num; (void)spu_num_per_nest;
#endif
}

void ggml_gtx_compute_rmsnorm_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    if (i_row == 0 || i_col == 0 || nest_num == 0 || spu_num_per_nest == 0) {
        return;
    }

    ggml_gtx_tile_plan_t plan;
    ggml_gtx_tile_plan_init(&plan, i_row, i_col, nest_num, spu_num_per_nest);
    if (plan.max_cal_per_thread == 0) {
        return;
    }

    __split();

    for (uint8_t nest_id = 0; nest_id < plan.eff_nest_int; nest_id++) {
        ggml_gtx_nest_tile_t * nest = &plan.nests[nest_id];

        if (nest->cal_num > 0) {
            __start_plan(nest_id);

            for (uint8_t thread_id = 0; thread_id < nest->eff_thread_int; thread_id++) {
                ggml_gtx_thread_tile_t * thread = &nest->threads[thread_id];

                if (thread->cal_num > 0) {
                    __start_thread(thread_id);

                    __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                    __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                    uint32_t l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                    uint32_t l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;
                    uint32_t offset_cal = 0;

                    for (uint16_t j = 0; j < thread->iteration; j++) {
                        uint32_t next_offset = thread->offset + offset_cal;

                        __load(l2_input + next_offset,
                               GGML_GTX_L1_BANK_A_OFFSET,
                               plan.unit_per_thread,
                               plan.unit_per_thread,
                               plan.max_cal_per_thread,
                               0xBEEF);

                        for (uint16_t k = 0; k < plan.max_cal_per_thread; k++) {
                            __rmsnorm(i_col,
                                      l1_A_addr,
                                      0,
                                      l1_R_addr,
                                      epsilon);

                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                        l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;
                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store(GGML_GTX_L1_BANK_R_OFFSET,
                                l2_result + next_offset,
                                plan.unit_per_thread,
                                plan.unit_per_thread,
                                plan.max_cal_per_thread,
                                0xBEEF);

                        offset_cal += plan.unit_per_thread * plan.max_cal_per_thread;
                    }

                    if (thread->remain > 0) {
                        uint32_t last_offset = thread->offset + offset_cal;

                        __load(l2_input + last_offset,
                               GGML_GTX_L1_BANK_A_OFFSET,
                               plan.unit_per_thread,
                               plan.unit_per_thread,
                               thread->remain,
                               0xBEEF);

                        for (uint16_t k = 0; k < thread->remain; k++) {
                            __rmsnorm(i_col,
                                      l1_A_addr,
                                      0,
                                      l1_R_addr,
                                      epsilon);

                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store(GGML_GTX_L1_BANK_R_OFFSET,
                                l2_result + last_offset,
                                plan.unit_per_thread,
                                plan.unit_per_thread,
                                thread->remain,
                                0xBEEF);
                    }

                    __end_thread(thread_id);
                }
            }

            __end_plan(nest_id);
        }
    }

    __join();
#else
    (void)l2_input; (void)l2_result;
    (void)i_row; (void)i_col; (void)epsilon;
    (void)nest_num; (void)spu_num_per_nest;
#endif
}

//==================================================================
// Matrix Multiplication
//==================================================================
