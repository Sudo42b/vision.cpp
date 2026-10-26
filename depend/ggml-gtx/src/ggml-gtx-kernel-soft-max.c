//==================================================================
// GGML GTX kernel: SOFT_MAX
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

static float ggml_gtx_softmax_bits_to_float(uint32_t bits);
static uint32_t ggml_gtx_softmax_float_to_bits(float value);
static float ggml_gtx_softmax_expf(float arg);

#define GGML_GTX_SOFTMAX_L1_A_BYTES (128u * 1024u)
#define GGML_GTX_SOFTMAX_L1_B_BYTES (64u * 1024u)

int ggml_gtx_kernel_soft_max_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_SOFT_MAX_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_SOFT_MAX_F16) {
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

    const float scale = ggml_gtx_softmax_bits_to_float((uint32_t) desc->op_params[0]);
    const float max_bias = ggml_gtx_softmax_bits_to_float((uint32_t) desc->op_params[1]);

    if (desc->memory_mode == GGML_GTX_MEMORY_MODE_L2_STASHED_N1) {
        const uint64_t row_bytes = (uint64_t) desc->ne[0] * sizeof(uint16_t);
        if (desc->nest_id >= GGML_GTX_NEST_NUM ||
            desc->src_addr[1] != 0 ||
            scale != 1.0f ||
            max_bias != 0.0f ||
            desc->dtype != GGML_GTX_TYPE_F16 ||
            desc->src_type[0] != GGML_GTX_TYPE_F16 ||
            desc->elem_size != GGML_GTX_DTYPE_SIZE ||
            desc->l2_src_addr[0] == 0 ||
            row_bytes == 0 ||
            row_bytes > GGML_GTX_L1_BANK_SIZE ||
            rows64 > UINT16_MAX) {
            return -3;
        }

        ggml_gtx_compute_softmax_l2((uint32_t) desc->l2_src_addr[0],
                                    (uint32_t) desc->l2_dst_addr,
                                    (uint16_t) rows64,
                                    (uint16_t) desc->ne[0],
                                    1u,
                                    GGML_GTX_SPU_NUM_PER_NEST);
        return 0;
    }

    if (desc->src_addr[1] == 0 && scale == 1.0f && max_bias == 0.0f) {
        const uint64_t row_bytes = (uint64_t) desc->ne[0] * sizeof(uint16_t);
        uint64_t row_base = 0;
        while (row_base < rows64) {
            const uint64_t remaining = rows64 - row_base;
            const uint16_t chunk_rows = (uint16_t)
                (remaining > UINT16_MAX ? UINT16_MAX : remaining);
            const uint64_t byte_off = row_base * row_bytes;
            ggml_gtx_compute_softmax(desc->src_addr[0] + byte_off,
                                     desc->dst_addr + byte_off,
                                     chunk_rows,
                                     (uint16_t) desc->ne[0],
                                     GGML_GTX_NEST_NUM,
                                     GGML_GTX_SPU_NUM_PER_NEST);
            row_base += chunk_rows;
        }
    } else {
        if (desc->ne[1] > UINT16_MAX ||
            desc->ne[2] > UINT16_MAX ||
            desc->ne[3] > UINT16_MAX ||
            desc->src_ne[1][2] > UINT16_MAX ||
            desc->src_ne[1][3] > UINT16_MAX) {
            return -3;
        }
        ggml_gtx_compute_softmax_ext(desc->src_addr[0],
                                     desc->src_addr[1],
                                     desc->dst_addr,
                                     (uint16_t) desc->ne[0],
                                     (uint16_t) desc->ne[1],
                                     (uint16_t) desc->ne[2],
                                     (uint16_t) desc->ne[3],
                                     desc->nb[1],
                                     desc->nb[2],
                                     desc->nb[3],
                                     (uint16_t) desc->src_ne[1][2],
                                     (uint16_t) desc->src_ne[1][3],
                                     desc->src_nb[1][1],
                                     desc->src_nb[1][2],
                                     desc->src_nb[1][3],
                                     scale,
                                     max_bias);
    }
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_soft_max_f16(arg);
}

static float ggml_gtx_softmax_bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value_bits;
    value_bits.u = bits;
    return value_bits.f;
}

static uint32_t ggml_gtx_softmax_float_to_bits(float value) {
    union { uint32_t u; float f; } value_bits;
    value_bits.f = value;
    return value_bits.u;
}

static float ggml_gtx_softmax_expf(float arg) {
    if (arg < -20.0f) {
        return 0.0f;
    }
    if (arg > 20.0f) {
        arg = 20.0f;
    }

    const float ln2 = 0.69314718056f;
    int exponent = (int)(arg / ln2);
    float reduced = arg - ((float) exponent * ln2);

    if (reduced < -0.5f * ln2) {
        reduced += ln2;
        --exponent;
    } else if (reduced > 0.5f * ln2) {
        reduced -= ln2;
        ++exponent;
    }

    const float reduced_sq = reduced * reduced;
    const float poly = 1.0f + reduced + reduced_sq * 0.5f +
                       reduced * reduced_sq * (1.0f / 6.0f) +
                       reduced_sq * reduced_sq * (1.0f / 24.0f);

    if (exponent > 30) {
        exponent = 30;
    }
    if (exponent < -30) {
        return 0.0f;
    }

    return exponent >= 0 ? poly * (float)(1u << exponent)
                         : poly / (float)(1u << -exponent);
}

void ggml_gtx_compute_softmax(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    // Following reference_usage.c __n4_softmax__ pattern
    ggml_gtx_tile_plan_t plan;
    ggml_gtx_tile_plan_init(&plan, i_row, i_col, nest_num, spu_num_per_nest);

    uint32_t offset_cal = 0;

    __split();

    for (uint8_t nest_id = 0; nest_id < plan.eff_nest_int; nest_id++) {
        ggml_gtx_nest_tile_t * nest = &plan.nests[nest_id];

        if (nest->cal_num > 0) {
            __start_plan(nest_id);
            __start_shared();

            // Load input tensor
            for (uint8_t thread_id = 0; thread_id < nest->eff_thread_int; thread_id++) {
                ggml_gtx_thread_tile_t * thread = &nest->threads[thread_id];
                __load_cr(
                    ddr_input + thread->offset,
                    GGML_GTX_L2_INPUT + GGML_GTX_L2_BANK_OFFSET * thread_id,
                    plan.unit_per_thread,
                    plan.unit_per_thread,
                    thread->cal_num,
                    plan.unit_per_thread,
                    1,
                    0x1 << thread_id,
                    0xBEEF
                );
            }

            // Store results
            for (uint8_t thread_id = 0; thread_id < nest->eff_thread_int; thread_id++) {
                ggml_gtx_thread_tile_t * thread = &nest->threads[thread_id];
                __credit_chk(0x1 << thread_id);
                __store_cr(
                    GGML_GTX_L2_RESULT + GGML_GTX_L2_BANK_OFFSET * thread_id,
                    ddr_result + thread->offset,
                    plan.unit_per_thread,
                    plan.unit_per_thread,
                    thread->cal_num,
                    plan.unit_per_thread,
                    1,
                    0x1 << thread_id
                );
            }

            __end_shared();

            // Thread: L2SPM <-> L1SPM + Compute
            for (uint8_t thread_id = 0; thread_id < nest->eff_thread_int; thread_id++) {
                ggml_gtx_thread_tile_t * thread = &nest->threads[thread_id];

                if (thread->cal_num > 0) {
                    __start_thread(thread_id);

                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);

                    uint32_t l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                    uint32_t l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;

                    __credit_chk(0xBEEF);

                    offset_cal = 0;

                    // Full iterations
                    for (uint16_t j = 0; j < thread->iteration; j++) {
                        uint32_t next_offset = GGML_GTX_L2_BANK_OFFSET * thread_id + offset_cal;

                        __load(
                            GGML_GTX_L2_INPUT + next_offset,
                            GGML_GTX_L1_BANK_A_OFFSET,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            plan.max_cal_per_thread,
                            0xBEEF
                        );

                        // Process each row in the tile
                        for (uint16_t k = 0; k < plan.max_cal_per_thread; k++) {
                            // [max.vs] find max value (L1 A bank -> SVR_0)
                            // 0xFBFF = -inf in FP16
                            __max_vs(i_col, 0xFBFF, 0x0, 0b0);

                            // [esum] exponential sum (L1 A bank & SVR_0 -> SVR_1)
                            __esum(i_col, 0xBEEF, 0xBEEF, 0x1, 0x180);

                            // [softmax] compute softmax (L1 A bank & SVR_1 -> L1 R bank)
                            __softmax(i_col, 0xBEEF, 0xBEEF, 0x184);

                            // Advance bank pointers
                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        // Reset bank pointers
                        l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                        l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;
                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store(
                            GGML_GTX_L1_BANK_R_OFFSET,
                            GGML_GTX_L2_RESULT + next_offset,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            plan.max_cal_per_thread,
                            0xBEEF
                        );

                        offset_cal += plan.unit_per_thread * plan.max_cal_per_thread;
                    }

                    // Remaining elements
                    if (thread->remain > 0) {
                        uint32_t last_offset = GGML_GTX_L2_BANK_OFFSET * thread_id + offset_cal;

                        __load_cr(
                            GGML_GTX_L2_INPUT + last_offset,
                            GGML_GTX_L1_BANK_A_OFFSET,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            thread->remain,
                            0xBEEF,
                            1,
                            0xBEEF,
                            0xBEEF
                        );

                        for (uint16_t k = 0; k < thread->remain; k++) {
                            __max_vs(i_col, 0xFBFF, 0x0, 0b0);
                            __esum(i_col, 0xBEEF, 0xBEEF, 0x1, 0x180);
                            __softmax(i_col, 0xBEEF, 0xBEEF, 0x184);

                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store_cr(
                            GGML_GTX_L1_BANK_R_OFFSET,
                            GGML_GTX_L2_RESULT + last_offset,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            thread->remain,
                            0xBEEF,
                            1,
                            0xBEEF
                        );
                    }

                    offset_cal = 0;

                    __end_thread(thread_id);
                }
            }

            __end_plan(nest_id);
        }
    }

    __join();
#else
    (void)ddr_input; (void)ddr_result;
    (void)i_row; (void)i_col;
    (void)nest_num; (void)spu_num_per_nest;
#endif
}

void ggml_gtx_compute_softmax_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t i_row,
    uint16_t i_col,
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

                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET, GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET, GGML_GTX_L1_BANK_A_OFFSET);

                    uint32_t l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                    uint32_t l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;
                    uint32_t offset_cal = 0;

                    for (uint16_t j = 0; j < thread->iteration; j++) {
                        uint32_t next_offset = thread->offset + offset_cal;

                        __load(
                            l2_input + next_offset,
                            GGML_GTX_L1_BANK_A_OFFSET,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            plan.max_cal_per_thread,
                            0xBEEF
                        );

                        for (uint16_t k = 0; k < plan.max_cal_per_thread; k++) {
                            __max_vs(i_col, 0xFBFF, 0x0, 0b0);
                            __esum(i_col, 0xBEEF, 0xBEEF, 0x1, 0x180);
                            __softmax(i_col, 0xBEEF, 0xBEEF, 0x184);

                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        l1_A_addr = GGML_GTX_L1_BANK_A_OFFSET;
                        l1_R_addr = GGML_GTX_L1_BANK_R_OFFSET;
                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store(
                            GGML_GTX_L1_BANK_R_OFFSET,
                            l2_result + next_offset,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            plan.max_cal_per_thread,
                            0xBEEF
                        );

                        offset_cal += plan.unit_per_thread * plan.max_cal_per_thread;
                    }

                    if (thread->remain > 0) {
                        uint32_t last_offset = thread->offset + offset_cal;

                        __load(
                            l2_input + last_offset,
                            GGML_GTX_L1_BANK_A_OFFSET,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            thread->remain,
                            0xBEEF
                        );

                        for (uint16_t k = 0; k < thread->remain; k++) {
                            __max_vs(i_col, 0xFBFF, 0x0, 0b0);
                            __esum(i_col, 0xBEEF, 0xBEEF, 0x1, 0x180);
                            __softmax(i_col, 0xBEEF, 0xBEEF, 0x184);

                            l1_A_addr += plan.unit_per_thread;
                            l1_R_addr += plan.unit_per_thread;
                            __set_spm_addr_A(l1_A_addr);
                            __set_spm_addr_R(l1_R_addr);
                        }

                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_R_OFFSET);

                        __store(
                            GGML_GTX_L1_BANK_R_OFFSET,
                            l2_result + last_offset,
                            plan.unit_per_thread,
                            plan.unit_per_thread,
                            thread->remain,
                            0xBEEF
                        );
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
    (void)i_row; (void)i_col;
    (void)nest_num; (void)spu_num_per_nest;
#endif
}

#ifdef GGML_GTX_USE_GTX
static void ggml_gtx_compute_softmax_ext_npu_plane(
    uint64_t ddr_input,
    uint64_t ddr_mask,
    uint64_t ddr_result,
    uint16_t rows,
    uint16_t cols,
    float scale)
{
    const uint32_t row_bytes = (uint32_t)cols * sizeof(uint16_t);
    const uint32_t l1_limit = ddr_mask != 0u ?
        GGML_GTX_SOFTMAX_L1_B_BYTES : GGML_GTX_SOFTMAX_L1_A_BYTES;
    uint32_t max_rows = l1_limit / row_bytes;
    const uint32_t dma_rows = UINT16_MAX / row_bytes;
    if (max_rows > dma_rows) {
        max_rows = dma_rows;
    }
    if (max_rows == 0u) {
        return;
    }

    const uint16_t scale_f16 = ggml_gtx_fp32_bits_to_fp16(
        ggml_gtx_softmax_float_to_bits(scale));

    for (uint32_t row0 = 0u; row0 < rows; row0 += max_rows) {
        const uint16_t block_rows = (uint16_t)
            (((uint32_t)rows - row0) < max_rows ? (uint32_t)rows - row0 : max_rows);
        const uint64_t byte_off = (uint64_t)row0 * row_bytes;

        /* Keep the three asynchronous domains in separate completed plans.
         * FPGA credit chaining is intentionally not used here: each join is
         * the completion boundary for the following phase. */
        __split();
            __start_plan(0u);
                __start_shared();
                    __load(ddr_input + byte_off,
                           GGML_GTX_L2_A,
                           row_bytes, row_bytes, block_rows, row_bytes);
                    if (ddr_mask != 0u) {
                        __load(ddr_mask + byte_off,
                               GGML_GTX_L2_B,
                               row_bytes, row_bytes, block_rows, row_bytes);
                    }
                __end_shared();
            __end_plan(0u);
        __join();

        __split();
            __start_plan(0u);
                __start_thread(0u);
                    /* The reference softmax intentionally aliases A and R.
                     * This makes scale/add results immediately visible as the
                     * next operation's A input without an R-to-A transfer. */
                    __set_spm_addr(GGML_GTX_L1_BANK_A_OFFSET,
                                   GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET,
                                   GGML_GTX_L1_BANK_A_OFFSET);
                    __load(GGML_GTX_L2_A,
                           GGML_GTX_L1_BANK_A_OFFSET,
                           row_bytes, (uint16_t)row_bytes,
                           block_rows, row_bytes);
                    if (ddr_mask != 0u) {
                        __load(GGML_GTX_L2_B,
                               GGML_GTX_L1_BANK_B_OFFSET,
                               row_bytes, (uint16_t)row_bytes,
                               block_rows, row_bytes);
                    }

                    if (scale != 1.0f) {
                        __mul_vs((uint32_t)cols * block_rows, scale_f16, 0u);
                    }
                    if (ddr_mask != 0u) {
                        __add_vv((uint32_t)cols * block_rows);
                    }

                    for (uint16_t row = 0u; row < block_rows; ++row) {
                        const uint32_t l1_off = (uint32_t)row * row_bytes;
                        __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET + l1_off);
                        __set_spm_addr_R(GGML_GTX_L1_BANK_A_OFFSET + l1_off);
                        __max_vs(cols, 0xFBFF, 0x0, 0x0);
                        __esum(cols, 0xBEEF, 0xBEEF, 0x1, 0x180);
                        __softmax(cols, 0xBEEF, 0xBEEF, 0x184);
                    }
                    __set_spm_addr_A(GGML_GTX_L1_BANK_A_OFFSET);
                    __set_spm_addr_R(GGML_GTX_L1_BANK_A_OFFSET);
                    __store(GGML_GTX_L1_BANK_A_OFFSET,
                            GGML_GTX_L2_RESULT,
                            row_bytes, (uint16_t)row_bytes,
                            block_rows, row_bytes);
                __end_thread(0u);
            __end_plan(0u);
        __join();

        __split();
            __start_plan(0u);
                __start_shared();
                    __store(GGML_GTX_L2_RESULT,
                            ddr_result + byte_off,
                            row_bytes, row_bytes, block_rows, row_bytes);
                __end_shared();
            __end_plan(0u);
        __join();
    }
}
#endif

void ggml_gtx_compute_softmax_ext(
    uint64_t ddr_input,
    uint64_t ddr_mask,
    uint64_t ddr_result,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint32_t dst_nb1,
    uint32_t dst_nb2,
    uint32_t dst_nb3,
    uint16_t mask_ne2,
    uint16_t mask_ne3,
    uint32_t mask_nb1,
    uint32_t mask_nb2,
    uint32_t mask_nb3,
    float scale,
    float max_bias)
{
    volatile uint16_t * src = (volatile uint16_t *)(uintptr_t)ddr_input;
    volatile uint16_t * dst = (volatile uint16_t *)(uintptr_t)ddr_result;
    volatile uint16_t * mask = (volatile uint16_t *)(uintptr_t)ddr_mask;

    const uint32_t src_nb1 = dst_nb1;
    const uint32_t src_nb2 = dst_nb2;
    const uint32_t src_nb3 = dst_nb3;
    const uint16_t safe_mask_ne2 = mask_ne2 == 0 ? 1 : mask_ne2;
    const uint16_t safe_mask_ne3 = mask_ne3 == 0 ? 1 : mask_ne3;

#ifdef GGML_GTX_USE_GTX
    if (max_bias == 0.0f && ne0 > 0u &&
        (uint32_t)ne0 * sizeof(uint16_t) <= GGML_GTX_SOFTMAX_L1_B_BYTES) {
        for (uint16_t i3 = 0; i3 < ne3; ++i3) {
            for (uint16_t i2 = 0; i2 < ne2; ++i2) {
                const uint64_t src_plane = ddr_input +
                    (uint64_t)i2 * src_nb2 + (uint64_t)i3 * src_nb3;
                const uint64_t dst_plane = ddr_result +
                    (uint64_t)i2 * dst_nb2 + (uint64_t)i3 * dst_nb3;
                uint64_t mask_plane = 0u;
                if (ddr_mask != 0u) {
                    const uint16_t mi2 = (uint16_t)(i2 % safe_mask_ne2);
                    const uint16_t mi3 = (uint16_t)(i3 % safe_mask_ne3);
                    mask_plane = ddr_mask +
                        (uint64_t)mi2 * mask_nb2 + (uint64_t)mi3 * mask_nb3;
                }
                ggml_gtx_compute_softmax_ext_npu_plane(
                    src_plane, mask_plane, dst_plane, ne1, ne0, scale);
            }
        }
        return;
    }
#endif

    for (uint16_t i3 = 0; i3 < ne3; ++i3) {
        for (uint16_t i2 = 0; i2 < ne2; ++i2) {
            for (uint16_t i1 = 0; i1 < ne1; ++i1) {
                const uint64_t row_off = ((uint64_t)i1 * src_nb1 +
                                          (uint64_t)i2 * src_nb2 +
                                          (uint64_t)i3 * src_nb3) / 2U;
                const uint64_t dst_off = ((uint64_t)i1 * dst_nb1 +
                                          (uint64_t)i2 * dst_nb2 +
                                          (uint64_t)i3 * dst_nb3) / 2U;

                uint64_t mask_off = 0;
                if (ddr_mask != 0) {
                    const uint16_t mi2 = (uint16_t)(i2 % safe_mask_ne2);
                    const uint16_t mi3 = (uint16_t)(i3 % safe_mask_ne3);
                    mask_off = ((uint64_t)i1 * mask_nb1 +
                                (uint64_t)mi2 * mask_nb2 +
                                (uint64_t)mi3 * mask_nb3) / 2U;
                }

                float row_max = -3.402823466e+38f;
                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    float v = ggml_gtx_softmax_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src[row_off + i0])) * scale;
                    if (ddr_mask != 0) {
                        v += ggml_gtx_softmax_bits_to_float(ggml_gtx_fp16_to_fp32_bits(mask[mask_off + i0]));
                    }
                    if (v > row_max) {
                        row_max = v;
                    }
                }

                float sum = 0.0f;
                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    float v = ggml_gtx_softmax_bits_to_float(ggml_gtx_fp16_to_fp32_bits(src[row_off + i0])) * scale;
                    if (ddr_mask != 0) {
                        v += ggml_gtx_softmax_bits_to_float(ggml_gtx_fp16_to_fp32_bits(mask[mask_off + i0]));
                    }
                    const float e = ggml_gtx_softmax_expf(v - row_max);
                    dst[dst_off + i0] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_softmax_float_to_bits(e));
                    sum += e;
                }

                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    const float e = ggml_gtx_softmax_bits_to_float(ggml_gtx_fp16_to_fp32_bits(dst[dst_off + i0]));
                    dst[dst_off + i0] = ggml_gtx_fp32_bits_to_fp16(ggml_gtx_softmax_float_to_bits(e * inv_sum));
                }
            }
        }
    }

    (void)max_bias;
}
