#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-plan.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#ifdef GGML_GTX_USE_GTX
#include "ggml-gtx/intrinsics/intrin.h"
#endif

int ggml_gtx_kernel_swiglu_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_SWIGLU_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_SWIGLU_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint64_t rows64 = (uint64_t) desc->src_ne[0][1] *
                            (uint64_t) desc->src_ne[0][2] *
                            (uint64_t) desc->src_ne[0][3];
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->src_ne[0][0] == 0 ||
        rows64 == 0 ||
        desc->src_ne[0][0] > UINT16_MAX) {
        return -2;
    }

    if (desc->memory_mode == GGML_GTX_MEMORY_MODE_L2_STASHED_N1) {
        const uint32_t n_cols = desc->src_ne[0][0];
        const uint32_t half = n_cols / 2u;
        const uint64_t half_bytes = (uint64_t)half * GGML_GTX_DTYPE_SIZE;
        if (desc->nest_id >= GGML_GTX_NEST_NUM ||
            desc->dtype != GGML_GTX_TYPE_F16 ||
            desc->src_type[0] != GGML_GTX_TYPE_F16 ||
            desc->src_addr[1] != 0 ||
            desc->l2_src_addr[0] == 0 ||
            desc->l2_dst_addr == 0 ||
            desc->elem_size != GGML_GTX_DTYPE_SIZE ||
            (n_cols & 1u) != 0u ||
            half == 0u ||
            desc->ne[0] != half ||
            desc->ne[1] != desc->src_ne[0][1] ||
            desc->ne[2] != desc->src_ne[0][2] ||
            desc->ne[3] != desc->src_ne[0][3] ||
            half > UINT16_MAX ||
            half_bytes > GGML_GTX_L1_BANK_SIZE ||
            rows64 > UINT16_MAX) {
            return -3;
        }

        ggml_gtx_compute_swiglu_l2((uint32_t)desc->l2_src_addr[0],
                                   (uint32_t)desc->l2_dst_addr,
                                   (uint16_t)rows64,
                                   (uint16_t)n_cols,
                                   (uint8_t)desc->op_params[1],
                                   (uint8_t)desc->nest_id,
                                   GGML_GTX_SPU_NUM_PER_NEST);
        return 0;
    }

    if (rows64 > UINT16_MAX) {
        return -4;
    }

    ggml_gtx_compute_swiglu(desc->src_addr[0],
                            desc->src_addr[1],
                            desc->dst_addr,
                            (uint16_t) rows64,
                            (uint16_t) desc->src_ne[0][0],
                            (uint8_t) desc->op_params[1],
                            GGML_GTX_NEST_NUM,
                            GGML_GTX_SPU_NUM_PER_NEST);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_swiglu_f16(arg);
}

//==================================================================
// GLU variants: SWIGLU, GEGLU, REGLU
//
// Single-tensor form: split row in half.
//   If NOT swapped:  input = first half,  gate = second half
//   If swapped:      gate  = first half,  input = second half
//
// Formula: activation(input) * gate
//   SWIGLU: SiLU(input) * gate
//   GEGLU:  GELU(input) * gate
//   REGLU:  ReLU(input) * gate
//
// Input:  [n_cols, n_rows, ...]  FP16 (n_cols = 2 * output_width)
// Output: [n_cols/2, n_rows, ...] FP16
//
// Strategy: Load input half → Bank A, apply activation → Bank R,
//           Load gate half → Bank B, __mul_vv → R = activation * gate.
//           Uses bank remapping for multi-step within single tile.
//
// For now: uses per-row GTX ISA processing. Each row processed as:
//   1. Load input_half (n_cols/2 elements) → L2 INPUT
//   2. Load gate_half  (n_cols/2 elements) → L2 second area
//   3. Thread: load input → A, apply activation → R, remap, load gate → B
//      → mul_vv → R, store R
//
// Actually, simplest correct approach: process per-row via a custom
// tile loop using single NEST/SPU, loading both halves.
//==================================================================

#define GLU_TYPE_SWIGLU  0
#define GLU_TYPE_GEGLU   1
#define GLU_TYPE_REGLU   2

#ifndef GGML_GTX_UNUSED
#define GGML_GTX_UNUSED(x) (void)(x)
#endif

static float ggml_gtx_glu_bare_expf(float x) {
    float term = 1.0f;
    float sum = 1.0f;

    for (int i = 1; i <= 10; ++i) {
        term *= x / (float) i;
        sum += term;
    }

    return sum;
}

static void ggml_gtx_compute_swiglu_packed(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t n_rows,
    uint16_t n_cols,
    uint8_t  swapped,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    GGML_GTX_UNUSED(nest_num);
    const uint16_t not_use = 0xBEEF;
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint32_t l1_bank_size = 64U * 1024U * 2U;
    const uint32_t offset_l2_bank = 0x100000U;
    const uint32_t base_l2_value = 0x000000U;
    const uint32_t base_l2_gate = 0x060000U;
    const uint32_t base_l2_result = 0x0C0000U;
    const uint32_t base_l1_a = 0x00000U;
    const uint32_t base_l1_b = 0x20000U;
    const uint32_t base_l1_r = 0x40000U;

    const uint32_t half = n_cols / 2U;
    const uint32_t cal_unit_origin_byte = n_cols * dtype_size;
    const uint32_t cal_unit = half;
    const uint32_t cal_unit_byte = cal_unit * dtype_size;
    const uint32_t cal_max = l1_bank_size / cal_unit_byte;
    const uint32_t cal_max_size = cal_max * cal_unit;
    const uint32_t offset_cal_max = cal_max * cal_unit_byte;
    const uint32_t cal_num = n_rows;
    const uint32_t eff_thread = cal_num > spu_num_per_nest ? spu_num_per_nest : cal_num;
    const uint32_t cal_quot = cal_num / eff_thread;
    const uint32_t cal_mod = cal_num % eff_thread;
    const uint32_t input_half_off = swapped ? cal_unit_byte : 0;
    const uint32_t gate_half_off = swapped ? 0 : cal_unit_byte;

    /*
     * Keep the DDR transfers and SPU work in separate plans.  The FPGA
     * runtime completes each split/join before the next stage starts, so
     * this does not depend on the shared/thread credit handshake used by
     * the original combined plan.
     */
    __split();
        __start_plani(0);
            __start_shared();
                for (uint8_t thread_id = 0; thread_id < eff_thread; ++thread_id) {
                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? cal_quot + 1U : cal_quot;
                    const uint32_t offset_load_thread =
                        ((thread_id < cal_mod ? thread_id : cal_mod) * (cal_quot + 1U) +
                         (thread_id > cal_mod ? (thread_id - cal_mod) : 0U) * cal_quot) * cal_unit_origin_byte;
                    const uint32_t bank_offset = (uint32_t)thread_id * offset_l2_bank;

                    __load(
                        ddr_input + offset_load_thread + input_half_off,
                        base_l2_value + bank_offset,
                        cal_unit_origin_byte,
                        cal_unit_byte,
                        cal_num_per_thread,
                        cal_unit_byte
                    );

                    __load(
                        ddr_input + offset_load_thread + gate_half_off,
                        base_l2_gate + bank_offset,
                        cal_unit_origin_byte,
                        cal_unit_byte,
                        cal_num_per_thread,
                        cal_unit_byte
                    );
                }
            __end_shared();
        __end_plani(0);
    __join();

    __split();
        __start_plani(0);
            for (uint8_t thread_id = 0; thread_id < eff_thread; ++thread_id) {
                __start_thread(thread_id);
                    __set_spm_addr_A(base_l1_a);
                    __set_spm_addr_R(base_l1_r);

                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? cal_quot + 1U : cal_quot;
                    const uint32_t sequence_num_per_thread = (cal_num_per_thread + cal_max - 1U) / cal_max;
                    const uint32_t iteration = sequence_num_per_thread - 1U;
                    const uint32_t cal_last = (cal_num_per_thread % cal_max) ? (cal_num_per_thread % cal_max) : cal_max;
                    const uint32_t bank_offset = (uint32_t)thread_id * offset_l2_bank;
                    const uint32_t l2_value_thread = base_l2_value + bank_offset;
                    const uint32_t l2_gate_thread = base_l2_gate + bank_offset;
                    const uint32_t l2_result_thread = base_l2_result + bank_offset;
                    uint32_t offset_cal = 0;

                    for (uint32_t iter = 0; iter < iteration; ++iter) {
                        __load(l2_value_thread + offset_cal, base_l1_r,
                               cal_unit_byte, cal_unit_byte, cal_max, not_use);
                        __load(l2_gate_thread + offset_cal, base_l1_b,
                               cal_unit_byte, cal_unit_byte, cal_max, not_use);

                        __sigm(cal_max_size);
                        __set_spm_addr_B(base_l1_r);
                        __mul_vv(cal_max_size);
                        __set_spm_addr_A(base_l1_b);
                        __mul_vv(cal_max_size);
                        __set_spm_addr_A(base_l1_a);
                        __set_spm_addr_B(base_l1_b);

                        __store(base_l1_r, l2_result_thread + offset_cal,
                                cal_unit_byte, cal_unit_byte, cal_max, not_use);

                        offset_cal += offset_cal_max;
                    }

                    __load(l2_value_thread + offset_cal, base_l1_r,
                           cal_unit_byte, cal_unit_byte, cal_last, not_use);
                    __load(l2_gate_thread + offset_cal, base_l1_b,
                           cal_unit_byte, cal_unit_byte, cal_last, not_use);

                    __sigm(cal_unit * cal_last);
                    __set_spm_addr_B(base_l1_r);
                    __mul_vv(cal_unit * cal_last);
                    __set_spm_addr_A(base_l1_b);
                    __mul_vv(cal_unit * cal_last);
                    __set_spm_addr_A(base_l1_a);
                    __set_spm_addr_B(base_l1_b);

                    __store(base_l1_r, l2_result_thread + offset_cal,
                            cal_unit_byte, cal_unit_byte, cal_last, not_use);
                __end_thread(thread_id);
            }
        __end_plani(0);
    __join();

    __split();
        __start_plani(0);
            __start_shared();
                for (uint8_t thread_id = 0; thread_id < eff_thread; ++thread_id) {
                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? cal_quot + 1U : cal_quot;
                    const uint32_t offset_store_thread =
                        ((thread_id < cal_mod ? thread_id : cal_mod) * (cal_quot + 1U) +
                         (thread_id > cal_mod ? (thread_id - cal_mod) : 0U) * cal_quot) * cal_unit_byte;
                    const uint32_t bank_offset = (uint32_t)thread_id * offset_l2_bank;

                    __store(
                        base_l2_result + bank_offset,
                        ddr_result + offset_store_thread,
                        cal_unit_byte,
                        cal_unit_byte,
                        cal_num_per_thread,
                        cal_unit_byte
                    );
                }
            __end_shared();
        __end_plani(0);
    __join();
#else
    GGML_GTX_UNUSED(nest_num);
    GGML_GTX_UNUSED(spu_num_per_nest);

    volatile uint32_t * src = (volatile uint32_t *)(uintptr_t)ddr_input;
    volatile uint32_t * dst = (volatile uint32_t *)(uintptr_t)ddr_result;
    uint16_t half = n_cols / 2;

    for (uint16_t row = 0; row < n_rows; ++row) {
        volatile uint32_t * row_base = src + (uint32_t)row * n_cols;
        volatile uint32_t * x_ptr = swapped ? row_base + half : row_base;
        volatile uint32_t * g_ptr = swapped ? row_base : row_base + half;
        volatile uint32_t * d_ptr = dst + (uint32_t)row * half;

        for (uint16_t k = 0; k < half; ++k) {
            union { uint32_t u; float f; } vx, vg, vo;
            vx.u = x_ptr[k];
            vg.u = g_ptr[k];
            float sig = 1.0f / (1.0f + ggml_gtx_glu_bare_expf(-vx.f));
            vo.f = (vx.f * sig) * vg.f;
            d_ptr[k] = vo.u;
        }
    }
#endif
}

void ggml_gtx_compute_swiglu_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t n_rows,
    uint16_t n_cols,
    uint8_t  swapped,
    uint8_t  nest_id,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    if (n_rows == 0 || n_cols == 0 || (n_cols & 1U) != 0U || spu_num_per_nest == 0) {
        return;
    }

    const uint16_t not_use = 0xBEEF;
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint32_t l1_bank_size = 64U * 1024U * 2U;
    const uint32_t base_l1_a = 0x00000U;
    const uint32_t base_l1_b = 0x20000U;
    const uint32_t base_l1_r = 0x40000U;

    const uint32_t half = n_cols / 2U;
    const uint32_t input_row_bytes = n_cols * dtype_size;
    const uint32_t output_row_bytes = half * dtype_size;
    const uint32_t cal_unit = half;
    const uint32_t cal_unit_byte = output_row_bytes;
    const uint32_t cal_max = l1_bank_size / cal_unit_byte;
    if (cal_max == 0U) {
        return;
    }

    const uint32_t cal_max_size = cal_max * cal_unit;
    const uint32_t input_offset_cal_max = cal_max * input_row_bytes;
    const uint32_t output_offset_cal_max = cal_max * output_row_bytes;
    const uint32_t cal_num = n_rows;
    const uint32_t eff_thread = cal_num > spu_num_per_nest ? spu_num_per_nest : cal_num;
    const uint32_t cal_quot = cal_num / eff_thread;
    const uint32_t cal_mod = cal_num % eff_thread;
    const uint32_t input_half_off = swapped ? output_row_bytes : 0U;
    const uint32_t gate_half_off = swapped ? 0U : output_row_bytes;

    __split();

        __start_plan(nest_id);

            for (uint8_t thread_id = 0; thread_id < eff_thread; ++thread_id) {

                __start_thread(thread_id);

                    __set_spm_addr_A(base_l1_a);
                    __set_spm_addr_B(base_l1_b);
                    __set_spm_addr_R(base_l1_r);

                    const uint32_t cal_num_per_thread = thread_id < cal_mod ? cal_quot + 1U : cal_quot;
                    const uint32_t start_row =
                        (thread_id < cal_mod ? thread_id : cal_mod) * (cal_quot + 1U) +
                        (thread_id > cal_mod ? (thread_id - cal_mod) : 0U) * cal_quot;
                    const uint32_t sequence_num_per_thread = (cal_num_per_thread + cal_max - 1U) / cal_max;
                    const uint32_t iteration = sequence_num_per_thread - 1U;
                    const uint32_t cal_last = (cal_num_per_thread % cal_max) ? (cal_num_per_thread % cal_max) : cal_max;
                    uint32_t input_offset_cal = start_row * input_row_bytes;
                    uint32_t output_offset_cal = start_row * output_row_bytes;

                    for (uint32_t iter = 0; iter < iteration; ++iter) {
                        __load(l2_input + input_offset_cal + input_half_off, base_l1_r,
                               input_row_bytes, cal_unit_byte, cal_max, not_use);
                        __load(l2_input + input_offset_cal + gate_half_off, base_l1_b,
                               input_row_bytes, cal_unit_byte, cal_max, not_use);

                        __sigm(cal_max_size);
                        __set_spm_addr_B(base_l1_r);
                        __mul_vv(cal_max_size);
                        __set_spm_addr_A(base_l1_b);
                        __mul_vv(cal_max_size);
                        __set_spm_addr_A(base_l1_a);
                        __set_spm_addr_B(base_l1_b);

                        __store(base_l1_r, l2_result + output_offset_cal,
                                cal_unit_byte, cal_unit_byte, cal_max, not_use);

                        input_offset_cal += input_offset_cal_max;
                        output_offset_cal += output_offset_cal_max;
                    }

                    __load(l2_input + input_offset_cal + input_half_off, base_l1_r,
                           input_row_bytes, cal_unit_byte, cal_last, not_use);
                    __load(l2_input + input_offset_cal + gate_half_off, base_l1_b,
                           input_row_bytes, cal_unit_byte, cal_last, not_use);

                    __sigm(cal_unit * cal_last);
                    __set_spm_addr_B(base_l1_r);
                    __mul_vv(cal_unit * cal_last);
                    __set_spm_addr_A(base_l1_b);
                    __mul_vv(cal_unit * cal_last);
                    __set_spm_addr_A(base_l1_a);
                    __set_spm_addr_B(base_l1_b);

                    __store(base_l1_r, l2_result + output_offset_cal,
                            cal_unit_byte, cal_unit_byte, cal_last, not_use);

                __end_thread(thread_id);
            }

        __end_plan(nest_id);

    __join();
#else
    GGML_GTX_UNUSED(l2_input);
    GGML_GTX_UNUSED(l2_result);
    GGML_GTX_UNUSED(n_rows);
    GGML_GTX_UNUSED(n_cols);
    GGML_GTX_UNUSED(swapped);
    GGML_GTX_UNUSED(nest_id);
    GGML_GTX_UNUSED(spu_num_per_nest);
#endif
}

void ggml_gtx_compute_swiglu(
    uint64_t ddr_input,
    uint64_t ddr_gate,
    uint64_t ddr_result,
    uint16_t n_rows,
    uint16_t n_cols,
    uint8_t  swapped,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
    if (ddr_gate == 0) {
        ggml_gtx_compute_swiglu_packed(
            ddr_input, ddr_result, n_rows, n_cols,
            swapped, nest_num, spu_num_per_nest);
        return;
    }

    const uint32_t half_bytes = (uint32_t)n_cols * 2;
    const uint32_t packed_row_bytes = half_bytes * 2;
    const uint32_t temp_bytes = 0x00200000U;
    const uint64_t ddr_packed = GGML_GTX_DDR_TEMP_0;
    const uint16_t max_rows_per_chunk =
        (uint16_t)(temp_bytes / packed_row_bytes);

    if (max_rows_per_chunk == 0 || n_cols > 32767U) {
        return;
    }

    for (uint32_t row_base = 0; row_base < n_rows; row_base += max_rows_per_chunk) {
        uint16_t rows_chunk = (uint16_t)((uint32_t)n_rows - row_base);
        if (rows_chunk > max_rows_per_chunk) {
            rows_chunk = max_rows_per_chunk;
        }

#ifdef GGML_GTX_USE_GTX
        for (uint16_t row = 0; row < rows_chunk; ++row) {
            const uint64_t absolute_row = row_base + row;
            const uint64_t src_row = ddr_input + absolute_row * half_bytes;
            const uint64_t gate_row = ddr_gate + absolute_row * half_bytes;
            const uint64_t packed_row = ddr_packed + (uint64_t)row * packed_row_bytes;

            __copy_mem(src_row,
                       packed_row,
                       half_bytes, half_bytes,
                       1, half_bytes, 0);
            __copy_mem(gate_row,
                       packed_row + half_bytes,
                       half_bytes, half_bytes,
                       1, half_bytes, 0);

        }
#else
        volatile uint16_t * packed = (volatile uint16_t *)(uintptr_t)ddr_packed;
        volatile uint16_t * src = (volatile uint16_t *)(uintptr_t)ddr_input;
        volatile uint16_t * gate = (volatile uint16_t *)(uintptr_t)ddr_gate;

        for (uint16_t row = 0; row < rows_chunk; ++row) {
            const uint32_t absolute_row = row_base + row;
            volatile uint16_t * packed_row = packed + (uint32_t)row * (uint32_t)(n_cols * 2);
            volatile uint16_t * src_row = src + absolute_row * n_cols;
            volatile uint16_t * gate_row = gate + absolute_row * n_cols;

            for (uint16_t col = 0; col < n_cols; ++col) {
                packed_row[col] = src_row[col];
                packed_row[n_cols + col] = gate_row[col];
            }
        }
#endif

        ggml_gtx_compute_swiglu_packed(
            ddr_packed,
            ddr_result + (uint64_t)row_base * half_bytes,
            rows_chunk,
            (uint16_t)(n_cols * 2),
            0, nest_num, spu_num_per_nest);
    }
}
