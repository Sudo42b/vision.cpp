//==================================================================
// GGML GTX kernel: GELU
//==================================================================
// Parameterized wrapper around reference/gelu.c.  The reference kernel
// structure is kept intact: load per-SPU vector tiles, run __gelu on each
// tile, and store the result.  Only fixed DDR addresses and compile-time
// tensor sizes are replaced by launch-descriptor parameters.

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_GELU_L2_INPUT       0x000000u
#define GGML_GTX_GELU_L2_RES         0x080000u
#define GGML_GTX_GELU_L2_BANK_OFFSET 0x100000u
#define GGML_GTX_GELU_L1_AR          0x00000u
#define GGML_GTX_GELU_NOT_USE        0xBEEFu

static inline uint32_t ggml_gtx_gelu_elems_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / GGML_GTX_SPU_NUM_PER_NEST;
    const uint32_t r = total_elems % GGML_GTX_SPU_NUM_PER_NEST;
    return tid < r ? q + 1u : q;
}

static inline uint32_t ggml_gtx_gelu_start_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / GGML_GTX_SPU_NUM_PER_NEST;
    const uint32_t r = total_elems % GGML_GTX_SPU_NUM_PER_NEST;
    if (tid < r) {
        return (uint32_t) tid * (q + 1u);
    }
    return r * (q + 1u) + ((uint32_t) tid - r) * q;
}

static uint16_t ggml_gtx_gelu_min_u16(uint64_t a, uint64_t b) {
    return (uint16_t) (a < b ? a : b);
}

void ggml_gtx_compute_gelu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    const uint32_t elem_count = (uint32_t) i_row * (uint32_t) i_col;
    if (elem_count == 0 || spu_num_per_nest == 0) {
        return;
    }

    const uint8_t nest_id = 0;
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint32_t max_chunk_elems = 65535u;

    (void) nest_num;

    for (uint32_t base_elem = 0; base_elem < elem_count; base_elem += max_chunk_elems) {
        const uint32_t chunk_elems = (elem_count - base_elem) < max_chunk_elems ?
            (elem_count - base_elem) : max_chunk_elems;
        const uint32_t cal_quot = chunk_elems / spu_num_per_nest;
        const uint32_t cal_mod  = chunk_elems % spu_num_per_nest;
        uint32_t bank_offset_shared = 0;
        uint32_t bank_offset_thread = 0;

        __split();

            __start_plani(nest_id);

                __set_spm_addr_A(GGML_GTX_GELU_L1_AR);
                __set_spm_addr_R(GGML_GTX_GELU_L1_AR);

                __start_shared();

                    for (uint8_t thread_id = 0; thread_id < spu_num_per_nest; thread_id++) {
                        const uint16_t cal_num_per_thread = (uint16_t) (thread_id < cal_mod ? (cal_quot + 1u) : cal_quot);
                        const uint32_t offset_thread =
                            (((thread_id < cal_mod) ?
                                  (uint32_t) thread_id * (cal_quot + 1u) :
                                  (uint32_t) cal_mod * (cal_quot + 1u) + (uint32_t) (thread_id - cal_mod) * cal_quot) *
                             dtype_size);

                        if (cal_num_per_thread > 0) {
                            __load(ddr_input + (uint64_t) base_elem * dtype_size + offset_thread,
                                   GGML_GTX_GELU_L2_INPUT + bank_offset_shared,
                                   cal_num_per_thread,
                                   cal_num_per_thread,
                                   dtype_size,
                                   cal_num_per_thread);
                            __credit_ld((uint16_t)(0x1u << thread_id), GGML_GTX_GELU_NOT_USE);
                        }

                        bank_offset_shared += GGML_GTX_GELU_L2_BANK_OFFSET;
                    }

                    bank_offset_shared = 0;

                    for (uint8_t thread_id = 0; thread_id < spu_num_per_nest; thread_id++) {
                        const uint16_t cal_num_per_thread = (uint16_t) (thread_id < cal_mod ? (cal_quot + 1u) : cal_quot);
                        const uint16_t target = (uint16_t)(0x1u << thread_id);
                        const uint32_t offset_thread =
                            (((thread_id < cal_mod) ?
                                  (uint32_t) thread_id * (cal_quot + 1u) :
                                  (uint32_t) cal_mod * (cal_quot + 1u) + (uint32_t) (thread_id - cal_mod) * cal_quot) *
                             dtype_size);

                        if (cal_num_per_thread > 0) {
                            __credit_chk(target);
                            __store(GGML_GTX_GELU_L2_RES + bank_offset_shared,
                                    ddr_result + (uint64_t) base_elem * dtype_size + offset_thread,
                                    cal_num_per_thread,
                                    cal_num_per_thread,
                                    dtype_size,
                                    cal_num_per_thread);
                            __credit_st(target);
                        }

                        bank_offset_shared += GGML_GTX_GELU_L2_BANK_OFFSET;
                    }

                __end_shared();

                for (uint8_t thread_id = 0; thread_id < spu_num_per_nest; thread_id++) {
                    const uint16_t cal_num_per_thread = (uint16_t) (thread_id < cal_mod ? (cal_quot + 1u) : cal_quot);

                    __start_thread(thread_id);

                        if (cal_num_per_thread > 0) {
                            const uint32_t l2_input_thread = GGML_GTX_GELU_L2_INPUT + bank_offset_thread;
                            const uint32_t l2_res_thread   = GGML_GTX_GELU_L2_RES   + bank_offset_thread;

                            __credit_chk(GGML_GTX_GELU_NOT_USE);
                            __load(l2_input_thread,
                                   GGML_GTX_GELU_L1_AR,
                                   cal_num_per_thread,
                                   cal_num_per_thread,
                                   dtype_size,
                                   GGML_GTX_GELU_NOT_USE);
                            __credit_ld(GGML_GTX_GELU_NOT_USE, GGML_GTX_GELU_NOT_USE);

                            __gelu(cal_num_per_thread);

                            __store(GGML_GTX_GELU_L1_AR,
                                    l2_res_thread,
                                    cal_num_per_thread,
                                    cal_num_per_thread,
                                    dtype_size,
                                    GGML_GTX_GELU_NOT_USE);
                            __credit_st(GGML_GTX_GELU_NOT_USE);
                        }

                        bank_offset_thread += GGML_GTX_GELU_L2_BANK_OFFSET;

                    __end_thread(thread_id);
                }

            __end_plani(nest_id);

        __join();
    }
#else
    (void) ddr_input;
    (void) ddr_result;
    (void) i_row;
    (void) i_col;
    (void) nest_num;
    (void) spu_num_per_nest;
#endif
}

int ggml_gtx_kernel_gelu_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_GELU_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_GELU_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint32_t elems = desc->ne[0];
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->nest_id >= GGML_GTX_NEST_NUM ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->elem_size != GGML_GTX_DTYPE_SIZE ||
        elems == 0 ||
        elems > UINT16_MAX) {
        return -2;
    }

    __split();
        __start_plan((uint8_t) desc->nest_id);
            __set_spm_addr_A(GGML_GTX_GELU_L1_AR);
            __set_spm_addr_R(GGML_GTX_GELU_L1_AR);

            for (uint8_t thread_id = 0; thread_id < GGML_GTX_SPU_NUM_PER_NEST; thread_id++) {
                const uint32_t elems_this_tid = ggml_gtx_gelu_elems_for_tid(elems, thread_id);

                __start_thread(thread_id);
                    if (elems_this_tid > 0u) {
                        const uint32_t elem_start = ggml_gtx_gelu_start_for_tid(elems, thread_id);
                        const uint32_t max_l1_elems = (UINT16_MAX / GGML_GTX_DTYPE_SIZE) & ~15u;
                        uint32_t local = 0u;

                        while (local < elems_this_tid) {
                            const uint32_t remaining = elems_this_tid - local;
                            const uint32_t chunk_elems = remaining < max_l1_elems ? remaining : max_l1_elems;
                            const uint32_t chunk_off = (elem_start + local) * GGML_GTX_DTYPE_SIZE;
                            const uint16_t chunk_bytes = (uint16_t) (chunk_elems * GGML_GTX_DTYPE_SIZE);

                            __load((uint32_t) desc->l2_src_addr[0] + chunk_off,
                                   GGML_GTX_GELU_L1_AR,
                                   chunk_bytes,
                                   chunk_bytes,
                                   1u,
                                   chunk_bytes);

                            __gelu(chunk_elems);

                            __store(GGML_GTX_GELU_L1_AR,
                                    (uint32_t) desc->l2_dst_addr + chunk_off,
                                    chunk_bytes,
                                    chunk_bytes,
                                    1u,
                                    chunk_bytes);
                            local += chunk_elems;
                        }
                    }
                __end_thread(thread_id);
            }
        __end_plan((uint8_t) desc->nest_id);
    __join();

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_gelu_f16(arg);
}

void ggml_gtx_printf(const char * fmt, ...) {
    (void) fmt;
}
