#ifndef GGML_GTX_KERNEL_UNARY_REF_H
#define GGML_GTX_KERNEL_UNARY_REF_H

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_UNARY_BANK_A    GGML_GTX_L1_BANK_A_OFFSET
#define GGML_GTX_UNARY_BANK_B    GGML_GTX_L1_BANK_B_OFFSET
#define GGML_GTX_UNARY_BANK_C    GGML_GTX_L1_BANK_C_OFFSET
#define GGML_GTX_UNARY_BANK_R    GGML_GTX_L1_BANK_R_OFFSET
#define GGML_GTX_UNARY_DTYPE     2u
#define GGML_GTX_UNARY_SPUS      GGML_GTX_SPU_NUM_PER_NEST
#define GGML_GTX_UNARY_TOKEN     0xBEEFu

static inline uint32_t ggml_gtx_unary_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t ggml_gtx_unary_elems_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / GGML_GTX_UNARY_SPUS;
    const uint32_t r = total_elems % GGML_GTX_UNARY_SPUS;
    return tid < r ? q + 1u : q;
}

static inline uint32_t ggml_gtx_unary_start_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / GGML_GTX_UNARY_SPUS;
    const uint32_t r = total_elems % GGML_GTX_UNARY_SPUS;
    if (tid < r) {
        return (uint32_t) tid * (q + 1u);
    }
    return r * (q + 1u) + ((uint32_t) tid - r) * q;
}

static int GGML_GTX_UNARY_KERNEL_FN(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_UNARY_KERNEL_ID ||
        karg->desc.kernel_id != GGML_GTX_UNARY_KERNEL_ID) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint32_t elems = desc->ne[0];
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->nest_id >= GGML_GTX_NEST_NUM ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->elem_size != GGML_GTX_UNARY_DTYPE ||
        elems == 0 ||
        elems > UINT16_MAX) {
        return -2;
    }

    __split();
        __start_plan((uint8_t) desc->nest_id);
            for (uint8_t tid = 0; tid < GGML_GTX_UNARY_SPUS; tid++) {
                const uint32_t elems_this_tid = ggml_gtx_unary_elems_for_tid(elems, tid);
                __start_thread(tid);
                    if (elems_this_tid > 0u) {
                        const uint32_t elem_start = ggml_gtx_unary_start_for_tid(elems, tid);
                        const uint32_t chunk_off = elem_start * GGML_GTX_UNARY_DTYPE;
                        const uint32_t chunk_bytes = elems_this_tid * GGML_GTX_UNARY_DTYPE;
                        const uint16_t chunk_bytes_u16 = (uint16_t) chunk_bytes;

                        __load((uint32_t) desc->l2_src_addr[0] + chunk_off,
                               GGML_GTX_UNARY_INPUT_BANK,
                               chunk_bytes,
                               chunk_bytes_u16,
                               1u,
                               chunk_bytes_u16);
                        __set_spm_addr(GGML_GTX_UNARY_BANK_R, GGML_GTX_UNARY_BANK_C,
                                       GGML_GTX_UNARY_BANK_B, GGML_GTX_UNARY_BANK_A);
                        GGML_GTX_UNARY_APPLY(elems_this_tid);
                        __store(GGML_GTX_UNARY_OUTPUT_BANK,
                                (uint32_t) desc->l2_dst_addr + chunk_off,
                                chunk_bytes,
                                chunk_bytes_u16,
                                1u,
                                chunk_bytes_u16);
                    }
                __end_thread(tid);
            }
        __end_plan((uint8_t) desc->nest_id);
    __join();

    return 0;
}

int kernel_main(void * arg) {
    return GGML_GTX_UNARY_KERNEL_FN(arg);
}

#endif
