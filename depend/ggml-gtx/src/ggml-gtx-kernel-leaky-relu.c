//==================================================================
// GGML GTX kernel: LEAKY_RELU
//
// Descriptor wrapper around reference/leaky_relu.c.
// dst[i] = src0[i] >= 0 ? src0[i] : negative_slope * src0[i]
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define BANK_A               GGML_GTX_L1_BANK_A_OFFSET
#define BANK_B               GGML_GTX_L1_BANK_B_OFFSET
#define BANK_C               GGML_GTX_L1_BANK_C_OFFSET
#define BANK_R               GGML_GTX_L1_BANK_R_OFFSET
#define DTYPE                2u
#define SPU_NUM_PER_NEST     GGML_GTX_SPU_NUM_PER_NEST
#define NOT_USE              0xBEEFu
#define FP16_ZERO            0x0000u

static inline uint32_t elems_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / SPU_NUM_PER_NEST;
    const uint32_t r = total_elems % SPU_NUM_PER_NEST;
    return tid < r ? q + 1u : q;
}

static inline uint32_t start_for_tid(uint32_t total_elems, uint8_t tid) {
    const uint32_t q = total_elems / SPU_NUM_PER_NEST;
    const uint32_t r = total_elems % SPU_NUM_PER_NEST;
    if (tid < r) {
        return (uint32_t) tid * (q + 1u);
    }
    return r * (q + 1u) + ((uint32_t) tid - r) * q;
}

int ggml_gtx_kernel_leaky_relu_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_LEAKY_RELU_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_LEAKY_RELU_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint32_t total_elems = desc->ne[0];
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->nest_id >= GGML_GTX_NEST_NUM ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->elem_size != DTYPE ||
        total_elems == 0 ||
        total_elems > UINT16_MAX) {
        return -2;
    }

    const uint16_t negative_slope = ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[0]);

    __split();
        __start_plan((uint8_t) desc->nest_id);
            for (uint8_t tid = 0; tid < SPU_NUM_PER_NEST; tid++) {
                const uint32_t elems_this_tid = elems_for_tid(total_elems, tid);
                __start_thread(tid);
                    if (elems_this_tid > 0u) {
                        const uint32_t elem_start = start_for_tid(total_elems, tid);
                        const uint32_t chunk_off = elem_start * DTYPE;
                        const uint32_t chunk_bytes = elems_this_tid * DTYPE;
                        const uint16_t chunk_bytes_u16 = (uint16_t) chunk_bytes;

                        __load((uint32_t) desc->l2_src_addr[0] + chunk_off, BANK_A,
                            chunk_bytes, chunk_bytes_u16, 1u, chunk_bytes_u16);

                        __set_spm_addr(BANK_C, BANK_C, BANK_B, BANK_A);
                        __clamp_min(elems_this_tid, FP16_ZERO, 0u);

                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_A);
                        __clamp_max(elems_this_tid, FP16_ZERO, 0u);

                        __set_spm_addr(BANK_R, BANK_C, BANK_B, BANK_R);
                        __mul_vs(elems_this_tid, negative_slope, 0u);

                        __set_spm_addr(BANK_R, BANK_C, BANK_C, BANK_R);
                        __add_vv(elems_this_tid);

                        __store(BANK_R, (uint32_t) desc->l2_dst_addr + chunk_off,
                            chunk_bytes, chunk_bytes_u16, 1u, chunk_bytes_u16);
                    }
                __end_thread(tid);
            }
        __end_plan((uint8_t) desc->nest_id);
    __join();

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_leaky_relu_f16(arg);
}
