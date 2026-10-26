//==================================================================
// GGML GTX kernel: SCALE, L2-stashed single-nest tile
//
// dst[i] = src0[i] * scale + bias
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

static void ggml_gtx_compute_scale_l2_n1(
    uint32_t l2_a,
    uint32_t l2_c,
    uint16_t elems,
    uint16_t scale,
    uint16_t bias,
    uint8_t  nest_id,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint16_t not_use = 0xBEEF;
    uint16_t spu_count = elems < spu_num_per_nest ? elems : (uint16_t) spu_num_per_nest;
    if (spu_count == 0) {
        spu_count = 1;
    }

    uint16_t elems_per_spu[GGML_GTX_SPU_NUM_PER_NEST];
    const uint16_t elem_quot = (uint16_t) (elems / spu_count);
    const uint16_t elem_mod = (uint16_t) (elems % spu_count);
    for (uint16_t thread_id = 0; thread_id < GGML_GTX_SPU_NUM_PER_NEST; ++thread_id) {
        elems_per_spu[thread_id] =
            thread_id < spu_count ? (uint16_t) (elem_quot + (thread_id < elem_mod ? 1 : 0)) : 0;
    }

    uint32_t elem_offset = 0;

    __split();

        __start_plan(nest_id);

            __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET,
                           GGML_GTX_L1_BANK_C_OFFSET,
                           GGML_GTX_L1_BANK_B_OFFSET,
                           GGML_GTX_L1_BANK_A_OFFSET);

            for (uint16_t thread_id = 0; thread_id < spu_count; ++thread_id) {
                const uint16_t thread_elems = elems_per_spu[thread_id];
                const uint32_t byte_offset = elem_offset * dtype_size;

                __start_thread(thread_id);

                    __load(l2_a + byte_offset,
                           GGML_GTX_L1_BANK_A_OFFSET,
                           thread_elems,
                           thread_elems,
                           dtype_size,
                           not_use);

                    __fmadd_vss(thread_elems, scale, bias, 0u);

                    __store(GGML_GTX_L1_BANK_R_OFFSET,
                            l2_c + byte_offset,
                            thread_elems,
                            thread_elems,
                            dtype_size,
                            not_use);

                __end_thread(thread_id);

                elem_offset += thread_elems;
            }

        __end_plan(nest_id);

    __join();
#else
    (void) l2_a;
    (void) l2_c;
    (void) elems;
    (void) scale;
    (void) bias;
    (void) nest_id;
    (void) spu_num_per_nest;
#endif
}

int ggml_gtx_kernel_scale_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_SCALE_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_SCALE_F16) {
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

    const uint16_t scale = ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[0]);
    const uint16_t bias  = ggml_gtx_fp32_bits_to_fp16((uint32_t) desc->op_params[1]);
    ggml_gtx_compute_scale_l2_n1(
        (uint32_t) desc->l2_src_addr[0],
        (uint32_t) desc->l2_dst_addr,
        (uint16_t) elems,
        scale,
        bias,
        (uint8_t) desc->nest_id,
        GGML_GTX_SPU_NUM_PER_NEST
    );
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_scale_f16(arg);
}
