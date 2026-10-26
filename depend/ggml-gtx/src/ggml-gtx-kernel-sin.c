//==================================================================
// GGML GTX kernel: SIN
//
// Descriptor wrapper around the host-owned GTX sine core.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#include "ggml-gtx-kernel-sin-core.h"

int ggml_gtx_kernel_sin_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_SIN_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_SIN_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->l2_scratch_capacity < 0x40200u ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->nb[0] != SINCOS_DTYPE ||
        desc->src_nb[0][0] != SINCOS_DTYPE) {
        return -2;
    }

    const uint64_t elems64 = (uint64_t) desc->ne[0] *
                             (uint64_t) desc->ne[1] *
                             (uint64_t) desc->ne[2] *
                             (uint64_t) desc->ne[3];
    if (elems64 == 0 || (elems64 % 16u) != 0u) {
        return -3;
    }

    for (int i = 0; i < 4; ++i) {
        if (desc->src_ne[0][i] != desc->ne[i]) {
            return -4;
        }
    }
    if (desc->nb[1] != desc->ne[0] * SINCOS_DTYPE ||
        desc->src_nb[0][1] != desc->src_ne[0][0] * SINCOS_DTYPE ||
        (desc->ne[2] > 1 && desc->nb[2] != desc->nb[1] * desc->ne[1]) ||
        (desc->ne[3] > 1 && desc->nb[3] != desc->nb[2] * desc->ne[2]) ||
        (desc->src_ne[0][2] > 1 && desc->src_nb[0][2] != desc->src_nb[0][1] * desc->src_ne[0][1]) ||
        (desc->src_ne[0][3] > 1 && desc->src_nb[0][3] != desc->src_nb[0][2] * desc->src_ne[0][2])) {
        return -5;
    }

    for (uint64_t elem = 0; elem < elems64; elem += 16u) {
        const uint64_t off = elem * SINCOS_DTYPE;
        sincos_vec_l2(0, 0, desc->l2_src_addr[0] + off, desc->l2_dst_addr + off, 0,
                      desc->l2_scratch_addr + 0x20000u,
                      desc->l2_scratch_addr + 0x40000u);
    }

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_sin_f16(arg);
}
