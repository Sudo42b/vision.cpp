//==================================================================
// GGML GTX kernel: PAD
//
// L2-only right-side zero padding for a host-lowered raw view.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

int ggml_gtx_kernel_pad_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_PAD_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_PAD_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16) {
        return -2;
    }

    const uint32_t src_cols = desc->src_ne[0][0];
    const uint32_t src_rows = desc->src_ne[0][1];
    const uint32_t src_channels = desc->src_ne[0][2] ? desc->src_ne[0][2] : 1u;
    const uint32_t dst_cols = desc->ne[0];
    const uint32_t dst_rows = desc->ne[1];
    const uint32_t dst_channels = desc->ne[2] ? desc->ne[2] : 1u;
    const uint32_t batches = desc->ne[3] ? desc->ne[3] : 1u;

    const int32_t lp0 = desc->op_params[0];
    const int32_t rp0 = desc->op_params[1];
    const int32_t lp1 = desc->op_params[2];
    const int32_t rp1 = desc->op_params[3];
    const int32_t lp2 = desc->op_params[4];
    const int32_t rp2 = desc->op_params[5];
    const int32_t lp3 = desc->op_params[6];
    const int32_t rp3 = desc->op_params[7];
    const int32_t circular = desc->op_params[8];

    if (src_cols == 0 || src_rows == 0 || src_channels == 0 ||
        dst_cols == 0 || dst_rows == 0 || dst_channels == 0 ||
        lp0 != 0 || lp1 != 0 || lp2 != 0 || lp3 != 0 ||
        rp0 < 0 || rp1 < 0 || rp2 < 0 || rp3 != 0 ||
        circular != 0 ||
        dst_cols != src_cols + (uint32_t) rp0 ||
        dst_rows != src_rows + (uint32_t) rp1 ||
        dst_channels != src_channels + (uint32_t) rp2 ||
        batches != (desc->src_ne[0][3] ? desc->src_ne[0][3] : 1u) ||
        desc->src_nb[0][0] != GGML_GTX_DTYPE_SIZE ||
        desc->nb[0] != GGML_GTX_DTYPE_SIZE) {
        return -3;
    }

    const uint32_t src_row_bytes = src_cols * sizeof(uint16_t);
    const uint32_t dst_row_bytes = dst_cols * sizeof(uint16_t);
    const uint64_t dst_rows_total = (uint64_t)dst_rows * dst_channels * batches;
    if (src_row_bytes > UINT16_MAX || dst_row_bytes > UINT16_MAX ||
        dst_rows_total == 0 || dst_rows_total > UINT16_MAX) {
        return -4;
    }

    __start_plan(0);
        __start_shared();
            __fill(desc->l2_dst_addr, dst_row_bytes, (uint16_t)dst_row_bytes,
                   (uint16_t)dst_rows_total, 0u, 0);
            for (uint32_t b = 0; b < batches; ++b) {
                for (uint32_t c = 0; c < src_channels; ++c) {
                    const uint32_t src_plane = (b * src_channels + c) * src_rows * src_row_bytes;
                    const uint32_t dst_plane = (b * dst_channels + c) * dst_rows * dst_row_bytes;
                    __copy((uint32_t)desc->l2_src_addr[0] + src_plane,
                           (uint32_t)desc->l2_dst_addr + dst_plane,
                           src_row_bytes, (uint16_t)src_row_bytes,
                           (uint16_t)src_rows, dst_row_bytes);
                }
            }
        __end_shared();
    __end_plan(0);

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_pad_f16(arg);
}
