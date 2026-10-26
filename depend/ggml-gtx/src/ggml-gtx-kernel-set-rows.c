//==================================================================
// GGML GTX kernel: SET_ROWS
//==================================================================

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#ifdef GGML_GTX_USE_GTX
#include "ggml-gtx-plan.h"
#include "ggml-gtx-config.h"
#include "ggml-gtx/intrinsics/intrin.h"
#endif

static int ggml_gtx_compute_set_rows_l2_n1(const ggml_gtx_kernel_launch_desc_t * desc) {
#ifdef GGML_GTX_USE_GTX
    if (desc == NULL ||
        desc->l2_src_addr[0] == 0u ||
        desc->l2_dst_addr == 0u ||
        desc->src_addr[1] == 0u ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        (desc->src_type[1] != GGML_GTX_TYPE_I32 && desc->src_type[1] != GGML_GTX_TYPE_I64) ||
        desc->ne[0] == 0u || desc->ne[1] == 0u ||
        desc->ne[2] == 0u || desc->ne[3] == 0u ||
        desc->src_ne[0][0] != desc->ne[0] ||
        desc->src_ne[0][1] == 0u ||
        desc->src_ne[0][2] != desc->ne[2] ||
        desc->src_ne[0][3] != desc->ne[3] ||
        desc->src_ne[0][1] > desc->ne[1]) {
        return -1;
    }

    const uint32_t row_bytes = desc->ne[0] * sizeof(uint16_t);
    if (row_bytes == 0u || row_bytes > UINT16_MAX) {
        return -2;
    }

    const uint32_t nr = desc->src_ne[0][1];
    const uint32_t ne2 = desc->ne[2];
    const uint32_t ne3 = desc->ne[3];
    const uint32_t n_vocab = desc->ne[1];
    const uint32_t ne_idx1 = desc->src_ne[1][1] == 0u ? 1u : desc->src_ne[1][1];
    const uint32_t ne_idx2 = desc->src_ne[1][2] == 0u ? 1u : desc->src_ne[1][2];
    const uint32_t idx_size = desc->src_type[1] == GGML_GTX_TYPE_I64 ? 8u : 4u;
    const uint64_t l2_src = desc->l2_src_addr[0];
    const uint64_t l2_dst = desc->l2_dst_addr;

    __start_plan(desc->nest_id);
        __start_shared();
            for (uint32_t i3 = 0; i3 < ne3; ++i3) {
                const uint32_t sel_batch = i3 % ne_idx2;
                for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                    const uint32_t sel_channel = i2 % ne_idx1;
                    for (uint32_t i = 0; i < nr; ++i) {
                        const uint64_t selected_index =
                            ((uint64_t)sel_batch * ne_idx1 + sel_channel) * nr + i;
                        const uint64_t row = idx_size == 8u
                            ? ((volatile const uint64_t *)(uintptr_t)desc->src_addr[1])[selected_index]
                            : ((volatile const uint32_t *)(uintptr_t)desc->src_addr[1])[selected_index];
                        if (row >= n_vocab) {
                            __end_shared();
                        __end_plan(desc->nest_id);
                            return -3;
                        }

                        const uint64_t src_off =
                            (((uint64_t)i3 * ne2 + i2) * nr + i) * row_bytes;
                        const uint64_t dst_off =
                            (((uint64_t)i3 * ne2 + i2) * n_vocab + row) * row_bytes;
                        __copy((uint32_t)(l2_src + src_off),
                               (uint32_t)(l2_dst + dst_off),
                               row_bytes, row_bytes, 1u, (uint16_t)row_bytes);
                    }
                }
            }
        __end_shared();
    __end_plan(desc->nest_id);
    return 0;
#else
    (void)desc;
    return -1;
#endif
}

int ggml_gtx_kernel_set_rows_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_SET_ROWS_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_SET_ROWS_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode == GGML_GTX_MEMORY_MODE_L2_STASHED_N1) {
        return ggml_gtx_compute_set_rows_l2_n1(desc);
    }

    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->src_addr[1] == 0 ||
        desc->src_addr[2] == 0) {
        return -2;
    }

    ggml_gtx_compute_set_rows(desc->src_addr[0],
                              desc->src_addr[1],
                              desc->src_addr[2],
                              desc->dst_addr,
                              (uint16_t) desc->src_ne[0][0],
                              (uint16_t) desc->src_ne[0][1],
                              (uint16_t) desc->ne[2],
                              (uint16_t) desc->ne[3],
                              (uint16_t) desc->ne[1],
                              (uint16_t) desc->src_ne[1][1],
                              (uint16_t) desc->src_ne[1][2],
                              desc->src_type[1] == GGML_GTX_TYPE_I64 ? 8 : 4);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_set_rows_f16(arg);
}

void ggml_gtx_compute_set_rows(
    uint64_t ddr_data,
    uint64_t ddr_idx,
    uint64_t ddr_src_base,
    uint64_t ddr_dst,
    uint16_t nc,
    uint16_t nr,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t n_vocab,
    uint16_t ne_idx1,
    uint16_t ne_idx2,
    uint16_t idx_size)
{
    const uint16_t dtype_size = 2;
    const uint32_t row_bytes = (uint32_t)nc * dtype_size;
    const uint32_t total_rows = (uint32_t)n_vocab * (uint32_t)ne2 * (uint32_t)ne3;
    if (ddr_src_base != ddr_dst) {
        __copy_mem(
            ddr_src_base,
            ddr_dst,
            row_bytes,
            row_bytes,
            total_rows,
            row_bytes,
            row_bytes >> 16);
    }

    for (uint16_t i3 = 0; i3 < ne3; i3++) {
        const uint16_t sel_batch = ne_idx2 != 0 ? (uint16_t)(i3 % ne_idx2) : 0;

        for (uint16_t i2 = 0; i2 < ne2; i2++) {
            const uint16_t sel_channel = ne_idx1 != 0 ? (uint16_t)(i2 % ne_idx1) : 0;

            for (uint16_t i = 0; i < nr; i++) {
                const uint16_t sel_height = i;
                const uint64_t selected_index = (uint64_t)((sel_batch * ne_idx1 + sel_channel) * nr + sel_height);
                const uint64_t row = (idx_size == 8)
                    ? ((volatile uint64_t *)(uintptr_t)ddr_idx)[selected_index]
                    : ((volatile uint32_t *)(uintptr_t)ddr_idx)[selected_index];

                __copy_mem(
                    ddr_data + (((uint64_t)i3 * ne2 + i2) * nr + i) * row_bytes,
                    ddr_dst + (((uint64_t)i3 * ne2 + i2) * n_vocab + row) * row_bytes,
                    row_bytes,
                    row_bytes,
                    1,
                    row_bytes,
                    row_bytes >> 16);
            }
        }
    }

}

//==================================================================
// OUT_PROD: Outer product (used in LoRA weight updates)
//
// src0: F32 [m, n, q0, r0]   (a)
// src1: F32 [p, n, q1, r1]   (b)
// dst:  F32 [m, p, q1, r1]
//
// Math: dst = src0 @ src1^T
// For each batch slice (i2, i3):
//   for i1 in [0, p):
//     for i01 in [0, n):
//       for i0 in [0, m):
//         dst[i0, i1, i2, i3] += src0[i0, i01, i02, i03] * src1[i1, i01, i2, i3]
//
// Broadcasting: src0 is broadcastable into src1 on dims 2,3
//   i02 = i2 / dps2, i03 = i3 / dps3
//==================================================================
