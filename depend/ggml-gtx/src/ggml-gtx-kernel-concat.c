//==================================================================
// GGML GTX kernel: CONCAT
//==================================================================

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

int ggml_gtx_kernel_concat_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CONCAT_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CONCAT_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const int32_t dim = desc->op_params[0];
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->l2_src_addr[1] == 0 ||
        dim < 0 ||
        dim >= 4) {
        return -2;
    }

    for (int i = 0; i < 4; ++i) {
        if (desc->ne[i] > UINT16_MAX ||
            desc->src_ne[0][i] > UINT16_MAX ||
            desc->src_ne[1][i] > UINT16_MAX) {
            return -3;
        }
        const uint32_t expected = i == dim ?
            desc->src_ne[0][i] + desc->src_ne[1][i] : desc->src_ne[0][i];
        if (desc->ne[i] != expected ||
            (i != dim && desc->src_ne[0][i] != desc->src_ne[1][i])) {
            return -4;
        }
    }
    const uint64_t dst_rows = (uint64_t)desc->ne[1] * desc->ne[2] * desc->ne[3];
    const uint64_t a_rows = (uint64_t)desc->src_ne[0][1] * desc->src_ne[0][2] * desc->src_ne[0][3];
    const uint64_t b_rows = (uint64_t)desc->src_ne[1][1] * desc->src_ne[1][2] * desc->src_ne[1][3];
    if (desc->ne[0] > UINT16_MAX / GGML_GTX_DTYPE_SIZE ||
        desc->src_ne[0][0] > UINT16_MAX / GGML_GTX_DTYPE_SIZE ||
        desc->src_ne[1][0] > UINT16_MAX / GGML_GTX_DTYPE_SIZE ||
        dst_rows > UINT16_MAX || a_rows > UINT16_MAX || b_rows > UINT16_MAX) {
        return -5;
    }

    ggml_gtx_compute_concat(desc->l2_src_addr[0],
                            desc->l2_src_addr[1],
                            desc->l2_dst_addr,
                            (uint16_t) desc->ne[0],
                            (uint16_t) desc->ne[1],
                            (uint16_t) desc->ne[2],
                            (uint16_t) desc->ne[3],
                            (uint16_t) desc->src_ne[0][dim],
                            dim,
                            GGML_GTX_DTYPE_SIZE,
                            1,
                            1);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_concat_f16(arg);
}

void ggml_gtx_compute_concat(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t a_ne_dim,
    int32_t dim,
    uint16_t elem_size,
    uint8_t nest_num,
    uint8_t spu_num_per_nest)
{
    (void) nest_num;
    (void) spu_num_per_nest;

    const uint32_t b_ne_dim = ((const uint32_t[]){ne0, ne1, ne2, ne3})[dim] - a_ne_dim;
    const uint32_t a_ne[4] = { ne0, ne1, ne2, ne3 };
    const uint32_t b_ne[4] = { ne0, ne1, ne2, ne3 };
    uint32_t a_shape[4] = { a_ne[0], a_ne[1], a_ne[2], a_ne[3] };
    uint32_t b_shape[4] = { b_ne[0], b_ne[1], b_ne[2], b_ne[3] };
    a_shape[dim] = a_ne_dim;
    b_shape[dim] = b_ne_dim;

#ifdef GGML_GTX_USE_GTX
    const uint32_t dst_row_bytes = (uint32_t)ne0 * elem_size;
    const uint32_t a_row_bytes = a_shape[0] * elem_size;
    const uint32_t b_row_bytes = b_shape[0] * elem_size;

    __start_plan(0);
        __start_shared();
            if (dim == 0) {
                const uint16_t rows = (uint16_t)((uint32_t)ne1 * ne2 * ne3);
                __copy((uint32_t)ddr_a, (uint32_t)ddr_dst,
                       a_row_bytes, (uint16_t)a_row_bytes, rows, dst_row_bytes);
                __copy((uint32_t)ddr_b, (uint32_t)ddr_dst + a_row_bytes,
                       b_row_bytes, (uint16_t)b_row_bytes, rows, dst_row_bytes);
            } else if (dim == 1) {
                const uint32_t dst_group_bytes = (uint32_t)ne1 * dst_row_bytes;
                const uint32_t a_group_bytes = a_shape[1] * a_row_bytes;
                const uint32_t b_group_bytes = b_shape[1] * b_row_bytes;
                const uint32_t groups = (uint32_t)ne2 * ne3;
                for (uint32_t g = 0; g < groups; ++g) {
                    __copy((uint32_t)ddr_a + g * a_group_bytes,
                           (uint32_t)ddr_dst + g * dst_group_bytes,
                           a_row_bytes, (uint16_t)a_row_bytes,
                           (uint16_t)a_shape[1], dst_row_bytes);
                    __copy((uint32_t)ddr_b + g * b_group_bytes,
                           (uint32_t)ddr_dst + g * dst_group_bytes + a_group_bytes,
                           b_row_bytes, (uint16_t)b_row_bytes,
                           (uint16_t)b_shape[1], dst_row_bytes);
                }
            } else if (dim == 2) {
                const uint32_t dst_batch_bytes = (uint32_t)ne2 * ne1 * dst_row_bytes;
                const uint32_t a_batch_bytes = a_shape[2] * a_shape[1] * a_row_bytes;
                const uint32_t b_batch_bytes = b_shape[2] * b_shape[1] * b_row_bytes;
                for (uint32_t i3 = 0; i3 < ne3; ++i3) {
                    __copy((uint32_t)ddr_a + i3 * a_batch_bytes,
                           (uint32_t)ddr_dst + i3 * dst_batch_bytes,
                           a_row_bytes, (uint16_t)a_row_bytes,
                           (uint16_t)(a_shape[2] * a_shape[1]), dst_row_bytes);
                    __copy((uint32_t)ddr_b + i3 * b_batch_bytes,
                           (uint32_t)ddr_dst + i3 * dst_batch_bytes + a_batch_bytes,
                           b_row_bytes, (uint16_t)b_row_bytes,
                           (uint16_t)(b_shape[2] * b_shape[1]), dst_row_bytes);
                }
            } else {
                const uint32_t a_rows = a_shape[3] * a_shape[2] * a_shape[1];
                const uint32_t b_rows = b_shape[3] * b_shape[2] * b_shape[1];
                __copy((uint32_t)ddr_a, (uint32_t)ddr_dst,
                       a_row_bytes, (uint16_t)a_row_bytes,
                       (uint16_t)a_rows, dst_row_bytes);
                __copy((uint32_t)ddr_b, (uint32_t)ddr_dst + a_rows * dst_row_bytes,
                       b_row_bytes, (uint16_t)b_row_bytes,
                       (uint16_t)b_rows, dst_row_bytes);
            }
        __end_shared();
    __end_plan(0);
#else

    volatile const uint8_t * a = (volatile const uint8_t *)(uintptr_t)ddr_a;
    volatile const uint8_t * b = (volatile const uint8_t *)(uintptr_t)ddr_b;
    volatile uint8_t * dst = (volatile uint8_t *)(uintptr_t)ddr_dst;

    const uint64_t a_nb0 = elem_size;
    const uint64_t a_nb1 = (uint64_t) a_shape[0] * a_nb0;
    const uint64_t a_nb2 = (uint64_t) a_shape[1] * a_nb1;
    const uint64_t a_nb3 = (uint64_t) a_shape[2] * a_nb2;
    const uint64_t b_nb0 = elem_size;
    const uint64_t b_nb1 = (uint64_t) b_shape[0] * b_nb0;
    const uint64_t b_nb2 = (uint64_t) b_shape[1] * b_nb1;
    const uint64_t b_nb3 = (uint64_t) b_shape[2] * b_nb2;

    uint64_t dst_off = 0;
    for (uint16_t i3 = 0; i3 < ne3; ++i3) {
        for (uint16_t i2 = 0; i2 < ne2; ++i2) {
            for (uint16_t i1 = 0; i1 < ne1; ++i1) {
                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    uint32_t idx[4] = { i0, i1, i2, i3 };
                    const bool from_a = idx[dim] < a_ne_dim;
                    if (!from_a) {
                        idx[dim] -= a_ne_dim;
                    }

                    const uint64_t src_off =
                        (uint64_t) idx[3] * (from_a ? a_nb3 : b_nb3) +
                        (uint64_t) idx[2] * (from_a ? a_nb2 : b_nb2) +
                        (uint64_t) idx[1] * (from_a ? a_nb1 : b_nb1) +
                        (uint64_t) idx[0] * (from_a ? a_nb0 : b_nb0);
                    volatile const uint8_t * src = from_a ? a : b;
                    for (uint16_t eb = 0; eb < elem_size; ++eb) {
                        dst[dst_off + eb] = src[src_off + eb];
                    }
                    dst_off += elem_size;
                }
            }
        }
    }
#endif
}
