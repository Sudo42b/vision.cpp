//==================================================================
// GGML GTX kernel: CPY
//==================================================================

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

int ggml_gtx_kernel_cpy_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CPY_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CPY_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0) {
        return -2;
    }

    const uint64_t dst_elements = (uint64_t) desc->ne[0] *
                                  (uint64_t) desc->ne[1] *
                                  (uint64_t) desc->ne[2] *
                                  (uint64_t) desc->ne[3];
    const uint64_t src_elements = (uint64_t) desc->src_ne[0][0] *
                                  (uint64_t) desc->src_ne[0][1] *
                                  (uint64_t) desc->src_ne[0][2] *
                                  (uint64_t) desc->src_ne[0][3];
    if (dst_elements != src_elements) {
        return -3;
    }

    const uint16_t elem_size =
        desc->src_type[0] == GGML_GTX_TYPE_F32 ? 4 : 2;
    ggml_gtx_compute_cpy(desc->src_addr[0],
                         desc->dst_addr,
                         desc->ne[0],
                         desc->ne[1],
                         desc->ne[2],
                         desc->ne[3],
                         desc->src_ne[0][0],
                         desc->src_ne[0][1],
                         desc->src_ne[0][2],
                         desc->src_ne[0][3],
                         desc->src_nb[0][0],
                         desc->src_nb[0][1],
                         desc->src_nb[0][2],
                         desc->src_nb[0][3],
                         desc->nb[0],
                         desc->nb[1],
                         desc->nb[2],
                         desc->nb[3],
                         elem_size);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_cpy_f16(arg);
}

void ggml_gtx_compute_cpy(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint32_t ne0,
    uint32_t ne1,
    uint32_t ne2,
    uint32_t ne3,
    uint32_t src_ne0,
    uint32_t src_ne1,
    uint32_t src_ne2,
    uint32_t src_ne3,
    uint32_t src_nb0,
    uint32_t src_nb1,
    uint32_t src_nb2,
    uint32_t src_nb3,
    uint32_t dst_nb0,
    uint32_t dst_nb1,
    uint32_t dst_nb2,
    uint32_t dst_nb3,
    uint16_t elem_size)
{
    volatile uint8_t * src = (volatile uint8_t *)(uintptr_t)ddr_src;
    volatile uint8_t * dst = (volatile uint8_t *)(uintptr_t)ddr_dst;

    const uint32_t row_bytes = (uint32_t) ne0 * elem_size;
    const uint32_t plane_bytes = (uint32_t) ne1 * row_bytes;
    const uint32_t block_bytes = (uint32_t) ne2 * plane_bytes;
    const bool same_shape =
        src_ne0 == ne0 &&
        src_ne1 == ne1 &&
        src_ne2 == ne2 &&
        src_ne3 == ne3;

    const bool src_contiguous =
        src_nb0 == elem_size &&
        src_nb1 == row_bytes &&
        src_nb2 == plane_bytes &&
        src_nb3 == block_bytes;
    const bool dst_contiguous =
        dst_nb0 == elem_size &&
        dst_nb1 == row_bytes &&
        dst_nb2 == plane_bytes &&
        dst_nb3 == block_bytes;

    if (src_contiguous && dst_contiguous) {
        const uint64_t total_bytes = (uint64_t) ne3 * block_bytes;
        for (uint64_t i = 0; i < total_bytes; ++i) {
            dst[i] = src[i];
        }
        return;
    }

    if (same_shape) {
        for (uint32_t i3 = 0; i3 < ne3; ++i3) {
            for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                for (uint32_t i1 = 0; i1 < ne1; ++i1) {
                    const uint64_t src_off = (uint64_t) i3 * src_nb3 +
                                             (uint64_t) i2 * src_nb2 +
                                             (uint64_t) i1 * src_nb1;
                    const uint64_t dst_off = (uint64_t) i3 * dst_nb3 +
                                             (uint64_t) i2 * dst_nb2 +
                                             (uint64_t) i1 * dst_nb1;
                    if (src_nb0 == elem_size && dst_nb0 == elem_size) {
                        for (uint32_t b = 0; b < row_bytes; ++b) {
                            dst[dst_off + b] = src[src_off + b];
                        }
                    } else {
                        for (uint32_t i0 = 0; i0 < ne0; ++i0) {
                            const uint64_t dst_col_off = dst_off + (uint64_t) i0 * dst_nb0;
                            const uint64_t src_col_off = src_off + (uint64_t) i0 * src_nb0;
                            for (uint16_t eb = 0; eb < elem_size; ++eb) {
                                dst[dst_col_off + eb] = src[src_col_off + eb];
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    const uint64_t total = (uint64_t) ne0 * ne1 * ne2 * ne3;
    for (uint64_t linear = 0; linear < total; ++linear) {
        uint64_t src_linear = linear;
        const uint64_t s0 = src_ne0 ? src_linear % src_ne0 : 0;
        src_linear = src_ne0 ? src_linear / src_ne0 : 0;
        const uint64_t s1 = src_ne1 ? src_linear % src_ne1 : 0;
        src_linear = src_ne1 ? src_linear / src_ne1 : 0;
        const uint64_t s2 = src_ne2 ? src_linear % src_ne2 : 0;
        src_linear = src_ne2 ? src_linear / src_ne2 : 0;
        const uint64_t s3 = src_ne3 ? src_linear % src_ne3 : 0;

        uint64_t dst_linear = linear;
        const uint64_t d0 = ne0 ? dst_linear % ne0 : 0;
        dst_linear = ne0 ? dst_linear / ne0 : 0;
        const uint64_t d1 = ne1 ? dst_linear % ne1 : 0;
        dst_linear = ne1 ? dst_linear / ne1 : 0;
        const uint64_t d2 = ne2 ? dst_linear % ne2 : 0;
        dst_linear = ne2 ? dst_linear / ne2 : 0;
        const uint64_t d3 = ne3 ? dst_linear % ne3 : 0;

        const uint64_t src_off = s3 * src_nb3 + s2 * src_nb2 + s1 * src_nb1 + s0 * src_nb0;
        const uint64_t dst_off = d3 * dst_nb3 + d2 * dst_nb2 + d1 * dst_nb1 + d0 * dst_nb0;
        for (uint16_t eb = 0; eb < elem_size; ++eb) {
            dst[dst_off + eb] = src[src_off + eb];
        }
    }
}
