//==================================================================
// GGML GTX kernel: CONT
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

int ggml_gtx_kernel_cont_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_CONT_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_CONT_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->ne[0] > UINT16_MAX ||
        desc->ne[1] > UINT16_MAX ||
        desc->ne[2] > UINT16_MAX ||
        desc->ne[3] > UINT16_MAX) {
        return -2;
    }

    const uint16_t elem_size =
        desc->src_type[0] == GGML_GTX_TYPE_F32 ? 4 : 2;
    ggml_gtx_compute_cont(desc->src_addr[0],
                          desc->dst_addr,
                          (uint16_t) desc->ne[0],
                          (uint16_t) desc->ne[1],
                          (uint16_t) desc->ne[2],
                          (uint16_t) desc->ne[3],
                          desc->src_nb[0][0],
                          desc->src_nb[0][1],
                          desc->src_nb[0][2],
                          desc->src_nb[0][3],
                          elem_size);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_cont_f16(arg);
}

void ggml_gtx_compute_cont(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t out_ne0,
    uint16_t out_ne1,
    uint16_t out_ne2,
    uint16_t out_ne3,
    uint32_t src_nb0,
    uint32_t src_nb1,
    uint32_t src_nb2,
    uint32_t src_nb3,
    uint16_t elem_size)
{
    uint8_t contiguous = 0;
    const uint32_t row_bytes = (uint32_t)out_ne0 * elem_size;
    const uint32_t plane_bytes = (uint32_t)out_ne1 * row_bytes;
    const uint32_t block_bytes = (uint32_t)out_ne2 * plane_bytes;

    if (src_nb0 == elem_size) contiguous |= 1;
    if (src_nb1 == row_bytes) contiguous |= 2;
    if (src_nb2 == plane_bytes) contiguous |= 4;
    if (src_nb3 == block_bytes) contiguous |= 8;

#ifdef GGML_GTX_USE_GTX
    if (row_bytes <= UINT16_MAX && contiguous == 0xF) {
        __copy_mem(ddr_src, ddr_dst,
                   row_bytes, row_bytes,
                   (uint32_t)out_ne1 * out_ne2 * out_ne3,
                   row_bytes, row_bytes >> 16);
        return;
    }

    uint64_t dst_addr = ddr_dst;

    if (row_bytes <= UINT16_MAX && contiguous == 0x7) {
        for (uint16_t b = 0; b < out_ne3; ++b) {
            __copy_mem(ddr_src + (uint64_t)b * src_nb3,
                       dst_addr,
                       row_bytes, row_bytes,
                       (uint32_t)out_ne1 * out_ne2,
                       row_bytes, row_bytes >> 16);
            dst_addr += block_bytes;
        }
        return;
    }

    if (row_bytes <= UINT16_MAX && contiguous == 0x3) {
        for (uint16_t b = 0; b < out_ne3; ++b) {
            for (uint16_t ch = 0; ch < out_ne2; ++ch) {
                __copy_mem(ddr_src + (uint64_t)b * src_nb3 + (uint64_t)ch * src_nb2,
                           dst_addr,
                           row_bytes, row_bytes,
                           out_ne1,
                           row_bytes, row_bytes >> 16);
                dst_addr += plane_bytes;
            }
        }
        return;
    }

    if (row_bytes <= UINT16_MAX && contiguous == 0x1) {
        for (uint16_t b = 0; b < out_ne3; ++b) {
            for (uint16_t ch = 0; ch < out_ne2; ++ch) {
                __copy_mem(ddr_src + (uint64_t)b * src_nb3 + (uint64_t)ch * src_nb2,
                           dst_addr,
                           src_nb1, (uint16_t)row_bytes,
                           out_ne1,
                           (uint16_t)row_bytes, row_bytes >> 16);
                dst_addr += plane_bytes;
            }
        }
        return;
    }

    /* General permutation/gather: one strided DMA command per logical row
     * instead of copying every element through the RISC-V core. */
    if (elem_size <= UINT16_MAX) {
        for (uint16_t b = 0; b < out_ne3; ++b) {
            for (uint16_t ch = 0; ch < out_ne2; ++ch) {
                for (uint16_t row = 0; row < out_ne1; ++row) {
                    __copy_mem(ddr_src + (uint64_t)b * src_nb3 +
                                   (uint64_t)ch * src_nb2 + (uint64_t)row * src_nb1,
                               dst_addr,
                               src_nb0, elem_size,
                               out_ne0,
                               elem_size, 0u);
                    dst_addr += row_bytes;
                }
            }
        }
        return;
    }
#endif

    volatile uint8_t * src = (volatile uint8_t *)(uintptr_t)ddr_src;
    volatile uint8_t * dst = (volatile uint8_t *)(uintptr_t)ddr_dst;
    uint64_t dst_off = 0;

    for (uint16_t b = 0; b < out_ne3; ++b) {
        for (uint16_t ch = 0; ch < out_ne2; ++ch) {
            for (uint16_t row = 0; row < out_ne1; ++row) {
                for (uint16_t col = 0; col < out_ne0; ++col) {
                    uint64_t src_off = (uint64_t)b * src_nb3 + (uint64_t)ch * src_nb2 +
                                       (uint64_t)row * src_nb1 + (uint64_t)col * src_nb0;
                    for (uint16_t eb = 0; eb < elem_size; ++eb) {
                        dst[dst_off + eb] = src[src_off + eb];
                    }
                    dst_off += elem_size;
                }
            }
        }
    }
}

//==================================================================
// SET: Embed src1 into a sub-view of src0 at a given byte offset
//
// 1. Copy src0 → dst (if not in-place)
// 2. Copy src1 into dst at (offset, nb1, nb2, nb3) view strides
//
// Both tensors are F32.
//==================================================================
