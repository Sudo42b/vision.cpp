//==================================================================
// GGML GTX kernel: REPEAT
//==================================================================

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

int ggml_gtx_kernel_repeat_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_REPEAT_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_REPEAT_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->ne[0] == 0 ||
        desc->ne[1] == 0 ||
        desc->ne[2] == 0 ||
        desc->ne[3] == 0 ||
        desc->src_ne[0][0] == 0 ||
        desc->src_ne[0][1] == 0 ||
        desc->src_ne[0][2] == 0 ||
        desc->src_ne[0][3] == 0 ||
        desc->ne[0] > UINT16_MAX / GGML_GTX_DTYPE_SIZE ||
        desc->src_ne[0][0] > UINT16_MAX / GGML_GTX_DTYPE_SIZE) {
        return -2;
    }

    for (int i = 0; i < 4; ++i) {
        if (desc->ne[i] > UINT16_MAX ||
            desc->src_ne[0][i] > UINT16_MAX ||
            desc->ne[i] < desc->src_ne[0][i] ||
            (desc->ne[i] % desc->src_ne[0][i]) != 0) {
            return -3;
        }
    }

    ggml_gtx_compute_repeat(desc->l2_src_addr[0],
                            desc->l2_dst_addr,
                            (uint16_t) desc->src_ne[0][0],
                            (uint16_t) desc->src_ne[0][1],
                            (uint16_t) desc->src_ne[0][2],
                            (uint16_t) desc->src_ne[0][3],
                            (uint16_t) desc->ne[0],
                            (uint16_t) desc->ne[1],
                            (uint16_t) desc->ne[2],
                            (uint16_t) desc->ne[3],
                            GGML_GTX_DTYPE_SIZE,
                            1,
                            1);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_repeat_f16(arg);
}

void ggml_gtx_compute_repeat(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne00,
    uint16_t ne01,
    uint16_t ne02,
    uint16_t ne03,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t elem_size,
    uint8_t nest_num,
    uint8_t spu_num_per_nest)
{
    (void) nest_num;
    (void) spu_num_per_nest;

#ifdef GGML_GTX_USE_GTX
    const uint32_t src_row_bytes = (uint32_t)ne00 * elem_size;
    const uint32_t dst_row_bytes = (uint32_t)ne0 * elem_size;
    const uint32_t src_plane_rows = (uint32_t)ne01;
    const uint32_t src_plane_count = (uint32_t)ne02;
    const uint32_t repeat0 = ne0 / ne00;
    const uint32_t repeat1 = ne1 / ne01;

    __start_plan(0);
        __start_shared();
            for (uint32_t i3 = 0; i3 < ne3; ++i3) {
                const uint32_t s3 = i3 % ne03;
                for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                    const uint32_t s2 = i2 % ne02;
                    const uint32_t src_plane = (s3 * src_plane_count + s2) * src_plane_rows;
                    const uint32_t dst_plane = (i3 * (uint32_t)ne2 + i2) * (uint32_t)ne1;
                    for (uint32_t r1 = 0; r1 < repeat1; ++r1) {
                        const uint32_t dst_row0 = dst_plane + r1 * src_plane_rows;
                        for (uint32_t r0 = 0; r0 < repeat0; ++r0) {
                            __copy((uint32_t)ddr_src + src_plane * src_row_bytes,
                                   (uint32_t)ddr_dst + dst_row0 * dst_row_bytes + r0 * src_row_bytes,
                                   src_row_bytes,
                                   (uint16_t)src_row_bytes,
                                   (uint16_t)src_plane_rows,
                                   dst_row_bytes);
                        }
                    }
                }
            }
        __end_shared();
    __end_plan(0);
#else

    volatile const uint8_t * src = (volatile const uint8_t *)(uintptr_t)ddr_src;
    volatile uint8_t * dst = (volatile uint8_t *)(uintptr_t)ddr_dst;

    const uint64_t src_nb0 = elem_size;
    const uint64_t src_nb1 = (uint64_t) ne00 * src_nb0;
    const uint64_t src_nb2 = (uint64_t) ne01 * src_nb1;
    const uint64_t src_nb3 = (uint64_t) ne02 * src_nb2;

    uint64_t dst_off = 0;
    for (uint16_t i3 = 0; i3 < ne3; ++i3) {
        const uint16_t s3 = i3 % ne03;
        for (uint16_t i2 = 0; i2 < ne2; ++i2) {
            const uint16_t s2 = i2 % ne02;
            for (uint16_t i1 = 0; i1 < ne1; ++i1) {
                const uint16_t s1 = i1 % ne01;
                for (uint16_t i0 = 0; i0 < ne0; ++i0) {
                    const uint16_t s0 = i0 % ne00;
                    const uint64_t src_off = (uint64_t) s3 * src_nb3 +
                                             (uint64_t) s2 * src_nb2 +
                                             (uint64_t) s1 * src_nb1 +
                                             (uint64_t) s0 * src_nb0;
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
