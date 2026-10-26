//==================================================================
// GGML GTX kernel: COS
//
// Descriptor wrapper around the verified n1s16 COS arithmetic sequence.
//
// The previous wrapper selected the COS branch in reference/sincos.c.  That
// branch is not numerically correct for part of the MobileSAM positional
// encoding range.  Keep the range-reduction and Horner sequence from
// gtx-firmware-stash/kernels/n1s16/n1s16_cos.c, with only the memory movement
// changed from fixed DDR buffers to descriptor-provided L2 addresses.
//==================================================================

#include <stdint.h>

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#include "gtx/intrin.h"

#define COS_DTYPE       2u
#define COS_VEC_ELEMS  16u
#define COS_VEC_BYTES  (COS_VEC_ELEMS * COS_DTYPE)

#define COS_BANK_A 0x00000u
#define COS_BANK_R 0x50000u

#define COS_FP16_TWO_PI  0x4648u
#define COS_FP16_INV_2PI 0x3118u
#define COS_FP16_C0       0x3C00u
#define COS_FP16_C1       0xB800u
#define COS_FP16_C2       0x2955u
#define COS_FP16_C3       0x95B4u
#define COS_FP16_C4       0x01A0u

static void ggml_gtx_cos_vec_l2(uint32_t src_addr, uint32_t dst_addr) {
    __start_plan(0);
        __start_thread(0);
            __load(src_addr, COS_BANK_A,
                   COS_VEC_BYTES, COS_VEC_BYTES, 1, COS_VEC_BYTES);
            __load_svr(COS_BANK_A, 0);

            // n = round(x / (2*pi)); x_red = x - n * 2*pi
            __mul_is(0, COS_FP16_INV_2PI, 1, 0);
            __rne_i(1, 2);
            __mul_is(2, COS_FP16_TWO_PI, 3, 0);
            __sub_ii(0, 3, 4);

            // Match the x^8 coefficient already used by reference/sincos.c:
            // cos(x_red) = 1 + x^2*(-1/2 + x^2*(1/24 +
            //                  x^2*(-1/720 + x^2/40320)))
            __mul_ii(4, 4, 5);
            __mul_is(5, COS_FP16_C4, 6, 0);
            __add_is(6, COS_FP16_C3, 7, 0);
            __mul_ii(5, 7, 8);
            __add_is(8, COS_FP16_C2, 9, 0);
            __mul_ii(5, 9, 10);
            __add_is(10, COS_FP16_C1, 11, 0);
            __mul_ii(5, 11, 12);
            __add_is(12, COS_FP16_C0, 13, 0);

            __store_svr(COS_BANK_R, 13);
            __store(COS_BANK_R, dst_addr,
                    COS_VEC_BYTES, COS_VEC_BYTES, 1, COS_VEC_BYTES);
        __end_thread(0);
    __end_plan(0);
}

int ggml_gtx_kernel_cos_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_COS_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_COS_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->l2_dst_addr == 0 ||
        desc->l2_src_addr[0] == 0 ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->nb[0] != COS_DTYPE ||
        desc->src_nb[0][0] != COS_DTYPE) {
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
    if (desc->nb[1] != desc->ne[0] * COS_DTYPE ||
        desc->src_nb[0][1] != desc->src_ne[0][0] * COS_DTYPE ||
        (desc->ne[2] > 1 && desc->nb[2] != desc->nb[1] * desc->ne[1]) ||
        (desc->ne[3] > 1 && desc->nb[3] != desc->nb[2] * desc->ne[2]) ||
        (desc->src_ne[0][2] > 1 && desc->src_nb[0][2] != desc->src_nb[0][1] * desc->src_ne[0][1]) ||
        (desc->src_ne[0][3] > 1 && desc->src_nb[0][3] != desc->src_nb[0][2] * desc->src_ne[0][2])) {
        return -5;
    }

    for (uint64_t elem = 0; elem < elems64; elem += COS_VEC_ELEMS) {
        const uint64_t off = elem * COS_DTYPE;
        ggml_gtx_cos_vec_l2(desc->l2_src_addr[0] + off,
                            desc->l2_dst_addr + off);
    }

    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_cos_f16(arg);
}
