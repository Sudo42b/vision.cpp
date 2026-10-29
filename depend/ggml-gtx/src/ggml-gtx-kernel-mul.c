//==================================================================
// GGML GTX kernel: MUL, L2-stashed single-nest tile
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-kernel-args.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_BINARY_L1_MAX_ELEMS ((64u * 1024u - 2u) / GGML_GTX_DTYPE_SIZE)

static uint32_t ggml_gtx_binary_rows(const ggml_gtx_kernel_launch_desc_t * desc) {
    return desc->ne[1] * desc->ne[2] * desc->ne[3];
}

static uint32_t ggml_gtx_binary_row_for_tid(uint32_t rows, uint32_t tid) {
    return (uint32_t) (((uint64_t) rows * tid) / GGML_GTX_SPU_NUM_PER_NEST);
}

static uint32_t ggml_gtx_binary_rhs_row(const ggml_gtx_kernel_launch_desc_t * desc,
                                        uint32_t row) {
    const uint32_t i1 = row % desc->ne[1];
    const uint32_t q = row / desc->ne[1];
    const uint32_t i2 = q % desc->ne[2];
    const uint32_t i3 = q / desc->ne[2];
    const uint32_t j1 = desc->src_ne[1][1] == 1u ? 0u : i1;
    const uint32_t j2 = desc->src_ne[1][2] == 1u ? 0u : i2;
    const uint32_t j3 = desc->src_ne[1][3] == 1u ? 0u : i3;
    return (j3 * desc->src_ne[1][2] + j2) * desc->src_ne[1][1] + j1;
}

static void ggml_gtx_compute_mul_vv_l2(const ggml_gtx_kernel_launch_desc_t * desc) {
#ifdef GGML_GTX_USE_GTX
    const uint32_t ne0 = desc->ne[0];
    const uint32_t rows = ggml_gtx_binary_rows(desc);
    const uint32_t row_bytes = ne0 * GGML_GTX_DTYPE_SIZE;
    const uint32_t rhs_row_bytes = desc->src_ne[1][0] * GGML_GTX_DTYPE_SIZE;

    __split();
        __start_plan((uint8_t) desc->nest_id);
            for (uint32_t tid = 0u; tid < GGML_GTX_SPU_NUM_PER_NEST; ++tid) {
                const uint32_t thread_row0 = ggml_gtx_binary_row_for_tid(rows, tid);
                const uint32_t thread_row1 = ggml_gtx_binary_row_for_tid(rows, tid + 1u);

                __start_thread((uint8_t) tid);
                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET,
                                   GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET,
                                   GGML_GTX_L1_BANK_A_OFFSET);

                    uint32_t row = thread_row0;
                    while (row < thread_row1) {
                        const uint32_t i1 = row % desc->ne[1];
                        const uint32_t rows_to_i1_boundary = desc->ne[1] - i1;
                        const uint32_t cols = ne0 < GGML_GTX_BINARY_L1_MAX_ELEMS ?
                                              ne0 : GGML_GTX_BINARY_L1_MAX_ELEMS;
                        const uint32_t max_rows_l1 = GGML_GTX_BINARY_L1_MAX_ELEMS / cols;
                        uint32_t block_rows = thread_row1 - row;
                        if (block_rows > rows_to_i1_boundary) {
                            block_rows = rows_to_i1_boundary;
                        }
                        if (block_rows > max_rows_l1) {
                            block_rows = max_rows_l1;
                        }

                        for (uint32_t col0 = 0u; col0 < ne0; col0 += cols) {
                            const uint32_t block_cols = ne0 - col0 < cols ? ne0 - col0 : cols;
                            const uint32_t block_elems = block_cols * block_rows;
                            const uint16_t block_bytes = (uint16_t) (block_cols * GGML_GTX_DTYPE_SIZE);
                            const uint32_t a_off = row * row_bytes + col0 * GGML_GTX_DTYPE_SIZE;
                            const uint32_t b_row = ggml_gtx_binary_rhs_row(desc, row);
                            const uint32_t b_off = b_row * rhs_row_bytes +
                                                   col0 * GGML_GTX_DTYPE_SIZE;
                            const uint32_t b_stride = desc->src_ne[1][1] == 1u ?
                                                      0u : rhs_row_bytes;

                            __load((uint32_t) desc->l2_src_addr[0] + a_off,
                                   GGML_GTX_L1_BANK_A_OFFSET,
                                   row_bytes,
                                   block_bytes,
                                   (uint16_t) block_rows,
                                   block_bytes);
                            __load((uint32_t) desc->l2_src_addr[1] + b_off,
                                   GGML_GTX_L1_BANK_B_OFFSET,
                                   b_stride,
                                   block_bytes,
                                   (uint16_t) block_rows,
                                   block_bytes);
                            __mul_vv(block_elems);
                            __store(GGML_GTX_L1_BANK_R_OFFSET,
                                    (uint32_t) desc->l2_dst_addr + a_off,
                                    row_bytes,
                                    block_bytes,
                                    (uint16_t) block_rows,
                                    block_bytes);
                        }
                        row += block_rows;
                    }
                __end_thread((uint8_t) tid);
            }
        __end_plan((uint8_t) desc->nest_id);
    __join();
#else
    (void) desc;
#endif
}

static void ggml_gtx_compute_mul_vs_l2(const ggml_gtx_kernel_launch_desc_t * desc,
                                       const uint16_t * scalars) {
    const uint32_t ne0 = desc->ne[0];
    const uint32_t rows = ggml_gtx_binary_rows(desc);
    const uint32_t row_bytes = ne0 * GGML_GTX_DTYPE_SIZE;
    const uint32_t cols = ne0 < GGML_GTX_BINARY_L1_MAX_ELEMS ?
                          ne0 : GGML_GTX_BINARY_L1_MAX_ELEMS;
    const uint32_t max_rows_l1 = GGML_GTX_BINARY_L1_MAX_ELEMS / cols;

    uint32_t row = 0u;
    while (row < rows) {
        const uint32_t i1 = row % desc->ne[1];
        uint32_t block_rows = rows - row;
        const uint32_t rows_to_i1_boundary = desc->ne[1] - i1;
        if (block_rows > rows_to_i1_boundary) block_rows = rows_to_i1_boundary;
        if (block_rows > max_rows_l1) block_rows = max_rows_l1;

        const uint16_t scalar = scalars[ggml_gtx_binary_rhs_row(desc, row)];
        for (uint32_t col0 = 0u; col0 < ne0; col0 += cols) {
            const uint32_t block_cols = ne0 - col0 < cols ? ne0 - col0 : cols;
            const uint32_t block_elems = block_cols * block_rows;
            const uint16_t block_bytes = (uint16_t)(block_cols * GGML_GTX_DTYPE_SIZE);
            const uint32_t a_off = row * row_bytes + col0 * GGML_GTX_DTYPE_SIZE;
            const uint32_t l2_src = (uint32_t)desc->l2_src_addr[0] + a_off;
            const uint32_t l2_dst = (uint32_t)desc->l2_dst_addr + a_off;

            __split();
                __start_plan((uint8_t)desc->nest_id);
                    __start_thread(0u);
                        __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET,
                                       GGML_GTX_L1_BANK_C_OFFSET,
                                       GGML_GTX_L1_BANK_B_OFFSET,
                                       GGML_GTX_L1_BANK_A_OFFSET);
                        __load(l2_src, GGML_GTX_L1_BANK_A_OFFSET,
                               row_bytes, block_bytes, (uint16_t)block_rows, block_bytes);
                        __mul_vs(block_elems, scalar, 0u);
                        __store(GGML_GTX_L1_BANK_R_OFFSET, l2_dst,
                                row_bytes, block_bytes, (uint16_t)block_rows, block_bytes);
                    __end_thread(0u);
                __end_plan((uint8_t)desc->nest_id);
            __join();
        }
        row += block_rows;
    }
}

int ggml_gtx_kernel_mul_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_MUL_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_MUL_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    const uint32_t elems = desc->ne[0];
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->nest_id >= GGML_GTX_NEST_NUM ||
        desc->dtype != GGML_GTX_TYPE_F16 ||
        desc->src_type[0] != GGML_GTX_TYPE_F16 ||
        desc->src_type[1] != GGML_GTX_TYPE_F16 ||
        desc->elem_size != GGML_GTX_DTYPE_SIZE ||
        elems == 0 ||
        desc->src_ne[0][0] != elems ||
        (desc->src_ne[1][0] != elems && desc->src_ne[1][0] != 1u) ||
        elems > UINT16_MAX) {
        return -2;
    }

    uint32_t rows = 1u;
    for (uint32_t d = 1u; d < 4u; ++d) {
        if (desc->src_ne[0][d] != desc->ne[d] ||
            (desc->src_ne[1][d] != 1u && desc->src_ne[1][d] != desc->ne[d]) ||
            desc->ne[d] == 0u || rows > UINT32_MAX / desc->ne[d]) {
            return -2;
        }
        rows *= desc->ne[d];
    }

    if (desc->src_ne[1][0] == elems) {
        ggml_gtx_compute_mul_vv_l2(desc);
        return 0;
    }

    const ggml_gtx_mul_args_t * params =
        (const ggml_gtx_mul_args_t *)ggml_gtx_kernel_args(
            karg, GGML_GTX_MUL_ARGS_HEADER_SIZE);
    const uint64_t expected_scalar_count =
        (uint64_t)desc->src_ne[1][1] * desc->src_ne[1][2] * desc->src_ne[1][3];
    const uint64_t expected_size = GGML_GTX_MUL_ARGS_HEADER_SIZE +
                                   expected_scalar_count * sizeof(uint16_t);
    if (params == 0 || params->version != GGML_GTX_MUL_ARGS_VERSION ||
        params->size != karg->desc.args_size || params->size != expected_size ||
        params->scalar_count != expected_scalar_count) {
        return -3;
    }
    ggml_gtx_compute_mul_vs_l2(desc, params->scalars);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_mul_f16(arg);
}
