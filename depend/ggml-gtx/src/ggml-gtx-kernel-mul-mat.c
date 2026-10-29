//==================================================================
// GGML GTX kernel: MUL_MAT, L2-stashed single-nest tile
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"
#include "ggml-gtx/ggml-gtx-kernel-args.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#define GGML_GTX_MUL_MAT_L1_A_BYTES       0x20000u
#define GGML_GTX_MUL_MAT_L1_B_BYTES       0x10000u
#define GGML_GTX_MUL_MAT_L1_R_BYTES       0x10000u
#define GGML_GTX_MUL_MAT_TILES_PER_PLAN   8u

static void ggml_gtx_compute_matmul_l2_n1(
    uint32_t l2_a,
    uint32_t l2_b,
    uint32_t l2_c,
    uint16_t a_row,
    uint16_t h_col,
    uint8_t  nest_id,
    uint8_t  spu_num_per_nest)
{
#ifdef GGML_GTX_USE_GTX
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint16_t not_use = 0xBEEF;
    const uint16_t byte_h_col = (uint16_t) (h_col * dtype_size);
    uint16_t rows_per_l1 = (uint16_t) (GGML_GTX_MUL_MAT_L1_A_BYTES / byte_h_col);
    if (rows_per_l1 == 0u) {
        rows_per_l1 = 1u;
    }

    __split();
        __start_plan(nest_id);
            __start_shared();
            __end_shared();

            /*
             * The active FPGA is n1s1.  Keep several full 64 KiB A tiles in
             * one plan and retain B in L1 across them.  This preserves the
             * verified N=1 result layout while avoiding one split/join and
             * one B load per M tile.
             */
            __start_thread(0u);
                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET,
                                   GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET,
                                   GGML_GTX_L1_BANK_A_OFFSET);

                    __load(
                        l2_b,
                        GGML_GTX_L1_BANK_B_OFFSET,
                        byte_h_col,
                        byte_h_col,
                        1,
                        not_use
                    );

                    for (uint16_t m0 = 0u; m0 < a_row;) {
                        uint16_t rows = (uint16_t) (a_row - m0);
                        if (rows > rows_per_l1) {
                            rows = rows_per_l1;
                        }
                        const uint32_t a_off = (uint32_t)m0 * byte_h_col;
                        const uint32_t c_off = (uint32_t)m0 * dtype_size;

                        __load(
                            l2_a + a_off,
                            GGML_GTX_L1_BANK_A_OFFSET,
                            byte_h_col,
                            byte_h_col,
                            rows,
                            not_use
                        );
                        __mm(rows, h_col, 1);
                        __store(
                            GGML_GTX_L1_BANK_R_OFFSET,
                            l2_c + c_off,
                            dtype_size,
                            dtype_size,
                            rows,
                            not_use
                        );
                        m0 = (uint16_t) (m0 + rows);
                    }
            __end_thread(0u);
        __end_plan(nest_id);
    __join();
#else
    (void) l2_a;
    (void) l2_b;
    (void) l2_c;
    (void) a_row;
    (void) h_col;
    (void) nest_id;
    (void) spu_num_per_nest;
#endif
}

static int ggml_gtx_compute_matmul_l2_block(
    uint32_t l2_a,
    uint32_t l2_b,
    uint32_t l2_c,
    uint16_t a_row,
    uint16_t h_col,
    uint16_t b_row,
    uint8_t  nest_id,
    uint8_t  spu_num_per_nest)
{
    const uint32_t dtype_size = GGML_GTX_DTYPE_SIZE;
    const uint32_t k_bytes = (uint32_t)h_col * dtype_size;
    if (a_row == 0u || h_col == 0u || b_row == 0u ||
        k_bytes == 0u || k_bytes > UINT16_MAX ||
        k_bytes > GGML_GTX_MUL_MAT_L1_B_BYTES) {
        return -1;
    }

    // Preserve the established decode/GEMV path exactly.  Large-N vision
    // workloads use the block path below.
    if (b_row == 1u) {
        ggml_gtx_compute_matmul_l2_n1(
            l2_a, l2_b, l2_c, a_row, h_col, nest_id, spu_num_per_nest);
        return 0;
    }

    // L1 A, B, and R are separate address ranges.  GGML_GTX_L1_BANK_SIZE is
    // the size of the complete L1 address space, not the capacity of each
    // bank: A is 128 KiB while B and R are 64 KiB.  Using the total 384 KiB
    // value here made a K=128, N=769 B load cross from B [0x20000,0x30000)
    // through C and into R at 0x50000; N=768 ended exactly at the R boundary.
    // Tile against the actual bank capacities instead.
    // Choose a KxN B
    // block first, then the largest M block that fits both A (MxK) and the
    // transposed result (NxM).  B is stored as N contiguous K-vectors, which
    // is the native GTX ntn input contract; __mm_t emits N contiguous
    // M-vectors, matching ggml's ne[0]-contiguous destination layout.
    uint32_t n_block = GGML_GTX_MUL_MAT_L1_B_BYTES / k_bytes;
    if (n_block > b_row) {
        n_block = b_row;
    }
    if (n_block == 0u || n_block > UINT16_MAX) {
        return -1;
    }

    uint32_t m_block = GGML_GTX_MUL_MAT_L1_A_BYTES / k_bytes;
    const uint32_t m_by_result = GGML_GTX_MUL_MAT_L1_R_BYTES /
                                 (n_block * dtype_size);
    if (m_block > m_by_result) {
        m_block = m_by_result;
    }
    if (m_block > a_row) {
        m_block = a_row;
    }
    // __store length is uint16 bytes.
    if (m_block > UINT16_MAX / dtype_size) {
        m_block = UINT16_MAX / dtype_size;
    }
    if (m_block == 0u) {
        return -1;
    }

    (void)spu_num_per_nest;
    for (uint32_t m0 = 0u; m0 < a_row; m0 += m_block) {
        const uint16_t cur_m = (uint16_t)
            (((uint32_t)a_row - m0) < m_block ? (uint32_t)a_row - m0 : m_block);

        __split();
            __start_plan(nest_id);
                __start_thread(0u);
                    __set_spm_addr(GGML_GTX_L1_BANK_R_OFFSET,
                                   GGML_GTX_L1_BANK_C_OFFSET,
                                   GGML_GTX_L1_BANK_B_OFFSET,
                                   GGML_GTX_L1_BANK_A_OFFSET);

                    // Reuse this A block for every N block in the descriptor.
                    __load(l2_a + m0 * k_bytes,
                           GGML_GTX_L1_BANK_A_OFFSET,
                           k_bytes,
                           (uint16_t)k_bytes,
                           cur_m,
                           k_bytes);

                    for (uint32_t n0 = 0u; n0 < b_row; n0 += n_block) {
                        const uint16_t cur_n = (uint16_t)
                            (((uint32_t)b_row - n0) < n_block ?
                                 (uint32_t)b_row - n0 : n_block);

                        __load(l2_b + n0 * k_bytes,
                               GGML_GTX_L1_BANK_B_OFFSET,
                               k_bytes,
                               (uint16_t)k_bytes,
                               cur_n,
                               k_bytes);

                        __mm_t(cur_m, h_col, cur_n);

                        __store(GGML_GTX_L1_BANK_R_OFFSET,
                                l2_c + (n0 * (uint32_t)a_row + m0) * dtype_size,
                                (uint32_t)a_row * dtype_size,
                                (uint16_t)((uint32_t)cur_m * dtype_size),
                                cur_n,
                                (uint32_t)cur_m * dtype_size);
                    }
                __end_thread(0u);
            __end_plan(nest_id);
        __join();
    }

    return 0;
}

int ggml_gtx_kernel_mul_mat_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_MUL_MAT_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_MUL_MAT_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->memory_mode != GGML_GTX_MEMORY_MODE_L2_STASHED_N1 ||
        desc->nest_id >= GGML_GTX_NEST_NUM) {
        return -2;
    }

    const ggml_gtx_mul_mat_args_t * params =
        (const ggml_gtx_mul_mat_args_t *)ggml_gtx_kernel_args(
            karg, sizeof(ggml_gtx_mul_mat_args_t));
    if (params == 0 ||
        params->version != GGML_GTX_MUL_MAT_ARGS_VERSION ||
        params->size < sizeof(*params)) {
        return -3;
    }

    const uint32_t h_col = params->k;
    const uint32_t a_row = params->m;
    const uint32_t b_row = params->n;
    if (h_col == 0 || a_row == 0 || b_row == 0 ||
        h_col != desc->src_ne[0][0] ||
        a_row != desc->src_ne[0][1] ||
        b_row != desc->src_ne[1][1] ||
        h_col > UINT16_MAX || a_row > UINT16_MAX || b_row > UINT16_MAX) {
        return -4;
    }

    if (ggml_gtx_compute_matmul_l2_block(
        (uint32_t) desc->l2_src_addr[0],
        (uint32_t) desc->l2_src_addr[1],
        (uint32_t) desc->l2_dst_addr,
        (uint16_t) a_row,
        (uint16_t) h_col,
        (uint16_t) b_row,
        (uint8_t) desc->nest_id,
        GGML_GTX_SPU_NUM_PER_NEST
    ) != 0) {
        return -5;
    }
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_mul_mat_f16(arg);
}
