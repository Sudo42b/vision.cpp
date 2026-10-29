//==================================================================
// GGML GTX kernel: GET_ROWS
//==================================================================

#include <stdint.h>

#include "ggml-gtx-baremetal.h"
#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/ggml-gtx-protocol.h"

#ifdef GGML_GTX_USE_GTX
#include "ggml-gtx/intrinsics/intrin.h"
#endif

int ggml_gtx_kernel_get_rows_f16(void * arg) {
    const ggml_gtx_kernel_arg_t * karg = (const ggml_gtx_kernel_arg_t *) arg;
    if (karg == 0 ||
        karg->magic != GGML_GTX_KERNEL_ARG_MAGIC ||
        karg->version != GGML_GTX_KERNEL_ARG_VERSION ||
        karg->size < sizeof(ggml_gtx_kernel_arg_t) ||
        karg->kernel_id != GGML_GTX_KERNEL_GET_ROWS_F16 ||
        karg->desc.kernel_id != GGML_GTX_KERNEL_GET_ROWS_F16) {
        return -1;
    }

    const ggml_gtx_kernel_launch_desc_t * desc = &karg->desc;
    if (desc->dst_addr == 0 ||
        desc->src_addr[0] == 0 ||
        desc->src_addr[1] == 0) {
        return -2;
    }

    ggml_gtx_compute_get_rows(desc->src_addr[0],
                              desc->src_addr[1],
                              desc->dst_addr,
                              (uint16_t) desc->src_ne[0][0],
                              (uint16_t) desc->src_ne[0][1],
                              (uint16_t) desc->src_ne[1][0],
                              (uint16_t) desc->src_ne[1][1],
                              (uint16_t) desc->src_ne[1][2],
                              desc->src_type[1] == GGML_GTX_TYPE_I64 ? 8 : 4,
                              GGML_GTX_NEST_NUM,
                              GGML_GTX_SPU_NUM_PER_NEST);
    return 0;
}

int kernel_main(void * arg) {
    return ggml_gtx_kernel_get_rows_f16(arg);
}

void ggml_gtx_printf(const char * fmt, ...) {
    (void) fmt;
}

static inline uint64_t ggml_gtx_read_rows_index(uint64_t ddr_indices, uint32_t idx_pos, uint16_t idx_size) {
    if (idx_size == 8) {
        volatile uint64_t * idx64 = (volatile uint64_t *)(uintptr_t)ddr_indices;
        return idx64[idx_pos];
    }

    volatile uint32_t * idx32 = (volatile uint32_t *)(uintptr_t)ddr_indices;
    return idx32[idx_pos];
}

void ggml_gtx_compute_get_rows(
    uint64_t ddr_data,
    uint64_t ddr_indices,
    uint64_t ddr_dst,
    uint16_t n_embd,
    uint16_t n_vocab,
    uint16_t n_rows,
    uint16_t n_channels,
    uint16_t n_batches,
    uint16_t idx_size,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
    GGML_GTX_UNUSED(nest_num);
    GGML_GTX_UNUSED(spu_num_per_nest);

    const uint32_t row_bytes = (uint32_t)n_embd * GGML_GTX_DTYPE_SIZE;

    for (uint16_t b = 0; b < n_batches; ++b) {
        for (uint16_t ch = 0; ch < n_channels; ++ch) {
            for (uint16_t r = 0; r < n_rows; ++r) {
                const uint32_t idx_pos = ((uint32_t)b * n_channels + ch) * n_rows + r;
                const uint32_t raw_word = *(const volatile uint32_t *)(uintptr_t)(ddr_indices + (uint64_t) idx_pos * idx_size);
                const uint64_t row_idx = ggml_gtx_read_rows_index(ddr_indices, idx_pos, idx_size);

                if (idx_pos == 0) {
                    ggml_gtx_printf("[GET_ROWS] data=0x%08llx idx=0x%08llx dst=0x%08llx n_embd=%u n_vocab=%u n_rows=%u n_channels=%u n_batches=%u idx_size=%u raw=0x%08x row_idx=%llu\n",
                                    (unsigned long long) ddr_data,
                                    (unsigned long long) ddr_indices,
                                    (unsigned long long) ddr_dst,
                                    n_embd, n_vocab, n_rows, n_channels, n_batches, idx_size,
                                    raw_word,
                                    (unsigned long long) row_idx);
                }

                if (row_idx >= n_vocab) {
                    ggml_gtx_printf("[GET_ROWS ERROR] row_idx=%llu out of range n_vocab=%u idx_pos=%u idx_addr=0x%08llx\n",
                                    (unsigned long long) row_idx,
                                    n_vocab,
                                    idx_pos,
                                    (unsigned long long) (ddr_indices + (uint64_t) idx_pos * idx_size));
                    return;
                }

                const uint64_t src_addr = ddr_data +
                    ((((uint64_t)b * n_channels + ch) * n_vocab + row_idx) * row_bytes);
                const uint64_t dst_addr = ddr_dst +
                    ((((uint64_t)b * n_channels + ch) * n_rows + r) * row_bytes);

#ifdef GGML_GTX_USE_GTX
                __copy_mem(src_addr,
                           dst_addr,
                           row_bytes,
                           row_bytes,
                           1,
                           row_bytes,
                           row_bytes >> 16);
#else
                volatile uint16_t * src = (volatile uint16_t *)(uintptr_t)src_addr;
                volatile uint16_t * dst = (volatile uint16_t *)(uintptr_t)dst_addr;
                for (uint16_t c = 0; c < n_embd; ++c) {
                    dst[c] = src[c];
                }
#endif
            }
        }
    }
}
