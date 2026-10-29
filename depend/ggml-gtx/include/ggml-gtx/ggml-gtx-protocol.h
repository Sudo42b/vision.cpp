#ifndef GGML_GTX_PROTOCOL_H
#define GGML_GTX_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#include "ggml-gtx-server.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_GTX_RPC_MAGIC       0x31585447u  /* "GTX1" little-endian */
#define GGML_GTX_RPC_VERSION     1u

#define GGML_GTX_KERNEL_ARG_MAGIC   0x31415847u  /* "GXA1" little-endian */
#define GGML_GTX_KERNEL_ARG_VERSION 1u

#define GTX_DDR_BASE                    0x370000000ull
#define GTX_FIXED_ARG_ADDR              (GTX_DDR_BASE + 0x01000000ull)
#define GTX_FIXED_ARG_REGION_SIZE       0x00100000ull
#define GTX_FIXED_SRC0_ADDR             (GTX_DDR_BASE + 0x01100000ull)
#define GTX_FIXED_SRC1_ADDR             (GTX_DDR_BASE + 0x01200000ull)
#define GTX_FIXED_DST_ADDR              (GTX_DDR_BASE + 0x01300000ull)
#define GTX_FIXED_TENSOR_ARENA_ADDR     GTX_FIXED_SRC0_ADDR
#define GTX_FIXED_TENSOR_ARENA_SIZE     0x7ea00000ull
#define GTX_FIXED_KERNEL_STAGING_ADDR   (GTX_DDR_BASE + 0x7ff00000ull)
#define GTX_FIXED_KERNEL_STAGING_SIZE   0x00100000ull
#define GTX_FIXED_KERNEL_WORK_ADDR      0x03100000ull
#define GTX_FIXED_KERNEL_WORK_SIZE      0x00100000ull
#define GTX_STASH_STATS_ADDR            (GTX_DDR_BASE + 0x7fe00000ull)
#define GTX_STASH_STATS_MAGIC           0x48534154u

/* Legacy DDR ROPE sin/cos helper workspace. L2 ROPE uses its dynamically
 * assigned scratch slot only for cosine and sine tables. */
#define GGML_GTX_ROPE_SCRATCH_WORK_B_OFFSET 0x00050000u

static inline uint32_t ggml_gtx_rope_scratch_size(uint32_t ne2, uint32_t n_dims) {
    if (ne2 == 0u || n_dims < 2u || (n_dims & 1u) != 0u) return 0u;
    const uint64_t pairs_padded = (((uint64_t)n_dims / 2u) + 15u) & ~15ull;
    const uint64_t table_bytes = (uint64_t)ne2 * pairs_padded * sizeof(uint16_t);
    const uint64_t bytes = 2u * table_bytes;
    return bytes <= UINT32_MAX - 31u ? (uint32_t)((bytes + 31u) & ~31ull) : 0u;
}

typedef enum ggml_gtx_rpc_cmd {
    GGML_GTX_RPC_PING = 1,
    GGML_GTX_RPC_UPLOAD_BYTES = 2,
    GGML_GTX_RPC_DOWNLOAD_BYTES = 3,
    GGML_GTX_RPC_LOAD_KERNEL = 4,
    GGML_GTX_RPC_LAUNCH = 5,
    GGML_GTX_RPC_COPY_TENSOR = 6,
    GGML_GTX_RPC_STASH_SYNC = 7,
    GGML_GTX_RPC_DISCONNECT = 8,
} ggml_gtx_rpc_cmd_t;

typedef enum ggml_gtx_rpc_status {
    GGML_GTX_RPC_STATUS_OK = 0,
    GGML_GTX_RPC_STATUS_ERROR = 1,
    GGML_GTX_RPC_STATUS_BAD_MAGIC = 2,
    GGML_GTX_RPC_STATUS_BAD_VERSION = 3,
    GGML_GTX_RPC_STATUS_BAD_SIZE = 4,
    GGML_GTX_RPC_STATUS_BAD_CMD = 5,
    GGML_GTX_RPC_STATUS_NOT_FOUND = 6,
} ggml_gtx_rpc_status_t;

typedef enum ggml_gtx_kernel_format {
    GGML_GTX_KERNEL_FORMAT_LLEXT = 1,
} ggml_gtx_kernel_format_t;

typedef struct ggml_gtx_rpc_frame {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint16_t cmd;
    uint16_t status;
    uint32_t seq;
    uint32_t payload_size;
} ggml_gtx_rpc_frame_t;

typedef struct ggml_gtx_rpc_upload_bytes {
    uint64_t addr;
    uint32_t size;
    uint32_t reserved;
} ggml_gtx_rpc_upload_bytes_t;

typedef struct ggml_gtx_rpc_download_bytes {
    uint64_t addr;
    uint32_t size;
    uint32_t reserved;
} ggml_gtx_rpc_download_bytes_t;

typedef struct ggml_gtx_rpc_load_kernel {
    uint32_t kernel_id;
    uint32_t format;
    uint64_t staging_addr;
    uint32_t binary_size;
    uint32_t reserved0;
    uint64_t work_addr;
    uint32_t work_size;
    char entry_symbol[32];
} ggml_gtx_rpc_load_kernel_t;

typedef struct ggml_gtx_rpc_launch {
    uint32_t kernel_id;
    uint32_t reserved;
    uint64_t arg_addr;
} ggml_gtx_rpc_launch_t;

typedef struct ggml_gtx_rpc_copy_tensor {
    ggml_gtx_rpc_tensor_t src;
    ggml_gtx_rpc_tensor_t dst;
} ggml_gtx_rpc_copy_tensor_t;

typedef struct ggml_gtx_rpc_copy_tensor_rsp {
    uint8_t result;
} ggml_gtx_rpc_copy_tensor_rsp_t;

typedef enum ggml_gtx_rpc_stash_sync_op {
    GGML_GTX_RPC_STASH_FLUSH_ALL = 1,
    GGML_GTX_RPC_STASH_INVALIDATE_RANGE = 2,
    // Write back only dirty stash slots that overlap addr..addr + size.
    GGML_GTX_RPC_STASH_FLUSH_RANGE = 3,
    // Make DDR authoritative, then discard all stash residency metadata.
    GGML_GTX_RPC_STASH_FLUSH_RESET = 4,
} ggml_gtx_rpc_stash_sync_op_t;

typedef struct ggml_gtx_rpc_stash_sync {
    uint32_t op;
    uint32_t reserved;
    uint64_t addr;
    uint64_t size;
} ggml_gtx_rpc_stash_sync_t;

typedef struct ggml_gtx_kernel_arg {
    uint32_t magic;
    uint16_t version;
    uint16_t size;
    uint32_t kernel_id;
    uint32_t flags;
    ggml_gtx_kernel_launch_desc_t desc;
} ggml_gtx_kernel_arg_t;

static inline const void * ggml_gtx_kernel_args(const ggml_gtx_kernel_arg_t * arg,
                                                 uint32_t minimum_size) {
    if (arg == NULL || arg->desc.args_addr == 0u ||
        arg->desc.args_size < minimum_size) {
        return NULL;
    }
    return (const void *)(uintptr_t)arg->desc.args_addr;
}

#define GGML_GTX_STATIC_ASSERT(cond, name) typedef char ggml_gtx_static_assert_##name[(cond) ? 1 : -1]
GGML_GTX_STATIC_ASSERT(sizeof(ggml_gtx_kernel_launch_desc_t) == 436, launch_desc_size);
GGML_GTX_STATIC_ASSERT(offsetof(ggml_gtx_kernel_launch_desc_t, memory_mode) == 340, launch_desc_memory_mode_offset);
GGML_GTX_STATIC_ASSERT(sizeof(ggml_gtx_kernel_arg_t) == 452, kernel_arg_size);
#undef GGML_GTX_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif
