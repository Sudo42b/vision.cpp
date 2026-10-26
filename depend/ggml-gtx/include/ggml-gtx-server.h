/**
 * @file ggml_gtx_server.h
 * @brief GGML Baremetal PCIe Server Header
 *
 * PCIe-based RPC server for GTX NPU.
 * Implements command handling similar to ggml-rpc.
 *
 * Communication Flow:
 *   1. Device waits in WFI (Wait For Interrupt)
 *   2. Host sends ELBI interrupt (BAR0 offset 0x0E40)
 *   3. Device wakes up, processes message queue
 *   4. Device sends MSI to host when complete
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GGML_GTX_SERVER_H
#define GGML_GTX_SERVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * RPC Commands (aligned with ggml-rpc)
 *============================================================================*/

typedef enum {
    /* Buffer Management */
    GGML_GTX_CMD_ALLOC_BUFFER     = 0,   /* Allocate device memory */
    GGML_GTX_CMD_GET_ALIGNMENT    = 1,   /* Get buffer alignment requirement */
    GGML_GTX_CMD_GET_MAX_SIZE     = 2,   /* Get maximum buffer size */
    GGML_GTX_CMD_BUFFER_GET_BASE  = 3,   /* Get buffer base address */
    GGML_GTX_CMD_FREE_BUFFER      = 4,   /* Free device memory */
    GGML_GTX_CMD_BUFFER_CLEAR     = 5,   /* Clear buffer to value */

    /* Tensor Operations */
    GGML_GTX_CMD_SET_TENSOR       = 6,   /* Upload tensor data (Host -> Device) */
    GGML_GTX_CMD_SET_TENSOR_HASH  = 7,   /* Upload with hash validation */
    GGML_GTX_CMD_GET_TENSOR       = 8,   /* Download tensor data (Device -> Host) */
    GGML_GTX_CMD_COPY_TENSOR      = 9,   /* Copy tensor on device */

    /* Compute */
    GGML_GTX_CMD_GRAPH_COMPUTE    = 10,  /* Execute compute graph */

    /* Device Info */
    GGML_GTX_CMD_GET_DEVICE_MEMORY = 11, /* Query free/total device memory */
    GGML_GTX_CMD_INIT_TENSOR      = 12,  /* Initialize tensor */
    GGML_GTX_CMD_GET_ALLOC_SIZE   = 13,  /* Get allocation size for tensor */

    /* Handshake */
    GGML_GTX_CMD_HELLO            = 14,  /* Handshake (must be 14 for ggml-rpc compat) */
    GGML_GTX_CMD_DEVICE_COUNT     = 15,  /* Query device count */

    /* Extended */
    GGML_GTX_CMD_GRAPH_RECOMPUTE  = 16,  /* Recompute graph */

    GGML_GTX_CMD_COUNT
} ggml_gtx_cmd_t;

/*=============================================================================
 * Server Configuration
 *============================================================================*/

/**
 * @brief Server configuration
 */
typedef struct {
    void *   msg_queue_addr;     /* Message queue base address */
    size_t   msg_queue_size;     /* Message queue size */
    void *   tensor_pool_addr;   /* Tensor memory pool base */
    size_t   tensor_pool_size;   /* Tensor memory pool size */
    uint32_t msi_vector;         /* MSI vector for completion notification */
} ggml_gtx_server_config_t;

/**
 * @brief Server statistics
 */
typedef struct {
    uint64_t commands_processed;
    uint64_t bytes_received;
    uint64_t bytes_sent;
    uint64_t compute_cycles;
    uint32_t errors;
    uint32_t last_error_cmd;
} ggml_gtx_server_stats_t;

/**
 * @brief Server state
 */
typedef enum {
    GGML_GTX_SERVER_STATE_UNINITIALIZED = 0,
    GGML_GTX_SERVER_STATE_IDLE,
    GGML_GTX_SERVER_STATE_PROCESSING,
    GGML_GTX_SERVER_STATE_COMPUTING,
    GGML_GTX_SERVER_STATE_ERROR,
} ggml_gtx_server_state_t;

/*=============================================================================
 * Graph Compute Limits
 *============================================================================*/

/** Maximum number of tensors in a serialized graph (nodes + all dependencies) */
#define GGML_GTX_GRAPH_MAX_TENSORS   4096

/* Descriptor-based graph execution ABI */
#define GGML_GTX_GRAPH_V2_MAGIC       0x32585447u  /* "GTX2" little-endian */
#define GGML_GTX_GRAPH_V2_VERSION     1u

/* Launch memory modes.
 *
 * DDR_DIRECT descriptors carry original DDR tensor addresses. Firmware-side
 * wrappers may materialize tiles into L2 and call kernels with L2_STASHED_N1.
 * L2_STASHED_N1 kernels must not perform DDR<->L2 transfers.
 */
#define GGML_GTX_MEMORY_MODE_DDR_DIRECT    0u
#define GGML_GTX_MEMORY_MODE_L2_STASHED_N1 1u

#define GGML_GTX_DESC_FLAG_STASH_CHAIN     0x00000001u
/* The host has already selected one contiguous tile.  Firmware maps raw
 * memory views into L2 but must not derive an OP-specific tile/schedule. */
#define GGML_GTX_DESC_FLAG_STASH_HOST_TILED 0x00000002u
/* Execution-memory policy chosen by the host lowerer.  Exactly one of these
 * flags is required for new descriptors: L2_REQUIRED admits only the generic
 * raw-view mapper, while DDR_ALLOWED executes as a coherent DDR boundary. */
#define GGML_GTX_DESC_FLAG_L2_REQUIRED      0x00000004u
#define GGML_GTX_DESC_FLAG_DDR_ALLOWED      0x00000008u

/* A small non-DDR-backed L2 slot is kept available even for kernels that
 * advertise zero scratch. This prevents ordinary scratch use from forcing a
 * stash mapping to be reclaimed and written back. */
#define GGML_GTX_L2_MIN_SCRATCH_BYTES (64u * 1024u)

typedef enum {
    GGML_GTX_KERNEL_INVALID = 0,
    GGML_GTX_KERNEL_ADD_F16 = 1,
    GGML_GTX_KERNEL_MUL_F16 = 2,
    GGML_GTX_KERNEL_SCALE_F16 = 3,
    GGML_GTX_KERNEL_RMS_NORM_F16 = 4,
    GGML_GTX_KERNEL_SOFT_MAX_F16 = 5,
    GGML_GTX_KERNEL_MUL_MAT_F16 = 6,
    GGML_GTX_KERNEL_ROPE_F16 = 7,
    GGML_GTX_KERNEL_GET_ROWS_F16 = 8,
    GGML_GTX_KERNEL_SET_ROWS_F16 = 9,
    GGML_GTX_KERNEL_IM2COL_F16 = 10,
    GGML_GTX_KERNEL_SWIGLU_F16 = 11,
    GGML_GTX_KERNEL_CONT_F16 = 12,
    GGML_GTX_KERNEL_CPY_F16 = 13,
    GGML_GTX_KERNEL_ADD_F16_F32_TO_F16 = 14,
    GGML_GTX_KERNEL_GELU_F16 = 15,
    GGML_GTX_KERNEL_NORM_F16 = 16,
    GGML_GTX_KERNEL_REPEAT_F16 = 17,
    GGML_GTX_KERNEL_CONCAT_F16 = 18,
    GGML_GTX_KERNEL_CONV_2D_F16 = 19,
    GGML_GTX_KERNEL_CONV_2D_DW_F16 = 20,
    GGML_GTX_KERNEL_UPSCALE_F16 = 21,
    GGML_GTX_KERNEL_SIN_F16 = 22,
    GGML_GTX_KERNEL_ABS_F16 = 23,
    GGML_GTX_KERNEL_NEG_F16 = 24,
    GGML_GTX_KERNEL_SGN_F16 = 25,
    GGML_GTX_KERNEL_CEIL_F16 = 26,
    GGML_GTX_KERNEL_FLOOR_F16 = 27,
    GGML_GTX_KERNEL_TANH_F16 = 28,
    GGML_GTX_KERNEL_SILU_F16 = 29,
    GGML_GTX_KERNEL_COS_F16 = 30,
    GGML_GTX_KERNEL_PAD_F16 = 31,
    GGML_GTX_KERNEL_RELU_F16 = 32,
    GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16 = 33,
    GGML_GTX_KERNEL_LEAKY_RELU_F16 = 34,
    GGML_GTX_KERNEL_CLAMP_F16 = 35,
} ggml_gtx_kernel_id_t;

/* GGML owns CONV op_params[0..5]. Host-lowered layout metadata uses the
 * remaining descriptor slots and is consumed only by the uploaded kernel. */
#define GGML_GTX_CONV_LAYOUT_PARAM       15
#define GGML_GTX_CONV_LAYOUT_C_FAST      1
#define GGML_GTX_CONV_LOGICAL_OW_PARAM   6
#define GGML_GTX_CONV_LOGICAL_OH_PARAM   7
#define GGML_GTX_CONV_LOGICAL_C_PARAM    8
#define GGML_GTX_CONV_LOGICAL_IW_PARAM   9
#define GGML_GTX_CONV_LOGICAL_IH_PARAM  10
#define GGML_GTX_CONV_LOGICAL_KW_PARAM  11
#define GGML_GTX_CONV_LOGICAL_KH_PARAM  12

/*=============================================================================
 * RPC Message Structures (packed, 8-byte aligned)
 *============================================================================*/

#pragma pack(push, 1)

/**
 * @brief RPC tensor descriptor for tensor transfer and copy commands.
 *
 * This remains compatible with the current tensor upload/download RPC path.
 * GRAPH_COMPUTE no longer consumes serialized ggml tensors; it uses the v2
 * descriptor payload below.
 */
typedef struct {
    uint64_t id;                    /* Tensor ID */
    uint32_t type;                  /* ggml_type */
    uint64_t buffer;                /* Buffer pointer */
    uint32_t ne[4];                 /* Shape */
    uint32_t nb[4];                 /* Strides */
    uint32_t op;                    /* Operation */
    int32_t  op_params[16];         /* Operation parameters */
    int32_t  flags;                 /* Flags */
    uint64_t src[10];               /* Source tensors */
    uint64_t view_src;              /* View source */
    uint64_t view_offs;             /* View offset */
    uint64_t data;                  /* Data pointer */
    char     name[64];              /* Tensor name */
    char     padding[4];            /* Alignment padding */
} ggml_gtx_rpc_tensor_t;

/**
 * @brief Descriptor-based graph payload header.
 *
 * Host staging format:
 *   | ggml_gtx_graph_v2_header_t |
 *   | desc[n_desc] (each sizeof(ggml_gtx_kernel_launch_desc_t)) |
 */
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint32_t device;
    uint32_t n_desc;
    uint32_t desc_size;
    uint32_t flags;
    uint32_t reserved;
} ggml_gtx_graph_v2_header_t;

/**
 * @brief Transport-independent kernel launch descriptor.
 *
 * Built-in firmware kernels and future uploaded kernels should share this
 * launch ABI shape: kernel_id selects executable code, args describe raw
 * device addresses and dimensions.
 */
typedef struct {
    uint32_t kernel_id;
    uint32_t abi_version;
    uint32_t dtype;
    uint32_t reserved0;
    uint64_t args_addr;
    uint32_t args_size;
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t shared_mem;
    uint32_t flags;

    uint64_t dst_addr;
    uint64_t src_addr[4];
    uint32_t ne[4];
    uint32_t nb[4];
    uint32_t src_type[4];
    uint32_t src_ne[4][4];
    uint32_t src_nb[4][4];
    int32_t  op_params[16];

    /* L2-stashed execution view.
     *
     * DDR tensor identity remains in src_addr/dst_addr plus ne/nb above.
     * When memory_mode is L2_STASHED_N1, l2_* fields describe compact
     * contiguous per-nest tiles and the kernel executes only L2->L1/SPU work.
     */
    uint32_t memory_mode;
    uint32_t nest_id;
    uint64_t l2_dst_addr;
    uint64_t l2_src_addr[4];
    uint32_t l2_ne[4];
    uint32_t elem_size;
    /* args_addr/args_size above identify an optional host-uploaded opaque
     * payload. The control core passes it directly to the uploaded kernel;
     * firmware and stash never stage or interpret it in L2.
     *
     * l2_scratch_size is the kernel-owned requirement advertised by the host.
     * Firmware fills addr/capacity without interpreting the kernel or its
     * arguments. Scratch is never DDR-backed or written back by stash. */
    uint32_t l2_scratch_size;
    uint32_t l2_scratch_addr;
    uint32_t l2_scratch_capacity;
    uint32_t reserved[4];
} ggml_gtx_kernel_launch_desc_t;

/**
 * @brief Hello response
 */
typedef struct {
    uint8_t major;                  /* Version major */
    uint8_t minor;                  /* Version minor */
    uint8_t patch;                  /* Version patch */
} ggml_gtx_msg_hello_rsp_t;

/**
 * @brief Device count response
 */
typedef struct {
    uint32_t device_count;
} ggml_gtx_msg_device_count_rsp_t;

/**
 * @brief Alloc buffer request
 */
typedef struct {
    uint32_t device;
    uint64_t size;
} ggml_gtx_msg_alloc_buffer_req_t;

/**
 * @brief Alloc buffer response
 */
typedef struct {
    uint64_t remote_ptr;
    uint64_t remote_size;
} ggml_gtx_msg_alloc_buffer_rsp_t;

/**
 * @brief Get alignment request
 */
typedef struct {
    uint32_t device;
} ggml_gtx_msg_get_alignment_req_t;

/**
 * @brief Get alignment response
 */
typedef struct {
    uint64_t alignment;
} ggml_gtx_msg_get_alignment_rsp_t;

/**
 * @brief Get max size request
 */
typedef struct {
    uint32_t device;
} ggml_gtx_msg_get_max_size_req_t;

/**
 * @brief Get max size response
 */
typedef struct {
    uint64_t max_size;
} ggml_gtx_msg_get_max_size_rsp_t;

/**
 * @brief Free buffer request
 */
typedef struct {
    uint64_t remote_ptr;
} ggml_gtx_msg_free_buffer_req_t;

/**
 * @brief Buffer clear request
 */
typedef struct {
    uint64_t remote_ptr;
    uint8_t  value;
} ggml_gtx_msg_buffer_clear_req_t;

/**
 * @brief Get tensor request
 */
typedef struct {
    ggml_gtx_rpc_tensor_t tensor;
    uint64_t offset;
    uint64_t size;
} ggml_gtx_msg_get_tensor_req_t;

/**
 * @brief Copy tensor request
 */
typedef struct {
    ggml_gtx_rpc_tensor_t src;
    ggml_gtx_rpc_tensor_t dst;
} ggml_gtx_msg_copy_tensor_req_t;

/**
 * @brief Copy tensor response
 */
typedef struct {
    uint8_t result;
} ggml_gtx_msg_copy_tensor_rsp_t;

/**
 * @brief Get device memory request
 */
typedef struct {
    uint32_t device;
} ggml_gtx_msg_get_device_memory_req_t;

/**
 * @brief Get device memory response
 */
typedef struct {
    uint64_t free_mem;
    uint64_t total_mem;
} ggml_gtx_msg_get_device_memory_rsp_t;

#pragma pack(pop)

/*=============================================================================
 * Server API
 *============================================================================*/

/**
 * @brief Initialize the RPC server
 * @param config Server configuration
 * @return 0 on success, negative on error
 */
int ggml_gtx_server_init(const ggml_gtx_server_config_t *config);

/**
 * @brief Run the server main loop
 *
 * This function never returns under normal operation.
 * It enters WFI when idle and wakes on ELBI interrupt.
 */
void ggml_gtx_server_run(void);

/**
 * @brief Process one command iteration
 *
 * Called from interrupt handler or polling loop.
 * Processes available messages and sends responses.
 *
 * @return Number of commands processed
 */
uint32_t ggml_gtx_server_process(void);

/**
 * @brief Get server state
 * @return Current server state
 */
ggml_gtx_server_state_t ggml_gtx_server_get_state(void);

/**
 * @brief Get server statistics
 * @param stats Pointer to store statistics
 */
void ggml_gtx_server_get_stats(ggml_gtx_server_stats_t *stats);

/**
 * @brief Reset server statistics
 */
void ggml_gtx_server_reset_stats(void);

/**
 * @brief Shutdown the server
 */
void ggml_gtx_server_shutdown(void);

/*=============================================================================
 * Interrupt Handlers
 *============================================================================*/

/**
 * @brief ELBI interrupt handler
 *
 * Called from vector_table.c when Host sends doorbell interrupt.
 * Sets flag to wake up server loop.
 *
 * @param source Interrupt source (0-31)
 * @param user_data User data (unused)
 */
void ggml_gtx_server_elbi_handler(uint32_t source, void *user_data);

/**
 * @brief Check if server has pending work
 * @return true if work pending
 */
bool ggml_gtx_server_has_pending_work(void);

/**
 * @brief Clear pending work flag
 */
void ggml_gtx_server_clear_pending(void);

/*=============================================================================
 * Version Info
 *============================================================================*/

#define GGML_GTX_SERVER_VERSION_MAJOR  0
#define GGML_GTX_SERVER_VERSION_MINOR  1
#define GGML_GTX_SERVER_VERSION_PATCH  0

#ifdef __cplusplus
}
#endif

#endif /* GGML_GTX_SERVER_H */
