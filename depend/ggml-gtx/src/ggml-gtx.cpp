/**
 * @file ggml-gtx.cpp
 * @brief GGML GTX NPU Backend Implementation
 *
 * Host-side GGML backend for GTX NPU accelerator.
 * Communicates with the NPU firmware via PCIe.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ggml-gtx.h"
#include "ggml-gtx-baremetal.h"
#include "ggml-gtx/ggml-gtx-kernel-args.h"
#include "ggml-gtx/ggml-gtx-protocol.h"
#include "ggml-gtx-server.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-backend-impl.h"
#include "pcie_irq_host.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <limits>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
/* Memory barrier for Windows */
#define MEMORY_BARRIER() _ReadWriteBarrier(); MemoryBarrier()
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <poll.h>
#include <signal.h>
/* Memory barrier for GCC/Clang */
#define MEMORY_BARRIER() __sync_synchronize()
#endif

/*=============================================================================
 * Debug Logging
 *============================================================================*/

static bool g_gtx_debug = false;

static std::string gtx_default_vfio_socket_path() {
    return "/tmp/gtx-ggml-cuda-vfio.sock";
}

static const char * gtx_selected_transport() {
    const char * transport = getenv("GTX_TRANSPORT");
    return transport != nullptr && transport[0] != '\0' ? transport : "xdma";
}

static bool gtx_use_uart_vp_transport() {
    const char * transport = gtx_selected_transport();
    return strcmp(transport, "uart-vp") == 0 || strcmp(transport, "xdma") == 0;
}

static bool gtx_use_xdma_stream_transport() {
    return strcmp(gtx_selected_transport(), "xdma") == 0;
}

static const char * gtx_env_or_default(const char * name, const char * fallback) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' ? value : fallback;
}

static bool gtx_env_truthy(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' &&
           strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

static bool gtx_log_ops_enabled() {
    return gtx_env_truthy("GTX_LOG_OPS");
}

static bool gtx_log_supports_ops_enabled() {
    return gtx_env_truthy("GTX_LOG_SUPPORTS_OPS");
}

static bool gtx_log_chain_ops_enabled() {
    return gtx_env_truthy("GTX_LOG_CHAIN_OPS");
}

static bool gtx_tiling_scheduler_enabled() {
    return gtx_env_truthy("GTX_ENABLE_TILING_SCHED");
}

static bool gtx_dump_graph_types_enabled() {
    return gtx_env_truthy("GTX_GRAPH_TYPES");
}

static bool gtx_log_softmax_enabled() {
    return gtx_env_truthy("GTX_LOG_SOFTMAX");
}

static bool gtx_dump_op_samples_enabled() {
    return gtx_env_truthy("GTX_DUMP_OP_SAMPLES");
}

static bool gtx_check_mul_mat_enabled() {
    return gtx_env_truthy("GTX_CHECK_MUL_MAT");
}

static bool gtx_check_ops_enabled() {
    return gtx_env_truthy("GTX_CHECK_OPS");
}

static uint64_t gtx_cpu_ref_graph_sequence = 0;
static uint64_t gtx_cpu_ref_current_graph = 0;
static bool gtx_cpu_ref_first_divergence_found = false;

static bool gtx_cpu_ref_enabled() {
    const char * dir = getenv("GTX_CPU_REF_DIR");
    return dir != nullptr && dir[0] != '\0';
}

static bool gtx_cpu_ref_path(char * path, size_t size, int node_index) {
    const char * dir = getenv("GTX_CPU_REF_DIR");
    if (dir == nullptr || dir[0] == '\0' || path == nullptr || size == 0 || node_index < 0) {
        return false;
    }
    const int n = snprintf(path, size, "%s/graph_%03llu_node_%04d.f32",
                           dir, (unsigned long long)gtx_cpu_ref_current_graph, node_index);
    return n > 0 && (size_t)n < size;
}

static bool gtx_cpu_ref_exists(int node_index) {
    char path[1024];
    if (!gtx_cpu_ref_path(path, sizeof(path), node_index)) {
        return false;
    }
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    fclose(file);
    return true;
}

static float gtx_float_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static const char * gtx_op_name_or_unknown(uint32_t raw_op) {
    if (raw_op < (uint32_t) GGML_OP_COUNT) {
        return ggml_op_name((ggml_op) raw_op);
    }
    if (raw_op == (uint32_t) GGML_OP_COUNT) {
        return "HOST_OP_COUNT_OR_SHIFTED_GLU";
    }
    return "UNKNOWN_HOST_OP";
}

static const char * gtx_normalized_op_name(uint32_t op) {
    if (op == UINT32_MAX) {
        return "UNSUPPORTED_HOST_OP";
    }
    if (op == (uint32_t) GGML_OP_CONV_TRANSPOSE_2D) {
        return "CONV_TRANSPOSE_2D";
    }
    if (op == (uint32_t) GGML_OP_UPSCALE) {
        return "UPSCALE";
    }
    if (op == (uint32_t) GGML_OP_PAD) {
        return "PAD";
    }
    if (op == (uint32_t) GGML_OP_LEAKY_RELU) {
        return "LEAKY_RELU";
    }
    if (op == (uint32_t) GGML_OP_CLAMP) {
        return "CLAMP";
    }
    if (op == (uint32_t) GGML_OP_GLU) {
        return "GLU";
    }
    if (op == (uint32_t) GGML_OP_UNARY) {
        return "UNARY";
    }
    return gtx_op_name_or_unknown(op);
}

static bool gtx_env_list_contains(const char * name, const char * needle) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0' || needle == nullptr || needle[0] == '\0') {
        return false;
    }

    const std::string haystack(value);
    size_t start = 0;
    while (start <= haystack.size()) {
        size_t end = haystack.find(',', start);
        if (end == std::string::npos) {
            end = haystack.size();
        }
        const std::string item = haystack.substr(start, end - start);
        if (item == needle) {
            return true;
        }
        if (end == haystack.size()) {
            break;
        }
        start = end + 1;
    }
    return false;
}

static bool gtx_env_substr_matches(const char * name, const char * value_to_check) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' &&
           value_to_check != nullptr && strstr(value_to_check, value) != nullptr;
}

static size_t gtx_env_size_kib_or_default(const char * name, size_t fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    unsigned long long kib = strtoull(value, &end, 10);
    if (end == value || kib == 0) {
        return fallback;
    }
    return (size_t) kib * 1024ULL;
}

static uint64_t gtx_env_u64_or_default(const char * name, uint64_t fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char * end = nullptr;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (end == value || errno != 0) {
        return fallback;
    }
    return (uint64_t) parsed;
}

#define GTX_LOG(...) \
    do { if (g_gtx_debug) fprintf(stderr, "[GTX] " __VA_ARGS__); } while(0)

#define GTX_LOG_WARN(...) \
    fprintf(stderr, "[GTX WARN] " __VA_ARGS__)

#define GTX_LOG_ERROR(...) \
    fprintf(stderr, "[GTX ERROR] " __VA_ARGS__)

static uint16_t gtx_fp32_to_fp16(float x) {
    return GGML_FP32_TO_FP16(x);
}

static float gtx_fp16_to_fp32(uint16_t h) {
    return GGML_FP16_TO_FP32(h);
}

static bool gtx_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool gtx_name_ends_with(const char * name, const char * suffix) {
    if (name == nullptr || suffix == nullptr) {
        return false;
    }
    const size_t name_len = strlen(name);
    const size_t suffix_len = strlen(suffix);
    return name_len >= suffix_len && strcmp(name + name_len - suffix_len, suffix) == 0;
}

static bool gtx_name_equals(const char * name, const char * expected) {
    return name != nullptr && expected != nullptr && strcmp(name, expected) == 0;
}

static float gtx_env_float_or_default(const char * name, float fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    const float parsed = strtof(value, &end);
    if (end == value || !std::isfinite(parsed) || parsed <= 0.0f) {
        return fallback;
    }
    return parsed;
}

static bool gtx_is_norm_weight_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr && gtx_name_ends_with(tensor->name, "norm.weight");
}

static bool gtx_is_bias_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr && strstr(tensor->name, "bias") != nullptr;
}

static bool gtx_is_output_head_weight_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           (gtx_name_equals(tensor->name, "output_norm.weight") ||
            gtx_name_equals(tensor->name, "token_embd.weight"));
}

static bool gtx_mul_norm_matmul_scale_enabled() {
    static const bool enabled = gtx_env_enabled("GTX_MUL_NORM_MATMUL_SCALE");
    return enabled;
}

static bool gtx_norm_weight_f32_upload_enabled() {
    return gtx_env_truthy("GTX_NORM_WEIGHT_F32_UPLOAD");
}

static bool gtx_bias_f32_upload_enabled() {
    return gtx_env_truthy("GTX_BIAS_F32_UPLOAD");
}

static float gtx_head_weight_scale() {
    static const float scale = gtx_env_float_or_default("GTX_HEAD_WEIGHT_SCALE", 1.0f);
    return scale;
}

static bool gtx_is_mul_norm_result(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           tensor->op == GGML_OP_MUL &&
           gtx_is_norm_weight_tensor(tensor->src[1]);
}

static bool gtx_is_mul_norm_matmul_result(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           tensor->op == GGML_OP_MUL_MAT &&
           gtx_is_mul_norm_result(tensor->src[1]);
}

#define GGML_GTX_MUL_NORM_UPLOAD_SCALE 0.001f
#define GGML_GTX_MUL_NORM_HOST_RESCALE 1000.0f

static float gtx_upload_f16_scale(const ggml_tensor * tensor) {
    const float head_scale = gtx_head_weight_scale();
    if (head_scale != 1.0f && gtx_is_output_head_weight_tensor(tensor)) {
        return head_scale;
    }

    if (gtx_mul_norm_matmul_scale_enabled() && gtx_is_norm_weight_tensor(tensor)) {
        return GGML_GTX_MUL_NORM_UPLOAD_SCALE;
    }

    return 1.0f;
}

static enum ggml_type gtx_remote_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_Q8_0:
            return GGML_TYPE_F16;
        default:
            return type;
    }
}

static bool gtx_preserve_remote_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }

    return (gtx_is_norm_weight_tensor(tensor) && gtx_norm_weight_f32_upload_enabled()) ||
           (gtx_is_bias_tensor(tensor) && gtx_bias_f32_upload_enabled());
}

static enum ggml_type gtx_remote_tensor_type(const ggml_tensor * tensor) {
    if (gtx_preserve_remote_f32(tensor)) {
        return GGML_TYPE_F32;
    }
    return tensor != nullptr ? gtx_remote_type(tensor->type) : GGML_TYPE_F32;
}

static size_t gtx_remote_size_bytes(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return 0;
    }

    if (gtx_preserve_remote_f32(tensor)) {
        return ggml_nbytes(tensor);
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_Q8_0) {
        return (size_t) ggml_nelements(tensor) * sizeof(uint16_t);
    }

    return ggml_nbytes(tensor);
}

static uint64_t gtx_remote_byte_offset(const ggml_tensor * tensor, uint64_t local_bytes) {
    if (tensor != nullptr) {
        if (gtx_preserve_remote_f32(tensor)) {
            return local_bytes;
        }

        if (tensor->type == GGML_TYPE_F32) {
            return (local_bytes / sizeof(float)) * sizeof(uint16_t);
        }

        if (tensor->type == GGML_TYPE_Q8_0) {
            const uint64_t type_size = (uint64_t) ggml_type_size(tensor->type);
            const uint64_t blck_size = (uint64_t) ggml_blck_size(tensor->type);
            return (local_bytes / type_size) * blck_size * sizeof(uint16_t);
        }
    }

    return local_bytes;
}

static uint64_t gtx_remote_data_ptr(const ggml_tensor * tensor);

static std::vector<uint8_t> gtx_pack_tensor_chunk(const ggml_tensor * tensor, const void * data, size_t size) {
    if (tensor->type == GGML_TYPE_Q8_0) {
        const size_t block_size = ggml_blck_size(tensor->type);
        const size_t type_size = ggml_type_size(tensor->type);
        const size_t n_blocks = size / type_size;
        const size_t n = n_blocks * block_size;

        std::vector<float> tmp(n);
        dequantize_row_q8_0((const block_q8_0 *) data, tmp.data(), (int64_t) n);

        std::vector<uint8_t> packed(n * sizeof(uint16_t));
        uint16_t * dst = (uint16_t *) packed.data();
        const float scale = gtx_upload_f16_scale(tensor);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = gtx_fp32_to_fp16(tmp[i] * scale);
        }

        return packed;
    }

    if (tensor->type != GGML_TYPE_F32 || gtx_preserve_remote_f32(tensor)) {
        const uint8_t * bytes = (const uint8_t *) data;
        return std::vector<uint8_t>(bytes, bytes + size);
    }

    const size_t n = size / sizeof(float);
    const float * src = (const float *) data;
    std::vector<uint8_t> packed(n * sizeof(uint16_t));
    uint16_t * dst = (uint16_t *) packed.data();
    const float scale = gtx_upload_f16_scale(tensor);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = gtx_fp32_to_fp16(src[i] * scale);
    }
    return packed;
}

static void gtx_unpack_tensor_chunk(const ggml_tensor * tensor, const uint8_t * src, void * data, size_t size) {
    if (tensor->type != GGML_TYPE_F32 || gtx_preserve_remote_f32(tensor)) {
        memcpy(data, src, size);
        return;
    }

    const size_t n = size / sizeof(float);
    const uint16_t * src16 = (const uint16_t *) src;
    float * dst = (float *) data;
    const float scale =
        gtx_mul_norm_matmul_scale_enabled() && gtx_is_mul_norm_matmul_result(tensor) ?
        GGML_GTX_MUL_NORM_HOST_RESCALE : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = gtx_fp16_to_fp32(src16[i]) * scale;
    }
}

static size_t gtx_upload_local_chunk_size(const ggml_tensor * tensor, size_t remaining, size_t max_chunk) {
    if (tensor != nullptr && tensor->type == GGML_TYPE_Q8_0) {
        const size_t type_size = ggml_type_size(tensor->type);
        const size_t packed_block_size = (size_t) ggml_blck_size(tensor->type) * sizeof(uint16_t);
        size_t chunk = std::min(remaining, (max_chunk / packed_block_size) * type_size);
        if (chunk == 0) {
            chunk = std::min(remaining, type_size);
        }
        return chunk;
    }

    if (tensor != nullptr && tensor->type == GGML_TYPE_F32) {
        const size_t packed_element_size =
            gtx_preserve_remote_f32(tensor) ? sizeof(float) : sizeof(uint16_t);
        const size_t max_elements = max_chunk / packed_element_size;
        size_t chunk = std::min(remaining, max_elements * sizeof(float));
        if (chunk == 0) {
            chunk = std::min(remaining, sizeof(float));
        }
        return chunk;
    }

    return std::min(remaining, max_chunk);
}

static size_t ggml_backend_gtx_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void) buft;
    return std::max(gtx_remote_size_bytes(tensor), ggml_nbytes(tensor));
}

/*=============================================================================
 * RPC Protocol (must match ggml_gtx_server.h)
 *============================================================================*/

enum gtx_cmd {
    GTX_CMD_ALLOC_BUFFER     = 0,
    GTX_CMD_GET_ALIGNMENT    = 1,
    GTX_CMD_GET_MAX_SIZE     = 2,
    GTX_CMD_BUFFER_GET_BASE  = 3,
    GTX_CMD_FREE_BUFFER      = 4,
    GTX_CMD_BUFFER_CLEAR     = 5,
    GTX_CMD_SET_TENSOR       = 6,
    GTX_CMD_SET_TENSOR_HASH  = 7,
    GTX_CMD_GET_TENSOR       = 8,
    GTX_CMD_COPY_TENSOR      = 9,
    GTX_CMD_GRAPH_COMPUTE    = 10,
    GTX_CMD_GET_DEVICE_MEMORY = 11,
    GTX_CMD_INIT_TENSOR      = 12,
    GTX_CMD_GET_ALLOC_SIZE   = 13,
    GTX_CMD_HELLO            = 14,
    GTX_CMD_DEVICE_COUNT     = 15,
    GTX_CMD_GRAPH_RECOMPUTE  = 16,
    GTX_CMD_COUNT
};

static const char * gtx_cmd_name(uint8_t cmd) {
    switch (cmd) {
        case GTX_CMD_ALLOC_BUFFER:      return "ALLOC_BUFFER";
        case GTX_CMD_GET_ALIGNMENT:     return "GET_ALIGNMENT";
        case GTX_CMD_GET_MAX_SIZE:      return "GET_MAX_SIZE";
        case GTX_CMD_BUFFER_GET_BASE:   return "BUFFER_GET_BASE";
        case GTX_CMD_FREE_BUFFER:       return "FREE_BUFFER";
        case GTX_CMD_BUFFER_CLEAR:      return "BUFFER_CLEAR";
        case GTX_CMD_SET_TENSOR:        return "SET_TENSOR";
        case GTX_CMD_SET_TENSOR_HASH:   return "SET_TENSOR_HASH";
        case GTX_CMD_GET_TENSOR:        return "GET_TENSOR";
        case GTX_CMD_COPY_TENSOR:       return "COPY_TENSOR";
        case GTX_CMD_GRAPH_COMPUTE:     return "GRAPH_COMPUTE";
        case GTX_CMD_GET_DEVICE_MEMORY: return "GET_DEVICE_MEMORY";
        case GTX_CMD_INIT_TENSOR:       return "INIT_TENSOR";
        case GTX_CMD_GET_ALLOC_SIZE:    return "GET_ALLOC_SIZE";
        case GTX_CMD_HELLO:             return "HELLO";
        case GTX_CMD_DEVICE_COUNT:      return "DEVICE_COUNT";
        case GTX_CMD_GRAPH_RECOMPUTE:   return "GRAPH_RECOMPUTE";
        default:                        return "UNKNOWN";
    }
}

#pragma pack(push, 1)

/* 64-byte message structure (must match device) */
struct gtx_message {
    uint32_t seq_num;
    uint32_t type;
    uint32_t resource;
    uint32_t size;
    uint32_t flags;
    uint32_t reserved;
    uint64_t address;
    uint64_t value;
    uint8_t  payload[24];
};
static_assert(sizeof(gtx_message) == 64, "gtx_message must be 64 bytes");

/* Response flag bit */
#define GTX_FLAG_RESPONSE (1U << 31)

/* RPC message structures */
struct gtx_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
};

struct gtx_msg_device_count_rsp {
    uint32_t device_count;
};

struct gtx_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct gtx_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct gtx_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct gtx_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct gtx_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct gtx_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t  value;
};

struct gtx_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

/* RPC tensor (serialized ggml_tensor) */
struct gtx_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    // ⚠️ **와이어 포맷이라 `GGML_MAX_NAME` 를 따라가면 안 된다.** 펌웨어 쪽
    //    `firmware/include/ggml_gtx_rpc_protocol.h` 가 `char name[64]` 로 하드코딩돼 있다.
    //    정본 트리(`device/zephyr-host`)는 자기 `ggml.h` 의 `GGML_MAX_NAME` 이 64라
    //    그대로 두어도 맞지만, vision.cpp 의 ggml 은 **128** 이라 옮겨오면 구조체가
    //    어긋나 RPC 가 조용히 깨진다. 정본을 다시 가져올 때마다 이 자리를 다시 박는다.
    char     name[64];
    char     padding[4];
};

struct gtx_msg_get_tensor_req {
    gtx_rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct gtx_msg_copy_tensor_req {
    gtx_rpc_tensor src;
    gtx_rpc_tensor dst;
};

struct gtx_msg_copy_tensor_rsp {
    uint8_t result;
};

#pragma pack(pop)

static gtx_rpc_tensor gtx_serialize_tensor(const ggml_tensor * tensor);

/*=============================================================================
 * PCIe Communication Layer (vfio-user based)
 *============================================================================*/

/* PCIe BAR memory map offsets */
#define GTX_BAR_ELBI_INT_GEN    0x0E40    /* Host -> Device interrupt (in BAR0) */

/* Message queue is accessed via BAR2 (maps to device memory 0x60000000)
 * Layout matches ggml_baremetal/drivers/message.c:
 *   [0-3]:   RX head (host writes, device reads)
 *   [4-7]:   RX tail (device writes, host reads)
 *   [8-11]:  TX head (device writes, host reads)
 *   [12-15]: TX tail (host writes, device reads)
 *   [64...]: RX queue data, then TX queue data
 */
#define GTX_MSG_QUEUE_BASE      0x00      /* Message queue at BAR2 offset 0 */
#define GTX_MSG_QUEUE_RX_HEAD   0x00      /* RX head */
#define GTX_MSG_QUEUE_RX_TAIL   0x04      /* RX tail */
#define GTX_MSG_QUEUE_TX_HEAD   0x08      /* TX head */
#define GTX_MSG_QUEUE_TX_TAIL   0x0C      /* TX tail */
#define GTX_MSG_QUEUE_HEADER    64        /* Header size (64 bytes aligned) */
#define GTX_MSG_QUEUE_SIZE      (64 * 1024)  /* 64KB total */
#define GTX_MSG_QUEUE_ENTRIES   ((GTX_MSG_QUEUE_SIZE - GTX_MSG_QUEUE_HEADER) / 2 / 64) /* ~511 entries per queue */
#define GTX_MSG_QUEUE_RX_DATA   GTX_MSG_QUEUE_HEADER   /* RX queue data starts at offset 64 */
#define GTX_MSG_QUEUE_TX_DATA   (GTX_MSG_QUEUE_RX_DATA + GTX_MSG_QUEUE_ENTRIES * 64) /* TX queue after RX */

/* BAR2 shared-memory staging area for large request/response payloads */
#define GTX_BAR2_DEVICE_BASE        0x60000000ULL
#define GTX_BAR2_STAGING_OFFSET     0x00010000ULL
#define GTX_BAR2_STAGING_SIZE       (8ULL * 1024ULL * 1024ULL)
#define GTX_BAR2_UPLOAD_CHUNK       (512ULL * 1024ULL)
#define GTX_BAR2_DOWNLOAD_CHUNK     (16ULL * 1024ULL)
#define GTX_XDMA_DRIVER_CHUNK       (255ULL * 4096ULL)
#define GTX_CMD_TIMEOUT_MS          30000
#define GTX_GRAPH_COMPUTE_TIMEOUT_MS 600000
#define GTX_UART_TRANSFER_TIMEOUT_MS 600000
#define GTX_UART_KERNEL_TIMEOUT_MS  600000

#define GTX_DIRECT_DDR_MAGIC          0x31524447u
#define GTX_DIRECT_DDR_OP_WRITE       1u
#define GTX_DIRECT_DDR_OP_READ        2u

struct gtx_direct_ddr_header {
    uint32_t magic;
    uint32_t op;
    uint64_t addr;
    uint32_t size;
    uint32_t status;
};

static_assert(GTX_BAR2_UPLOAD_CHUNK <= GTX_BAR2_STAGING_SIZE,
              "BAR2 upload chunk must fit in staging area");
static_assert(GTX_BAR2_DOWNLOAD_CHUNK <= GTX_BAR2_STAGING_SIZE,
              "BAR2 download chunk must fit in staging area");

static size_t gtx_upload_chunk_size() {
    const size_t default_chunk = gtx_use_xdma_stream_transport() ?
        GTX_XDMA_DRIVER_CHUNK : GTX_BAR2_UPLOAD_CHUNK;
    size_t chunk = gtx_env_size_kib_or_default("GTX_UPLOAD_CHUNK_KIB", default_chunk / 1024ULL);
    if (chunk < 4ULL * 1024ULL) {
        chunk = 4ULL * 1024ULL;
    }
    if (chunk > GTX_BAR2_STAGING_SIZE) {
        chunk = GTX_BAR2_STAGING_SIZE;
    }
    return chunk;
}

/* BAR region indices */
#define GTX_BAR0_REGION         0         /* Registers (doorbell, etc.) */
#define GTX_BAR2_REGION         2         /* Shared memory / Message queue */

class gtx_pcie_connection {
public:
    gtx_pcie_connection() : m_irq_ctx(nullptr), m_seq_num(0) {}
    ~gtx_pcie_connection() { close(); }

    bool open(const char* device_path, int bar_index = 0);
    void close();
    bool is_open() const { return m_irq_ctx != nullptr; }

    /* Send command and wait for response */
    bool send_cmd(uint8_t cmd, const void* request, size_t req_size,
                  void* response, size_t rsp_size);

    bool send_cmd_staged(uint8_t cmd, uint64_t address, uint32_t size,
                         void* response, size_t rsp_size);

    /* Send data to device BAR2 (for large transfers) */
    bool send_data(uint64_t offset, const void* data, size_t size);

    /* Receive data from device BAR2 */
    bool recv_data(uint64_t offset, void* data, size_t size);

    /* Send doorbell interrupt to device */
    void send_doorbell(uint32_t source = 0);

    /* Wait for MSI from device */
    bool wait_msi(int vector, int timeout_ms);

    /* Get IRQ context (for advanced use) */
    gtx_pcie_irq_ctx* get_irq_ctx() const { return m_irq_ctx; }

private:
    /* vfio-user IRQ context */
    gtx_pcie_irq_ctx* m_irq_ctx;

    /* Sequence number for request matching */
    uint32_t m_seq_num;

    /* Mutex for thread safety */
    std::mutex m_mutex;
};

struct ggml_backend_gtx_buffer_context;

#ifndef _WIN32
struct gtx_uart_kernel_bundle {
    uint32_t kernel_id;
    const char * relative_path;
    const char * entry_symbol;
    const char * name;
};

static constexpr uint32_t GTX_KERNEL_WORK_SLOT_COUNT = 32;
static constexpr uint64_t GTX_FPGA_KERNEL_WORK_ADDR = 0x00040000ull;
static constexpr uint64_t GTX_FPGA_KERNEL_WORK_SIZE = 0x00010000ull;
static constexpr uint32_t GTX_FPGA_KERNEL_WORK_SLOT_COUNT = 28;
// Keep the first 256 MiB for the FPGA RPC rings, arguments, stats, and kernel
// staging.  The CPU-visible DDR window continues from 0x380000000 through
// 0x3ffffffff, providing a separate contiguous 2 GiB tensor arena.
static constexpr uint64_t GTX_FPGA_TENSOR_ARENA_OFFSET = 0x10000000ull;
static constexpr uint64_t GTX_FPGA_TENSOR_ARENA_SIZE   = 0x80000000ull;
static constexpr uint64_t GTX_STASH_L2_BYTES = 10ull * 1024ull * 1024ull;
static constexpr uint64_t GTX_STASH_MIN_SCRATCH_BYTES = GGML_GTX_L2_MIN_SCRATCH_BYTES;
static constexpr uint64_t GTX_STASH_SLOT_COUNT = 3ull;

static std::string gtx_kernel_bundle_dir() {
    return gtx_env_or_default("GTX_KERNEL_DIR", "build-kernels");
}

static std::string gtx_join_path(const std::string & dir, const char * file) {
    if (file == nullptr || file[0] == '\0') {
        return dir;
    }
    if (file[0] == '/') {
        return file;
    }
    if (dir.empty() || dir == ".") {
        return file;
    }
    if (dir.back() == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

static const gtx_uart_kernel_bundle * gtx_uart_kernel_bundle_for(uint32_t kernel_id) {
    static const gtx_uart_kernel_bundle bundles[] = {
        {
            GGML_GTX_KERNEL_ADD_F16,
            "add.llext",
            "ggml_gtx_kernel_add_f16",
            "add-f16",
        },
        {
            GGML_GTX_KERNEL_MUL_F16,
            "mul.llext",
            "ggml_gtx_kernel_mul_f16",
            "mul-f16",
        },
        {
            GGML_GTX_KERNEL_SCALE_F16,
            "scale.llext",
            "ggml_gtx_kernel_scale_f16",
            "scale-f16",
        },
        {
            GGML_GTX_KERNEL_RMS_NORM_F16,
            "rms-norm.llext",
            "ggml_gtx_kernel_rms_norm_f16",
            "rms-norm-f16",
        },
        {
            GGML_GTX_KERNEL_NORM_F16,
            "norm.llext",
            "ggml_gtx_kernel_norm_f16",
            "norm-f16",
        },
        {
            GGML_GTX_KERNEL_REPEAT_F16,
            "repeat.llext",
            "ggml_gtx_kernel_repeat_f16",
            "repeat-f16",
        },
        {
            GGML_GTX_KERNEL_CONCAT_F16,
            "concat.llext",
            "ggml_gtx_kernel_concat_f16",
            "concat-f16",
        },
        {
            GGML_GTX_KERNEL_SOFT_MAX_F16,
            "soft-max.llext",
            "ggml_gtx_kernel_soft_max_f16",
            "soft-max-f16",
        },
        {
            GGML_GTX_KERNEL_GELU_F16,
            "gelu.llext",
            "ggml_gtx_kernel_gelu_f16",
            "gelu-f16",
        },
        {
            GGML_GTX_KERNEL_MUL_MAT_F16,
            "mul-mat.llext",
            "ggml_gtx_kernel_mul_mat_f16",
            "mul-mat-f16",
        },
        {
            GGML_GTX_KERNEL_ROPE_F16,
            "rope.llext",
            "ggml_gtx_kernel_rope_f16",
            "rope-f16",
        },
        {
            GGML_GTX_KERNEL_GET_ROWS_F16,
            "get-rows.llext",
            "ggml_gtx_kernel_get_rows_f16",
            "get-rows-f16",
        },
        {
            GGML_GTX_KERNEL_SET_ROWS_F16,
            "set-rows.llext",
            "ggml_gtx_kernel_set_rows_f16",
            "set-rows-f16",
        },
        {
            GGML_GTX_KERNEL_IM2COL_F16,
            "im2col.llext",
            "ggml_gtx_kernel_im2col_f16",
            "im2col-f16",
        },
        {
            GGML_GTX_KERNEL_SWIGLU_F16,
            "swiglu.llext",
            "ggml_gtx_kernel_swiglu_f16",
            "swiglu-f16",
        },
        {
            GGML_GTX_KERNEL_CONT_F16,
            "cont.llext",
            "ggml_gtx_kernel_cont_f16",
            "cont-f16",
        },
        {
            GGML_GTX_KERNEL_CPY_F16,
            "cpy.llext",
            "ggml_gtx_kernel_cpy_f16",
            "cpy-f16",
        },
        {
            GGML_GTX_KERNEL_ADD_F16_F32_TO_F16,
            "add-fp32.llext",
            "ggml_gtx_kernel_add_f16_f32",
            "add-f16-f32",
        },
        {
            GGML_GTX_KERNEL_CONV_2D_F16,
            "conv-2d.llext",
            "ggml_gtx_kernel_conv_2d_f16",
            "conv-2d-f16",
        },
        {
            GGML_GTX_KERNEL_CONV_2D_DW_F16,
            "conv-2d-dw.llext",
            "ggml_gtx_kernel_conv_2d_dw_f16",
            "conv-2d-dw-f16",
        },
        {
            GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16,
            "conv-transpose-2d.llext",
            "kernel_main",
            "conv-transpose-2d-f16",
        },
        {
            GGML_GTX_KERNEL_UPSCALE_F16,
            "upscale.llext",
            "ggml_gtx_kernel_upscale_f16",
            "upscale-f16",
        },
        {
            GGML_GTX_KERNEL_SIN_F16,
            "sin.llext",
            "ggml_gtx_kernel_sin_f16",
            "sin-f16",
        },
        {
            GGML_GTX_KERNEL_ABS_F16,
            "abs.llext",
            "ggml_gtx_kernel_abs_f16",
            "abs-f16",
        },
        {
            GGML_GTX_KERNEL_NEG_F16,
            "neg.llext",
            "ggml_gtx_kernel_neg_f16",
            "neg-f16",
        },
        {
            GGML_GTX_KERNEL_SGN_F16,
            "sgn.llext",
            "ggml_gtx_kernel_sgn_f16",
            "sgn-f16",
        },
        {
            GGML_GTX_KERNEL_CEIL_F16,
            "ceil.llext",
            "ggml_gtx_kernel_ceil_f16",
            "ceil-f16",
        },
        {
            GGML_GTX_KERNEL_FLOOR_F16,
            "floor.llext",
            "ggml_gtx_kernel_floor_f16",
            "floor-f16",
        },
        {
            GGML_GTX_KERNEL_TANH_F16,
            "tanh.llext",
            "ggml_gtx_kernel_tanh_f16",
            "tanh-f16",
        },
        {
            GGML_GTX_KERNEL_SILU_F16,
            "silu.llext",
            "ggml_gtx_kernel_silu_f16",
            "silu-f16",
        },
        {
            GGML_GTX_KERNEL_RELU_F16,
            "relu.llext",
            "ggml_gtx_kernel_relu_f16",
            "relu-f16",
        },
        {
            GGML_GTX_KERNEL_LEAKY_RELU_F16,
            "leaky-relu.llext",
            "ggml_gtx_kernel_leaky_relu_f16",
            "leaky-relu-f16",
        },
        {
            GGML_GTX_KERNEL_CLAMP_F16,
            "clamp.llext",
            "ggml_gtx_kernel_clamp_f16",
            "clamp-f16",
        },
        {
            GGML_GTX_KERNEL_COS_F16,
            "cos.llext",
            "ggml_gtx_kernel_cos_f16",
            "cos-f16",
        },
        {
            GGML_GTX_KERNEL_PAD_F16,
            "pad.llext",
            "ggml_gtx_kernel_pad_f16",
            "pad-f16",
        },
    };

    for (const gtx_uart_kernel_bundle & bundle : bundles) {
        if (bundle.kernel_id == kernel_id) {
            return &bundle;
        }
    }
    return nullptr;
}

static uint64_t gtx_kernel_work_size() {
    return gtx_use_xdma_stream_transport() ? GTX_FPGA_KERNEL_WORK_SIZE
                                           : GTX_FIXED_KERNEL_WORK_SIZE;
}

static uint64_t gtx_kernel_work_base_addr() {
    const uint64_t fallback = gtx_use_xdma_stream_transport() ?
                              GTX_FPGA_KERNEL_WORK_ADDR :
                              GTX_FIXED_KERNEL_WORK_ADDR;
    return gtx_env_u64_or_default("GTX_KERNEL_WORK_ADDR", fallback);
}

static uint64_t gtx_kernel_work_addr_for_slot(uint32_t slot) {
    return gtx_kernel_work_base_addr() + (uint64_t) slot * gtx_kernel_work_size();
}

static uint64_t gtx_fixed_ddr_base() {
    return gtx_env_u64_or_default("GTX_FIXED_DDR_BASE", GTX_DDR_BASE);
}

static uint64_t gtx_fixed_arg_addr() {
    return gtx_fixed_ddr_base() +
           (gtx_use_xdma_stream_transport() ? 0x00300000ull : 0x01000000ull);
}

static uint64_t gtx_fixed_tensor_arena_addr() {
    return gtx_fixed_ddr_base() +
           (gtx_use_xdma_stream_transport() ? GTX_FPGA_TENSOR_ARENA_OFFSET
                                            : 0x01100000ull);
}

static uint64_t gtx_fixed_tensor_arena_size() {
    return gtx_use_xdma_stream_transport() ? GTX_FPGA_TENSOR_ARENA_SIZE
                                           : GTX_FIXED_TENSOR_ARENA_SIZE;
}

static bool gtx_range_overlaps(uint64_t a_addr, uint64_t a_size,
                               uint64_t b_addr, uint64_t b_size) {
    if (a_size == 0 || b_size == 0) {
        return false;
    }
    return a_addr < b_addr + b_size && b_addr < a_addr + a_size;
}

static bool gtx_range_overlaps_tensor_arena(uint64_t addr, uint64_t size) {
    return gtx_range_overlaps(addr,
                              size,
                              gtx_fixed_tensor_arena_addr(),
                              gtx_fixed_tensor_arena_size());
}

static uint64_t gtx_fixed_kernel_staging_addr() {
    return gtx_fixed_ddr_base() +
           (gtx_use_xdma_stream_transport() ? 0x0ff00000ull : 0x7ff00000ull);
}

#ifndef _WIN32
static constexpr uint64_t GTX_XDMA_HOST_DDR_BASE = 0x400000000ull;
static constexpr uint64_t GTX_XDMA_RX_OFFSET = 0x00100000ull;
static constexpr uint64_t GTX_XDMA_TX_OFFSET = 0x00200000ull;
static constexpr uint32_t GTX_XDMA_RING_SIZE = 0x00100000u;
static constexpr uint32_t GTX_XDMA_STREAM_MAGIC = 0x31534458u;

struct gtx_xdma_stream_ctrl_host {
    uint32_t magic;
    uint32_t version;
    uint32_t rx_head;
    uint32_t rx_tail;
    uint32_t tx_head;
    uint32_t tx_tail;
    uint32_t state;
    uint32_t heartbeat;
    uint64_t last_mcause;
    uint64_t last_mepc;
    uint64_t last_mtval;
    uint8_t reserved[64];
};
#endif

class gtx_uart_vp_connection {
public:
    gtx_uart_vp_connection() = default;
    ~gtx_uart_vp_connection() { close(); }

    bool open();
    void close();
    bool is_open() const { return m_open; }

    bool upload_bytes(uint64_t addr, const void * data, size_t size);
    bool download_bytes(uint64_t addr, void * data, size_t size);
    bool copy_tensor(const ggml_tensor * src, const ggml_tensor * dst);
    bool stash_flush_reset();
    bool launch(const ggml_gtx_kernel_launch_desc_t & desc,
                const void * kernel_args = nullptr,
                size_t kernel_args_size = 0);

private:
    bool write_all(const void * data, size_t size);
    bool read_byte(uint8_t & byte, int timeout_ms);
    bool read_exact(void * data, size_t size, int timeout_ms);
    bool direct_ddr_enabled() const;
    bool direct_ddr_open();
    void direct_ddr_close();
    bool direct_ddr_write_all(const void * data, size_t size);
    bool direct_ddr_read_all(void * data, size_t size);
    bool direct_ddr_transfer(uint32_t op, uint64_t addr, void * data, size_t size);
    bool stash_flush_range(uint64_t addr, size_t size);
    bool stash_invalidate_range(uint64_t addr, size_t size);
    bool read_frame(ggml_gtx_rpc_frame_t & frame, std::vector<uint8_t> & payload, int timeout_ms);
    bool rpc(uint16_t cmd, const void * payload, size_t payload_size,
             std::vector<uint8_t> * response_payload, int timeout_ms = 30000);
    bool assign_kernel_work_addr(uint32_t kernel_id, uint64_t & work_addr);
    bool ensure_kernel_loaded(const gtx_uart_kernel_bundle & bundle);
    static std::vector<uint8_t> read_file(const char * path);

    int m_fd = -1;
    int m_direct_fd = -1;
#ifndef _WIN32
    int m_xdma_h2c_fd = -1;
    int m_xdma_c2h_fd = -1;
    int m_xdma_user_fd = -1;
    volatile uint32_t * m_xdma_user_bar = nullptr;
    uint32_t m_xdma_rx_head = 0;
    uint32_t m_xdma_tx_tail = 0;
    bool m_xdma = false;
#endif
    uint32_t m_seq = 0;
    bool m_open = false;
    bool m_direct_disabled = false;
    bool m_stash_may_be_resident = false;
    uint32_t m_next_kernel_work_slot = 0;
    std::unordered_map<uint32_t, uint64_t> m_kernel_work_addrs;
    std::unordered_set<uint32_t> m_loaded_kernels;
    std::string m_uart_log;
};

bool gtx_uart_vp_connection::open() {
    if (m_open) {
        return true;
    }

    if (gtx_use_xdma_stream_transport()) {
        const char * h2c = gtx_env_or_default("GTX_XDMA_H2C", "/dev/xdma0_h2c_0");
        const char * c2h = gtx_env_or_default("GTX_XDMA_C2H", "/dev/xdma0_c2h_0");
        const char * user = gtx_env_or_default("GTX_XDMA_USER", "/dev/xdma0_user");
        m_xdma_h2c_fd = ::open(h2c, O_WRONLY);
        m_xdma_c2h_fd = ::open(c2h, O_RDONLY);
        m_xdma_user_fd = ::open(user, O_RDWR | O_SYNC);
        if (m_xdma_h2c_fd < 0 || m_xdma_c2h_fd < 0 || m_xdma_user_fd < 0) {
            GTX_LOG_ERROR("failed to open XDMA stream devices: errno=%d (%s)\n",
                          errno, strerror(errno));
            close();
            return false;
        }
        void * bar = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                          m_xdma_user_fd, 0);
        if (bar == MAP_FAILED) {
            m_xdma_user_bar = nullptr;
            GTX_LOG_ERROR("failed to map XDMA user BAR: errno=%d (%s)\n",
                          errno, strerror(errno));
            close();
            return false;
        }
        m_xdma_user_bar = (volatile uint32_t *) bar;

        gtx_xdma_stream_ctrl_host ctrl = {};
        if (pread(m_xdma_c2h_fd, &ctrl, sizeof(ctrl),
                  (off_t) GTX_XDMA_HOST_DDR_BASE) != (ssize_t) sizeof(ctrl) ||
            ctrl.magic != GTX_XDMA_STREAM_MAGIC || ctrl.version != 1u) {
            GTX_LOG_ERROR("XDMA firmware stream is not ready (magic=0x%08x version=%u)\n",
                          ctrl.magic, ctrl.version);
            close();
            return false;
        }
        m_xdma_rx_head = ctrl.rx_head;
        m_xdma_tx_tail = ctrl.tx_tail;
        m_xdma = true;
        m_open = true;
        GTX_LOG("connected FPGA XDMA RPC stream\n");
        return true;
    }

    const char * endpoint = gtx_env_or_default("GTX_RPC_ENDPOINT", "unix:/tmp/gtx-vp-uart.sock");
    const char * prefix = "unix:";
    if (strncmp(endpoint, prefix, strlen(prefix)) != 0) {
        GTX_LOG_ERROR("unsupported GTX_RPC_ENDPOINT '%s' (expected unix:/path)\n", endpoint);
        return false;
    }

    const char * path = endpoint + strlen(prefix);
    if (strlen(path) >= sizeof(sockaddr_un::sun_path)) {
        GTX_LOG_ERROR("GTX_RPC_ENDPOINT path too long: %s\n", path);
        return false;
    }

    m_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_fd < 0) {
        GTX_LOG_ERROR("failed to create UART proxy socket\n");
        return false;
    }

    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);

    if (connect(m_fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        GTX_LOG_ERROR("failed to connect UART proxy endpoint: %s: errno=%d (%s)\n",
                      endpoint, errno, strerror(errno));
        close();
        return false;
    }

    int flags = fcntl(m_fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(m_fd, F_SETFL, flags | O_NONBLOCK);
    }

    m_open = true;
    return true;
}

void gtx_uart_vp_connection::close() {
    if (m_open) {
        // XDMA has no socket EOF visible to firmware. Send a bounded explicit
        // session marker so the physical UART trace has a reliable endpoint.
        (void) rpc(GGML_GTX_RPC_DISCONNECT, nullptr, 0, nullptr, 1000);
    }
    direct_ddr_close();
#ifndef _WIN32
    if (m_xdma_user_bar != nullptr) {
        munmap((void *) m_xdma_user_bar, 4096);
        m_xdma_user_bar = nullptr;
    }
    if (m_xdma_h2c_fd >= 0) ::close(m_xdma_h2c_fd);
    if (m_xdma_c2h_fd >= 0) ::close(m_xdma_c2h_fd);
    if (m_xdma_user_fd >= 0) ::close(m_xdma_user_fd);
    m_xdma_h2c_fd = m_xdma_c2h_fd = m_xdma_user_fd = -1;
    m_xdma = false;
#endif
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
    m_open = false;
}

bool gtx_uart_vp_connection::write_all(const void * data, size_t size) {
#ifndef _WIN32
    if (m_xdma) {
        const uint8_t * bytes = (const uint8_t *) data;
        while (size > 0) {
            uint32_t rx_tail = 0;
            if (pread(m_xdma_c2h_fd, &rx_tail, sizeof(rx_tail),
                      (off_t) (GTX_XDMA_HOST_DDR_BASE +
                               offsetof(gtx_xdma_stream_ctrl_host, rx_tail))) !=
                (ssize_t) sizeof(rx_tail)) {
                return false;
            }
            uint32_t free_bytes = GTX_XDMA_RING_SIZE - (m_xdma_rx_head - rx_tail);
            if (free_bytes == 0) {
                usleep(100);
                continue;
            }
            size_t chunk = std::min<size_t>(size, free_bytes);
            uint32_t pos = m_xdma_rx_head & (GTX_XDMA_RING_SIZE - 1u);
            chunk = std::min<size_t>(chunk, GTX_XDMA_RING_SIZE - pos);
            chunk = std::min<size_t>(chunk, GTX_XDMA_DRIVER_CHUNK);
            if (pwrite(m_xdma_h2c_fd, bytes, chunk,
                       (off_t) (GTX_XDMA_HOST_DDR_BASE + GTX_XDMA_RX_OFFSET + pos)) !=
                (ssize_t) chunk) {
                return false;
            }
            m_xdma_rx_head += (uint32_t) chunk;
            if (pwrite(m_xdma_h2c_fd, &m_xdma_rx_head, sizeof(m_xdma_rx_head),
                       (off_t) (GTX_XDMA_HOST_DDR_BASE +
                                offsetof(gtx_xdma_stream_ctrl_host, rx_head))) !=
                (ssize_t) sizeof(m_xdma_rx_head)) {
                return false;
            }
            MEMORY_BARRIER();
            m_xdma_user_bar[0x20 / sizeof(uint32_t)] = 1u;
            bytes += chunk;
            size -= chunk;
        }
        return true;
    }
#endif
    const uint8_t * bytes = (const uint8_t *) data;
    while (size > 0) {
        ssize_t n = write(m_fd, bytes, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd pfd = {};
            pfd.fd = m_fd;
            pfd.events = POLLOUT;
            if (poll(&pfd, 1, 30000) <= 0) {
                return false;
            }
            continue;
        }
        if (n <= 0) {
            return false;
        }
        bytes += n;
        size -= (size_t) n;
    }
    return true;
}


// ⚠️ **direct-DDR 이 꺼지면 조용히 UART RPC 로 떨어진다.** 그러면 전송이 수십~수천배
//    느려지는데 경고가 한 줄도 없다(위키 pitfall/15시간-로그없이-느림).
//    어느 자리에서 꺼졌는지 한 번만 알린다.
static void gtx_note_direct_ddr_off(int site) {
    static bool told = false;
    if (!told) {
        told = true;
        fprintf(stderr,
                "[GTX WARN] direct-DDR disabled at site %d — falling back to UART RPC "
                "(transfers get much slower)\n", site);
    }
}

bool gtx_uart_vp_connection::direct_ddr_enabled() const {
    if (gtx_use_xdma_stream_transport()) {
        return false;
    }
    const char * value = getenv("GTX_UART_DIRECT_DDR");
    return value == nullptr || strcmp(value, "0") != 0;
}

bool gtx_uart_vp_connection::direct_ddr_open() {
    if (!direct_ddr_enabled() || m_direct_disabled) {
        return false;
    }
    if (m_direct_fd >= 0) {
        return true;
    }

    const char * endpoint = gtx_env_or_default("GTX_VP_DIRECT_ENDPOINT", "unix:/tmp/gtx-vp-ddr.sock");
    const char * prefix = "unix:";
    const char * path = endpoint;
    if (strncmp(endpoint, prefix, strlen(prefix)) == 0) {
        path = endpoint + strlen(prefix);
    }
    if (strlen(path) >= sizeof(sockaddr_un::sun_path)) {
        GTX_LOG_ERROR("GTX_VP_DIRECT_ENDPOINT path too long: %s\n", path);
        gtx_note_direct_ddr_off(1);
        m_direct_disabled = true;
        return false;
    }

    m_direct_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_direct_fd < 0) {
        gtx_note_direct_ddr_off(2);
        m_direct_disabled = true;
        return false;
    }

    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
    if (connect(m_direct_fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        direct_ddr_close();
        gtx_note_direct_ddr_off(3);
        m_direct_disabled = true;
        GTX_LOG("direct DDR endpoint unavailable: %s\n", path);
        return false;
    }

    GTX_LOG("connected direct DDR endpoint: %s\n", path);
    return true;
}

void gtx_uart_vp_connection::direct_ddr_close() {
    if (m_direct_fd >= 0) {
        ::close(m_direct_fd);
        m_direct_fd = -1;
    }
}

bool gtx_uart_vp_connection::direct_ddr_write_all(const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    while (size > 0) {
        ssize_t n = write(m_direct_fd, bytes, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        bytes += n;
        size -= (size_t) n;
    }
    return true;
}

bool gtx_uart_vp_connection::direct_ddr_read_all(void * data, size_t size) {
    uint8_t * bytes = (uint8_t *) data;
    while (size > 0) {
        ssize_t n = read(m_direct_fd, bytes, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        bytes += n;
        size -= (size_t) n;
    }
    return true;
}

bool gtx_uart_vp_connection::direct_ddr_transfer(uint32_t op, uint64_t addr, void * data, size_t size) {
#ifndef _WIN32
    if (m_xdma) {
        const uint64_t ddr_base = gtx_fixed_ddr_base();
        if ((op != GTX_DIRECT_DDR_OP_WRITE && op != GTX_DIRECT_DDR_OP_READ) ||
            addr < ddr_base || size > UINT64_MAX - addr) {
            return false;
        }

        const uint64_t ddr_offset = addr - ddr_base;
        if (ddr_offset > (uint64_t) std::numeric_limits<off_t>::max() -
                             GTX_XDMA_HOST_DDR_BASE ||
            size > (uint64_t) std::numeric_limits<off_t>::max() -
                       (GTX_XDMA_HOST_DDR_BASE + ddr_offset)) {
            return false;
        }

        uint8_t * bytes = (uint8_t *) data;
        size_t done = 0;
        while (done < size) {
            const size_t chunk =
                std::min<size_t>(size - done, GTX_XDMA_DRIVER_CHUNK);
            const off_t offset =
                (off_t) (GTX_XDMA_HOST_DDR_BASE + ddr_offset + done);
            const ssize_t transferred =
                op == GTX_DIRECT_DDR_OP_WRITE ?
                    pwrite(m_xdma_h2c_fd, bytes + done, chunk, offset) :
                    pread(m_xdma_c2h_fd, bytes + done, chunk, offset);
            if (transferred != (ssize_t) chunk) {
                return false;
            }
            done += chunk;
        }
        return true;
    }
#endif

    if (size > UINT32_MAX || !direct_ddr_open()) {
        return false;
    }

    gtx_direct_ddr_header hdr = {};
    hdr.magic = GTX_DIRECT_DDR_MAGIC;
    hdr.op = op;
    hdr.addr = addr;
    hdr.size = (uint32_t) size;
    hdr.status = 0;

    if (!direct_ddr_write_all(&hdr, sizeof(hdr)) ||
        (op == GTX_DIRECT_DDR_OP_WRITE && size > 0 && !direct_ddr_write_all(data, size))) {
        direct_ddr_close();
        gtx_note_direct_ddr_off(4);
        m_direct_disabled = true;
        return false;
    }

    gtx_direct_ddr_header rsp = {};
    if (!direct_ddr_read_all(&rsp, sizeof(rsp)) ||
        rsp.magic != GTX_DIRECT_DDR_MAGIC ||
        rsp.op != op ||
        rsp.addr != addr ||
        rsp.size != size ||
        rsp.status != 0) {
        direct_ddr_close();
        gtx_note_direct_ddr_off(5);
        m_direct_disabled = true;
        return false;
    }

    if (op == GTX_DIRECT_DDR_OP_READ && size > 0 && !direct_ddr_read_all(data, size)) {
        direct_ddr_close();
        gtx_note_direct_ddr_off(6);
        m_direct_disabled = true;
        return false;
    }

    return true;
}

bool gtx_uart_vp_connection::stash_flush_range(uint64_t addr, size_t size) {
    ggml_gtx_rpc_stash_sync_t req = {};
    req.op = GGML_GTX_RPC_STASH_FLUSH_RANGE;
    req.addr = addr;
    req.size = (uint64_t) size;
    return rpc(GGML_GTX_RPC_STASH_SYNC, &req, sizeof(req), nullptr,
               GTX_UART_KERNEL_TIMEOUT_MS);
}

bool gtx_uart_vp_connection::stash_flush_reset() {
    ggml_gtx_rpc_stash_sync_t req = {};
    req.op = GGML_GTX_RPC_STASH_FLUSH_RESET;
    const bool ok = rpc(GGML_GTX_RPC_STASH_SYNC, &req, sizeof(req), nullptr,
                        GTX_UART_KERNEL_TIMEOUT_MS);
    if (ok) {
        m_stash_may_be_resident = false;
    }
    return ok;
}

bool gtx_uart_vp_connection::stash_invalidate_range(uint64_t addr, size_t size) {
    ggml_gtx_rpc_stash_sync_t req = {};
    req.op = GGML_GTX_RPC_STASH_INVALIDATE_RANGE;
    req.addr = addr;
    req.size = (uint64_t) size;
    return rpc(GGML_GTX_RPC_STASH_SYNC, &req, sizeof(req), nullptr,
               GTX_UART_KERNEL_TIMEOUT_MS);
}

bool gtx_uart_vp_connection::read_byte(uint8_t & byte, int timeout_ms) {
#ifndef _WIN32
    if (m_xdma) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        for (;;) {
            uint32_t tx_head = 0;
            if (pread(m_xdma_c2h_fd, &tx_head, sizeof(tx_head),
                      (off_t) (GTX_XDMA_HOST_DDR_BASE +
                               offsetof(gtx_xdma_stream_ctrl_host, tx_head))) !=
                (ssize_t) sizeof(tx_head)) {
                return false;
            }
            if (m_xdma_tx_tail != tx_head) {
                uint32_t pos = m_xdma_tx_tail & (GTX_XDMA_RING_SIZE - 1u);
                if (pread(m_xdma_c2h_fd, &byte, 1,
                          (off_t) (GTX_XDMA_HOST_DDR_BASE + GTX_XDMA_TX_OFFSET + pos)) != 1) {
                    return false;
                }
                ++m_xdma_tx_tail;
                return pwrite(m_xdma_h2c_fd, &m_xdma_tx_tail, sizeof(m_xdma_tx_tail),
                              (off_t) (GTX_XDMA_HOST_DDR_BASE +
                                       offsetof(gtx_xdma_stream_ctrl_host, tx_tail))) ==
                       (ssize_t) sizeof(m_xdma_tx_tail);
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            usleep(100);
        }
    }
#endif
    struct pollfd pfd = {};
    pfd.fd = m_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0) {
        return false;
    }
    ssize_t n = read(m_fd, &byte, 1);
    if (n < 0 && errno == EINTR) {
        return read_byte(byte, timeout_ms);
    }
    return n == 1;
}

bool gtx_uart_vp_connection::read_exact(void * data, size_t size, int timeout_ms) {
    uint8_t * bytes = (uint8_t *) data;
    while (size > 0) {
        uint8_t ch = 0;
        if (!read_byte(ch, timeout_ms)) {
            return false;
        }
        *bytes++ = ch;
        --size;
    }
    return true;
}

bool gtx_uart_vp_connection::read_frame(ggml_gtx_rpc_frame_t & frame,
                                        std::vector<uint8_t> & payload,
                                        int timeout_ms) {
    uint32_t magic = 0;
    while (true) {
        while (magic != GGML_GTX_RPC_MAGIC) {
            uint8_t ch = 0;
            if (!read_byte(ch, timeout_ms)) {
                return false;
            }
            if (ch == '\n' || (ch >= 0x20 && ch <= 0x7e)) {
                m_uart_log.push_back((char) ch);
                if (m_uart_log.size() > 8192) {
                    m_uart_log.erase(0, m_uart_log.size() - 8192);
                }
            }
            magic = (magic >> 8) | ((uint32_t) ch << 24);
        }

        frame.magic = GGML_GTX_RPC_MAGIC;
        if (!read_exact(((uint8_t *) &frame) + sizeof(frame.magic),
                        sizeof(frame) - sizeof(frame.magic), timeout_ms)) {
            return false;
        }

        if (frame.version == GGML_GTX_RPC_VERSION &&
            frame.header_size == sizeof(frame) &&
            frame.payload_size <= (16u * 1024u * 1024u)) {
            break;
        }

        magic = 0;
        m_uart_log.append("[discarded malformed GTX RPC candidate]\n");
        if (m_uart_log.size() > 8192) {
            m_uart_log.erase(0, m_uart_log.size() - 8192);
        }
    }

    payload.resize(frame.payload_size);
    return payload.empty() || read_exact(payload.data(), payload.size(), timeout_ms);
}

bool gtx_uart_vp_connection::rpc(uint16_t cmd, const void * payload, size_t payload_size,
                                 std::vector<uint8_t> * response_payload, int timeout_ms) {
    ggml_gtx_rpc_frame_t frame = {};
    frame.magic = GGML_GTX_RPC_MAGIC;
    frame.version = GGML_GTX_RPC_VERSION;
    frame.header_size = sizeof(frame);
    frame.cmd = cmd;
    frame.seq = ++m_seq;
    frame.payload_size = (uint32_t) payload_size;

    if (!write_all(&frame, sizeof(frame)) ||
        (payload_size > 0 && !write_all(payload, payload_size))) {
        return false;
    }

    while (true) {
        ggml_gtx_rpc_frame_t rsp = {};
        std::vector<uint8_t> rsp_payload;
        if (!read_frame(rsp, rsp_payload, timeout_ms)) {
            GTX_LOG_ERROR("UART RPC cmd=%u seq=%u timed out or received malformed frame\n",
                          (unsigned) cmd, (unsigned) frame.seq);
            if (!m_uart_log.empty()) {
                GTX_LOG_ERROR("UART log tail:\n%s\n", m_uart_log.c_str());
            }
            return false;
        }
        if (rsp.seq != frame.seq) {
            continue;
        }
        if (rsp.status != GGML_GTX_RPC_STATUS_OK) {
            GTX_LOG_ERROR("UART RPC cmd=%u seq=%u failed status=%u\n",
                          (unsigned) cmd, (unsigned) frame.seq, (unsigned) rsp.status);
            if (!m_uart_log.empty()) {
                GTX_LOG_ERROR("UART log tail:\n%s\n", m_uart_log.c_str());
            }
            return false;
        }
        if (response_payload != nullptr) {
            *response_payload = std::move(rsp_payload);
        }
        return true;
    }
}

std::vector<uint8_t> gtx_uart_vp_connection::read_file(const char * path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data((size_t) size);
    if (!data.empty()) {
        input.read((char *) data.data(), size);
    }
    return data;
}

bool gtx_uart_vp_connection::upload_bytes(uint64_t addr, const void * data, size_t size) {
    const bool overlaps_stash = m_stash_may_be_resident &&
                                gtx_range_overlaps_tensor_arena(addr, size);

    // A host write may partially replace a dirty resident tile.  Preserve any
    // untouched bytes before the direct-DDR write, then discard the stale L2
    // mapping after it.  This is intentionally range-scoped: unrelated dirty
    // stash outputs remain resident.
    if (overlaps_stash && !stash_flush_range(addr, size)) {
        return false;
    }

    if (direct_ddr_transfer(GTX_DIRECT_DDR_OP_WRITE, addr, const_cast<void *>(data), size)) {
        if (overlaps_stash && !stash_invalidate_range(addr, size)) {
            return false;
        }
        return true;
    }

    ggml_gtx_rpc_upload_bytes_t req = {};
    req.addr = addr;
    req.size = (uint32_t) size;

    std::vector<uint8_t> payload(sizeof(req) + size);
    memcpy(payload.data(), &req, sizeof(req));
    if (size > 0) {
        memcpy(payload.data() + sizeof(req), data, size);
    }
    return rpc(GGML_GTX_RPC_UPLOAD_BYTES, payload.data(), payload.size(), nullptr,
               GTX_UART_TRANSFER_TIMEOUT_MS);
}

bool gtx_uart_vp_connection::download_bytes(uint64_t addr, void * data, size_t size) {
    if ((!m_stash_may_be_resident || stash_flush_range(addr, size)) &&
        direct_ddr_transfer(GTX_DIRECT_DDR_OP_READ, addr, data, size)) {
        return true;
    }

    ggml_gtx_rpc_download_bytes_t req = {};
    req.addr = addr;
    req.size = (uint32_t) size;

    std::vector<uint8_t> payload;
    if (!rpc(GGML_GTX_RPC_DOWNLOAD_BYTES, &req, sizeof(req), &payload,
             GTX_UART_TRANSFER_TIMEOUT_MS) ||
        payload.size() != size) {
        return false;
    }
    if (size > 0) {
        memcpy(data, payload.data(), size);
    }
    return true;
}

bool gtx_uart_vp_connection::copy_tensor(const ggml_tensor * src, const ggml_tensor * dst) {
    if (src == nullptr || dst == nullptr) {
        return false;
    }

    gtx_msg_copy_tensor_req req = {};
    req.src = gtx_serialize_tensor(src);
    req.dst = gtx_serialize_tensor(dst);

    if (req.src.data == 0 || req.dst.data == 0) {
        GTX_LOG_ERROR("UART COPY_TENSOR requires GTX-backed src and dst tensors\n");
        return false;
    }

    std::vector<uint8_t> payload;
    if (!rpc(GGML_GTX_RPC_COPY_TENSOR, &req, sizeof(req), &payload,
             GTX_UART_TRANSFER_TIMEOUT_MS) ||
        payload.size() != sizeof(gtx_msg_copy_tensor_rsp)) {
        GTX_LOG_ERROR("UART COPY_TENSOR command failed\n");
        return false;
    }

    gtx_msg_copy_tensor_rsp rsp = {};
    memcpy(&rsp, payload.data(), sizeof(rsp));
    return rsp.result != 0;
}

bool gtx_uart_vp_connection::assign_kernel_work_addr(uint32_t kernel_id, uint64_t & work_addr) {
    auto it = m_kernel_work_addrs.find(kernel_id);
    if (it != m_kernel_work_addrs.end()) {
        work_addr = it->second;
        return true;
    }

    const uint32_t default_slot_count = gtx_use_xdma_stream_transport() ?
                                        GTX_FPGA_KERNEL_WORK_SLOT_COUNT :
                                        GTX_KERNEL_WORK_SLOT_COUNT;
    const uint32_t slot_count = (uint32_t) gtx_env_u64_or_default(
        "GTX_KERNEL_WORK_SLOT_COUNT", default_slot_count);
    if (m_next_kernel_work_slot >= slot_count) {
        GTX_LOG_ERROR("no free LLEXT kernel work slot: kernel_id=%u slots=%u\n",
                      (unsigned) kernel_id,
                      (unsigned) slot_count);
        return false;
    }

    work_addr = gtx_kernel_work_addr_for_slot(m_next_kernel_work_slot++);
    m_kernel_work_addrs[kernel_id] = work_addr;
    return true;
}

bool gtx_uart_vp_connection::ensure_kernel_loaded(const gtx_uart_kernel_bundle & bundle) {
    if (m_loaded_kernels.find(bundle.kernel_id) != m_loaded_kernels.end()) {
        if (gtx_log_ops_enabled()) {
            fprintf(stderr, "[GTX KERNEL] reuse %s id=%u\n",
                    bundle.name, (unsigned) bundle.kernel_id);
        }
        return true;
    }

    const std::string kernel_path = gtx_join_path(gtx_kernel_bundle_dir(), bundle.relative_path);
    std::vector<uint8_t> binary = read_file(kernel_path.c_str());
    if (binary.empty() || binary.size() > GTX_FIXED_KERNEL_STAGING_SIZE) {
        GTX_LOG_ERROR("failed to read %s kernel: %s\n", bundle.name, kernel_path.c_str());
        return false;
    }
    const uint64_t staging_addr = gtx_fixed_kernel_staging_addr();
    if (!upload_bytes(staging_addr, binary.data(), binary.size())) {
        return false;
    }

    uint64_t work_addr = 0;
    if (!assign_kernel_work_addr(bundle.kernel_id, work_addr)) {
        return false;
    }

    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX KERNEL] load begin %s id=%u path=%s size=%zu staging=0x%llx work=0x%llx\n",
                bundle.name,
                (unsigned) bundle.kernel_id,
                kernel_path.c_str(),
                binary.size(),
                (unsigned long long) staging_addr,
                (unsigned long long) work_addr);
    }

    ggml_gtx_rpc_load_kernel_t req = {};
    req.kernel_id = bundle.kernel_id;
    req.format = GGML_GTX_KERNEL_FORMAT_LLEXT;
    req.staging_addr = staging_addr;
    req.work_addr = work_addr;
    req.work_size = gtx_kernel_work_size();
    req.binary_size = (uint32_t) binary.size();
    snprintf(req.entry_symbol, sizeof(req.entry_symbol), "%s", bundle.entry_symbol);

    if (!rpc(GGML_GTX_RPC_LOAD_KERNEL, &req, sizeof(req), nullptr, 120000)) {
        if (gtx_log_ops_enabled()) {
            fprintf(stderr, "[GTX KERNEL] load failed %s id=%u\n",
                    bundle.name, (unsigned) bundle.kernel_id);
        }
        m_kernel_work_addrs.erase(bundle.kernel_id);
        const uint64_t work_base = gtx_kernel_work_base_addr();
        const uint32_t failed_slot = (uint32_t) ((work_addr - work_base) /
                                                gtx_kernel_work_size());
        if (failed_slot + 1 == m_next_kernel_work_slot) {
            m_next_kernel_work_slot = failed_slot;
        }
        return false;
    }

    m_loaded_kernels.insert(bundle.kernel_id);
    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX KERNEL] load done %s id=%u\n",
                bundle.name, (unsigned) bundle.kernel_id);
    }
    return true;
}

bool gtx_uart_vp_connection::launch(const ggml_gtx_kernel_launch_desc_t & desc,
                                    const void * kernel_args,
                                    size_t kernel_args_size) {
    const gtx_uart_kernel_bundle * bundle = gtx_uart_kernel_bundle_for(desc.kernel_id);
    if (bundle == nullptr || !ensure_kernel_loaded(*bundle)) {
        return false;
    }

    if ((kernel_args == nullptr) != (kernel_args_size == 0)) {
        GTX_LOG_ERROR("kernel argument payload pointer/size mismatch: kernel_id=%u size=%zu\n",
                      (unsigned) desc.kernel_id, kernel_args_size);
        return false;
    }
    if (kernel_args_size > UINT32_MAX) {
        GTX_LOG_ERROR("kernel argument payload is too large: kernel_id=%u size=%zu\n",
                      (unsigned) desc.kernel_id, kernel_args_size);
        return false;
    }

    if (gtx_log_ops_enabled()) {
        fprintf(stderr,
                "[GTX LAUNCH] UART launch %s: kernel_id=%u flags=0x%x dtype=%u(%s) src_type=[%u(%s),%u(%s),%u(%s),%u(%s)] src0=0x%llx src1=0x%llx dst=0x%llx ne=%u,%u,%u,%u\n",
                bundle->name,
                (unsigned) desc.kernel_id,
                (unsigned) desc.flags,
                (unsigned) desc.dtype,
                ggml_type_name((ggml_type) desc.dtype),
                (unsigned) desc.src_type[0],
                ggml_type_name((ggml_type) desc.src_type[0]),
                (unsigned) desc.src_type[1],
                ggml_type_name((ggml_type) desc.src_type[1]),
                (unsigned) desc.src_type[2],
                ggml_type_name((ggml_type) desc.src_type[2]),
                (unsigned) desc.src_type[3],
                ggml_type_name((ggml_type) desc.src_type[3]),
                (unsigned long long) desc.src_addr[0],
                (unsigned long long) desc.src_addr[1],
                (unsigned long long) desc.dst_addr,
                desc.ne[0], desc.ne[1], desc.ne[2], desc.ne[3]);
    }
    ggml_gtx_kernel_arg_t arg = {};
    arg.magic = GGML_GTX_KERNEL_ARG_MAGIC;
    arg.version = GGML_GTX_KERNEL_ARG_VERSION;
    arg.size = sizeof(arg);
    arg.kernel_id = desc.kernel_id;
    arg.desc = desc;

    const uint64_t arg_addr = gtx_fixed_arg_addr();
    const size_t payload_offset = (sizeof(arg) + 15u) & ~size_t(15u);
    size_t upload_size = sizeof(arg);
    if (kernel_args_size != 0) {
        if (payload_offset > GTX_FIXED_ARG_REGION_SIZE ||
            kernel_args_size > GTX_FIXED_ARG_REGION_SIZE - payload_offset) {
            GTX_LOG_ERROR("kernel argument payload exceeds fixed argument region: kernel_id=%u size=%zu\n",
                          (unsigned) desc.kernel_id, kernel_args_size);
            return false;
        }
        arg.desc.args_addr = arg_addr + payload_offset;
        arg.desc.args_size = (uint32_t) kernel_args_size;
        upload_size = payload_offset + kernel_args_size;
    } else if ((arg.desc.args_addr == 0u) != (arg.desc.args_size == 0u)) {
        GTX_LOG_ERROR("remote kernel argument address/size mismatch: kernel_id=%u\n",
                      (unsigned) desc.kernel_id);
        return false;
    }

    std::vector<uint8_t> arg_blob(upload_size, 0u);
    memcpy(arg_blob.data(), &arg, sizeof(arg));
    if (kernel_args_size != 0) {
        memcpy(arg_blob.data() + payload_offset, kernel_args, kernel_args_size);
    }

    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX LAUNCH] arg upload begin %s id=%u arg=0x%llx\n",
                bundle->name,
                (unsigned) desc.kernel_id,
                (unsigned long long) gtx_fixed_arg_addr());
    }

    if (!upload_bytes(arg_addr, arg_blob.data(), arg_blob.size())) {
        if (gtx_log_ops_enabled()) {
            fprintf(stderr, "[GTX LAUNCH] arg upload failed %s id=%u\n",
                    bundle->name, (unsigned) desc.kernel_id);
        }
        return false;
    }

    ggml_gtx_rpc_launch_t req = {};
    req.kernel_id = desc.kernel_id;
    req.arg_addr = arg_addr;

    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX LAUNCH] rpc begin %s id=%u\n",
                bundle->name, (unsigned) desc.kernel_id);
    }

    const bool ok = rpc(GGML_GTX_RPC_LAUNCH, &req, sizeof(req), nullptr, GTX_UART_KERNEL_TIMEOUT_MS);
    if (ok) {
        m_stash_may_be_resident = true;
    }
    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX LAUNCH] rpc %s %s id=%u\n",
                ok ? "done" : "failed", bundle->name, (unsigned) desc.kernel_id);
    }
    return ok;
}
#else
class gtx_uart_vp_connection {
public:
    bool open() { return false; }
    bool is_open() const { return false; }
    bool upload_bytes(uint64_t, const void *, size_t) { return false; }
    bool download_bytes(uint64_t, void *, size_t) { return false; }
    bool copy_tensor(const ggml_tensor *, const ggml_tensor *) { return false; }
    bool stash_flush_reset() { return false; }
    bool launch(const ggml_gtx_kernel_launch_desc_t &, const void * = nullptr, size_t = 0) { return false; }
};
#endif

static bool gtx_build_kernel_args(gtx_uart_vp_connection & connection,
                                  const ggml_gtx_kernel_launch_desc_t & desc,
                                  std::vector<uint8_t> & args) {
    args.clear();
    if ((desc.kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
         desc.kernel_id == GGML_GTX_KERNEL_MUL_F16) &&
        desc.src_ne[1][0] == 1u) {
        uint64_t scalar_count = 1u;
        for (uint32_t d = 1u; d < GGML_MAX_DIMS; ++d) {
            if (desc.src_ne[1][d] == 0u ||
                scalar_count > UINT32_MAX / desc.src_ne[1][d]) {
                return false;
            }
            scalar_count *= desc.src_ne[1][d];
        }

        uint64_t expected_stride = sizeof(uint16_t);
        for (uint32_t d = 0u; d < GGML_MAX_DIMS; ++d) {
            if (desc.src_nb[1][d] != expected_stride) {
                GTX_LOG_ERROR("ADD/MUL scalar payload requires compact RHS: dim=%u nb=%u expected=%llu\n",
                              d, desc.src_nb[1][d],
                              (unsigned long long) expected_stride);
                return false;
            }
            expected_stride *= desc.src_ne[1][d];
        }

        const size_t scalar_bytes = (size_t) scalar_count * sizeof(uint16_t);
        const size_t header_size = GGML_GTX_ADD_ARGS_HEADER_SIZE;
        static_assert(GGML_GTX_ADD_ARGS_HEADER_SIZE == GGML_GTX_MUL_ARGS_HEADER_SIZE,
                      "ADD/MUL scalar payload headers must match");
        args.resize(header_size + scalar_bytes, 0u);
        ggml_gtx_add_args_t params = {};
        params.version = desc.kernel_id == GGML_GTX_KERNEL_ADD_F16 ?
                         GGML_GTX_ADD_ARGS_VERSION : GGML_GTX_MUL_ARGS_VERSION;
        params.size = (uint32_t) args.size();
        params.scalar_count = (uint32_t) scalar_count;
        memcpy(args.data(), &params, header_size);
        if (!connection.download_bytes(desc.src_addr[1],
                                       args.data() + header_size,
                                       scalar_bytes)) {
            GTX_LOG_ERROR("failed to download ADD/MUL scalar payload: addr=0x%llx size=%zu\n",
                          (unsigned long long) desc.src_addr[1], scalar_bytes);
            args.clear();
            return false;
        }
        return true;
    }
    if (desc.kernel_id != GGML_GTX_KERNEL_MUL_MAT_F16) {
        return true;
    }

    ggml_gtx_mul_mat_args_t params = {};
    params.version = GGML_GTX_MUL_MAT_ARGS_VERSION;
    params.size = sizeof(params);
    params.m = desc.src_ne[0][1];
    params.k = desc.src_ne[0][0];
    params.n = desc.src_ne[1][1];
    args.resize(sizeof(params));
    memcpy(args.data(), &params, sizeof(params));
    return true;
}

static bool gtx_uart_launch_desc(gtx_uart_vp_connection & connection,
                                 const ggml_gtx_kernel_launch_desc_t & desc) {
    std::vector<uint8_t> args;
    if (!gtx_build_kernel_args(connection, desc, args)) {
        return false;
    }
    return connection.launch(desc,
                             args.empty() ? nullptr : args.data(),
                             args.size());
}

static std::shared_ptr<gtx_uart_vp_connection> ggml_backend_gtx_shared_uart_connection(int device_id) {
    static std::mutex s_mutex;
    static std::vector<std::shared_ptr<gtx_uart_vp_connection>> s_connections;

    std::lock_guard<std::mutex> lock(s_mutex);
    if ((int) s_connections.size() <= device_id) {
        s_connections.resize(device_id + 1);
    }
    if (!s_connections[device_id]) {
        s_connections[device_id] = std::make_shared<gtx_uart_vp_connection>();
    }
    return s_connections[device_id];
}

static std::shared_ptr<gtx_pcie_connection> ggml_backend_gtx_shared_connection(int device_id) {
    static std::mutex s_mutex;
    static std::vector<std::shared_ptr<gtx_pcie_connection>> s_connections;

    std::lock_guard<std::mutex> lock(s_mutex);
    if ((int) s_connections.size() <= device_id) {
        s_connections.resize(device_id + 1);
    }
    if (!s_connections[device_id]) {
        s_connections[device_id] = std::make_shared<gtx_pcie_connection>();
    }
    return s_connections[device_id];
}

bool gtx_pcie_connection::open(const char* device_path, int bar_index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    (void)bar_index;

    if (m_irq_ctx != nullptr) {
        return true; /* Already open */
    }
    /* Try vfio-user socket paths in order */
    const std::string default_socket = gtx_default_vfio_socket_path();
    std::vector<std::string> socket_paths = {
        default_socket,
        "./gtx-ggml-cuda-vfio.sock",
    };
    if (device_path != nullptr && device_path[0] != '\0') {
        socket_paths.emplace_back(device_path);
    }

    for (const std::string & path_str : socket_paths) {
        const char * path = path_str.c_str();

#ifdef __linux__
        struct stat st;
        if (stat(path, &st) == 0 && S_ISSOCK(st.st_mode)) {
            GTX_LOG("Trying vfio-user socket: %s\n", path);
            m_irq_ctx = gtx_pcie_irq_create_vfio(path);
            if (m_irq_ctx != nullptr) {
                GTX_LOG("Connected to GTX device via vfio-user: %s\n", path);
                return true;
            }
        }
#endif
    }

    /* If no vfio-user socket found, try hardware VFIO */
    if (device_path && strncmp(device_path, "/dev/vfio/", 10) == 0) {
        m_irq_ctx = gtx_pcie_irq_create_hw(device_path, 0);
        if (m_irq_ctx != nullptr) {
            GTX_LOG("Connected to GTX device via hardware VFIO: %s\n", device_path);
            return true;
        }
    }

    GTX_LOG_ERROR("Failed to connect to GTX device\n");
    return false;
}

void gtx_pcie_connection::close() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_irq_ctx != nullptr) {
        gtx_pcie_irq_destroy(m_irq_ctx);
        m_irq_ctx = nullptr;
        GTX_LOG("GTX connection closed\n");
    }

}

void gtx_pcie_connection::send_doorbell(uint32_t source) {
    if (m_irq_ctx == nullptr) return;
    gtx_pcie_irq_send_doorbell(m_irq_ctx, source);
}

bool gtx_pcie_connection::wait_msi(int vector, int timeout_ms) {
    if (m_irq_ctx == nullptr) return false;
    return gtx_pcie_irq_wait_msi(m_irq_ctx, vector, timeout_ms) >= 0;
}

bool gtx_pcie_connection::send_data(uint64_t offset, const void* data, size_t size) {
    if (m_irq_ctx == nullptr || data == nullptr) return false;
    return gtx_pcie_irq_region_write(m_irq_ctx, GTX_BAR2_REGION, offset, data, size) == 0;
}

bool gtx_pcie_connection::recv_data(uint64_t offset, void* data, size_t size) {
    if (m_irq_ctx == nullptr || data == nullptr) return false;
    return gtx_pcie_irq_region_read(m_irq_ctx, GTX_BAR2_REGION, offset, data, size) == 0;
}

bool gtx_pcie_connection::send_cmd(uint8_t cmd, const void* request, size_t req_size,
                                     void* response, size_t rsp_size) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_irq_ctx == nullptr) {
        GTX_LOG_ERROR("send_cmd: not connected\n");
        return false;
    }

    auto region_read = [&](uint64_t offset, void * data, size_t size) {
        return gtx_pcie_irq_region_read(m_irq_ctx, GTX_BAR2_REGION, offset, data, size);
    };

    auto region_write = [&](uint64_t offset, const void * data, size_t size) {
        return gtx_pcie_irq_region_write(m_irq_ctx, GTX_BAR2_REGION, offset, data, size);
    };

    /* Build 64-byte message */
    gtx_message msg = {};
    msg.seq_num = ++m_seq_num;
    msg.type = cmd;
    msg.size = (uint32_t)req_size;

    /* Copy request data to payload (up to 24 bytes) */
    if (request != nullptr && req_size > 0) {
        size_t copy_size = req_size < sizeof(msg.payload) ? req_size : sizeof(msg.payload);
        memcpy(msg.payload, request, copy_size);
    }

    /* Message queue is in BAR2 (maps to device memory 0x60000000) */

    /* Read current RX head from device */
    uint32_t rx_head = 0;
    if (region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_HEAD,
                    &rx_head, sizeof(rx_head)) < 0) {
        GTX_LOG_ERROR("Failed to read RX head\n");
        return false;
    }

    GTX_LOG("RX head before write: %u\n", rx_head);

    /* Write message to RX queue */
    uint64_t msg_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_DATA + (rx_head * 64);
    if (region_write(msg_offset, &msg, sizeof(msg)) < 0) {
        GTX_LOG_ERROR("Failed to write message to RX queue\n");
        return false;
    }

    /* Update RX head */
    uint32_t next_head = (rx_head + 1) % GTX_MSG_QUEUE_ENTRIES;
    if (region_write(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_HEAD,
                     &next_head, sizeof(next_head)) < 0) {
        GTX_LOG_ERROR("Failed to update RX head\n");
        return false;
    }

    GTX_LOG("RX head after write: %u\n", next_head);

    GTX_LOG("send_cmd: cmd=%s(%u) seq=%u req_size=%zu\n", gtx_cmd_name(cmd), cmd, msg.seq_num, req_size);

    const bool log_alloc = (cmd == GTX_CMD_ALLOC_BUFFER);
    auto log_alloc_rsp = [&](const char * tag, const gtx_message & rsp, uint32_t tx_head, uint32_t tx_tail) {
        gtx_msg_alloc_buffer_rsp alloc_rsp = {};
        if (rsp.size >= sizeof(alloc_rsp)) {
            memcpy(&alloc_rsp, rsp.payload, sizeof(alloc_rsp));
        }

        GTX_LOG("ALLOC_BUFFER %s: got_seq=%u expect_seq=%u tx_head=%u tx_tail=%u type=%u flags=0x%x size=%u rsp=%u remote_ptr=0x%llx remote_size=%llu\n",
                tag,
                rsp.seq_num,
                msg.seq_num,
                tx_head,
                tx_tail,
                rsp.type,
                rsp.flags,
                rsp.size,
                (rsp.flags & GTX_FLAG_RESPONSE) != 0,
                (unsigned long long) alloc_rsp.remote_ptr,
                (unsigned long long) alloc_rsp.remote_size);
    };

    /* Send doorbell to wake device */
    send_doorbell(GTX_DOORBELL_CMD_READY);

    /* Wait for response in TX queue */
    auto start = std::chrono::steady_clock::now();
    const int timeout_ms = (cmd == GTX_CMD_GRAPH_COMPUTE) ?
        GTX_GRAPH_COMPUTE_TIMEOUT_MS : GTX_CMD_TIMEOUT_MS;

    while (true) {
        /* Read TX head and tail from BAR2 */
        uint32_t tx_head = 0, tx_tail = 0;
        region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_HEAD,
                    &tx_head, sizeof(tx_head));
        region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                    &tx_tail, sizeof(tx_tail));

        if (tx_head != tx_tail) {
            /* Read response message */
            gtx_message rsp = {};
            uint64_t rsp_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_DATA + (tx_tail * 64);
            region_read(rsp_offset, &rsp, sizeof(rsp));

            if (log_alloc) {
                log_alloc_rsp("queue rsp", rsp, tx_head, tx_tail);
            }

            /* Update TX tail */
            uint32_t next_tail = (tx_tail + 1) % GTX_MSG_QUEUE_ENTRIES;
            gtx_pcie_irq_region_write(m_irq_ctx, GTX_BAR2_REGION,
                                       GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                                       &next_tail, sizeof(next_tail));

            /* Check sequence number */
            if (rsp.seq_num == msg.seq_num) {
                /* Copy response data */
                if (response != nullptr && rsp_size > 0) {
                    size_t copy_size = rsp_size < sizeof(rsp.payload) ? rsp_size : sizeof(rsp.payload);
                    memcpy(response, rsp.payload, copy_size);
                }
                GTX_LOG("Response received: cmd=%s seq=%u\n", gtx_cmd_name(cmd), rsp.seq_num);
                return true;
            } else if (log_alloc) {
                GTX_LOG_ERROR("ALLOC_BUFFER queue mismatch: got_seq=%u expect_seq=%u type=%u flags=0x%x size=%u rsp=%u tx_head=%u tx_tail=%u\n",
                              rsp.seq_num,
                              msg.seq_num,
                              rsp.type,
                              rsp.flags,
                              rsp.size,
                              (rsp.flags & GTX_FLAG_RESPONSE) != 0,
                              tx_head,
                              tx_tail);
            }
        }

        gtx_message rsp = {};
        uint64_t rsp_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_DATA + (tx_tail * 64);
        if (region_read(rsp_offset, &rsp, sizeof(rsp)) == 0 &&
            rsp.seq_num == msg.seq_num) {
            if (log_alloc) {
                log_alloc_rsp("fallback rsp", rsp, tx_head, tx_tail);
            } else {
                GTX_LOG("Response slot fallback hit: cmd=%s seq=%u\n", gtx_cmd_name(cmd), rsp.seq_num);
            }

            uint32_t next_tail = (tx_tail + 1) % GTX_MSG_QUEUE_ENTRIES;
            region_write(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                         &next_tail, sizeof(next_tail));

            if (response != nullptr && rsp_size > 0) {
                size_t copy_size = rsp_size < sizeof(rsp.payload) ? rsp_size : sizeof(rsp.payload);
                memcpy(response, rsp.payload, copy_size);
            }
            return true;
        }

        /* Check timeout */
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > timeout_ms) {
            GTX_LOG_ERROR("Command timeout: cmd=%d, seq=%u\n", cmd, msg.seq_num);
            return false;
        }

        /* Small sleep to avoid busy-waiting */
#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }
}

bool gtx_pcie_connection::send_cmd_staged(uint8_t cmd, uint64_t address, uint32_t size,
                                          void* response, size_t rsp_size) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_irq_ctx == nullptr) {
        GTX_LOG_ERROR("send_cmd_staged: not connected\n");
        return false;
    }

    auto region_read = [&](uint64_t offset, void * data, size_t read_size) {
        return gtx_pcie_irq_region_read(m_irq_ctx, GTX_BAR2_REGION, offset, data, read_size);
    };

    auto region_write = [&](uint64_t offset, const void * data, size_t write_size) {
        return gtx_pcie_irq_region_write(m_irq_ctx, GTX_BAR2_REGION, offset, data, write_size);
    };

    gtx_message msg = {};
    msg.seq_num = ++m_seq_num;
    msg.type = cmd;
    msg.size = size;
    msg.address = address;

    uint32_t rx_head = 0;
    if (region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_HEAD,
                    &rx_head, sizeof(rx_head)) < 0) {
        GTX_LOG_ERROR("Failed to read RX head\n");
        return false;
    }

    uint64_t msg_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_DATA + (rx_head * 64);
    if (region_write(msg_offset, &msg, sizeof(msg)) < 0) {
        GTX_LOG_ERROR("Failed to write staged message to RX queue\n");
        return false;
    }

    uint32_t next_head = (rx_head + 1) % GTX_MSG_QUEUE_ENTRIES;
    if (region_write(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_RX_HEAD,
                     &next_head, sizeof(next_head)) < 0) {
        GTX_LOG_ERROR("Failed to update RX head\n");
        return false;
    }

    send_doorbell(GTX_DOORBELL_CMD_READY);

    auto start = std::chrono::steady_clock::now();
    const int timeout_ms = 30000;

    while (true) {
        uint32_t tx_head = 0, tx_tail = 0;
        region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_HEAD,
                    &tx_head, sizeof(tx_head));
        region_read(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                    &tx_tail, sizeof(tx_tail));

        if (tx_head != tx_tail) {
            gtx_message rsp = {};
            uint64_t rsp_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_DATA + (tx_tail * 64);
            region_read(rsp_offset, &rsp, sizeof(rsp));

            uint32_t next_tail = (tx_tail + 1) % GTX_MSG_QUEUE_ENTRIES;
            region_write(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                         &next_tail, sizeof(next_tail));

            if (rsp.seq_num == msg.seq_num) {
                if (response != nullptr && rsp_size > 0) {
                    size_t copy_size = rsp_size < sizeof(rsp.payload) ? rsp_size : sizeof(rsp.payload);
                    memcpy(response, rsp.payload, copy_size);
                }
                return true;
            }
        }

        gtx_message rsp = {};
        uint64_t rsp_offset = GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_DATA + (tx_tail * 64);
        if (region_read(rsp_offset, &rsp, sizeof(rsp)) == 0 &&
            rsp.seq_num == msg.seq_num) {
            uint32_t next_tail = (tx_tail + 1) % GTX_MSG_QUEUE_ENTRIES;
            region_write(GTX_MSG_QUEUE_BASE + GTX_MSG_QUEUE_TX_TAIL,
                         &next_tail, sizeof(next_tail));

            if (response != nullptr && rsp_size > 0) {
                size_t copy_size = rsp_size < sizeof(rsp.payload) ? rsp_size : sizeof(rsp.payload);
                memcpy(response, rsp.payload, copy_size);
            }
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed > timeout_ms) {
            GTX_LOG_ERROR("Staged command timeout: cmd=%d, seq=%u\n", cmd, msg.seq_num);
            return false;
        }

#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }
}

/*=============================================================================
 * GTX Backend Context
 *============================================================================*/

struct ggml_backend_gtx_context {
    int device_id;
    std::shared_ptr<gtx_pcie_connection> conn;
    std::shared_ptr<gtx_uart_vp_connection> uart;
    bool use_uart;
    ggml_gtx_stats_t stats;
    std::string last_error;
    uint8_t firmware_version[3];

    /* Buffer tracking */
    std::unordered_map<uint64_t, size_t> allocated_buffers; /* ptr -> size */
    size_t uart_arena_next = 0;

    ggml_backend_gtx_context(int device) : device_id(device), use_uart(gtx_use_uart_vp_transport()) {
        conn = ggml_backend_gtx_shared_connection(device);
        if (use_uart) {
            uart = ggml_backend_gtx_shared_uart_connection(device);
        }
        memset(&stats, 0, sizeof(stats));
        memset(firmware_version, 0, sizeof(firmware_version));
    }

    ~ggml_backend_gtx_context() = default;

    /* Get IRQ context from connection */
    gtx_pcie_irq_ctx* get_irq_ctx() const {
        return conn ? conn->get_irq_ctx() : nullptr;
    }
};

#if 0
/* Materialize channel-wise ADD/MUL broadcasting as an ordinary dense remote
 * tensor.  The graph tensor does not know its consumer shape when set_tensor
 * runs, so this is the consumer-side completion of the normal upload path.
 * The temporary occupies the top of the fixed tensor arena and is reused
 * after each synchronous descriptor launch. */
static bool gtx_materialize_binary_rhs(ggml_backend_gtx_context * ctx,
                                       ggml_gtx_kernel_launch_desc_t & desc) {
    if (ctx == nullptr || !ctx->use_uart || ctx->uart == nullptr ||
        (desc.kernel_id != GGML_GTX_KERNEL_ADD_F16 &&
         desc.kernel_id != GGML_GTX_KERNEL_MUL_F16)) {
        return true;
    }

    bool same_shape = true;
    for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
        same_shape = same_shape && desc.src_ne[1][d] == desc.ne[d];
    }
    if (same_shape) {
        return true;
    }

    /* This path intentionally covers scalar/channel-wise tensors such as
     * [1,1,C,1]. Other broadcast layouts remain rejected by lowering. */
    if (desc.src_ne[1][0] != 1u || desc.src_ne[1][1] != 1u) {
        GTX_LOG_ERROR("ADD/MUL remote materialization only supports channel-wise RHS\n");
        return false;
    }

    uint64_t src_elems = 1u;
    uint64_t dst_elems = 1u;
    uint64_t dense_stride = sizeof(uint16_t);
    for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
        if (desc.ne[d] == 0u || desc.src_ne[1][d] == 0u ||
            (desc.src_ne[1][d] != 1u && desc.src_ne[1][d] != desc.ne[d]) ||
            desc.src_nb[1][d] != dense_stride ||
            src_elems > SIZE_MAX / desc.src_ne[1][d] ||
            dst_elems > SIZE_MAX / desc.ne[d]) {
            GTX_LOG_ERROR("invalid ADD/MUL channel-wise RHS descriptor at dim=%u\n", d);
            return false;
        }
        src_elems *= desc.src_ne[1][d];
        dst_elems *= desc.ne[d];
        dense_stride *= desc.src_ne[1][d];
    }
    if (dst_elems > SIZE_MAX / sizeof(uint16_t)) {
        return false;
    }

    std::vector<uint16_t> source((size_t) src_elems);
    if (!ctx->uart->download_bytes(desc.src_addr[1], source.data(),
                                   source.size() * sizeof(uint16_t))) {
        GTX_LOG_ERROR("failed to download ADD/MUL channel-wise RHS\n");
        return false;
    }

    std::vector<uint16_t> expanded((size_t) dst_elems);
    size_t out = 0;
    for (uint32_t i3 = 0; i3 < desc.ne[3]; ++i3) {
        const uint32_t j3 = desc.src_ne[1][3] == 1u ? 0u : i3;
        for (uint32_t i2 = 0; i2 < desc.ne[2]; ++i2) {
            const uint32_t j2 = desc.src_ne[1][2] == 1u ? 0u : i2;
            const size_t src_index =
                (size_t) j3 * desc.src_ne[1][2] * desc.src_ne[1][1] +
                (size_t) j2 * desc.src_ne[1][1];
            const uint16_t value = source[src_index];
            for (uint32_t i1 = 0; i1 < desc.ne[1]; ++i1) {
                for (uint32_t i0 = 0; i0 < desc.ne[0]; ++i0) {
                    expanded[out++] = value;
                }
            }
        }
    }

    const uint64_t bytes = dst_elems * sizeof(uint16_t);
    const uint64_t aligned_bytes = (bytes + 63u) & ~63ull;
    const uint64_t arena_base = gtx_fixed_tensor_arena_addr();
    const uint64_t arena_size = gtx_fixed_tensor_arena_size();
    if (aligned_bytes > arena_size || ctx->uart_arena_next > arena_size - aligned_bytes) {
        GTX_LOG_ERROR("tensor arena has no room for broadcast RHS: bytes=%llu used=%zu arena=%llu\n",
                      (unsigned long long) bytes, ctx->uart_arena_next,
                      (unsigned long long) arena_size);
        return false;
    }
    const uint64_t remote = arena_base + arena_size - aligned_bytes;
    if (!ctx->uart->upload_bytes(remote, expanded.data(), (size_t) bytes)) {
        GTX_LOG_ERROR("failed to upload materialized ADD/MUL RHS\n");
        return false;
    }

    desc.src_addr[1] = remote;
    uint64_t stride = sizeof(uint16_t);
    for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
        desc.src_ne[1][d] = desc.ne[d];
        desc.src_nb[1][d] = (uint32_t) stride;
        stride *= desc.ne[d];
    }
    if (gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX BROADCAST] kernel=%u rhs=0x%llx bytes=%llu shape=[%u,%u,%u,%u]\n",
                desc.kernel_id, (unsigned long long) remote,
                (unsigned long long) bytes, desc.ne[0], desc.ne[1],
                desc.ne[2], desc.ne[3]);
    }
    return true;
}

static bool gtx_submit_uart_launch_desc(ggml_backend_gtx_context * ctx,
                                        const ggml_gtx_kernel_launch_desc_t & input) {
    ggml_gtx_kernel_launch_desc_t desc = input;
    return gtx_materialize_binary_rhs(ctx, desc) &&
           gtx_uart_launch_desc(*ctx->uart, desc);
}
#endif

struct ggml_backend_gtx_buffer_context {
    int device_id;
    uint64_t remote_ptr;
    size_t size;
    gtx_pcie_connection* conn;
    ggml_backend_gtx_context* backend_ctx;

    /* Local shadow buffer for small tensors */
    std::vector<uint8_t> shadow_buffer;
};

/*=============================================================================
 * Buffer Type Implementation
 *============================================================================*/

static const char* ggml_backend_gtx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)buft->context;
    static char name[64];
    snprintf(name, sizeof(name), "GTX%d", ctx->device_id);
    return name;
}

static ggml_backend_buffer_t ggml_backend_gtx_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size);

static void ggml_backend_gtx_buffer_free_buffer(ggml_backend_buffer_t buffer);

static bool ggml_backend_gtx_open_device_connection(gtx_pcie_connection * conn, int device_id) {
    if (conn == nullptr) {
        return false;
    }
    if (conn->is_open()) {
        return true;
    }

    char device_path[64];
#ifdef _WIN32
    snprintf(device_path, sizeof(device_path), "\\.\\gtx%d", device_id);
#else
    snprintf(device_path, sizeof(device_path), "/dev/gtx%d", device_id);
#endif

    return conn->open(device_path);
}

static size_t ggml_backend_gtx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64; /* 64-byte alignment for cache efficiency */
}

static size_t ggml_backend_gtx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 16ULL * 1024 * 1024 * 1024; /* 16 GB max */
}

static size_t gtx_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static bool ggml_backend_gtx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return false; /* Device memory, not host */
}

static bool ggml_backend_buffer_is_gtx(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.free_buffer == ggml_backend_gtx_buffer_free_buffer;
}

static uint64_t gtx_remote_data_ptr(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return 0;
    }

    if (tensor->view_src != nullptr) {
        return gtx_remote_data_ptr(tensor->view_src) +
               gtx_remote_byte_offset(tensor, tensor->view_offs);
    }

    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_gtx(tensor->buffer)) {
        return 0;
    }

    ggml_backend_gtx_buffer_context * ctx =
        (ggml_backend_gtx_buffer_context *) tensor->buffer->context;
    uint64_t base = ctx != nullptr ? ctx->remote_ptr : 0;
    uintptr_t local_base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    uintptr_t local_ptr = (uintptr_t) tensor->data;
    return base + (local_ptr - local_base);
}

static bool gtx_is_valid_unary_op(int32_t op) {
    return op >= 0 && op < GGML_UNARY_OP_COUNT;
}

static bool gtx_is_valid_glu_op(int32_t op) {
    return op >= 0 && op < GGML_GLU_OP_COUNT;
}

static constexpr uint32_t GGML_GTX_OP_UNSUPPORTED_HOST = UINT32_MAX;

static uint32_t gtx_normalize_host_op(const ggml_tensor * tensor) {
    const uint32_t raw_op = tensor != nullptr ? (uint32_t) tensor->op : (uint32_t) GGML_OP_NONE;

    if (tensor == nullptr) {
        return raw_op;
    }

    const char * host_op_name = ggml_op_name((ggml_op) raw_op);

    // Newer llama.cpp ggml inserts GATED_DELTA_NET before UNARY, so raw UNARY/GLU
    // values arrive shifted by +1 compared to this backend's bundled ggml headers.
    if (raw_op == (uint32_t) (GGML_OP_UNARY + 1)) {
        const int32_t unary_op = tensor->op_params[0];
        if (tensor->src[1] == nullptr && gtx_is_valid_unary_op(unary_op)) {
            return (uint32_t) GGML_OP_UNARY;
        }
    }

    // Dynamic backends can be loaded by a newer ggml host whose op enum has
    // drifted from this backend's bundled headers.  Prefer the host's op name
    // when it clearly identifies a parameterized op we can still execute.
    if (raw_op != (uint32_t) GGML_OP_UNARY &&
        strcmp(host_op_name, "UNARY") == 0) {
        const int32_t unary_op = tensor->op_params[0];
        if (tensor->src[1] == nullptr && gtx_is_valid_unary_op(unary_op)) {
            return (uint32_t) GGML_OP_UNARY;
        }
    }

    if (raw_op == (uint32_t) GGML_OP_COUNT) {
        const int32_t glu_op = tensor->op_params[0];
        if (tensor->src[1] != nullptr && gtx_is_valid_glu_op(glu_op)) {
            return (uint32_t) GGML_OP_GLU;
        }
    }

    if (raw_op != (uint32_t) GGML_OP_GLU &&
        strcmp(host_op_name, "GLU") == 0) {
        const int32_t glu_op = tensor->op_params[0];
        if (tensor->src[1] != nullptr && gtx_is_valid_glu_op(glu_op)) {
            return (uint32_t) GGML_OP_GLU;
        }
    }

    if (raw_op != (uint32_t) GGML_OP_UPSCALE &&
        strcmp(host_op_name, "UPSCALE") == 0) {
        return (uint32_t) GGML_OP_UPSCALE;
    }

    if (raw_op != (uint32_t) GGML_OP_PAD &&
        strcmp(host_op_name, "PAD") == 0) {
        return (uint32_t) GGML_OP_PAD;
    }

    if (raw_op != (uint32_t) GGML_OP_CONV_TRANSPOSE_2D &&
        strcmp(host_op_name, "CONV_TRANSPOSE_2D") == 0) {
        return (uint32_t) GGML_OP_CONV_TRANSPOSE_2D;
    }

    if (raw_op != (uint32_t) GGML_OP_LEAKY_RELU &&
        strcmp(host_op_name, "LEAKY_RELU") == 0) {
        return (uint32_t) GGML_OP_LEAKY_RELU;
    }

    if (raw_op != (uint32_t) GGML_OP_CLAMP &&
        strcmp(host_op_name, "CLAMP") == 0) {
        return (uint32_t) GGML_OP_CLAMP;
    }

    return raw_op;
}

static const char * gtx_tensor_op_display_name(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "NULL";
    }
    const uint32_t normalized_op = gtx_normalize_host_op(tensor);
    if (normalized_op == GGML_GTX_OP_UNSUPPORTED_HOST) {
        return ggml_op_name(tensor->op);
    }
    return gtx_normalized_op_name(normalized_op);
}

static bool gtx_is_swiglu_node(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_normalize_host_op(tensor) == (uint32_t) GGML_OP_GLU &&
           tensor->op_params[0] == GGML_GLU_OP_SWIGLU;
}

static bool gtx_is_gelu_node(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_normalize_host_op(tensor) == (uint32_t) GGML_OP_UNARY &&
           tensor->op_params[0] == GGML_UNARY_OP_GELU;
}

static bool gtx_add_has_f32_operand(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_normalize_host_op(tensor) == (uint32_t) GGML_OP_ADD &&
           (tensor->type == GGML_TYPE_F32 ||
            (tensor->src[0] != nullptr && tensor->src[0]->type == GGML_TYPE_F32) ||
            (tensor->src[1] != nullptr && tensor->src[1]->type == GGML_TYPE_F32));
}

static bool gtx_is_add_f16_f32_vector_to_f16(const ggml_tensor * tensor) {
    if (tensor == nullptr ||
        gtx_normalize_host_op(tensor) != (uint32_t) GGML_OP_ADD ||
        tensor->src[0] == nullptr ||
        tensor->src[1] == nullptr ||
        gtx_remote_tensor_type(tensor) != GGML_TYPE_F16 ||
        gtx_remote_tensor_type(tensor->src[0]) != GGML_TYPE_F16 ||
        gtx_remote_tensor_type(tensor->src[1]) != GGML_TYPE_F32 ||
        !gtx_preserve_remote_f32(tensor->src[1]) ||
        tensor->src[0]->ne[0] != tensor->ne[0] ||
        tensor->src[1]->ne[0] != tensor->ne[0]) {
        return false;
    }

    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->src[0]->ne[i] != tensor->ne[i]) {
            return false;
        }
        if (tensor->src[1]->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool gtx_is_add_f16_bias_vector_basic(const ggml_tensor * tensor) {
    if (tensor == nullptr ||
        gtx_normalize_host_op(tensor) != (uint32_t) GGML_OP_ADD ||
        tensor->type != GGML_TYPE_F32 ||
        tensor->src[0] == nullptr ||
        tensor->src[1] == nullptr ||
        tensor->src[0]->type != GGML_TYPE_F32 ||
        tensor->src[1]->type != GGML_TYPE_F32 ||
        strstr(tensor->src[1]->name, "bias") == nullptr ||
        tensor->src[0]->ne[0] != tensor->ne[0] ||
        tensor->src[1]->ne[0] != tensor->ne[0]) {
        return false;
    }

    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->src[0]->ne[i] != tensor->ne[i]) {
            return false;
        }
        if (tensor->src[1]->ne[i] != 1) {
            return false;
        }
    }

    return gtx_remote_tensor_type(tensor) == GGML_TYPE_F16 &&
           gtx_remote_tensor_type(tensor->src[0]) == GGML_TYPE_F16 &&
           gtx_remote_tensor_type(tensor->src[1]) == GGML_TYPE_F16;
}

static bool gtx_is_cont_same_shape(const ggml_tensor * tensor) {
    if (tensor == nullptr ||
        gtx_normalize_host_op(tensor) != (uint32_t) GGML_OP_CONT ||
        tensor->src[0] == nullptr) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != tensor->src[0]->ne[i]) {
            return false;
        }
    }
    return true;
}

static uint32_t gtx_wire_op_from_tensor(const ggml_tensor * tensor) {
    const uint32_t op = gtx_normalize_host_op(tensor);

    if (tensor == nullptr) {
        return op;
    }

    if (op == (uint32_t) GGML_OP_UNARY) {
        switch ((ggml_unary_op) tensor->op_params[0]) {
            case GGML_UNARY_OP_ABS:          return GGML_GTX_OP_ABS;
            case GGML_UNARY_OP_SGN:          return GGML_GTX_OP_SGN;
            case GGML_UNARY_OP_NEG:          return GGML_GTX_OP_NEG;
            case GGML_UNARY_OP_STEP:         return GGML_GTX_OP_STEP;
            case GGML_UNARY_OP_TANH:         return GGML_GTX_OP_TANH;
            case GGML_UNARY_OP_ELU:          return GGML_GTX_OP_ELU;
            case GGML_UNARY_OP_RELU:         return GGML_GTX_OP_RELU;
            case GGML_UNARY_OP_SIGMOID:      return GGML_GTX_OP_SIGMOID;
            case GGML_UNARY_OP_GELU:         return GGML_GTX_OP_GELU;
            case GGML_UNARY_OP_GELU_QUICK:   return GGML_GTX_OP_GELU_QUICK;
            case GGML_UNARY_OP_SILU:         return GGML_GTX_OP_SILU;
            case GGML_UNARY_OP_HARDSWISH:    return GGML_GTX_OP_HARDSWISH;
            case GGML_UNARY_OP_HARDSIGMOID:  return GGML_GTX_OP_HARDSIGMOID;
            case GGML_UNARY_OP_EXP:          return GGML_GTX_OP_EXP;
            case GGML_UNARY_OP_EXPM1:        return GGML_GTX_OP_EXPM1;
            case GGML_UNARY_OP_SOFTPLUS:     return GGML_GTX_OP_SOFTPLUS;
            case GGML_UNARY_OP_GELU_ERF:     return GGML_GTX_OP_GELU_ERF;
            case GGML_UNARY_OP_XIELU:        return GGML_GTX_OP_XIELU;
            case GGML_UNARY_OP_FLOOR:        return GGML_GTX_OP_FLOOR;
            case GGML_UNARY_OP_CEIL:         return GGML_GTX_OP_CEIL;
            case GGML_UNARY_OP_ROUND:        return GGML_GTX_OP_ROUND;
            case GGML_UNARY_OP_TRUNC:        return GGML_GTX_OP_TRUNC;
            default:                         return op;
        }
    }

    if (op == (uint32_t) GGML_OP_GLU) {
        return GGML_GTX_OP_GLU;
    }

    return op;
}

static gtx_rpc_tensor gtx_serialize_tensor(const ggml_tensor * tensor) {
    gtx_rpc_tensor result = {};

    if (tensor == nullptr) {
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = gtx_remote_type(tensor->type);

    if (tensor->buffer && ggml_backend_buffer_is_gtx(tensor->buffer)) {
        ggml_backend_gtx_buffer_context * ctx =
            (ggml_backend_gtx_buffer_context *) tensor->buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = gtx_remote_data_ptr(tensor);
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = (uint32_t) gtx_remote_byte_offset(tensor, tensor->nb[i]);
    }

    if (tensor->type == GGML_TYPE_Q8_0 && tensor->view_src == nullptr) {
        result.nb[0] = sizeof(uint16_t);
        for (uint32_t i = 1; i < GGML_MAX_DIMS; ++i) {
            result.nb[i] = result.nb[i - 1] * result.ne[i - 1];
        }
    }

    result.op = gtx_wire_op_from_tensor(tensor);
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;

    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = gtx_remote_byte_offset(tensor, tensor->view_offs);

    memset(result.name, 0, sizeof(result.name));
    snprintf(result.name, 64, "%s", tensor->name);   // 위 name[64] 와 같은 이유
    return result;
}

static bool gtx_fill_tensor_desc(const ggml_tensor * tensor,
                                 uint64_t * addr,
                                 uint32_t * type,
                                 uint32_t ne[GGML_MAX_DIMS],
                                 uint32_t nb[GGML_MAX_DIMS]) {
    if (tensor == nullptr) {
        *addr = 0;
        *type = 0;
        memset(ne, 0, GGML_MAX_DIMS * sizeof(uint32_t));
        memset(nb, 0, GGML_MAX_DIMS * sizeof(uint32_t));
        return true;
    }

    *addr = gtx_remote_data_ptr(tensor);
    *type = (uint32_t) gtx_remote_tensor_type(tensor);
    if (*addr == 0) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] < 0 || tensor->ne[i] > UINT32_MAX) {
            return false;
        }
        const uint64_t rnb = gtx_remote_byte_offset(tensor, tensor->nb[i]);
        if (rnb > UINT32_MAX) {
            return false;
        }
        ne[i] = (uint32_t) tensor->ne[i];
        nb[i] = (uint32_t) rnb;
    }

    return true;
}

static uint32_t gtx_kernel_id_from_node(const ggml_tensor * node) {
    if (node == nullptr) {
        return GGML_GTX_KERNEL_INVALID;
    }

    switch (gtx_normalize_host_op(node)) {
        case GGML_OP_ADD:
            return gtx_is_add_f16_f32_vector_to_f16(node) ?
                GGML_GTX_KERNEL_ADD_F16_F32_TO_F16 : GGML_GTX_KERNEL_ADD_F16;
        case GGML_OP_MUL:      return GGML_GTX_KERNEL_MUL_F16;
        case GGML_OP_SCALE:    return GGML_GTX_KERNEL_SCALE_F16;
        case GGML_OP_NORM:     return GGML_GTX_KERNEL_NORM_F16;
        case GGML_OP_RMS_NORM: return GGML_GTX_KERNEL_RMS_NORM_F16;
        case GGML_OP_SOFT_MAX: return GGML_GTX_KERNEL_SOFT_MAX_F16;
        case GGML_OP_MUL_MAT:  return GGML_GTX_KERNEL_MUL_MAT_F16;
        case GGML_OP_ROPE:     return GGML_GTX_KERNEL_ROPE_F16;
        case GGML_OP_GET_ROWS: return GGML_GTX_KERNEL_GET_ROWS_F16;
        case GGML_OP_SET_ROWS: return GGML_GTX_KERNEL_SET_ROWS_F16;
        case GGML_OP_IM2COL:   return GGML_GTX_KERNEL_IM2COL_F16;
        case GGML_OP_REPEAT:   return GGML_GTX_KERNEL_REPEAT_F16;
        case GGML_OP_CONCAT:   return GGML_GTX_KERNEL_CONCAT_F16;
        case GGML_OP_CONV_2D:  return GGML_GTX_KERNEL_CONV_2D_F16;
        case GGML_OP_CONV_2D_DW: return GGML_GTX_KERNEL_CONV_2D_DW_F16;
        case GGML_OP_CONV_TRANSPOSE_2D: return GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16;
        case GGML_OP_UPSCALE:  return GGML_GTX_KERNEL_UPSCALE_F16;
        case GGML_OP_PAD:      return GGML_GTX_KERNEL_PAD_F16;
        case GGML_OP_SIN:      return GGML_GTX_KERNEL_SIN_F16;
        case GGML_OP_COS:      return GGML_GTX_KERNEL_COS_F16;
        case GGML_OP_CONT:     return GGML_GTX_KERNEL_CONT_F16;
        case GGML_OP_CPY:      return GGML_GTX_KERNEL_CPY_F16;
        case GGML_GTX_OP_GELU:  return GGML_GTX_KERNEL_GELU_F16;
        case GGML_GTX_OP_ABS:   return GGML_GTX_KERNEL_ABS_F16;
        case GGML_GTX_OP_NEG:   return GGML_GTX_KERNEL_NEG_F16;
        case GGML_GTX_OP_SGN:   return GGML_GTX_KERNEL_SGN_F16;
        case GGML_GTX_OP_CEIL:  return GGML_GTX_KERNEL_CEIL_F16;
        case GGML_GTX_OP_FLOOR: return GGML_GTX_KERNEL_FLOOR_F16;
        case GGML_GTX_OP_TANH:  return GGML_GTX_KERNEL_TANH_F16;
        case GGML_GTX_OP_SILU:  return GGML_GTX_KERNEL_SILU_F16;
        case GGML_GTX_OP_RELU:  return GGML_GTX_KERNEL_RELU_F16;
        case GGML_OP_LEAKY_RELU: return GGML_GTX_KERNEL_LEAKY_RELU_F16;
        case GGML_OP_CLAMP:     return GGML_GTX_KERNEL_CLAMP_F16;
        case GGML_OP_GLU:
            return gtx_is_swiglu_node(node) ? GGML_GTX_KERNEL_SWIGLU_F16 : GGML_GTX_KERNEL_INVALID;
        case GGML_OP_UNARY:
            switch ((ggml_unary_op) node->op_params[0]) {
                case GGML_UNARY_OP_GELU:  return GGML_GTX_KERNEL_GELU_F16;
                case GGML_UNARY_OP_ABS:   return GGML_GTX_KERNEL_ABS_F16;
                case GGML_UNARY_OP_NEG:   return GGML_GTX_KERNEL_NEG_F16;
                case GGML_UNARY_OP_SGN:   return GGML_GTX_KERNEL_SGN_F16;
                case GGML_UNARY_OP_CEIL:  return GGML_GTX_KERNEL_CEIL_F16;
                case GGML_UNARY_OP_FLOOR: return GGML_GTX_KERNEL_FLOOR_F16;
                case GGML_UNARY_OP_TANH:  return GGML_GTX_KERNEL_TANH_F16;
                case GGML_UNARY_OP_SILU:  return GGML_GTX_KERNEL_SILU_F16;
                case GGML_UNARY_OP_RELU:  return GGML_GTX_KERNEL_RELU_F16;
                default:                  return GGML_GTX_KERNEL_INVALID;
            }
        default:
            return GGML_GTX_KERNEL_INVALID;
    }
}

/* Kernel-owned resource metadata. Current L2 micro-kernels advertise zero;
 * the function is the single host extension point for a future size formula. */
static uint32_t gtx_kernel_l2_scratch_size(const ggml_tensor * node) {
    if (node != nullptr && gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_ROPE_F16 &&
        node->ne[2] > 0 && node->ne[2] <= UINT32_MAX) {
        return ggml_gtx_rope_scratch_size((uint32_t)node->ne[2],
                                          (uint32_t)node->op_params[1]);
    }
    if (node != nullptr &&
        (gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_SIN_F16 ||
         gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_COS_F16)) {
        return 0x40200u;
    }
    if (node != nullptr &&
        (gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_CONV_2D_F16 ||
         gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_CONV_2D_DW_F16)) {
        return 8u * 1024u * 1024u;
    }
    return 0u;
}

static uint64_t gtx_stash_data_budget(const ggml_tensor * node) {
    const uint64_t required = gtx_kernel_l2_scratch_size(node);
    const uint64_t scratch = std::max<uint64_t>(required, GTX_STASH_MIN_SCRATCH_BYTES);
    return scratch < GTX_STASH_L2_BYTES ? GTX_STASH_L2_BYTES - scratch : 0u;
}

static uint64_t gtx_stash_data_budget(const std::vector<ggml_tensor *> & nodes) {
    uint64_t required = 0u;
    for (const ggml_tensor * node : nodes) {
        required = std::max<uint64_t>(required, gtx_kernel_l2_scratch_size(node));
    }
    const uint64_t scratch = std::max<uint64_t>(required, GTX_STASH_MIN_SCRATCH_BYTES);
    return scratch < GTX_STASH_L2_BYTES ? GTX_STASH_L2_BYTES - scratch : 0u;
}

enum class gtx_l2_chain_kind : uint32_t {
    unsupported = 0,
    native      = 1,
    producer    = 2,
    boundary    = 3,
};

struct gtx_l2_chainability {
    gtx_l2_chain_kind kind;
    uint32_t kernel_id;
    const char * reason;
};

static const char * gtx_l2_chain_kind_name(gtx_l2_chain_kind kind) {
    switch (kind) {
        case gtx_l2_chain_kind::native:      return "native";
        case gtx_l2_chain_kind::producer:    return "producer";
        case gtx_l2_chain_kind::boundary:    return "boundary";
        case gtx_l2_chain_kind::unsupported: return "unsupported";
    }
    return "unsupported";
}

static bool gtx_l2_chain_participates(gtx_l2_chain_kind kind) {
    return kind == gtx_l2_chain_kind::native ||
           kind == gtx_l2_chain_kind::producer;
}

static bool gtx_tensor_remote_f16_dense(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_remote_tensor_type(tensor) == GGML_TYPE_F16 &&
           ggml_is_contiguous(tensor);
}

static bool gtx_tensor_remote_f16_vector_contiguous(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_remote_tensor_type(tensor) == GGML_TYPE_F16 &&
           tensor->ne[0] > 0 &&
           tensor->ne[0] <= UINT16_MAX &&
           gtx_remote_byte_offset(tensor, tensor->nb[0]) == sizeof(uint16_t);
}

// A matrix's logical row count can be larger than a single kernel tile.  This
// deliberately checks only the element stride; callers which expose a whole
// row to a kernel must still enforce their own `ne[0]` bound.
static bool gtx_tensor_remote_f16_element_contiguous(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           gtx_remote_tensor_type(tensor) == GGML_TYPE_F16 &&
           tensor->ne[0] > 0 &&
           gtx_remote_byte_offset(tensor, tensor->nb[0]) == sizeof(uint16_t);
}

static bool gtx_tensor_remote_f16_c_fast_whc(const ggml_tensor * tensor) {
    if (tensor == nullptr || gtx_remote_tensor_type(tensor) != GGML_TYPE_F16 ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1) {
        return false;
    }
    const uint64_t c_bytes = (uint64_t)tensor->ne[2] * sizeof(uint16_t);
    return gtx_remote_byte_offset(tensor, tensor->nb[2]) == sizeof(uint16_t) &&
           gtx_remote_byte_offset(tensor, tensor->nb[0]) == c_bytes &&
           gtx_remote_byte_offset(tensor, tensor->nb[1]) == c_bytes * (uint64_t)tensor->ne[0];
}

static bool gtx_tensor_remote_f16_c_fast_dw_weight(const ggml_tensor * tensor) {
    if (tensor == nullptr || gtx_remote_tensor_type(tensor) != GGML_TYPE_F16 ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] <= 0) {
        return false;
    }
    const uint64_t c_bytes = (uint64_t)tensor->ne[3] * sizeof(uint16_t);
    return gtx_remote_byte_offset(tensor, tensor->nb[3]) == sizeof(uint16_t) &&
           gtx_remote_byte_offset(tensor, tensor->nb[0]) == c_bytes &&
           gtx_remote_byte_offset(tensor, tensor->nb[1]) == c_bytes * (uint64_t)tensor->ne[0];
}

static bool gtx_l2_conv_layout_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr) {
        return false;
    }
    if (gtx_tensor_remote_f16_element_contiguous(node) &&
        gtx_tensor_remote_f16_element_contiguous(node->src[0]) &&
        gtx_tensor_remote_f16_element_contiguous(node->src[1])) {
        return true;
    }
    return gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_CONV_2D_DW_F16 &&
           gtx_tensor_remote_f16_c_fast_whc(node) &&
           gtx_tensor_remote_f16_c_fast_dw_weight(node->src[0]) &&
           gtx_tensor_remote_f16_c_fast_whc(node->src[1]);
}

static bool gtx_tensor_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

static bool gtx_l2_dense_unary_shape_ok(const ggml_tensor * node) {
    return node != nullptr &&
           node->src[0] != nullptr &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_tensor_remote_f16_dense(node->src[0]) &&
           gtx_tensor_same_shape(node, node->src[0]);
}

static bool gtx_l2_dense_binary_shape_ok(const ggml_tensor * node) {
    return node != nullptr &&
           node->src[0] != nullptr &&
           node->src[1] != nullptr &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_tensor_remote_f16_dense(node->src[0]) &&
           gtx_tensor_remote_f16_dense(node->src[1]) &&
           gtx_tensor_same_shape(node, node->src[0]) &&
           gtx_tensor_same_shape(node, node->src[1]);
}

static bool gtx_l2_rhs_broadcast_shape_ok(const ggml_tensor * node, bool allow_scalar_dim0) {
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_remote_f16_dense(node->src[1]) ||
        !gtx_tensor_same_shape(node, node->src[0])) {
        return false;
    }

    const ggml_tensor * rhs = node->src[1];
    if (rhs->ne[0] != node->ne[0] &&
        (!allow_scalar_dim0 || rhs->ne[0] != 1)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (rhs->ne[i] != 1 && rhs->ne[i] != node->ne[i]) {
            return false;
        }
    }
    return true;
}

static bool gtx_l2_rhs_col_vector_broadcast_shape_ok(const ggml_tensor * node,
                                                     bool allow_scalar_dim0) {
    if (!gtx_l2_rhs_broadcast_shape_ok(node, allow_scalar_dim0) ||
        node == nullptr ||
        node->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * rhs = node->src[1];
    if (rhs->ne[0] != node->ne[0] &&
        (!allow_scalar_dim0 || rhs->ne[0] != 1)) {
        return false;
    }
    return rhs->ne[1] == 1 &&
           rhs->ne[2] == 1 &&
           rhs->ne[3] == 1;
}

static bool gtx_l2_scale_shape_ok(const ggml_tensor * node) {
    return node != nullptr &&
           node->src[0] != nullptr &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_tensor_remote_f16_dense(node->src[0]) &&
           gtx_tensor_same_shape(node, node->src[0]);
}

static bool gtx_l2_add_shape_ok(const ggml_tensor * node) {
    return gtx_l2_dense_binary_shape_ok(node) ||
           gtx_l2_rhs_broadcast_shape_ok(node, true) ||
           gtx_is_add_f16_bias_vector_basic(node);
}

static bool gtx_l2_repeat_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0])) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->src[0]->ne[i] <= 0 || node->ne[i] < node->src[0]->ne[i] ||
            (node->ne[i] % node->src[0]->ne[i]) != 0 ||
            node->ne[i] > UINT16_MAX || node->src[0]->ne[i] > UINT16_MAX) {
            return false;
        }
    }
    return (uint64_t)node->ne[0] * sizeof(ggml_fp16_t) <= UINT16_MAX &&
           (uint64_t)node->src[0]->ne[0] * sizeof(ggml_fp16_t) <= UINT16_MAX;
}

static bool gtx_l2_concat_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_remote_f16_dense(node->src[1])) {
        return false;
    }
    const int32_t dim = ((const int32_t *)node->op_params)[0];
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t expected = i == dim ?
            node->src[0]->ne[i] + node->src[1]->ne[i] : node->src[0]->ne[i];
        if (node->ne[i] != expected ||
            (i != dim && node->src[0]->ne[i] != node->src[1]->ne[i]) ||
            node->ne[i] > UINT16_MAX || node->src[0]->ne[i] > UINT16_MAX ||
            node->src[1]->ne[i] > UINT16_MAX) {
            return false;
        }
    }
    const auto copy_shape_ok = [](const ggml_tensor * tensor) {
        const uint64_t row_bytes = (uint64_t)tensor->ne[0] * sizeof(ggml_fp16_t);
        const uint64_t rows = (uint64_t)tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
        return row_bytes > 0 && row_bytes <= UINT16_MAX && rows > 0 && rows <= UINT16_MAX;
    };
    return copy_shape_ok(node) && copy_shape_ok(node->src[0]) && copy_shape_ok(node->src[1]);
}

static bool gtx_l2_pad_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0])) {
        return false;
    }
    const int32_t * p = (const int32_t *)node->op_params;
    if (p[0] != 0 || p[2] != 0 || p[4] != 0 || p[6] != 0 ||
        p[1] < 0 || p[3] < 0 || p[5] < 0 || p[7] != 0 || p[8] != 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t expected = node->src[0]->ne[i] + (i < 3 ? p[i * 2 + 1] : 0);
        if (node->ne[i] != expected || node->ne[i] > UINT16_MAX ||
            node->src[0]->ne[i] > UINT16_MAX) {
            return false;
        }
    }
    const uint64_t src_row_bytes = (uint64_t)node->src[0]->ne[0] * sizeof(ggml_fp16_t);
    const uint64_t dst_row_bytes = (uint64_t)node->ne[0] * sizeof(ggml_fp16_t);
    const uint64_t dst_rows = (uint64_t)node->ne[1] * node->ne[2] * node->ne[3];
    return src_row_bytes > 0 && src_row_bytes <= UINT16_MAX &&
           dst_row_bytes > 0 && dst_row_bytes <= UINT16_MAX &&
           dst_rows > 0 && dst_rows <= UINT16_MAX;
}

static bool gtx_l2_conv_transpose_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_remote_f16_dense(node->src[1])) {
        return false;
    }
    const uint64_t slot_budget = (gtx_stash_data_budget(node) / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint64_t dst_bytes = gtx_remote_size_bytes(node);
    const uint64_t weight_bytes = gtx_remote_size_bytes(node->src[0]);
    const uint64_t input_bytes = gtx_remote_size_bytes(node->src[1]);
    return dst_bytes > 0 && dst_bytes <= slot_budget &&
           weight_bytes > 0 && weight_bytes <= slot_budget &&
           input_bytes > 0 && input_bytes <= slot_budget &&
           dst_bytes <= UINT64_MAX - weight_bytes &&
           dst_bytes + weight_bytes <= UINT64_MAX - input_bytes &&
           dst_bytes + weight_bytes + input_bytes <= gtx_stash_data_budget(node);
}

static bool gtx_l2_conv_whole_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        !gtx_l2_conv_layout_ok(node) ||
        node->ne[3] != 1 || node->src[1]->ne[3] != 1) {
        return false;
    }
    return node->ne[0] > 0 && node->ne[1] > 0 && node->ne[2] > 0 &&
           node->src[0]->ne[0] > 0 && node->src[0]->ne[1] > 0 &&
           node->src[1]->ne[0] > 0 && node->src[1]->ne[1] > 0 &&
           node->src[1]->ne[2] > 0;
}

static bool gtx_l2_softmax_basic_shape_ok(const ggml_tensor * node) {
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] != nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_same_shape(node, node->src[0]) ||
        node->ne[0] <= 0 ||
        node->ne[0] > UINT16_MAX) {
        return false;
    }

    const int32_t scale_bits = node->op_params[0];
    const int32_t max_bias_bits = node->op_params[1];
    return scale_bits == (int32_t)0x3f800000u &&
           max_bias_bits == 0;
}

static bool gtx_l2_rms_norm_shape_ok(const ggml_tensor * node) {
    return gtx_l2_dense_unary_shape_ok(node) &&
           node->ne[0] > 0 &&
           node->ne[0] <= UINT16_MAX &&
           (uint64_t)node->ne[0] * sizeof(uint16_t) <= 384ull * 1024ull;
}

static bool gtx_l2_swiglu_shape_ok(const ggml_tensor * node) {
    return node != nullptr &&
           node->src[0] != nullptr &&
           node->src[1] == nullptr &&
           gtx_is_swiglu_node(node) &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_tensor_remote_f16_dense(node->src[0]) &&
           node->ne[0] > 0 &&
           node->ne[0] <= UINT16_MAX &&
           node->src[0]->ne[0] == node->ne[0] * 2 &&
           node->src[0]->ne[0] <= UINT16_MAX &&
           node->src[0]->ne[1] == node->ne[1] &&
           node->src[0]->ne[2] == node->ne[2] &&
           node->src[0]->ne[3] == node->ne[3] &&
           (uint64_t)node->ne[0] * sizeof(uint16_t) <= 384ull * 1024ull;
}

static bool gtx_l2_mul_mat_shape_ok(const ggml_tensor * node) {
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_remote_f16_dense(node->src[1])) {
        return false;
    }

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t batch2 = node->ne[2];
    const int64_t batch3 = node->ne[3];

    return k > 0 && rows > 0 && cols > 0 &&
           batch2 > 0 && batch3 > 0 &&
           k <= UINT16_MAX &&
           rows <= UINT16_MAX &&
           cols <= UINT16_MAX &&
           batch2 <= UINT16_MAX &&
           batch3 <= UINT16_MAX &&
           src1->ne[0] == k &&
           node->ne[0] == rows &&
           node->ne[1] == cols &&
           src1->ne[2] == batch2 &&
           src1->ne[3] == batch3 &&
           src0->ne[2] > 0 &&
           src0->ne[3] > 0 &&
           (batch2 % src0->ne[2]) == 0 &&
           (batch3 % src0->ne[3]) == 0;
}

static bool gtx_l2_mul_mat_vector_shape_ok(const ggml_tensor * node) {
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_vector_contiguous(node) ||
        !gtx_tensor_remote_f16_vector_contiguous(node->src[0]) ||
        !gtx_tensor_remote_f16_vector_contiguous(node->src[1])) {
        return false;
    }

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t batch2 = node->ne[2];
    const int64_t batch3 = node->ne[3];

    return k > 0 && rows > 0 && cols > 0 &&
           batch2 > 0 && batch3 > 0 &&
           k <= UINT16_MAX &&
           rows <= UINT16_MAX &&
           cols <= UINT16_MAX &&
           batch2 <= UINT16_MAX &&
           batch3 <= UINT16_MAX &&
           src1->ne[0] == k &&
           node->ne[0] == rows &&
           node->ne[1] == cols &&
           src1->ne[2] == batch2 &&
           src1->ne[3] == batch3 &&
           src0->ne[2] > 0 &&
           src0->ne[3] > 0 &&
           (batch2 % src0->ne[2]) == 0 &&
           (batch3 % src0->ne[3]) == 0;
}

static bool gtx_l2_materialize_producer_shape_ok(const ggml_tensor * node) {
    constexpr uint64_t gtx_host_stash_slot_bytes = 5ull * 1024ull * 1024ull;
    return node != nullptr &&
           node->src[0] != nullptr &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_remote_tensor_type(node->src[0]) == GGML_TYPE_F16 &&
           gtx_tensor_same_shape(node, node->src[0]) &&
           node->ne[0] > 0 &&
           node->ne[0] <= UINT16_MAX &&
           (uint64_t) gtx_remote_size_bytes(node) > 0 &&
           (uint64_t) gtx_remote_size_bytes(node) <= gtx_host_stash_slot_bytes;
}

static bool gtx_l2_materialize_producer_tile_shape_ok(const ggml_tensor * node) {
    return node != nullptr &&
           node->src[0] != nullptr &&
           gtx_tensor_remote_f16_dense(node) &&
           gtx_remote_tensor_type(node->src[0]) == GGML_TYPE_F16 &&
           gtx_tensor_same_shape(node, node->src[0]) &&
           node->ne[0] > 0 &&
           node->ne[0] <= UINT16_MAX;
}

static bool gtx_l2_get_rows_producer_shape_ok(const ggml_tensor * node) {
    constexpr uint64_t gtx_host_stash_slot_bytes = 5ull * 1024ull * 1024ull;
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        gtx_remote_tensor_type(node->src[0]) != GGML_TYPE_F16 ||
        (node->src[1]->type != GGML_TYPE_I32 && node->src[1]->type != GGML_TYPE_I64) ||
        node->src[0]->ne[0] <= 0 ||
        node->src[0]->ne[0] > UINT16_MAX) {
        return false;
    }

    const uint64_t row_bytes = (uint64_t)node->src[0]->ne[0] * sizeof(uint16_t);
    const uint64_t dst_bytes = (uint64_t)gtx_remote_size_bytes(node);
    return row_bytes > 0 &&
           row_bytes <= UINT16_MAX &&
           dst_bytes > 0 &&
           dst_bytes <= gtx_host_stash_slot_bytes;
}

// Initial L2 SET_ROWS support is intentionally narrow: src2 already aliases
// the full destination backing region, so the device can keep that resident
// slot and overwrite only the indexed rows from src0.
static bool gtx_l2_set_rows_producer_shape_ok(const ggml_tensor * node) {
    constexpr uint64_t gtx_host_stash_slot_bytes = 5ull * 1024ull * 1024ull;
    if (node == nullptr ||
        node->src[0] == nullptr || node->src[1] == nullptr || node->src[2] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_remote_f16_dense(node->src[2]) ||
        gtx_remote_tensor_type(node->src[0]) != GGML_TYPE_F16 ||
        gtx_remote_tensor_type(node->src[2]) != GGML_TYPE_F16 ||
        (node->src[1]->type != GGML_TYPE_I32 && node->src[1]->type != GGML_TYPE_I64) ||
        gtx_remote_data_ptr(node) != gtx_remote_data_ptr(node->src[2]) ||
        !gtx_tensor_same_shape(node, node->src[2]) ||
        node->ne[0] <= 0 || node->ne[0] > UINT16_MAX ||
        node->src[0]->ne[0] != node->ne[0] ||
        node->src[0]->ne[1] <= 0 || node->src[0]->ne[1] > node->ne[1] ||
        node->src[0]->ne[2] != node->ne[2] ||
        node->src[0]->ne[3] != node->ne[3]) {
        return false;
    }

    const uint64_t row_bytes = (uint64_t)node->ne[0] * sizeof(uint16_t);
    const uint64_t data_bytes = row_bytes * (uint64_t)node->src[0]->ne[1] *
                                (uint64_t)node->ne[2] * (uint64_t)node->ne[3];
    const uint64_t dst_bytes = (uint64_t)gtx_remote_size_bytes(node);
    return row_bytes > 0 && row_bytes <= UINT16_MAX &&
           data_bytes > 0 && data_bytes <= gtx_host_stash_slot_bytes &&
           dst_bytes > 0 && dst_bytes <= gtx_host_stash_slot_bytes;
}

static bool gtx_l2_rope_shape_ok(const ggml_tensor * node) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_same_shape(node, node->src[0]) ||
        gtx_remote_tensor_type(node->src[1]) != GGML_TYPE_I32 ||
        !ggml_is_contiguous(node->src[1]) ||
        node->op_params[2] != GGML_ROPE_TYPE_NORMAL ||
        node->ne[0] <= 0 || node->ne[0] > UINT16_MAX ||
        node->ne[1] <= 0 || node->ne[1] > UINT16_MAX ||
        node->ne[2] <= 0 || node->ne[2] > UINT16_MAX ||
        node->ne[3] <= 0 || node->ne[3] > UINT16_MAX ||
        node->op_params[1] < 2 || node->op_params[1] > node->ne[0] ||
        (node->op_params[1] & 1) != 0 ||
        node->src[1]->ne[0] < node->ne[2]) {
        return false;
    }

    const uint64_t rows = (uint64_t)node->ne[1] * node->ne[2] * node->ne[3];
    const uint64_t padded_pairs = (((uint64_t)node->op_params[1] / 2u) + 15u) & ~15ull;
    const uint64_t table_bytes = (uint64_t)node->ne[2] * padded_pairs * sizeof(uint16_t);
    return (uint64_t)node->ne[0] * sizeof(uint16_t) <= UINT16_MAX &&
           (uint64_t)node->ne[2] * sizeof(int32_t) <= UINT16_MAX &&
           rows <= UINT16_MAX && table_bytes <= 64u * 1024u &&
           ggml_gtx_rope_scratch_size((uint32_t)node->ne[2],
                                       (uint32_t)node->op_params[1]) != 0u;
}

static gtx_l2_chainability gtx_l2_chainability_from_node(const ggml_tensor * node) {
    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if (node == nullptr || kernel_id == GGML_GTX_KERNEL_INVALID) {
        return { gtx_l2_chain_kind::unsupported, kernel_id, "unsupported_kernel" };
    }

    switch (kernel_id) {
        case GGML_GTX_KERNEL_ADD_F16:
            return { gtx_l2_add_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_add_shape_ok(node) ? "add_l2_native" : "add_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_MUL_F16:
            return { (gtx_l2_dense_binary_shape_ok(node) ||
                      gtx_l2_rhs_broadcast_shape_ok(node, true)) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     (gtx_l2_dense_binary_shape_ok(node) ||
                      gtx_l2_rhs_broadcast_shape_ok(node, true)) ? "mul_l2_native" : "mul_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_SCALE_F16:
            return { gtx_l2_scale_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_scale_shape_ok(node) ? "scale_l2_native" : "scale_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_GELU_F16:
        case GGML_GTX_KERNEL_ABS_F16:
        case GGML_GTX_KERNEL_NEG_F16:
        case GGML_GTX_KERNEL_SGN_F16:
        case GGML_GTX_KERNEL_CEIL_F16:
        case GGML_GTX_KERNEL_FLOOR_F16:
        case GGML_GTX_KERNEL_TANH_F16:
        case GGML_GTX_KERNEL_SILU_F16:
        case GGML_GTX_KERNEL_RELU_F16:
        case GGML_GTX_KERNEL_LEAKY_RELU_F16:
        case GGML_GTX_KERNEL_CLAMP_F16:
            return { gtx_l2_dense_unary_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_dense_unary_shape_ok(node) ? "unary_l2_native" : "unary_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_SOFT_MAX_F16:
            return { gtx_l2_softmax_basic_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_softmax_basic_shape_ok(node) ? "softmax_basic_l2_native" : "softmax_mask_or_param_boundary" };
        case GGML_GTX_KERNEL_MUL_MAT_F16:
            if (gtx_l2_mul_mat_shape_ok(node)) {
                return { gtx_l2_chain_kind::native, kernel_id, "mul_mat_l2_native" };
            }
            return { gtx_l2_mul_mat_vector_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_mul_mat_vector_shape_ok(node) ? "mul_mat_vector_l2_native" : "mul_mat_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_RMS_NORM_F16:
            return { gtx_l2_rms_norm_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_rms_norm_shape_ok(node) ? "rms_norm_l2_native" : "rms_norm_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_SWIGLU_F16:
            return { gtx_l2_swiglu_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_swiglu_shape_ok(node) ? "swiglu_l2_native" : "swiglu_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_REPEAT_F16:
            return { gtx_l2_repeat_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_repeat_shape_ok(node) ? "repeat_l2_native" : "repeat_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_CONCAT_F16:
            return { gtx_l2_concat_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_concat_shape_ok(node) ? "concat_l2_native" : "concat_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_PAD_F16:
            return { gtx_l2_pad_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_pad_shape_ok(node) ? "pad_l2_native" : "pad_shape_or_layout_boundary" };
        case GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16:
            return { gtx_l2_conv_transpose_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_conv_transpose_shape_ok(node) ? "conv_transpose_l2_native" : "conv_transpose_shape_or_budget_boundary" };
        case GGML_GTX_KERNEL_CONV_2D_F16:
        case GGML_GTX_KERNEL_CONV_2D_DW_F16:
            return { gtx_l2_conv_whole_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_conv_whole_shape_ok(node) ? "conv_l2_native" : "conv_needs_host_spatial_tile" };
        case GGML_GTX_KERNEL_CONT_F16:
        case GGML_GTX_KERNEL_CPY_F16:
            return { gtx_l2_chain_kind::boundary, kernel_id,
                     "layout_materialize_boundary" };
        case GGML_GTX_KERNEL_GET_ROWS_F16:
            return { gtx_l2_chain_kind::boundary, kernel_id,
                     "gather_boundary" };
        case GGML_GTX_KERNEL_SET_ROWS_F16:
            return { gtx_l2_chain_kind::boundary, kernel_id,
                     "scatter_alias_boundary" };
        case GGML_GTX_KERNEL_ROPE_F16:
            return { gtx_l2_rope_shape_ok(node) ? gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_rope_shape_ok(node) ? "rope_l2_native" : "rope_shape_or_mode_boundary" };
        case GGML_GTX_KERNEL_SIN_F16:
        case GGML_GTX_KERNEL_COS_F16:
            return { gtx_l2_dense_unary_shape_ok(node) && (ggml_nelements(node) % 16) == 0 ?
                         gtx_l2_chain_kind::native : gtx_l2_chain_kind::boundary,
                     kernel_id,
                     gtx_l2_dense_unary_shape_ok(node) && (ggml_nelements(node) % 16) == 0 ?
                         "sincos_l2_native" : "sincos_shape_boundary" };
        case GGML_GTX_KERNEL_NORM_F16:
        case GGML_GTX_KERNEL_ADD_F16_F32_TO_F16:
            return { gtx_l2_chain_kind::boundary, kernel_id, "coherence_boundary" };
        default:
            return { gtx_l2_chain_kind::unsupported, kernel_id, "not_chain_advertised" };
    }
}

static void gtx_log_chainability_decision(const ggml_tensor * node) {
    if (!gtx_log_chain_ops_enabled() || node == nullptr) {
        return;
    }
    const gtx_l2_chainability chain = gtx_l2_chainability_from_node(node);
    fprintf(stderr,
            "[GTX CHAIN] op=%s tensor=%s kernel_id=%u kind=%s participates=%s reason=%s shape=[%lld,%lld,%lld,%lld]\n",
            gtx_tensor_op_display_name(node),
            node->name[0] != '\0' ? node->name : "(anon)",
            (unsigned) chain.kernel_id,
            gtx_l2_chain_kind_name(chain.kind),
            gtx_l2_chain_participates(chain.kind) ? "yes" : "no",
            chain.reason,
            (long long) node->ne[0],
            (long long) node->ne[1],
            (long long) node->ne[2],
            (long long) node->ne[3]);
    if ((chain.kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
         chain.kernel_id == GGML_GTX_KERNEL_MUL_F16) &&
        node->src[0] != nullptr && node->src[1] != nullptr) {
        fprintf(stderr,
                "[GTX BINARY SHAPE] op=%s dst=[%lld,%lld,%lld,%lld] "
                "src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld]\n",
                gtx_tensor_op_display_name(node),
                (long long)node->ne[0], (long long)node->ne[1],
                (long long)node->ne[2], (long long)node->ne[3],
                (long long)node->src[0]->ne[0], (long long)node->src[0]->ne[1],
                (long long)node->src[0]->ne[2], (long long)node->src[0]->ne[3],
                (long long)node->src[1]->ne[0], (long long)node->src[1]->ne[1],
                (long long)node->src[1]->ne[2], (long long)node->src[1]->ne[3]);
    }
    if ((chain.kind == gtx_l2_chain_kind::boundary &&
         (chain.kernel_id == GGML_GTX_KERNEL_CONV_2D_F16 ||
          chain.kernel_id == GGML_GTX_KERNEL_CONV_2D_DW_F16)) ||
        chain.kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
        chain.kernel_id == GGML_GTX_KERNEL_MUL_F16) {
        const auto log_layout = [](const char * label, const ggml_tensor * tensor) {
            if (tensor == nullptr) {
                fprintf(stderr, "[GTX CHAIN]   %s=-\n", label);
                return;
            }
            fprintf(stderr,
                    "[GTX CHAIN]   %s remote_type=%s ne=[%lld,%lld,%lld,%lld] "
                    "remote_nb=[%llu,%llu,%llu,%llu]\n",
                    label, ggml_type_name(gtx_remote_tensor_type(tensor)),
                    (long long)tensor->ne[0], (long long)tensor->ne[1],
                    (long long)tensor->ne[2], (long long)tensor->ne[3],
                    (unsigned long long)gtx_remote_byte_offset(tensor, tensor->nb[0]),
                    (unsigned long long)gtx_remote_byte_offset(tensor, tensor->nb[1]),
                    (unsigned long long)gtx_remote_byte_offset(tensor, tensor->nb[2]),
                    (unsigned long long)gtx_remote_byte_offset(tensor, tensor->nb[3]));
        };
        log_layout("dst", node);
        log_layout("src0", node->src[0]);
        log_layout("src1", node->src[1]);
    }
}

static void gtx_log_graph_chainability(const ggml_cgraph * cgraph) {
    if (!gtx_log_chain_ops_enabled() || cgraph == nullptr) {
        return;
    }
    fprintf(stderr, "[GTX CHAIN] n_nodes=%d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        gtx_log_chainability_decision(cgraph->nodes[i]);
    }
}

static bool gtx_build_kernel_launch_desc(const ggml_tensor * node,
                                         ggml_gtx_kernel_launch_desc_t & desc) {
    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if (kernel_id == GGML_GTX_KERNEL_INVALID) {
        if (node != nullptr) {
            GTX_LOG_ERROR("unsupported descriptor op: %s (%d)\n",
                          gtx_tensor_op_display_name(node), (int) node->op);
        }
        return false;
    }

    desc = {};
    desc.kernel_id = kernel_id;
    desc.abi_version = GGML_GTX_GRAPH_V2_VERSION;
    // A normal, untiled launch is an explicit coherent DDR boundary.  The
    // scheduler replaces this with L2_REQUIRED only after it has lowered a
    // complete contiguous raw-view tile.
    desc.flags = GGML_GTX_DESC_FLAG_DDR_ALLOWED;

    if (!gtx_fill_tensor_desc(node, &desc.dst_addr, &desc.dtype, desc.ne, desc.nb)) {
        GTX_LOG_ERROR("Failed to describe destination tensor for op: %s\n",
                      gtx_tensor_op_display_name(node));
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        if (!gtx_fill_tensor_desc(node->src[i],
                                  &desc.src_addr[i],
                                  &desc.src_type[i],
                                  desc.src_ne[i],
                                  desc.src_nb[i])) {
            GTX_LOG_ERROR("Failed to describe source %d for op: %s\n",
                          i, gtx_tensor_op_display_name(node));
            return false;
        }
    }

    const int n_params = GGML_MAX_OP_PARAMS / sizeof(int32_t);
    for (int i = 0; i < n_params && i < 16; ++i) {
        desc.op_params[i] = node->op_params[i];
    }

    desc.l2_scratch_size = gtx_kernel_l2_scratch_size(node);

    return desc.dst_addr != 0;
}

static bool gtx_build_launch_payload(uint32_t device,
                                     const ggml_gtx_kernel_launch_desc_t * descs,
                                     size_t n_desc,
                                     std::vector<uint8_t> & output) {
    if (descs == nullptr || n_desc == 0 || n_desc > UINT32_MAX) {
        return false;
    }

    ggml_gtx_graph_v2_header_t hdr = {};
    hdr.magic = GGML_GTX_GRAPH_V2_MAGIC;
    hdr.version = GGML_GTX_GRAPH_V2_VERSION;
    hdr.header_size = sizeof(hdr);
    hdr.device = device;
    hdr.n_desc = (uint32_t) n_desc;
    hdr.desc_size = sizeof(ggml_gtx_kernel_launch_desc_t);

    output.clear();
    output.reserve(sizeof(hdr) + n_desc * sizeof(ggml_gtx_kernel_launch_desc_t));
    const uint8_t * hdr_bytes = reinterpret_cast<const uint8_t *>(&hdr);
    output.insert(output.end(), hdr_bytes, hdr_bytes + sizeof(hdr));
    const uint8_t * desc_bytes = reinterpret_cast<const uint8_t *>(descs);
    output.insert(output.end(), desc_bytes, desc_bytes + n_desc * sizeof(ggml_gtx_kernel_launch_desc_t));
    return true;
}

static bool gtx_build_single_launch_payload(uint32_t device,
                                            const ggml_gtx_kernel_launch_desc_t & desc,
                                            std::vector<uint8_t> & output) {
    return gtx_build_launch_payload(device, &desc, 1, output);
}

static bool gtx_submit_launch_desc(ggml_backend_gtx_context * ctx,
                                   const ggml_gtx_kernel_launch_desc_t & desc) {
    std::vector<uint8_t> graph_blob;
    if (!gtx_build_single_launch_payload((uint32_t) ctx->device_id, desc, graph_blob)) {
        return false;
    }

    if (!ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, graph_blob.data(), graph_blob.size()) ||
        !ctx->conn->send_cmd_staged(GTX_CMD_GRAPH_COMPUTE,
                                    GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                    (uint32_t) graph_blob.size(), nullptr, 0)) {
        return false;
    }

    gtx_pcie_irq_ctx * irq_ctx = ctx->get_irq_ctx();
    return irq_ctx == nullptr || gtx_wait_infer_done(irq_ctx, GTX_GRAPH_COMPUTE_TIMEOUT_MS) >= 0;
}

static bool gtx_submit_launch_descs(ggml_backend_gtx_context * ctx,
                                    const std::vector<ggml_gtx_kernel_launch_desc_t> & descs) {
    if (ctx == nullptr || descs.empty()) {
        return false;
    }

    if (ctx->use_uart) {
        for (size_t i = 0; i < descs.size(); ++i) {
            const ggml_gtx_kernel_launch_desc_t & desc = descs[i];
            if (!gtx_uart_launch_desc(*ctx->uart, desc)) {
                GTX_LOG_ERROR("descriptor launch failed: index=%zu/%zu kernel=%u flags=0x%x "
                              "dst=[%u,%u,%u,%u] src0=[%u,%u,%u,%u] "
                              "src1=[%u,%u,%u,%u] scratch=%u\n",
                              i, descs.size(), desc.kernel_id, desc.flags,
                              desc.ne[0], desc.ne[1], desc.ne[2], desc.ne[3],
                              desc.src_ne[0][0], desc.src_ne[0][1],
                              desc.src_ne[0][2], desc.src_ne[0][3],
                              desc.src_ne[1][0], desc.src_ne[1][1],
                              desc.src_ne[1][2], desc.src_ne[1][3],
                              desc.l2_scratch_size);
                return false;
            }
        }
        return true;
    }

    std::vector<uint8_t> graph_blob;
    if (!gtx_build_launch_payload((uint32_t) ctx->device_id, descs.data(), descs.size(), graph_blob)) {
        return false;
    }

    if (!ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, graph_blob.data(), graph_blob.size()) ||
        !ctx->conn->send_cmd_staged(GTX_CMD_GRAPH_COMPUTE,
                                    GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                    (uint32_t) graph_blob.size(), nullptr, 0)) {
        return false;
    }

    gtx_pcie_irq_ctx * irq_ctx = ctx->get_irq_ctx();
    return irq_ctx == nullptr || gtx_wait_infer_done(irq_ctx, GTX_GRAPH_COMPUTE_TIMEOUT_MS) >= 0;
}

static bool gtx_copy_tensor_rpc(gtx_pcie_connection * conn,
                                const ggml_tensor * src,
                                ggml_tensor * dst) {
    if (conn == nullptr || !conn->is_open() || src == nullptr || dst == nullptr) {
        return false;
    }

    gtx_msg_copy_tensor_req req = {};
    req.src = gtx_serialize_tensor(src);
    req.dst = gtx_serialize_tensor(dst);

    if (req.src.data == 0 || req.dst.data == 0) {
        GTX_LOG_ERROR("COPY_TENSOR requires GTX-backed src and dst tensors\n");
        return false;
    }

    gtx_msg_copy_tensor_rsp rsp = {};
    if (!conn->send_data(GTX_BAR2_STAGING_OFFSET, &req, sizeof(req)) ||
        !conn->send_cmd_staged(GTX_CMD_COPY_TENSOR,
                               GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                               (uint32_t) sizeof(req), &rsp, sizeof(rsp))) {
        GTX_LOG_ERROR("COPY_TENSOR command failed\n");
        return false;
    }

    return rsp.result != 0;
}

static bool gtx_is_metadata_op(enum ggml_op op) {
    return op == GGML_OP_NONE ||
           op == GGML_OP_RESHAPE ||
           op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

static void gtx_format_shape(const ggml_tensor * tensor, char * buf, size_t buf_size) {
    if (tensor == nullptr) {
        snprintf(buf, buf_size, "-");
        return;
    }

    snprintf(buf, buf_size, "[%lld,%lld,%lld,%lld]",
             (long long) tensor->ne[0],
             (long long) tensor->ne[1],
             (long long) tensor->ne[2],
             (long long) tensor->ne[3]);
}

static void gtx_format_tensor_detail(const ggml_tensor * tensor, char * buf, size_t buf_size) {
    if (tensor == nullptr) {
        snprintf(buf, buf_size, "-");
        return;
    }

    snprintf(buf, buf_size,
             "%s type=%s->%s shape=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] addr=0x%llx",
             tensor->name[0] != '\0' ? tensor->name : "(anon)",
             ggml_type_name(tensor->type),
             ggml_type_name(gtx_remote_tensor_type(tensor)),
             (long long) tensor->ne[0],
             (long long) tensor->ne[1],
             (long long) tensor->ne[2],
             (long long) tensor->ne[3],
             (size_t) tensor->nb[0],
             (size_t) tensor->nb[1],
             (size_t) tensor->nb[2],
             (size_t) tensor->nb[3],
             (unsigned long long) gtx_remote_data_ptr(tensor));
}

static void gtx_log_op_shapes(const ggml_tensor * node) {
    if (!gtx_log_ops_enabled()) {
        return;
    }

    char dst[256];
    char src0[256];
    char src1[256];
    char src2[256];

    gtx_format_tensor_detail(node, dst, sizeof(dst));
    gtx_format_tensor_detail(node != nullptr ? node->src[0] : nullptr, src0, sizeof(src0));
    gtx_format_tensor_detail(node != nullptr ? node->src[1] : nullptr, src1, sizeof(src1));
    gtx_format_tensor_detail(node != nullptr ? node->src[2] : nullptr, src2, sizeof(src2));

    if (node != nullptr &&
        gtx_normalize_host_op(node) == (uint32_t) GGML_OP_UNARY) {
        const int32_t unary_op = node->op_params[0];
        const char * unary_name =
            unary_op >= 0 && unary_op < GGML_UNARY_OP_COUNT ?
            ggml_unary_op_name((ggml_unary_op) unary_op) : "unknown";
        fprintf(stderr,
                "[GTX OP] %s unary=%s(%d)\n  dst=%s\n  src0=%s\n  src1=%s\n  src2=%s\n",
                gtx_tensor_op_display_name(node),
                unary_name, unary_op,
                dst, src0, src1, src2);
        return;
    }

    fprintf(stderr, "[GTX OP] %s\n  dst=%s\n  src0=%s\n  src1=%s\n  src2=%s\n",
            gtx_tensor_op_display_name(node),
            dst, src0, src1, src2);
}

static void gtx_dump_graph_node_types(const ggml_cgraph * cgraph) {
    if (!gtx_dump_graph_types_enabled() || cgraph == nullptr) {
        return;
    }

    const char * path = getenv("GTX_GRAPH_TYPES_FILE");
    FILE * out = stderr;
    if (path != nullptr && path[0] != '\0') {
        out = fopen(path, "a");
        if (out == nullptr) {
            out = stderr;
            fprintf(stderr, "[GTX WARN] failed to open GTX_GRAPH_TYPES_FILE=%s\n", path);
        }
    }

    fprintf(out, "[GTX GRAPH] n_nodes=%d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            fprintf(out, "[GTX GRAPH] node=%d op=NULL dst=-\n", i);
            continue;
        }

        fprintf(out,
                "[GTX GRAPH] node=%d op=%s dst=%s:%s->%s",
                i,
                gtx_tensor_op_display_name(node),
                node->name[0] != '\0' ? node->name : "(anon)",
                ggml_type_name(node->type),
                ggml_type_name(gtx_remote_tensor_type(node)));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            fprintf(out,
                    " src%d=%s",
                    j,
                    src != nullptr ? (src->name[0] != '\0' ? src->name : "(anon)") : "-");
            if (src != nullptr) {
                fprintf(out, ":%s->%s",
                        ggml_type_name(src->type),
                        ggml_type_name(gtx_remote_tensor_type(src)));
            }
        }
        fprintf(out, "\n");
    }

    if (out != stderr) {
        fclose(out);
    }
}

static void gtx_log_supports_op_decision(const ggml_tensor * node, bool supported, const char * reason) {
    if (!gtx_log_supports_ops_enabled() || node == nullptr) {
        return;
    }

    char dst[256];
    char src0[256];
    char src1[256];
    char src2[256];

    gtx_format_tensor_detail(node, dst, sizeof(dst));
    gtx_format_tensor_detail(node->src[0], src0, sizeof(src0));
    gtx_format_tensor_detail(node->src[1], src1, sizeof(src1));
    gtx_format_tensor_detail(node->src[2], src2, sizeof(src2));

    const uint32_t raw_op = (uint32_t) node->op;
    const uint32_t normalized_op = gtx_normalize_host_op(node);
    const char * display_op = gtx_tensor_op_display_name(node);

    fprintf(stderr,
            "[GTX SUPPORT] %s op=%s raw_op=%u(%s) normalized_op=%u(%s) reason=%s\n"
            "  dst=%s\n"
            "  src0=%s\n"
            "  src1=%s\n"
            "  src2=%s\n",
            supported ? "yes" : "no",
            display_op,
            raw_op, gtx_op_name_or_unknown(raw_op),
            normalized_op, gtx_normalized_op_name(normalized_op),
            reason != nullptr ? reason : "-",
            dst, src0, src1, src2);
}

static bool gtx_download_tensor_prefix_as_f32(ggml_backend_gtx_context * ctx,
                                             const ggml_tensor * tensor,
                                             size_t max_elems,
                                             std::vector<float> & out) {
    out.clear();
    if (ctx == nullptr || tensor == nullptr || max_elems == 0) {
        return false;
    }

    const size_t elems = (size_t) std::min<int64_t>((int64_t) max_elems, ggml_nelements(tensor));
    if (elems == 0) {
        return false;
    }

    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(elems);
        const size_t local_bytes = elems * sizeof(float);
        const size_t remote_bytes = (size_t) gtx_remote_byte_offset(tensor, local_bytes);
        std::vector<uint8_t> packed(remote_bytes);
        const uint64_t remote = gtx_remote_data_ptr(tensor);

        bool ok = false;
        if (ctx->use_uart) {
            ok = ctx->uart != nullptr &&
                 ctx->uart->download_bytes(remote, packed.data(), packed.size());
        } else if (ctx->conn != nullptr && ctx->conn->is_open()) {
            gtx_msg_get_tensor_req req = {};
            req.tensor = gtx_serialize_tensor(tensor);
            req.offset = 0;
            req.size = (uint32_t) packed.size();
            ok = ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, &req, sizeof(req)) &&
                 ctx->conn->send_cmd_staged(GTX_CMD_GET_TENSOR,
                                            GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                            (uint32_t) sizeof(req), nullptr, 0) &&
                 ctx->conn->recv_data(GTX_BAR2_STAGING_OFFSET, packed.data(), packed.size());
        }
        if (!ok) {
            return false;
        }

        gtx_unpack_tensor_chunk(tensor, packed.data(), tmp.data(), local_bytes);
        out.swap(tmp);
        return true;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(elems);
        const size_t bytes = elems * sizeof(ggml_fp16_t);
        const uint64_t remote = gtx_remote_data_ptr(tensor);

        bool ok = false;
        if (ctx->use_uart) {
            ok = ctx->uart != nullptr &&
                 ctx->uart->download_bytes(remote, tmp.data(), bytes);
        } else if (ctx->conn != nullptr && ctx->conn->is_open()) {
            gtx_msg_get_tensor_req req = {};
            req.tensor = gtx_serialize_tensor(tensor);
            req.offset = 0;
            req.size = (uint32_t) bytes;
            ok = ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, &req, sizeof(req)) &&
                 ctx->conn->send_cmd_staged(GTX_CMD_GET_TENSOR,
                                            GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                            (uint32_t) sizeof(req), nullptr, 0) &&
                 ctx->conn->recv_data(GTX_BAR2_STAGING_OFFSET, tmp.data(), bytes);
        }
        if (!ok) {
            return false;
        }

        out.resize(elems);
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t) elems);
        return true;
    }

    return false;
}

static void gtx_log_softmax_tensor_stats(ggml_backend_gtx_context * ctx,
                                         const ggml_tensor * tensor,
                                         const char * tag) {
    if (!gtx_log_softmax_enabled() || tensor == nullptr) {
        return;
    }

    const size_t row = tensor->ne[0] > 0 ? (size_t) tensor->ne[0] : 0;
    if (row == 0) {
        return;
    }

    const uint64_t env_limit = gtx_env_u64_or_default("GTX_LOG_SOFTMAX_ELEMS", 65536);
    const size_t elems = (size_t) std::min<uint64_t>((uint64_t) ggml_nelements(tensor),
                                                    std::max<uint64_t>(row, env_limit));
    std::vector<float> values;
    if (!gtx_download_tensor_prefix_as_f32(ctx, tensor, elems, values) || values.empty()) {
        fprintf(stderr, "[GTX SOFTMAX] %s tensor=%s download failed\n",
                tag, tensor->name[0] != '\0' ? tensor->name : "(anon)");
        return;
    }

    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t neg_count = 0;
    size_t gt1_count = 0;
    for (float v : values) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count++;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        if (v < 0.0f) {
            neg_count++;
        }
        if (v > 1.0f) {
            gt1_count++;
        }
    }

    const size_t row_elems = std::min(row, values.size());
    double row_sum = 0.0;
    float row_min = std::numeric_limits<float>::infinity();
    float row_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < row_elems; ++i) {
        const float v = values[i];
        if (std::isfinite(v)) {
            row_sum += (double) v;
            row_min = std::min(row_min, v);
            row_max = std::max(row_max, v);
        }
    }

    const size_t head = std::min<size_t>(8, values.size());
    fprintf(stderr,
            "[GTX SOFTMAX] %s tensor=%s graph_type=%s remote_type=%s shape=[%lld,%lld,%lld,%lld] elems=%zu/%lld row0_sum=%.9g row0_min=%.9g row0_max=%.9g all_min=%.9g all_max=%.9g nan=%zu inf=%zu neg=%zu gt1=%zu head=[",
            tag,
            tensor->name[0] != '\0' ? tensor->name : "(anon)",
            ggml_type_name(tensor->type),
            ggml_type_name(gtx_remote_tensor_type(tensor)),
            (long long) tensor->ne[0],
            (long long) tensor->ne[1],
            (long long) tensor->ne[2],
            (long long) tensor->ne[3],
            values.size(),
            (long long) ggml_nelements(tensor),
            row_sum,
            row_min,
            row_max,
            min_v,
            max_v,
            nan_count,
            inf_count,
            neg_count,
            gt1_count);
    for (size_t i = 0; i < head; ++i) {
        fprintf(stderr, "%s%.9g", i == 0 ? "" : " ", values[i]);
    }
    fprintf(stderr, "]\n");
}

static bool gtx_dump_op_sample_should_dump(const ggml_tensor * node) {
    if (!gtx_dump_op_samples_enabled() || node == nullptr) {
        return false;
    }

    const uint32_t normalized_op = gtx_normalize_host_op(node);
    const char * op_name = gtx_normalized_op_name(normalized_op);
    const char * value = getenv("GTX_DUMP_OP_SAMPLES");
    const bool all_default = value != nullptr &&
                             (strcmp(value, "1") == 0 ||
                              strcmp(value, "true") == 0 ||
                              strcmp(value, "TRUE") == 0 ||
                              strcmp(value, "on") == 0 ||
                              strcmp(value, "ON") == 0 ||
                              strcmp(value, "all") == 0 ||
                              strcmp(value, "ALL") == 0);
    const bool op_selected = all_default ?
                             (normalized_op == (uint32_t) GGML_OP_MUL ||
                              normalized_op == (uint32_t) GGML_OP_MUL_MAT) :
                             gtx_env_list_contains("GTX_DUMP_OP_SAMPLES", op_name);
    if (!op_selected) {
        return false;
    }

    const char * filter = getenv("GTX_DUMP_OP_SAMPLE_FILTER");
    if (filter != nullptr && filter[0] != '\0') {
        const char * name = node->name[0] != '\0' ? node->name : "";
        bool matched = strstr(name, filter) != nullptr;
        for (int i = 0; i < GGML_MAX_SRC && !matched; ++i) {
            const ggml_tensor * src = node->src[i];
            const char * src_name = src != nullptr && src->name[0] != '\0' ? src->name : "";
            matched = strstr(src_name, filter) != nullptr;
        }
        if (!matched) {
            return false;
        }
    }

    return true;
}

static void gtx_dump_op_sample_tensor(ggml_backend_gtx_context * ctx,
                                      const ggml_tensor * tensor,
                                      const char * tag) {
    if (ctx == nullptr || tensor == nullptr) {
        return;
    }

    const uint64_t max_elems_env = gtx_env_u64_or_default("GTX_DUMP_OP_SAMPLE_ELEMS", 64);
    const size_t max_elems = max_elems_env == 0 ?
                             (size_t) ggml_nelements(tensor) :
                             (size_t) std::min<uint64_t>(max_elems_env, (uint64_t) ggml_nelements(tensor));
    std::vector<float> values;
    if (!gtx_download_tensor_prefix_as_f32(ctx, tensor, max_elems, values) || values.empty()) {
        fprintf(stderr, "[GTX SAMPLE] %s tensor=%s download failed\n",
                tag != nullptr ? tag : "-",
                tensor->name[0] != '\0' ? tensor->name : "(anon)");
        return;
    }

    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (float v : values) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count++;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += (double) v;
        finite_count++;
    }
    const double mean = finite_count > 0 ? sum / (double) finite_count : 0.0;
    const size_t top_k = std::min<size_t>((size_t) gtx_env_u64_or_default("GTX_DUMP_OP_SAMPLE_TOPK", 8),
                                          values.size());
    std::vector<size_t> top_indices(values.size());
    for (size_t i = 0; i < top_indices.size(); ++i) {
        top_indices[i] = i;
    }
    if (top_k > 0) {
        std::partial_sort(top_indices.begin(),
                          top_indices.begin() + top_k,
                          top_indices.end(),
                          [&values](size_t a, size_t b) {
                              const float va = values[a];
                              const float vb = values[b];
                              if (std::isnan(va)) {
                                  return false;
                              }
                              if (std::isnan(vb)) {
                                  return true;
                              }
                              return va > vb;
                          });
    }

    FILE * out = stderr;
    const char * path = getenv("GTX_DUMP_OP_SAMPLE_FILE");
    if (path != nullptr && path[0] != '\0') {
        out = fopen(path, "a");
        if (out == nullptr) {
            out = stderr;
            fprintf(stderr, "[GTX WARN] failed to open GTX_DUMP_OP_SAMPLE_FILE=%s\n", path);
        }
    }

    const size_t head = std::min<size_t>(16, values.size());
    fprintf(out,
            "[GTX SAMPLE] %s op=%s tensor=%s graph_type=%s remote_type=%s shape=[%lld,%lld,%lld,%lld] elems=%zu/%lld finite=%zu nan=%zu inf=%zu min=%.9g max=%.9g mean=%.9g head=[",
            tag != nullptr ? tag : "-",
            ggml_op_name(tensor->op),
            tensor->name[0] != '\0' ? tensor->name : "(anon)",
            ggml_type_name(tensor->type),
            ggml_type_name(gtx_remote_tensor_type(tensor)),
            (long long) tensor->ne[0],
            (long long) tensor->ne[1],
            (long long) tensor->ne[2],
            (long long) tensor->ne[3],
            values.size(),
            (long long) ggml_nelements(tensor),
            finite_count,
            nan_count,
            inf_count,
            finite_count > 0 ? min_v : 0.0f,
            finite_count > 0 ? max_v : 0.0f,
            mean);
    for (size_t i = 0; i < head; ++i) {
        fprintf(out, "%s%.9g", i == 0 ? "" : " ", values[i]);
    }
    fprintf(out, "] top=[");
    for (size_t i = 0; i < top_k; ++i) {
        const size_t idx = top_indices[i];
        fprintf(out, "%s%zu:%.9g", i == 0 ? "" : " ", idx, values[idx]);
    }
    fprintf(out, "]\n");

    if (out != stderr) {
        fclose(out);
    }
}

static bool gtx_tensor_is_remote_compact_f16(const ggml_tensor * tensor) {
    if (tensor == nullptr || gtx_remote_tensor_type(tensor) != GGML_TYPE_F16) {
        return false;
    }

    const gtx_rpc_tensor rpc = gtx_serialize_tensor(tensor);
    uint64_t stride = sizeof(uint16_t);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (rpc.nb[i] != stride) {
            return false;
        }
        stride *= (uint64_t) std::max<int64_t>(tensor->ne[i], 1);
    }
    return true;
}

static bool gtx_download_compact_f16_tensor_as_f32(ggml_backend_gtx_context * ctx,
                                                   const ggml_tensor * tensor,
                                                   std::vector<float> & out) {
    out.clear();
    if (ctx == nullptr || tensor == nullptr || !gtx_tensor_is_remote_compact_f16(tensor)) {
        return false;
    }

    const size_t elems = (size_t) ggml_nelements(tensor);
    if (elems == 0) {
        return false;
    }

    std::vector<ggml_fp16_t> tmp(elems);
    const uint64_t remote = gtx_remote_data_ptr(tensor);
    const size_t bytes = tmp.size() * sizeof(ggml_fp16_t);
    bool ok = false;
    if (ctx->use_uart) {
        ok = ctx->uart != nullptr && ctx->uart->download_bytes(remote, tmp.data(), bytes);
    } else if (ctx->conn != nullptr && ctx->conn->is_open()) {
        gtx_msg_get_tensor_req req = {};
        req.tensor = gtx_serialize_tensor(tensor);
        req.offset = 0;
        req.size = (uint32_t) bytes;
        ok = ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, &req, sizeof(req)) &&
             ctx->conn->send_cmd_staged(GTX_CMD_GET_TENSOR,
                                        GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                        (uint32_t) sizeof(req), nullptr, 0) &&
             ctx->conn->recv_data(GTX_BAR2_STAGING_OFFSET, tmp.data(), bytes);
    }
    if (!ok) {
        return false;
    }

    out.resize(elems);
    ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t) elems);
    return true;
}

static bool gtx_download_f16_bytes_as_f32(ggml_backend_gtx_context * ctx,
                                          uint64_t remote,
                                          size_t elems,
                                          float * out) {
    if (ctx == nullptr || remote == 0 || elems == 0 || out == nullptr) {
        return false;
    }

    std::vector<ggml_fp16_t> tmp(elems);
    const size_t bytes = tmp.size() * sizeof(ggml_fp16_t);
    bool ok = false;
    if (ctx->use_uart) {
        ok = ctx->uart != nullptr && ctx->uart->download_bytes(remote, tmp.data(), bytes);
    } else if (ctx->conn != nullptr && ctx->conn->is_open()) {
        ok = ctx->conn->recv_data(remote, tmp.data(), bytes);
    }
    if (!ok) {
        return false;
    }

    ggml_fp16_to_fp32_row(tmp.data(), out, (int64_t) elems);
    return true;
}

static bool gtx_download_tensor_strided_f16_as_f32(ggml_backend_gtx_context * ctx,
                                                   const ggml_tensor * tensor,
                                                   std::vector<float> & out) {
    out.clear();
    if (ctx == nullptr ||
        tensor == nullptr ||
        gtx_remote_tensor_type(tensor) != GGML_TYPE_F16 ||
        tensor->ne[0] <= 0 ||
        tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0 ||
        tensor->ne[3] <= 0) {
        return false;
    }

    const uint64_t max_elems = gtx_cpu_ref_enabled() ?
        gtx_env_u64_or_default("GTX_CPU_REF_MAX_ELEMS", 32ull * 1024ull * 1024ull) :
        gtx_env_u64_or_default("GTX_CHECK_MUL_MAT_MAX_ELEMS", 4ull * 1024ull * 1024ull);
    const uint64_t elems64 = (uint64_t) ggml_nelements(tensor);
    if (max_elems != 0 && elems64 > max_elems) {
        return false;
    }

    const uint64_t base = gtx_remote_data_ptr(tensor);
    const gtx_rpc_tensor rpc = gtx_serialize_tensor(tensor);
    out.resize((size_t) elems64);

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                const uint64_t remote = base +
                    (uint64_t) i3 * rpc.nb[3] +
                    (uint64_t) i2 * rpc.nb[2] +
                    (uint64_t) i1 * rpc.nb[1];
                const size_t dst_off =
                    (((size_t) i3 * (size_t) tensor->ne[2] + (size_t) i2) *
                     (size_t) tensor->ne[1] + (size_t) i1) *
                    (size_t) tensor->ne[0];
                if (!gtx_download_f16_bytes_as_f32(ctx, remote, (size_t) tensor->ne[0], out.data() + dst_off)) {
                    out.clear();
                    return false;
                }
            }
        }
    }
    return true;
}

static inline float gtx_tensor4_value(const std::vector<float> & data,
                                      const ggml_tensor * tensor,
                                      int64_t i0,
                                      int64_t i1,
                                      int64_t i2,
                                      int64_t i3) {
    const size_t idx =
        (((size_t) i3 * (size_t) tensor->ne[2] + (size_t) i2) *
         (size_t) tensor->ne[1] + (size_t) i1) *
        (size_t) tensor->ne[0] + (size_t) i0;
    return data[idx];
}

static void gtx_check_mul_mat_after_launch(ggml_backend_gtx_context * ctx,
                                           const ggml_tensor * node) {
    if (!gtx_check_mul_mat_enabled() ||
        ctx == nullptr ||
        node == nullptr ||
        gtx_normalize_host_op(node) != (uint32_t) GGML_OP_MUL_MAT ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr) {
        return;
    }

    const char * filter = getenv("GTX_CHECK_MUL_MAT_FILTER");
    if (filter != nullptr && filter[0] != '\0') {
        const char * node_name = node->name[0] != '\0' ? node->name : "";
        const char * src0_name = node->src[0]->name[0] != '\0' ? node->src[0]->name : "";
        const char * src1_name = node->src[1]->name[0] != '\0' ? node->src[1]->name : "";
        if (strstr(node_name, filter) == nullptr &&
            strstr(src0_name, filter) == nullptr &&
            strstr(src1_name, filter) == nullptr) {
            return;
        }
    }

    static uint64_t checked_nodes = 0;
    const uint64_t max_nodes = gtx_env_u64_or_default("GTX_CHECK_MUL_MAT_MAX_NODES", 256);
    if (max_nodes != 0 && checked_nodes >= max_nodes) {
        return;
    }
    checked_nodes++;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t batch2 = node->ne[2];
    const int64_t batch3 = node->ne[3];
    if (k <= 0 || rows <= 0 || cols <= 0 ||
        batch2 <= 0 || batch3 <= 0 ||
        src1->ne[0] != k ||
        node->ne[0] != rows ||
        node->ne[1] != cols ||
        src1->ne[2] != batch2 || src1->ne[3] != batch3 ||
        src0->ne[2] <= 0 || src0->ne[3] <= 0 ||
        (batch2 % src0->ne[2]) != 0 ||
        (batch3 % src0->ne[3]) != 0) {
        fprintf(stderr,
                "[GTX MUL_MAT CHECK] skip tensor=%s reason=unsupported-shape dst=[%lld,%lld,%lld,%lld] src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld]\n",
                node->name[0] != '\0' ? node->name : "(anon)",
                (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3],
                (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
                (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3]);
        return;
    }

    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> dst;
    if (!gtx_download_tensor_strided_f16_as_f32(ctx, src0, a) ||
        !gtx_download_tensor_strided_f16_as_f32(ctx, src1, b) ||
        !gtx_download_tensor_strided_f16_as_f32(ctx, node, dst)) {
        fprintf(stderr,
                "[GTX MUL_MAT CHECK] skip tensor=%s reason=download-failed-or-too-large\n",
                node->name[0] != '\0' ? node->name : "(anon)");
        return;
    }

    const int64_t r2 = batch2 / src0->ne[2];
    const int64_t r3 = batch3 / src0->ne[3];
    const int64_t max_rows = (int64_t) gtx_env_u64_or_default("GTX_CHECK_MUL_MAT_ROWS", 128);
    const int64_t row_limit = max_rows <= 0 ? rows : std::min<int64_t>(rows, max_rows);
    const int64_t col_limit = cols;
    float max_diff_f32 = 0.0f;
    float max_diff_f16 = 0.0f;
    int64_t max_row = 0;
    int64_t max_col = 0;
    float max_ref = 0.0f;
    float max_ref_f16 = 0.0f;
    float max_got = 0.0f;

    for (int64_t i3 = 0; i3 < batch3; ++i3) {
        for (int64_t i2 = 0; i2 < batch2; ++i2) {
            const int64_t src0_i2 = i2 / r2;
            const int64_t src0_i3 = i3 / r3;
            for (int64_t s = 0; s < row_limit; ++s) {
                const int64_t r = row_limit == rows || row_limit <= 1 ?
                                  s :
                                  (s * (rows - 1)) / (row_limit - 1);
                for (int64_t c = 0; c < col_limit; ++c) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < k; ++i) {
                        acc += gtx_tensor4_value(a, src0, i, r, src0_i2, src0_i3) *
                               gtx_tensor4_value(b, src1, i, c, i2, i3);
                    }
                    const float acc_f16 = gtx_fp16_to_fp32(gtx_fp32_to_fp16(acc));
                    const float got = gtx_tensor4_value(dst, node, r, c, i2, i3);
                    const float diff_f32 = fabsf(got - acc);
                    const float diff_f16 = fabsf(got - acc_f16);
                    if (diff_f16 > max_diff_f16) {
                        max_diff_f16 = diff_f16;
                        max_diff_f32 = diff_f32;
                        max_row = r;
                        max_col = c;
                        max_ref = acc;
                        max_ref_f16 = acc_f16;
                        max_got = got;
                    }
                }
            }
        }
    }

    fprintf(stderr,
            "[GTX MUL_MAT CHECK] tensor=%s src0=%s src1=%s dst=[%lld,%lld,%lld,%lld] k=%lld checked_rows=%lld cols=%lld max_diff_f16=%.9g max_diff_f32=%.9g at=[%lld,%lld] ref=%.9g ref_f16=%.9g got=%.9g\n",
            node->name[0] != '\0' ? node->name : "(anon)",
            src0->name[0] != '\0' ? src0->name : "(anon)",
            src1->name[0] != '\0' ? src1->name : "(anon)",
            (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3],
            (long long) k,
            (long long) row_limit,
            (long long) col_limit,
            max_diff_f16,
            max_diff_f32,
            (long long) max_row,
            (long long) max_col,
            max_ref,
            max_ref_f16,
            max_got);
}

static bool gtx_check_ops_should_check(const ggml_tensor * node) {
    if (!gtx_check_ops_enabled() || node == nullptr) {
        return false;
    }

    const uint32_t normalized_op = gtx_normalize_host_op(node);
    const char * op_name = gtx_normalized_op_name(normalized_op);
    const char * value = getenv("GTX_CHECK_OPS");
    const bool all_default = value != nullptr &&
                             (strcmp(value, "1") == 0 ||
                              strcmp(value, "true") == 0 ||
                              strcmp(value, "TRUE") == 0 ||
                              strcmp(value, "on") == 0 ||
                              strcmp(value, "ON") == 0 ||
                              strcmp(value, "all") == 0 ||
                              strcmp(value, "ALL") == 0);
    const bool supported = normalized_op == (uint32_t) GGML_OP_NORM ||
                           normalized_op == (uint32_t) GGML_OP_SOFT_MAX;
    return supported && (all_default || gtx_env_list_contains("GTX_CHECK_OPS", op_name));
}

static bool gtx_check_ops_download(ggml_backend_gtx_context * ctx,
                                   const ggml_tensor * tensor,
                                   std::vector<float> & out,
                                   const char * tag,
                                   const char * node_name) {
    const uint64_t max_elems = gtx_env_u64_or_default("GTX_CHECK_OP_MAX_ELEMS", 4ull * 1024ull * 1024ull);
    const uint64_t elems = tensor != nullptr ? (uint64_t) ggml_nelements(tensor) : 0;
    if (max_elems != 0 && elems > max_elems) {
        fprintf(stderr,
                "[GTX OP CHECK] skip tensor=%s reason=%s-too-large elems=%llu max=%llu\n",
                node_name != nullptr ? node_name : "(anon)",
                tag != nullptr ? tag : "tensor",
                (unsigned long long) elems,
                (unsigned long long) max_elems);
        return false;
    }
    return gtx_download_tensor_strided_f16_as_f32(ctx, tensor, out);
}

struct gtx_op_check_pre_state {
    bool valid = false;
    uint32_t normalized_op = 0;
    std::vector<float> src0;
    std::vector<float> src1;
};

static gtx_op_check_pre_state gtx_check_ops_before_launch(ggml_backend_gtx_context * ctx,
                                                          const ggml_tensor * node) {
    gtx_op_check_pre_state pre;
    if (!gtx_check_ops_should_check(node) || node == nullptr || node->src[0] == nullptr) {
        return pre;
    }

    pre.normalized_op = gtx_normalize_host_op(node);
    const char * name = node->name[0] != '\0' ? node->name : "(anon)";
    if (!gtx_check_ops_download(ctx, node->src[0], pre.src0, "src0", name)) {
        fprintf(stderr, "[GTX OP CHECK] skip %s tensor=%s reason=pre-src0-download-failed-or-too-large\n",
                gtx_normalized_op_name(pre.normalized_op), name);
        pre.src0.clear();
        return pre;
    }
    if (pre.normalized_op == (uint32_t) GGML_OP_SOFT_MAX && node->src[1] != nullptr &&
        !gtx_check_ops_download(ctx, node->src[1], pre.src1, "mask", name)) {
        fprintf(stderr, "[GTX OP CHECK] skip SOFT_MAX tensor=%s reason=pre-mask-download-failed-or-too-large\n",
                name);
        pre.src0.clear();
        pre.src1.clear();
        return pre;
    }
    pre.valid = true;
    return pre;
}

static void gtx_check_norm_after_launch(ggml_backend_gtx_context * ctx,
                                        const ggml_tensor * node,
                                        const gtx_op_check_pre_state & pre) {
    if (node == nullptr || node->src[0] == nullptr) {
        return;
    }

    std::vector<float> dst;
    const char * name = node->name[0] != '\0' ? node->name : "(anon)";
    if (!pre.valid ||
        pre.normalized_op != (uint32_t) GGML_OP_NORM ||
        !gtx_check_ops_download(ctx, node, dst, "dst", name)) {
        fprintf(stderr, "[GTX OP CHECK] skip NORM tensor=%s reason=download-failed-or-too-large\n", name);
        return;
    }
    const std::vector<float> & src = pre.src0;

    const int64_t cols = node->ne[0];
    const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];
    if (cols <= 0 || rows <= 0 || src.size() != dst.size()) {
        return;
    }

    const float eps = gtx_float_from_bits((uint32_t) node->op_params[0]);
    float max_diff = 0.0f;
    float max_diff_f16 = 0.0f;
    size_t max_idx = 0;
    float max_ref = 0.0f;
    float max_ref_f16 = 0.0f;
    float max_got = 0.0f;

    for (int64_t row = 0; row < rows; ++row) {
        const size_t base = (size_t) row * (size_t) cols;
        double mean = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            mean += (double) src[base + (size_t) col];
        }
        mean /= (double) cols;

        double var = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            const double d = (double) src[base + (size_t) col] - mean;
            var += d * d;
        }
        var /= (double) cols;
        const float inv_std = 1.0f / sqrtf((float) var + eps);

        for (int64_t col = 0; col < cols; ++col) {
            const size_t idx = base + (size_t) col;
            const float ref = (src[idx] - (float) mean) * inv_std;
            const float ref_f16 = gtx_fp16_to_fp32(gtx_fp32_to_fp16(ref));
            const float got = dst[idx];
            const float diff = fabsf(got - ref);
            const float diff_f16 = fabsf(got - ref_f16);
            if (diff_f16 > max_diff_f16) {
                max_diff = diff;
                max_diff_f16 = diff_f16;
                max_idx = idx;
                max_ref = ref;
                max_ref_f16 = ref_f16;
                max_got = got;
            }
        }
    }

    fprintf(stderr,
            "[GTX OP CHECK] NORM tensor=%s shape=[%lld,%lld,%lld,%lld] eps=%.9g max_diff_f16=%.9g max_diff=%.9g idx=%zu ref=%.9g ref_f16=%.9g got=%.9g\n",
            name,
            (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3],
            eps,
            max_diff_f16,
            max_diff,
            max_idx,
            max_ref,
            max_ref_f16,
            max_got);
}

static void gtx_check_softmax_after_launch(ggml_backend_gtx_context * ctx,
                                           const ggml_tensor * node,
                                           const gtx_op_check_pre_state & pre) {
    if (node == nullptr || node->src[0] == nullptr) {
        return;
    }

    std::vector<float> dst;
    const char * name = node->name[0] != '\0' ? node->name : "(anon)";
    if (!pre.valid ||
        pre.normalized_op != (uint32_t) GGML_OP_SOFT_MAX ||
        !gtx_check_ops_download(ctx, node, dst, "dst", name)) {
        fprintf(stderr, "[GTX OP CHECK] skip SOFT_MAX tensor=%s reason=download-failed-or-too-large\n", name);
        return;
    }
    const std::vector<float> & src = pre.src0;
    const std::vector<float> & mask = pre.src1;

    const int64_t ne0 = node->ne[0];
    const int64_t ne1 = node->ne[1];
    const int64_t ne2 = node->ne[2];
    const int64_t ne3 = node->ne[3];
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0 || src.size() != dst.size()) {
        return;
    }

    const float scale = gtx_float_from_bits((uint32_t) node->op_params[0]);
    const int64_t mne2 = node->src[1] != nullptr ? node->src[1]->ne[2] : 1;
    const int64_t mne3 = node->src[1] != nullptr ? node->src[1]->ne[3] : 1;
    float max_diff = 0.0f;
    float max_diff_f16 = 0.0f;
    size_t max_idx = 0;
    float max_ref = 0.0f;
    float max_ref_f16 = 0.0f;
    float max_got = 0.0f;

    std::vector<float> tmp((size_t) ne0);
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const size_t base = (((size_t) i3 * (size_t) ne2 + (size_t) i2) *
                                     (size_t) ne1 + (size_t) i1) *
                                    (size_t) ne0;
                const int64_t mi2 = mne2 > 0 ? i2 % mne2 : 0;
                const int64_t mi3 = mne3 > 0 ? i3 % mne3 : 0;
                const size_t mask_base = node->src[1] != nullptr ?
                    (((size_t) mi3 * (size_t) node->src[1]->ne[2] + (size_t) mi2) *
                     (size_t) node->src[1]->ne[1] + (size_t) i1) *
                    (size_t) node->src[1]->ne[0] : 0;

                float row_max = -std::numeric_limits<float>::infinity();
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    float v = src[base + (size_t) i0] * scale;
                    if (node->src[1] != nullptr) {
                        v += mask[mask_base + (size_t) i0];
                    }
                    tmp[(size_t) i0] = v;
                    row_max = std::max(row_max, v);
                }

                float sum = 0.0f;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float e = expf(tmp[(size_t) i0] - row_max);
                    tmp[(size_t) i0] = e;
                    sum += e;
                }
                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t idx = base + (size_t) i0;
                    const float ref = tmp[(size_t) i0] * inv_sum;
                    const float ref_f16 = gtx_fp16_to_fp32(gtx_fp32_to_fp16(ref));
                    const float got = dst[idx];
                    const float diff = fabsf(got - ref);
                    const float diff_f16 = fabsf(got - ref_f16);
                    if (diff_f16 > max_diff_f16) {
                        max_diff = diff;
                        max_diff_f16 = diff_f16;
                        max_idx = idx;
                        max_ref = ref;
                        max_ref_f16 = ref_f16;
                        max_got = got;
                    }
                }
            }
        }
    }

    fprintf(stderr,
            "[GTX OP CHECK] SOFT_MAX tensor=%s shape=[%lld,%lld,%lld,%lld] scale=%.9g mask=%d max_diff_f16=%.9g max_diff=%.9g idx=%zu ref=%.9g ref_f16=%.9g got=%.9g\n",
            name,
            (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
            scale,
            node->src[1] != nullptr ? 1 : 0,
            max_diff_f16,
            max_diff,
            max_idx,
            max_ref,
            max_ref_f16,
            max_got);
}

static void gtx_check_ops_after_launch(ggml_backend_gtx_context * ctx,
                                       const ggml_tensor * node,
                                       const gtx_op_check_pre_state & pre) {
    if (!gtx_check_ops_should_check(node)) {
        return;
    }

    static uint64_t checked_nodes = 0;
    const uint64_t max_nodes = gtx_env_u64_or_default("GTX_CHECK_OP_MAX_NODES", 256);
    if (max_nodes != 0 && checked_nodes >= max_nodes) {
        return;
    }
    checked_nodes++;

    const uint32_t normalized_op = gtx_normalize_host_op(node);
    if (normalized_op == (uint32_t) GGML_OP_NORM) {
        gtx_check_norm_after_launch(ctx, node, pre);
    } else if (normalized_op == (uint32_t) GGML_OP_SOFT_MAX) {
        gtx_check_softmax_after_launch(ctx, node, pre);
    }
}

static void gtx_dump_op_sample_after_launch(ggml_backend_gtx_context * ctx,
                                            const ggml_tensor * node) {
    if (!gtx_dump_op_sample_should_dump(node)) {
        return;
    }

    gtx_dump_op_sample_tensor(ctx, node, "post-dst");
}

static void gtx_dump_op_sample_before_launch(ggml_backend_gtx_context * ctx,
                                             const ggml_tensor * node) {
    if (!gtx_dump_op_sample_should_dump(node)) {
        return;
    }

    gtx_dump_op_sample_tensor(ctx, node->src[0], "pre-src0");
    const uint32_t normalized_op = gtx_normalize_host_op(node);
    if ((normalized_op == (uint32_t) GGML_OP_MUL_MAT ||
         normalized_op == (uint32_t) GGML_OP_ADD ||
         normalized_op == (uint32_t) GGML_OP_MUL) &&
        node->src[1] != nullptr) {
        gtx_dump_op_sample_tensor(ctx, node->src[1], "pre-src1");
    }
}

static void gtx_diagnostic_report_first_nonfinite(ggml_backend_gtx_context * ctx,
                                                  const ggml_tensor * node) {
    if (!gtx_env_enabled("GTX_DIAGNOSTIC_FIRST_NONFINITE") ||
        ctx == nullptr || node == nullptr) {
        return;
    }

    static bool reported = false;
    if (reported) {
        return;
    }

    const uint64_t limit_env =
        gtx_env_u64_or_default("GTX_DIAGNOSTIC_NONFINITE_ELEMS", 4096);
    const size_t limit = limit_env == 0 ?
        (size_t) ggml_nelements(node) :
        (size_t) std::min<uint64_t>(limit_env, (uint64_t) ggml_nelements(node));
    std::vector<float> values;
    if (limit == 0 ||
        !gtx_download_tensor_prefix_as_f32(ctx, node, limit, values)) {
        return;
    }

    size_t first = values.size();
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            first = i;
            break;
        }
    }
    if (first == values.size()) {
        return;
    }

    reported = true;
    fprintf(stderr,
            "[GTX FIRST NONFINITE] op=%s tensor=%s index=%zu value=%.9g "
            "shape=[%lld,%lld,%lld,%lld] scanned=%zu/%lld\n",
            gtx_tensor_op_display_name(node),
            node->name[0] != '\0' ? node->name : "(anon)",
            first, values[first],
            (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3],
            values.size(), (long long) ggml_nelements(node));
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            char tag[32];
            snprintf(tag, sizeof(tag), "first-bad-src%d", i);
            gtx_dump_op_sample_tensor(ctx, node->src[i], tag);
        }
    }
    gtx_dump_op_sample_tensor(ctx, node, "first-bad-dst");

    const char * capture_prefix = getenv("GTX_DIAGNOSTIC_CAPTURE_PREFIX");
    if (capture_prefix == nullptr || capture_prefix[0] == '\0') {
        return;
    }

    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s-meta.txt", capture_prefix);
    FILE * meta = fopen(meta_path, "w");
    if (meta != nullptr) {
        fprintf(meta,
                "op=%s\ntensor=%s\nshape=%lld,%lld,%lld,%lld\nfirst_nonfinite=%zu\n",
                gtx_tensor_op_display_name(node),
                node->name[0] != '\0' ? node->name : "(anon)",
                (long long) node->ne[0], (long long) node->ne[1],
                (long long) node->ne[2], (long long) node->ne[3], first);
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        const size_t elems = (size_t) ggml_nelements(src);
        std::vector<float> captured;
        if (!gtx_download_tensor_prefix_as_f32(ctx, src, elems, captured) ||
            captured.size() != elems) {
            fprintf(stderr,
                    "[GTX CAPTURE] failed src%d tensor=%s elems=%zu\n",
                    i, src->name[0] != '\0' ? src->name : "(anon)", elems);
            if (meta != nullptr) {
                fprintf(meta, "src%d_status=download_failed\n", i);
            }
            continue;
        }

        char data_path[1024];
        snprintf(data_path, sizeof(data_path), "%s-src%d.f32", capture_prefix, i);
        FILE * data = fopen(data_path, "wb");
        const bool wrote = data != nullptr &&
            fwrite(captured.data(), sizeof(float), captured.size(), data) == captured.size();
        if (data != nullptr) {
            fclose(data);
        }
        fprintf(stderr,
                "[GTX CAPTURE] %s src%d tensor=%s shape=[%lld,%lld,%lld,%lld] "
                "elems=%zu path=%s\n",
                wrote ? "saved" : "failed",
                i, src->name[0] != '\0' ? src->name : "(anon)",
                (long long) src->ne[0], (long long) src->ne[1],
                (long long) src->ne[2], (long long) src->ne[3],
                captured.size(), data_path);
        if (meta != nullptr) {
            fprintf(meta,
                    "src%d_name=%s\nsrc%d_shape=%lld,%lld,%lld,%lld\n"
                    "src%d_graph_type=%s\nsrc%d_remote_type=%s\n"
                    "src%d_elems=%zu\nsrc%d_file=%s\nsrc%d_status=%s\n",
                    i, src->name[0] != '\0' ? src->name : "(anon)",
                    i, (long long) src->ne[0], (long long) src->ne[1],
                    (long long) src->ne[2], (long long) src->ne[3],
                    i, ggml_type_name(src->type),
                    i, ggml_type_name(gtx_remote_tensor_type(src)),
                    i, captured.size(), i, data_path, i, wrote ? "saved" : "write_failed");
        }
    }
    if (meta != nullptr) {
        fclose(meta);
    }
}

static bool gtx_tiling_scheduler_diagnostics_active() {
    return gtx_check_ops_enabled() ||
           gtx_check_mul_mat_enabled() ||
           gtx_dump_op_samples_enabled() ||
           gtx_env_enabled("GTX_DIAGNOSTIC_FIRST_NONFINITE") ||
           gtx_log_softmax_enabled();
}

static bool gtx_diagnostic_materialize_stash(ggml_backend_gtx_context * ctx,
                                             const ggml_tensor * node) {
    if (!gtx_env_enabled("GTX_DIAGNOSTIC_STASH_FLUSH_RESET")) {
        return true;
    }
    if (ctx == nullptr || !ctx->use_uart || ctx->uart == nullptr) {
        return true;
    }
    const bool ok = ctx->uart->stash_flush_reset();
    if (!ok || gtx_log_ops_enabled()) {
        fprintf(stderr, "[GTX DIAG STASH] flush_reset %s tensor=%s op=%s\n",
                ok ? "done" : "failed",
                node != nullptr && node->name[0] != '\0' ? node->name : "(anon)",
                node != nullptr ? gtx_tensor_op_display_name(node) : "-");
    }
    return ok;
}

static bool gtx_scheduler_node_skippable(const ggml_tensor * node) {
    return node == nullptr ||
           gtx_is_metadata_op(node->op) ||
           ggml_nelements(node) == 0;
}

static bool gtx_scheduler_dense_f16_2d(const ggml_tensor * tensor) {
    return gtx_tensor_remote_f16_dense(tensor) &&
           tensor->ne[0] > 0 &&
           tensor->ne[1] > 0 &&
           tensor->ne[2] == 1 &&
           tensor->ne[3] == 1 &&
           tensor->ne[0] <= UINT32_MAX &&
           tensor->ne[1] <= UINT32_MAX;
}

static bool gtx_scheduler_same_2d_shape(const ggml_tensor * a, const ggml_tensor * b) {
    return a != nullptr && b != nullptr &&
           a->ne[0] == b->ne[0] &&
           a->ne[1] == b->ne[1] &&
           a->ne[2] == 1 &&
           a->ne[3] == 1 &&
           b->ne[2] == 1 &&
           b->ne[3] == 1;
}

struct gtx_scheduler_col_tile_shape {
    uint32_t i1;
    uint32_t i2;
    uint32_t i3;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
};

static bool gtx_scheduler_make_col_tile_shape(const ggml_tensor * node,
                                              uint32_t col0,
                                              uint32_t cols,
                                              gtx_scheduler_col_tile_shape & shape) {
    if (node == nullptr ||
        cols == 0 ||
        node->ne[1] <= 0 ||
        node->ne[2] <= 0 ||
        node->ne[3] <= 0 ||
        node->ne[1] > UINT32_MAX ||
        node->ne[2] > UINT32_MAX ||
        node->ne[3] > UINT32_MAX) {
        return false;
    }

    const uint32_t ne1 = (uint32_t)node->ne[1];
    const uint32_t ne2 = (uint32_t)node->ne[2];
    const uint32_t i1 = col0 % ne1;
    const uint32_t tmp = col0 / ne1;
    const uint32_t i2 = tmp % ne2;
    const uint32_t i3 = tmp / ne2;

    if (i1 + cols <= ne1) {
        shape = { i1, i2, i3, cols, 1u, 1u };
        return true;
    }

    if (i1 == 0 && cols % ne1 == 0) {
        const uint32_t tile_ne2 = cols / ne1;
        if (tile_ne2 > 0 && i2 + tile_ne2 <= ne2) {
            shape = { i1, i2, i3, ne1, tile_ne2, 1u };
            return true;
        }
    }

    return false;
}

static uint64_t gtx_scheduler_col_offset(const ggml_tensor * tensor,
                                         const gtx_scheduler_col_tile_shape & shape) {
    return (uint64_t)shape.i1 * gtx_remote_byte_offset(tensor, tensor->nb[1]) +
           (uint64_t)shape.i2 * gtx_remote_byte_offset(tensor, tensor->nb[2]) +
           (uint64_t)shape.i3 * gtx_remote_byte_offset(tensor, tensor->nb[3]);
}

static bool gtx_scheduler_col_span_contiguous(const ggml_tensor * tensor,
                                              uint32_t col0,
                                              uint32_t cols) {
    gtx_scheduler_col_tile_shape shape = {};
    if (!gtx_scheduler_make_col_tile_shape(tensor, col0, cols, shape) ||
        !gtx_tensor_remote_f16_vector_contiguous(tensor)) {
        return false;
    }

    const uint64_t row_bytes = (uint64_t)tensor->ne[0] * sizeof(uint16_t);
    return (shape.ne1 <= 1u ||
            gtx_remote_byte_offset(tensor, tensor->nb[1]) == row_bytes) &&
           (shape.ne2 <= 1u ||
            gtx_remote_byte_offset(tensor, tensor->nb[2]) == row_bytes * (uint64_t)tensor->ne[1]) &&
           (shape.ne3 <= 1u ||
            gtx_remote_byte_offset(tensor, tensor->nb[3]) == row_bytes * (uint64_t)tensor->ne[1] * (uint64_t)tensor->ne[2]);
}

static bool gtx_scheduler_col_span_strided_rows_supported(const ggml_tensor * tensor,
                                                          uint32_t col0,
                                                          uint32_t cols) {
    gtx_scheduler_col_tile_shape shape = {};
    if (!gtx_scheduler_make_col_tile_shape(tensor, col0, cols, shape) ||
        !gtx_tensor_remote_f16_element_contiguous(tensor)) {
        return false;
    }

    const uint64_t nb1 = gtx_remote_byte_offset(tensor, tensor->nb[1]);
    const uint64_t nb2 = gtx_remote_byte_offset(tensor, tensor->nb[2]);
    const uint64_t nb3 = gtx_remote_byte_offset(tensor, tensor->nb[3]);
    return nb1 > 0 && nb1 <= UINT32_MAX &&
           nb2 <= UINT32_MAX &&
           nb3 <= UINT32_MAX;
}

static void gtx_scheduler_make_dense_tile_shape_4d(ggml_gtx_kernel_launch_desc_t & desc,
                                                   uint32_t ne0,
                                                   uint32_t ne1,
                                                   uint32_t ne2,
                                                   uint32_t ne3) {
    desc.ne[0] = ne0;
    desc.ne[1] = ne1;
    desc.ne[2] = ne2;
    desc.ne[3] = ne3;
    desc.nb[0] = sizeof(uint16_t);
    desc.nb[1] = ne0 * sizeof(uint16_t);
    desc.nb[2] = desc.nb[1] * ne1;
    desc.nb[3] = desc.nb[2] * ne2;
}

static void gtx_scheduler_make_strided_dst_tile_shape_4d(
        ggml_gtx_kernel_launch_desc_t & desc,
        const ggml_tensor * dst,
        uint32_t ne0,
        uint32_t ne1,
        uint32_t ne2,
        uint32_t ne3) {
    desc.ne[0] = ne0;
    desc.ne[1] = ne1;
    desc.ne[2] = ne2;
    desc.ne[3] = ne3;
    if (dst != nullptr) {
        desc.nb[0] = (uint32_t)gtx_remote_byte_offset(dst, dst->nb[0]);
        desc.nb[1] = (uint32_t)gtx_remote_byte_offset(dst, dst->nb[1]);
        desc.nb[2] = (uint32_t)gtx_remote_byte_offset(dst, dst->nb[2]);
        desc.nb[3] = (uint32_t)gtx_remote_byte_offset(dst, dst->nb[3]);
    }
}

static void gtx_scheduler_make_dense_src_tile_shape_4d(ggml_gtx_kernel_launch_desc_t & desc,
                                                       uint32_t src_idx,
                                                       uint32_t ne0,
                                                       uint32_t ne1,
                                                       uint32_t ne2,
                                                       uint32_t ne3) {
    desc.src_ne[src_idx][0] = ne0;
    desc.src_ne[src_idx][1] = ne1;
    desc.src_ne[src_idx][2] = ne2;
    desc.src_ne[src_idx][3] = ne3;
    desc.src_nb[src_idx][0] = sizeof(uint16_t);
    desc.src_nb[src_idx][1] = ne0 * sizeof(uint16_t);
    desc.src_nb[src_idx][2] = desc.src_nb[src_idx][1] * ne1;
    desc.src_nb[src_idx][3] = desc.src_nb[src_idx][2] * ne2;
}

static void gtx_scheduler_make_strided_src_tile_shape_4d(ggml_gtx_kernel_launch_desc_t & desc,
                                                         uint32_t src_idx,
                                                         const ggml_tensor * src,
                                                         uint32_t ne0,
                                                         uint32_t ne1,
                                                         uint32_t ne2,
                                                         uint32_t ne3) {
    desc.src_ne[src_idx][0] = ne0;
    desc.src_ne[src_idx][1] = ne1;
    desc.src_ne[src_idx][2] = ne2;
    desc.src_ne[src_idx][3] = ne3;
    if (src != nullptr) {
        desc.src_nb[src_idx][0] = (uint32_t)gtx_remote_byte_offset(src, src->nb[0]);
        desc.src_nb[src_idx][1] = (uint32_t)gtx_remote_byte_offset(src, src->nb[1]);
        desc.src_nb[src_idx][2] = (uint32_t)gtx_remote_byte_offset(src, src->nb[2]);
        desc.src_nb[src_idx][3] = (uint32_t)gtx_remote_byte_offset(src, src->nb[3]);
    }
}

static bool gtx_scheduler_tileable_row_node(const ggml_tensor * node) {
    if (!gtx_scheduler_dense_f16_2d(node)) {
        return false;
    }

    const gtx_l2_chainability chain = gtx_l2_chainability_from_node(node);
    if (chain.kind == gtx_l2_chain_kind::producer) {
        // Materialization requires a host-provided strided-copy access plan.
        return false;
    }
    if (chain.kind != gtx_l2_chain_kind::native) {
        return false;
    }

    const uint32_t kernel_id = chain.kernel_id;
    if (kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16) {
        return false;
    }

    if (kernel_id == GGML_GTX_KERNEL_SOFT_MAX_F16) {
        return false;
    }
    if (kernel_id == GGML_GTX_KERNEL_RMS_NORM_F16) {
        return gtx_l2_rms_norm_shape_ok(node) &&
               gtx_scheduler_same_2d_shape(node, node->src[0]);
    }
    if (kernel_id == GGML_GTX_KERNEL_SWIGLU_F16) {
        return false;
    }

    if (node->src[0] == nullptr ||
        !gtx_scheduler_same_2d_shape(node, node->src[0])) {
        return false;
    }

    if (kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
        kernel_id == GGML_GTX_KERNEL_MUL_F16) {
        if (node->src[1] == nullptr) {
            return false;
        }
        if (gtx_scheduler_same_2d_shape(node, node->src[1])) {
            return true;
        }
        return false;
    }

    return kernel_id == GGML_GTX_KERNEL_SCALE_F16 ||
           kernel_id == GGML_GTX_KERNEL_GELU_F16 ||
           kernel_id == GGML_GTX_KERNEL_ABS_F16 ||
           kernel_id == GGML_GTX_KERNEL_NEG_F16 ||
           kernel_id == GGML_GTX_KERNEL_SGN_F16 ||
           kernel_id == GGML_GTX_KERNEL_CEIL_F16 ||
           kernel_id == GGML_GTX_KERNEL_FLOOR_F16 ||
           kernel_id == GGML_GTX_KERNEL_TANH_F16 ||
           kernel_id == GGML_GTX_KERNEL_SILU_F16 ||
           kernel_id == GGML_GTX_KERNEL_RELU_F16 ||
           kernel_id == GGML_GTX_KERNEL_LEAKY_RELU_F16 ||
           kernel_id == GGML_GTX_KERNEL_CLAMP_F16;
}

static uint64_t gtx_scheduler_row_offset(const ggml_tensor * tensor, uint32_t row0) {
    return (uint64_t)row0 * gtx_remote_byte_offset(tensor, tensor->nb[1]);
}

static void gtx_scheduler_make_dense_tile_shape(ggml_gtx_kernel_launch_desc_t & desc,
                                                uint32_t ne0,
                                                uint32_t rows) {
    gtx_scheduler_make_dense_tile_shape_4d(desc, ne0, rows, 1u, 1u);
}

static void gtx_scheduler_make_dense_src_tile_shape(ggml_gtx_kernel_launch_desc_t & desc,
                                                    uint32_t src_idx,
                                                    uint32_t ne0,
                                                    uint32_t rows) {
    gtx_scheduler_make_dense_src_tile_shape_4d(desc, src_idx, ne0, rows, 1u, 1u);
}

// These uploaded L2 kernels consume a single flat vector. Keep this ABI
// adaptation host-side: firmware receives an already-final raw tile and never
// derives it from the ggml operation.
static bool gtx_scheduler_row_tile_kernel_is_flat(uint32_t kernel_id) {
    switch (kernel_id) {
        case GGML_GTX_KERNEL_ADD_F16:
        case GGML_GTX_KERNEL_MUL_F16:
        case GGML_GTX_KERNEL_SCALE_F16:
        case GGML_GTX_KERNEL_GELU_F16:
        case GGML_GTX_KERNEL_ABS_F16:
        case GGML_GTX_KERNEL_NEG_F16:
        case GGML_GTX_KERNEL_SGN_F16:
        case GGML_GTX_KERNEL_CEIL_F16:
        case GGML_GTX_KERNEL_FLOOR_F16:
        case GGML_GTX_KERNEL_TANH_F16:
        case GGML_GTX_KERNEL_SILU_F16:
        case GGML_GTX_KERNEL_RELU_F16:
        case GGML_GTX_KERNEL_LEAKY_RELU_F16:
        case GGML_GTX_KERNEL_CLAMP_F16:
            return true;
        default:
            return false;
    }
}

static bool gtx_scheduler_build_row_tile_desc(const ggml_tensor * node,
                                              uint32_t row0,
                                              uint32_t rows,
                                              ggml_gtx_kernel_launch_desc_t & tile) {
    if (!gtx_build_kernel_launch_desc(node, tile) ||
        node->ne[0] <= 0 ||
        node->ne[0] > UINT32_MAX ||
        rows == 0) {
        return false;
    }

    const uint32_t ne0 = (uint32_t)node->ne[0];
    tile.dst_addr += gtx_scheduler_row_offset(node, row0);
    gtx_scheduler_make_dense_tile_shape(tile, ne0, rows);

    if (gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_SWIGLU_F16) {
        const ggml_tensor * src = node->src[0];
        const uint32_t src_ne0 = ne0 * 2u;
        tile.src_addr[0] += gtx_scheduler_row_offset(src, row0);
        gtx_scheduler_make_dense_src_tile_shape(tile, 0, src_ne0, rows);
        tile.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
        return true;
    }

    for (uint32_t i = 0; i < 4; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src == nullptr || tile.src_addr[i] == 0) {
            continue;
        }
        if (gtx_scheduler_same_2d_shape(node, src)) {
            tile.src_addr[i] += gtx_scheduler_row_offset(src, row0);
            gtx_scheduler_make_dense_src_tile_shape(tile, i, ne0, rows);
        }
    }

    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if (gtx_scheduler_row_tile_kernel_is_flat(kernel_id)) {
        const uint64_t tile_elems = (uint64_t)ne0 * rows;
        if (tile_elems == 0u || tile_elems > UINT32_MAX) {
            return false;
        }
        gtx_scheduler_make_dense_tile_shape(tile, (uint32_t)tile_elems, 1u);
        for (uint32_t i = 0; i < 4; ++i) {
            if (node->src[i] != nullptr && tile.src_addr[i] != 0u) {
                gtx_scheduler_make_dense_src_tile_shape(tile, i,
                                                         (uint32_t)tile_elems, 1u);
            }
        }
    }
    return true;
}

static bool gtx_compare_cpu_reference_after_launch(ggml_backend_gtx_context * ctx,
                                                   const ggml_tensor * node,
                                                   int node_index) {
    if (gtx_cpu_ref_first_divergence_found &&
        gtx_env_truthy("GTX_CPU_REF_STOP_AFTER_DIVERGENCE")) {
        return true;
    }
    char path[1024];
    if (!gtx_cpu_ref_path(path, sizeof(path), node_index)) {
        return true;
    }
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        return true;
    }

    const size_t elems = (size_t)ggml_nelements(node);
    std::vector<float> reference(elems);
    const size_t got = fread(reference.data(), sizeof(float), elems, file);
    const int trailing = fgetc(file);
    fclose(file);
    if (got != elems || trailing != EOF) {
        fprintf(stderr,
                "[GTX CPU REF] graph=%llu node=%d tensor=%s invalid_reference_size path=%s expected_elems=%zu got=%zu\n",
                (unsigned long long)gtx_cpu_ref_current_graph, node_index,
                node->name[0] != '\0' ? node->name : "(anon)", path, elems, got);
        return false;
    }

    std::vector<float> actual;
    if (!gtx_download_tensor_strided_f16_as_f32(ctx, node, actual) || actual.size() != elems) {
        fprintf(stderr,
                "[GTX CPU REF] graph=%llu node=%d tensor=%s download_failed elems=%zu\n",
                (unsigned long long)gtx_cpu_ref_current_graph, node_index,
                node->name[0] != '\0' ? node->name : "(anon)", elems);
        return false;
    }

    float max_diff = 0.0f;
    size_t max_index = 0;
    double diff_sumsq = 0.0;
    double ref_sumsq = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < elems; ++i) {
        const float ref = gtx_fp16_to_fp32(gtx_fp32_to_fp16(reference[i]));
        reference[i] = ref;
        const float value = actual[i];
        if (!std::isfinite(ref) || !std::isfinite(value)) {
            ++nonfinite;
            max_diff = INFINITY;
            max_index = i;
            continue;
        }
        const float diff = fabsf(value - ref);
        diff_sumsq += (double)diff * (double)diff;
        ref_sumsq += (double)ref * (double)ref;
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    const double rmse = elems != 0 ? sqrt(diff_sumsq / (double)elems) : 0.0;
    const double ref_rms = elems != 0 ? sqrt(ref_sumsq / (double)elems) : 0.0;
    const double scale = std::max(ref_rms, 1.0e-6);
    const double max_rel_rms = (double)max_diff / scale;
    const double rmse_rel = rmse / scale;
    const float abs_threshold = gtx_env_float_or_default("GTX_CPU_REF_THRESHOLD", 0.02f);
    const float max_rel_threshold = gtx_env_float_or_default("GTX_CPU_REF_MAX_REL_RMS", 0.05f);
    const float rmse_rel_threshold = gtx_env_float_or_default("GTX_CPU_REF_RMSE_REL", 0.01f);
    const bool diverged = nonfinite != 0 ||
        (max_diff > abs_threshold &&
         max_rel_rms > (double)max_rel_threshold &&
         rmse_rel > (double)rmse_rel_threshold);
    fprintf(stderr,
            "[GTX CPU REF] graph=%llu node=%d op=%s tensor=%s shape=[%lld,%lld,%lld,%lld] "
            "max_diff=%.9g index=%zu ref=%.9g got=%.9g rmse=%.9g ref_rms=%.9g "
            "max_rel_rms=%.9g rmse_rel=%.9g nonfinite=%zu status=%s\n",
            (unsigned long long)gtx_cpu_ref_current_graph, node_index,
            gtx_tensor_op_display_name(node), node->name[0] != '\0' ? node->name : "(anon)",
            (long long)node->ne[0], (long long)node->ne[1],
            (long long)node->ne[2], (long long)node->ne[3],
            max_diff, max_index, reference[max_index], actual[max_index],
            rmse, ref_rms, max_rel_rms, rmse_rel, nonfinite, diverged ? "DIVERGED" : "ok");
    if (diverged) {
        gtx_cpu_ref_first_divergence_found = true;
    }
    return !diverged;
}

static bool gtx_scheduler_lower_row_island(const std::vector<ggml_tensor *> & nodes,
                                           std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (nodes.empty() || nodes[0] == nullptr ||
        !gtx_scheduler_dense_f16_2d(nodes[0])) {
        return false;
    }

    const int64_t ne0_64 = nodes[0]->ne[0];
    const int64_t rows64 = nodes[0]->ne[1];
    if (ne0_64 <= 0 || rows64 <= 0 || ne0_64 > UINT32_MAX || rows64 > UINT32_MAX) {
        return false;
    }

    for (const ggml_tensor * node : nodes) {
        if (!gtx_scheduler_tileable_row_node(node) ||
            !gtx_scheduler_same_2d_shape(nodes[0], node)) {
            return false;
        }
    }

    const uint32_t ne0 = (uint32_t)ne0_64;
    const uint32_t total_rows = (uint32_t)rows64;
    const uint64_t row_bytes = (uint64_t)ne0 * sizeof(uint16_t);
    const uint64_t slot_budget = (gtx_stash_data_budget(nodes) / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint64_t max_tile_bytes = std::min<uint64_t>(
        gtx_env_u64_or_default("GTX_TILE_MAX_BYTES", slot_budget), slot_budget);
    if (row_bytes == 0 || max_tile_bytes < row_bytes) {
        return false;
    }

    uint32_t rows_per_tile = (uint32_t)std::min<uint64_t>(total_rows, max_tile_bytes / row_bytes);
    if (rows_per_tile == 0) {
        return false;
    }

    out_descs.clear();
    for (uint32_t row0 = 0; row0 < total_rows; row0 += rows_per_tile) {
        const uint32_t rows = std::min<uint32_t>(rows_per_tile, total_rows - row0);
        for (const ggml_tensor * node : nodes) {
            ggml_gtx_kernel_launch_desc_t desc = {};
            if (!gtx_scheduler_build_row_tile_desc(node, row0, rows, desc)) {
                out_descs.clear();
                return false;
            }
            out_descs.push_back(desc);
        }
    }
    return !out_descs.empty();
}

static bool gtx_scheduler_find_row_island(ggml_cgraph * cgraph,
                                          int start,
                                          int & end,
                                          std::vector<ggml_tensor *> & nodes) {
    nodes.clear();
    end = start;
    if (cgraph == nullptr || start < 0 || start >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * first = cgraph->nodes[start];
    if (!gtx_scheduler_tileable_row_node(first)) {
        return false;
    }

    for (int i = start; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!gtx_scheduler_tileable_row_node(node) ||
            !gtx_scheduler_same_2d_shape(first, node)) {
            break;
        }
        nodes.push_back(node);
        end = i + 1;
    }

    return nodes.size() >= 2;
}

static bool gtx_scheduler_tileable_col_node(const ggml_tensor * node) {
    if (node == nullptr ||
        node->ne[0] <= 0 ||
        node->ne[1] <= 0 ||
        node->ne[2] <= 0 ||
        node->ne[3] <= 0 ||
        node->ne[0] > UINT32_MAX ||
        node->ne[1] > UINT32_MAX ||
        node->ne[2] > UINT32_MAX ||
        node->ne[3] > UINT32_MAX) {
        return false;
    }

    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if (kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16) {
        // `src0.ne[1]` is output-row count and may exceed the L1/kernel
        // limit in the complete tensor.  The host row tiler applies that
        // limit to each emitted tile; do not reject it before tiling.
        return node->src[0] != nullptr &&
               node->src[1] != nullptr &&
               gtx_tensor_remote_f16_element_contiguous(node) &&
               node->ne[1] <= UINT16_MAX &&
               node->ne[2] <= UINT16_MAX &&
               node->ne[3] <= UINT16_MAX &&
               node->src[0]->ne[0] == node->src[1]->ne[0] &&
               node->src[0]->ne[0] <= UINT16_MAX &&
               node->src[1]->ne[1] == node->ne[1] &&
               node->src[1]->ne[2] == node->ne[2] &&
               node->src[1]->ne[3] == node->ne[3] &&
               node->src[0]->ne[2] > 0 && node->src[0]->ne[3] > 0 &&
               node->ne[2] % node->src[0]->ne[2] == 0 &&
               node->ne[3] % node->src[0]->ne[3] == 0 &&
               gtx_tensor_remote_f16_vector_contiguous(node->src[0]) &&
               gtx_tensor_remote_f16_vector_contiguous(node->src[1]) &&
               node->src[1]->ne[2] == node->ne[2] &&
               node->src[1]->ne[3] == node->ne[3] &&
               node->src[1]->ne[1] == node->ne[1];
    }

    if (!gtx_tensor_remote_f16_dense(node)) {
        return false;
    }

    const gtx_l2_chainability chain = gtx_l2_chainability_from_node(node);
    if (chain.kind != gtx_l2_chain_kind::native) {
        return false;
    }

    if (chain.kernel_id == GGML_GTX_KERNEL_SWIGLU_F16) {
        return gtx_l2_swiglu_shape_ok(node);
    }

    if (node->src[0] == nullptr ||
        !gtx_tensor_remote_f16_dense(node->src[0]) ||
        !gtx_tensor_same_shape(node, node->src[0])) {
        return false;
    }
    if (chain.kernel_id == GGML_GTX_KERNEL_SOFT_MAX_F16) {
        return gtx_l2_softmax_basic_shape_ok(node);
    }
    if (chain.kernel_id == GGML_GTX_KERNEL_ROPE_F16) {
        return gtx_l2_rope_shape_ok(node);
    }
    if (chain.kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
        chain.kernel_id == GGML_GTX_KERNEL_MUL_F16) {
        return node->src[1] != nullptr &&
               gtx_tensor_remote_f16_dense(node->src[1]) &&
               gtx_tensor_same_shape(node, node->src[1]);
    }
    return chain.kernel_id == GGML_GTX_KERNEL_SCALE_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_GELU_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_RMS_NORM_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_ROPE_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_SWIGLU_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_ABS_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_NEG_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_SGN_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_CEIL_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_FLOOR_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_TANH_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_SILU_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_RELU_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_LEAKY_RELU_F16 ||
           chain.kernel_id == GGML_GTX_KERNEL_CLAMP_F16;
}

static bool gtx_scheduler_node_consumes_prev(const ggml_tensor * node,
                                             const ggml_tensor * prev) {
    if (node == nullptr || prev == nullptr) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        const ggml_tensor * src = node->src[i];
        while (src != nullptr && gtx_is_metadata_op(src->op) && src->src[0] != nullptr) {
            src = src->src[0];
        }
        if (src == prev) {
            return true;
        }
    }
    return false;
}

static bool gtx_scheduler_build_col_tile_desc(const ggml_tensor * node,
                                              uint32_t col0,
                                              uint32_t cols,
                                              ggml_gtx_kernel_launch_desc_t & tile) {
    gtx_scheduler_col_tile_shape shape = {};
    if (node == nullptr ||
        !gtx_scheduler_make_col_tile_shape(node, col0, cols, shape)) {
        return false;
    }

    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    const uint64_t dst_off = gtx_scheduler_col_offset(node, shape);

    if (kernel_id == GGML_GTX_KERNEL_SWIGLU_F16) {
        if (!gtx_build_kernel_launch_desc(node, tile) ||
            node->src[0] == nullptr ||
            node->ne[0] > UINT32_MAX ||
            node->src[0]->ne[0] > UINT32_MAX) {
            return false;
        }

        const uint32_t ne0 = (uint32_t)node->ne[0];
        const uint32_t src_ne0 = (uint32_t)node->src[0]->ne[0];
        const ggml_tensor * src = node->src[0];
        const uint64_t src_off = gtx_scheduler_col_offset(src, shape);
        tile.dst_addr += dst_off;
        tile.src_addr[0] += src_off;
        gtx_scheduler_make_dense_tile_shape_4d(tile, ne0, shape.ne1, shape.ne2, shape.ne3);
        gtx_scheduler_make_dense_src_tile_shape_4d(tile, 0, src_ne0, shape.ne1, shape.ne2, shape.ne3);
        tile.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
        return true;
    }

    if (kernel_id != GGML_GTX_KERNEL_MUL_MAT_F16) {
        if (!gtx_build_kernel_launch_desc(node, tile) ||
            node->ne[0] > UINT32_MAX) {
            return false;
        }
        const uint32_t ne0 = (uint32_t)node->ne[0];
        tile.dst_addr += dst_off;
        gtx_scheduler_make_dense_tile_shape_4d(tile, ne0, shape.ne1, shape.ne2, shape.ne3);
        for (uint32_t s = 0; s < 4; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr || tile.src_addr[s] == 0) {
                continue;
            }
            if (gtx_tensor_same_shape(node, src)) {
                const uint64_t src_off = gtx_scheduler_col_offset(src, shape);
                tile.src_addr[s] += src_off;
                gtx_scheduler_make_dense_src_tile_shape_4d(tile, s, ne0, shape.ne1, shape.ne2, shape.ne3);
            }
        }
        const bool flat_kernel = gtx_scheduler_row_tile_kernel_is_flat(kernel_id) &&
                                 kernel_id != GGML_GTX_KERNEL_ADD_F16 &&
                                 kernel_id != GGML_GTX_KERNEL_MUL_F16;
        if (flat_kernel) {
            const uint64_t tile_elems = (uint64_t)ne0 * shape.ne1 * shape.ne2 * shape.ne3;
            if (tile_elems == 0u || tile_elems > UINT16_MAX) {
                return false;
            }
            gtx_scheduler_make_dense_tile_shape(tile, (uint32_t)tile_elems, 1u);
            for (uint32_t s = 0; s < 4; ++s) {
                if (node->src[s] != nullptr && tile.src_addr[s] != 0u &&
                    gtx_tensor_same_shape(node, node->src[s])) {
                    gtx_scheduler_make_dense_src_tile_shape(
                        tile, s, (uint32_t)tile_elems, 1u);
                }
            }
        }
        if (kernel_id == GGML_GTX_KERNEL_ROPE_F16) {
            const ggml_tensor * pos = node->src[1];
            if (pos == nullptr || shape.i2 + shape.ne2 > (uint32_t)pos->ne[0]) {
                return false;
            }
            tile.src_addr[1] += (uint64_t)shape.i2 * gtx_remote_byte_offset(pos, pos->nb[0]);
            tile.src_ne[1][0] = shape.ne2;
            tile.src_ne[1][1] = 1u;
            tile.src_ne[1][2] = 1u;
            tile.src_ne[1][3] = 1u;
            tile.src_nb[1][0] = sizeof(int32_t);
            tile.src_nb[1][1] = shape.ne2 * sizeof(int32_t);
            tile.src_nb[1][2] = tile.src_nb[1][1];
            tile.src_nb[1][3] = tile.src_nb[1][1];
            tile.l2_scratch_size = ggml_gtx_rope_scratch_size(shape.ne2,
                                                               (uint32_t)tile.op_params[1]);
        }
        tile.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
        return true;
    }

    if (!gtx_build_kernel_launch_desc(node, tile) ||
        node->src[1] == nullptr ||
        node->ne[0] <= 0 ||
        node->src[1]->ne[0] <= 0 ||
        node->ne[0] > UINT32_MAX ||
        node->src[1]->ne[0] > UINT32_MAX) {
        return false;
    }

    const uint32_t dst_ne0 = (uint32_t)node->ne[0];
    const uint32_t src1_ne0 = (uint32_t)node->src[1]->ne[0];
    const uint32_t src0_ne2 = (uint32_t)node->src[0]->ne[2];
    const uint32_t src0_ne3 = (uint32_t)node->src[0]->ne[3];
    const uint32_t src0_r2 = (uint32_t)node->ne[2] / src0_ne2;
    const uint32_t src0_r3 = (uint32_t)node->ne[3] / src0_ne3;
    const uint32_t src0_i2 = shape.i2 / src0_r2;
    const uint32_t src0_i3 = shape.i3 / src0_r3;
    const uint64_t src0_off =
        (uint64_t)src0_i2 * gtx_remote_byte_offset(node->src[0], node->src[0]->nb[2]) +
        (uint64_t)src0_i3 * gtx_remote_byte_offset(node->src[0], node->src[0]->nb[3]);
    const uint64_t src1_off = gtx_scheduler_col_offset(node->src[1], shape);
    tile.dst_addr += dst_off;
    tile.src_addr[0] += src0_off;
    tile.src_addr[1] += src1_off;
    // Preserve the raw DDR strides. Firmware stash packs this view into a
    // dense L2 allocation before invoking the uploaded kernel.
    gtx_scheduler_make_strided_dst_tile_shape_4d(
        tile, node, dst_ne0, shape.ne1, shape.ne2, shape.ne3);
    if (shape.ne2 > 1u) {
        tile.src_ne[0][2] = (uint32_t)node->src[0]->ne[2];
    } else {
        tile.src_ne[0][2] = 1;
    }
    tile.src_ne[0][3] = 1;
    gtx_scheduler_make_strided_src_tile_shape_4d(tile, 1, node->src[1], src1_ne0, shape.ne1, shape.ne2, shape.ne3);
    tile.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
    return true;
}

static bool gtx_scheduler_build_mul_mat_col_row_tile_desc(const ggml_tensor * node,
                                                          uint32_t col0,
                                                          uint32_t cols,
                                                          uint32_t row0,
                                                          uint32_t rows,
                                                          ggml_gtx_kernel_launch_desc_t & tile) {
    gtx_scheduler_col_tile_shape shape = {};
    if (node == nullptr ||
        gtx_kernel_id_from_node(node) != GGML_GTX_KERNEL_MUL_MAT_F16 ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        rows == 0 ||
        !gtx_scheduler_make_col_tile_shape(node, col0, cols, shape) ||
        !gtx_build_kernel_launch_desc(node, tile) ||
        node->ne[0] <= 0 ||
        node->src[1]->ne[0] <= 0 ||
        node->ne[0] > UINT32_MAX ||
        node->src[1]->ne[0] > UINT32_MAX ||
        row0 >= (uint32_t)node->ne[0] ||
        rows > (uint32_t)node->ne[0] - row0) {
        return false;
    }

    const uint32_t dst_ne0 = rows;
    const uint32_t src1_ne0 = (uint32_t)node->src[1]->ne[0];
    const uint32_t src0_ne2 = (uint32_t)node->src[0]->ne[2];
    const uint32_t src0_ne3 = (uint32_t)node->src[0]->ne[3];
    const uint32_t src0_r2 = (uint32_t)node->ne[2] / src0_ne2;
    const uint32_t src0_r3 = (uint32_t)node->ne[3] / src0_ne3;
    const uint32_t src0_i2 = shape.i2 / src0_r2;
    const uint32_t src0_i3 = shape.i3 / src0_r3;
    const uint64_t src0_off =
        (uint64_t)src0_i2 * gtx_remote_byte_offset(node->src[0], node->src[0]->nb[2]) +
        (uint64_t)src0_i3 * gtx_remote_byte_offset(node->src[0], node->src[0]->nb[3]) +
        (uint64_t)row0 * gtx_remote_byte_offset(node->src[0], node->src[0]->nb[1]);
    const uint64_t src1_off = gtx_scheduler_col_offset(node->src[1], shape);
    const uint64_t dst_off = gtx_scheduler_col_offset(node, shape) +
                             (uint64_t)row0 * sizeof(uint16_t);

    tile.dst_addr += dst_off;
    tile.src_addr[0] += src0_off;
    tile.src_addr[1] += src1_off;
    // A row slice spanning multiple output columns retains the full output
    // column stride in DDR. Stash scatters the compact L2 result through it.
    gtx_scheduler_make_strided_dst_tile_shape_4d(
        tile, node, dst_ne0, shape.ne1, shape.ne2, shape.ne3);
    tile.src_ne[0][1] = rows;
    if (shape.ne2 > 1u) {
        tile.src_ne[0][2] = (uint32_t)node->src[0]->ne[2];
    } else {
        tile.src_ne[0][2] = 1;
    }
    tile.src_ne[0][3] = 1;
    gtx_scheduler_make_strided_src_tile_shape_4d(tile, 1, node->src[1], src1_ne0, shape.ne1, shape.ne2, shape.ne3);
    tile.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
    return true;
}

static uint32_t gtx_scheduler_mul_mat_rows_per_output_tile(const std::vector<ggml_tensor *> & nodes,
                                                           size_t node_idx,
                                                           uint32_t cols) {
    const ggml_tensor * node = node_idx < nodes.size() ? nodes[node_idx] : nullptr;
    if (node == nullptr ||
        node->src[0] == nullptr ||
        node->src[1] == nullptr ||
        node->ne[0] <= 0 ||
        node->src[0]->ne[0] <= 0 ||
        node->src[1]->ne[0] <= 0 ||
        cols == 0 ||
        node->ne[0] > UINT32_MAX ||
        node->src[0]->ne[0] > UINT32_MAX ||
        node->src[1]->ne[0] > UINT32_MAX) {
        return 0;
    }

    const uint64_t data_budget = gtx_stash_data_budget(node);
    const uint64_t l2_budget_bytes = std::min<uint64_t>(
        gtx_env_u64_or_default("GTX_TILE_MAX_BYTES", data_budget), data_budget);
    const uint64_t reserve_bytes = gtx_env_u64_or_default("GTX_TILE_RESERVE_BYTES", 64ull * 1024ull);
    const uint64_t dst_row_bytes = (uint64_t)cols * sizeof(uint16_t);
    const uint64_t weight_row_bytes = (uint64_t)node->src[0]->ne[0] * sizeof(uint16_t);
    const uint64_t src1_bytes = (uint64_t)node->src[1]->ne[0] * (uint64_t)cols * sizeof(uint16_t);
    const uint64_t fixed_bytes = gtx_align_up((size_t)src1_bytes, 32) + reserve_bytes;
    const uint64_t bytes_per_output_row = weight_row_bytes + dst_row_bytes;
    if (bytes_per_output_row == 0 ||
        l2_budget_bytes <= fixed_bytes ||
        l2_budget_bytes - fixed_bytes < bytes_per_output_row) {
        return 1;
    }

    uint64_t rows = (l2_budget_bytes - fixed_bytes) / bytes_per_output_row;
    rows = std::min<uint64_t>(rows, (uint64_t)node->ne[0]);
    // Keep an optional override for focused hardware-boundary experiments,
    // but derive the normal row tile solely from the actual L2 budget.  A
    // single __mm_t(64, 128, 2) is verified on the active n1s1 FPGA, so a
    // blanket 32-row cap would only multiply descriptor count.
    rows = std::min<uint64_t>(
        rows,
        gtx_env_u64_or_default("GTX_MUL_MAT_ROW_TILE_MAX_ROWS", (uint64_t)node->ne[0]));
    rows = std::min<uint64_t>(rows, UINT16_MAX);
    while (rows > 1) {
        const uint64_t src0_bytes = weight_row_bytes * rows;
        const uint64_t dst_bytes = dst_row_bytes * rows;
        const uint64_t total_bytes =
            gtx_align_up((size_t)src0_bytes, 32) +
            gtx_align_up((size_t)dst_bytes, 32) +
            gtx_align_up((size_t)src1_bytes, 32) +
            reserve_bytes;
        if (total_bytes <= l2_budget_bytes) {
            break;
        }
        --rows;
    }
    return rows == 0 ? 1u : (uint32_t)rows;
}

static bool gtx_scheduler_col_tile_contiguous_for_node(const ggml_tensor * node,
                                                       uint32_t col0,
                                                       uint32_t cols) {
    if (node == nullptr) {
        return false;
    }

    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if (kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16) {
        gtx_scheduler_col_tile_shape shape = {};
        if (!gtx_scheduler_make_col_tile_shape(node, col0, cols, shape) ||
            node->src[0] == nullptr ||
            node->src[1] == nullptr ||
            !gtx_tensor_remote_f16_element_contiguous(node) ||
            !gtx_scheduler_col_span_strided_rows_supported(node->src[1], col0, cols)) {
            return false;
        }

        // A row-sliced, multi-column destination is strided in DDR. Firmware
        // stash already packs/unpacks raw strided views and exposes a dense L2
        // view to the kernel, so it is safe to retain the computed col tile.
        if (!gtx_scheduler_col_span_strided_rows_supported(node, col0, cols)) {
            return false;
        }
        if (shape.ne2 > 1u) {
            const uint32_t src0_ne2 = (uint32_t)node->src[0]->ne[2];
            if (src0_ne2 == 0u ||
                shape.i2 % src0_ne2 != 0u ||
                shape.ne2 % src0_ne2 != 0u) {
                return false;
            }
        }
        return true;
    }

    if (!gtx_scheduler_col_span_contiguous(node, col0, cols)) {
        return false;
    }

    if (kernel_id == GGML_GTX_KERNEL_SWIGLU_F16) {
        return node->src[0] != nullptr &&
               gtx_scheduler_col_span_contiguous(node->src[0], col0, cols);
    }

    for (uint32_t s = 0; s < 4; ++s) {
        const ggml_tensor * src = node->src[s];
        if (src != nullptr &&
            gtx_tensor_same_shape(node, src) &&
            !gtx_scheduler_col_span_contiguous(src, col0, cols)) {
            return false;
        }
    }
    return true;
}

static bool gtx_scheduler_col_tile_contiguous_for_island(const std::vector<ggml_tensor *> & nodes,
                                                         uint32_t col0,
                                                         uint32_t cols) {
    for (const ggml_tensor * node : nodes) {
        if (!gtx_scheduler_col_tile_contiguous_for_node(node, col0, cols)) {
            return false;
        }
    }
    return true;
}


// ⚠️ **flat 커널의 한계는 원소 수가 아니라 바이트다.** 업로드 커널(relu·clamp·scale 등)은
//    전송 폭을 `(uint16_t)(elems * 2)` 로 만들어, 바이트가 65536 을 넘으면 감긴다.
//    f16 이므로 원소로는 **32768** 이 한계다. 아래 타일러들이 `UINT16_MAX` 를 원소 수로
//    쓰고 있어 정확히 두 배를 허용했고, 그만큼이 조용히 틀린 값으로 나갔다.
//    실측(`gtx_smoke --limits`): relu·clamp·leaky_relu·neg·scale 모두 32768 까지 맞고
//    32769 부터 틀린다. ResNet-18 에서는 argmax 282 → 228 로 나타났다.
//    ADD/MUL 은 4-D 모양을 유지하는 별도 커널이라 이 한계를 안 탄다.
static constexpr uint64_t GTX_FLAT_KERNEL_MAX_ELEMS = 32768u;

static uint32_t gtx_scheduler_max_cols_per_tile(
        const std::vector<ggml_tensor *> & nodes,
        uint32_t total_cols) {
    if (nodes.empty() || nodes[0] == nullptr || total_cols == 0u) {
        return 0u;
    }

    const uint64_t data_budget = gtx_stash_data_budget(nodes);
    const uint64_t slot_budget = (data_budget / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint64_t configured_budget =
        gtx_env_u64_or_default("GTX_TILE_MAX_BYTES", slot_budget);
    const uint64_t view_budget = std::min<uint64_t>(slot_budget, configured_budget);
    if (view_budget == 0u) {
        return 0u;
    }

    // Do not cross logical ne1 boundaries. Crossing into ne2 can select a
    // different head/batch weight matrix and requires a separate raw view.
    uint64_t layout_limit = total_cols;
    uint64_t max_col_bytes = 0u;
    for (const ggml_tensor * node : nodes) {
        if (node == nullptr || node->ne[0] <= 0 || node->ne[1] <= 0) {
            return 0u;
        }
        layout_limit = std::min<uint64_t>(layout_limit, (uint64_t)node->ne[1]);
        const uint32_t kernel_id = gtx_kernel_id_from_node(node);
        if (kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16) {
            layout_limit = std::min<uint64_t>(
                layout_limit,
                gtx_env_u64_or_default("GTX_MUL_MAT_COL_TILE_MAX_COLS", layout_limit));
        }
        const bool flat_kernel = gtx_scheduler_row_tile_kernel_is_flat(kernel_id) &&
                                 kernel_id != GGML_GTX_KERNEL_ADD_F16 &&
                                 kernel_id != GGML_GTX_KERNEL_MUL_F16;
        if (flat_kernel) {
            const uint64_t flat_cols_limit = GTX_FLAT_KERNEL_MAX_ELEMS / (uint64_t)node->ne[0];
            if (flat_cols_limit == 0u) {
                return 0u;
            }
            layout_limit = std::min<uint64_t>(layout_limit, flat_cols_limit);
        }
        max_col_bytes = std::max<uint64_t>(
            max_col_bytes, (uint64_t)node->ne[0] * sizeof(uint16_t));

        if (gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_MUL_MAT_F16) {
            if (node->src[1] == nullptr || node->src[1]->ne[0] <= 0) {
                return 0u;
            }
            max_col_bytes = std::max<uint64_t>(
                max_col_bytes, (uint64_t)node->src[1]->ne[0] * sizeof(uint16_t));
        } else {
            for (uint32_t i = 0; i < 4u; ++i) {
                const ggml_tensor * src = node->src[i];
                if (src != nullptr && src->ne[0] > 0 &&
                    src->ne[1] * src->ne[2] * src->ne[3] ==
                        node->ne[1] * node->ne[2] * node->ne[3]) {
                    max_col_bytes = std::max<uint64_t>(
                        max_col_bytes, (uint64_t)src->ne[0] * sizeof(uint16_t));
                }
            }
        }
    }

    if (max_col_bytes == 0u || view_budget < max_col_bytes) {
        return 1u;
    }
    const uint64_t budget_limit = view_budget / max_col_bytes;
    return (uint32_t)std::max<uint64_t>(
        1u, std::min<uint64_t>(layout_limit, budget_limit));
}

static bool gtx_scheduler_lower_col_island(const std::vector<ggml_tensor *> & nodes,
                                           std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (nodes.empty() || nodes[0] == nullptr) {
        return false;
    }

    const int64_t total_cols64 = nodes[0]->ne[1] * nodes[0]->ne[2] * nodes[0]->ne[3];
    if (total_cols64 <= 0 || total_cols64 > UINT32_MAX) {
        return false;
    }

    bool has_mul_mat = false;
    for (const ggml_tensor * node : nodes) {
        if (!gtx_scheduler_tileable_col_node(node) ||
            node->ne[1] * node->ne[2] * node->ne[3] != total_cols64) {
            return false;
        }
        has_mul_mat = has_mul_mat ||
                      gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_MUL_MAT_F16;
    }
    if (!has_mul_mat) {
        return false;
    }

    const uint32_t total_cols = (uint32_t)total_cols64;
    // Coalesce consecutive N columns up to both the generic stash slot budget
    // and the logical ne1 boundary. The uploaded MUL_MAT kernel consumes the
    // resulting b_row directly; firmware does not split the tile.
    const uint32_t max_cols_per_tile =
        gtx_scheduler_max_cols_per_tile(nodes, total_cols);
    if (max_cols_per_tile == 0u) {
        return false;
    }

    out_descs.clear();
    for (uint32_t col0 = 0; col0 < total_cols;) {
        uint32_t cols = std::min<uint32_t>(max_cols_per_tile, total_cols - col0);
        while (cols > 1u && !gtx_scheduler_col_tile_contiguous_for_island(nodes, col0, cols)) {
            --cols;
        }
        if (!gtx_scheduler_col_tile_contiguous_for_island(nodes, col0, cols)) {
            out_descs.clear();
            return false;
        }
        for (size_t node_idx = 0; node_idx < nodes.size(); ++node_idx) {
            const ggml_tensor * node = nodes[node_idx];
            ggml_gtx_kernel_launch_desc_t desc = {};
            const bool last_node = node_idx + 1 == nodes.size();
            if (last_node &&
                gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_MUL_MAT_F16) {
                const uint32_t rows_per_tile = gtx_scheduler_mul_mat_rows_per_output_tile(nodes, node_idx, cols);
                if (rows_per_tile == 0) {
                    out_descs.clear();
                    return false;
                }
                for (uint32_t row0 = 0; row0 < (uint32_t)node->ne[0]; row0 += rows_per_tile) {
                    const uint32_t rows = std::min<uint32_t>(rows_per_tile, (uint32_t)node->ne[0] - row0);
                    if (!gtx_scheduler_build_mul_mat_col_row_tile_desc(node, col0, cols, row0, rows, desc)) {
                        out_descs.clear();
                        return false;
                    }
                    out_descs.push_back(desc);
                }
            } else {
                if (!gtx_scheduler_build_col_tile_desc(node, col0, cols, desc)) {
                    out_descs.clear();
                    return false;
                }
                out_descs.push_back(desc);
            }
        }
        col0 += cols;
    }
    return !out_descs.empty();
}

static bool gtx_scheduler_find_col_island(ggml_cgraph * cgraph,
                                          int start,
                                          int & end,
                                          std::vector<ggml_tensor *> & nodes) {
    nodes.clear();
    end = start;
    if (cgraph == nullptr || start < 0 || start >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * first = cgraph->nodes[start];
    if (!gtx_scheduler_tileable_col_node(first)) {
        return false;
    }

    bool has_mul_mat = false;
    ggml_tensor * prev = nullptr;
    const int64_t total_cols = first->ne[1] * first->ne[2] * first->ne[3];
    for (int i = start; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!gtx_scheduler_tileable_col_node(node) ||
            node->ne[1] * node->ne[2] * node->ne[3] != total_cols ||
            (prev != nullptr && !gtx_scheduler_node_consumes_prev(node, prev))) {
            break;
        }
        has_mul_mat = has_mul_mat ||
                      gtx_kernel_id_from_node(node) == GGML_GTX_KERNEL_MUL_MAT_F16;
        nodes.push_back(node);
        prev = node;
        end = i + 1;
    }

    return has_mul_mat && !nodes.empty();
}

static bool gtx_scheduler_lower_conv_l2_node(
        const ggml_tensor * node,
        std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr) {
        return false;
    }
    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    const bool depthwise = kernel_id == GGML_GTX_KERNEL_CONV_2D_DW_F16;
    if (!depthwise && kernel_id != GGML_GTX_KERNEL_CONV_2D_F16) {
        return false;
    }
    const bool c_fast = depthwise &&
                        gtx_tensor_remote_f16_c_fast_whc(node) &&
                        gtx_tensor_remote_f16_c_fast_dw_weight(node->src[0]) &&
                        gtx_tensor_remote_f16_c_fast_whc(node->src[1]);

    ggml_gtx_kernel_launch_desc_t base = {};
    if (!gtx_build_kernel_launch_desc(node, base)) {
        return false;
    }
    const int32_t s1 = base.op_params[1];
    const int32_t p1 = base.op_params[3];
    const int32_t d1 = base.op_params[5];
    const uint32_t kh = base.src_ne[0][1];
    if (s1 <= 0 || p1 < 0 || d1 <= 0 || kh == 0 || base.ne[3] != 1u ||
        base.src_ne[1][3] != 1u) {
        return false;
    }
    const uint64_t k_eff = (uint64_t)d1 * (kh - 1u) + 1u;
    const uint64_t data_budget = gtx_stash_data_budget(node);
    const uint64_t slot_budget = (data_budget / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint32_t total_channels = base.ne[2];
    const uint32_t total_oh = base.ne[1];
    const uint32_t iw = base.src_ne[1][0];
    const uint32_t ih = base.src_ne[1][1];
    const uint32_t ic = base.src_ne[1][2];
    const uint32_t kw = base.src_ne[0][0];
    if (slot_budget == 0 || total_channels == 0 || total_oh == 0 ||
        iw == 0 || ih == 0 || ic == 0 || kw == 0) {
        return false;
    }

    const auto aligned = [](uint64_t bytes) {
        return (bytes + 31u) & ~31ull;
    };
    const auto make_tile_shape = [&](uint32_t oh0, uint32_t oh_count,
                                     uint32_t channels,
                                     uint32_t & clip_h0, uint32_t & clip_h,
                                     uint32_t & pad_top,
                                     uint64_t & input_bytes,
                                     uint64_t & weight_bytes,
                                     uint64_t & output_bytes) {
        const int64_t raw_h0 = (int64_t)oh0 * s1 - p1;
        const int64_t raw_h1 = (int64_t)(oh0 + oh_count - 1u) * s1 - p1 + (int64_t)k_eff;
        const int64_t clipped_h0 = std::max<int64_t>(0, raw_h0);
        const int64_t clipped_h1 = std::min<int64_t>((int64_t)ih, raw_h1);
        if (clipped_h1 <= clipped_h0) {
            return false;
        }
        clip_h0 = (uint32_t)clipped_h0;
        clip_h = (uint32_t)(clipped_h1 - clipped_h0);
        pad_top = raw_h0 < 0 ? (uint32_t)(-raw_h0) : 0u;
        const uint64_t tile_ic = depthwise ? channels : ic;
        input_bytes = aligned((uint64_t)iw * clip_h * tile_ic * sizeof(uint16_t));
        weight_bytes = aligned(depthwise ?
            (uint64_t)kw * kh * channels * sizeof(uint16_t) :
            (uint64_t)kw * kh * ic * channels * sizeof(uint16_t));
        output_bytes = aligned((uint64_t)base.ne[0] * oh_count * channels * sizeof(uint16_t));
        return input_bytes <= slot_budget && weight_bytes <= slot_budget &&
               output_bytes <= slot_budget &&
               input_bytes + weight_bytes + output_bytes <= data_budget;
    };

    out_descs.clear();
    for (uint32_t c0 = 0; c0 < total_channels;) {
        uint32_t channels = total_channels - c0;
        uint32_t dummy_h0 = 0, dummy_h = 0, dummy_pad = 0;
        uint64_t input_bytes = 0, weight_bytes = 0, output_bytes = 0;
        while (channels > 0 &&
               !make_tile_shape(0, 1, channels, dummy_h0, dummy_h, dummy_pad,
                                input_bytes, weight_bytes, output_bytes)) {
            channels = channels == 1 ? 0 : (channels + 1u) / 2u;
        }
        if (channels == 0) {
            out_descs.clear();
            return false;
        }

        for (uint32_t oh0 = 0; oh0 < total_oh;) {
            uint32_t oh_count = total_oh - oh0;
            uint32_t clip_h0 = 0, clip_h = 0, pad_top = 0;
            while (oh_count > 0 &&
                   !make_tile_shape(oh0, oh_count, channels,
                                    clip_h0, clip_h, pad_top,
                                    input_bytes, weight_bytes, output_bytes)) {
                oh_count = oh_count == 1 ? 0 : (oh_count + 1u) / 2u;
            }
            if (oh_count == 0) {
                out_descs.clear();
                return false;
            }

            ggml_gtx_kernel_launch_desc_t tile = base;
            tile.dst_addr += (uint64_t)oh0 * base.nb[1] + (uint64_t)c0 * base.nb[2];
            tile.ne[1] = oh_count;
            tile.ne[2] = channels;
            tile.src_addr[1] += (uint64_t)clip_h0 * base.src_nb[1][1];
            tile.src_ne[1][1] = clip_h;
            if (depthwise) {
                tile.src_addr[1] += (uint64_t)c0 * base.src_nb[1][2];
                tile.src_ne[1][2] = channels;
                tile.src_addr[0] += (uint64_t)c0 * base.src_nb[0][3];
                tile.src_ne[0][3] = channels;
            } else {
                tile.src_addr[0] += (uint64_t)c0 * base.src_nb[0][3];
                tile.src_ne[0][3] = channels;
            }
            tile.op_params[3] = (int32_t)pad_top;
            if (c_fast) {
                tile.op_params[GGML_GTX_CONV_LAYOUT_PARAM] = GGML_GTX_CONV_LAYOUT_C_FAST;
                tile.op_params[GGML_GTX_CONV_LOGICAL_OW_PARAM] = (int32_t)base.ne[0];
                tile.op_params[GGML_GTX_CONV_LOGICAL_OH_PARAM] = (int32_t)oh_count;
                tile.op_params[GGML_GTX_CONV_LOGICAL_C_PARAM] = (int32_t)channels;
                tile.op_params[GGML_GTX_CONV_LOGICAL_IW_PARAM] = (int32_t)iw;
                tile.op_params[GGML_GTX_CONV_LOGICAL_IH_PARAM] = (int32_t)clip_h;
                tile.op_params[GGML_GTX_CONV_LOGICAL_KW_PARAM] = (int32_t)kw;
                tile.op_params[GGML_GTX_CONV_LOGICAL_KH_PARAM] = (int32_t)kh;

                // Re-express each C-fast remote tile with C as raw ne0. The
                // generic stash mapper can then pack it without op knowledge;
                // the DW kernel reconstructs the logical WHC view in scratch.
                tile.ne[0] = channels;
                tile.ne[1] = base.ne[0];
                tile.ne[2] = oh_count;
                tile.ne[3] = 1u;
                tile.nb[0] = base.nb[2];
                tile.nb[1] = base.nb[0];
                tile.nb[2] = base.nb[1];
                tile.nb[3] = base.nb[3];

                tile.src_ne[1][0] = channels;
                tile.src_ne[1][1] = iw;
                tile.src_ne[1][2] = clip_h;
                tile.src_ne[1][3] = 1u;
                tile.src_nb[1][0] = base.src_nb[1][2];
                tile.src_nb[1][1] = base.src_nb[1][0];
                tile.src_nb[1][2] = base.src_nb[1][1];
                tile.src_nb[1][3] = base.src_nb[1][3];

                tile.src_ne[0][0] = channels;
                tile.src_ne[0][1] = kw;
                tile.src_ne[0][2] = kh;
                tile.src_ne[0][3] = 1u;
                tile.src_nb[0][0] = base.src_nb[0][3];
                tile.src_nb[0][1] = base.src_nb[0][0];
                tile.src_nb[0][2] = base.src_nb[0][1];
                tile.src_nb[0][3] = base.src_nb[0][2];
            }
            tile.l2_scratch_size = 8u * 1024u * 1024u;
            out_descs.push_back(tile);
            oh0 += oh_count;
        }
        c0 += channels;
    }
    return !out_descs.empty();
}

static bool gtx_scheduler_lower_conv_transpose_l2_node(
        const ggml_tensor * node,
        std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        gtx_kernel_id_from_node(node) != GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16 ||
        node->ne[0] <= 0 || node->ne[1] <= 0 || node->ne[2] <= 0 || node->ne[3] <= 0) {
        return false;
    }

    ggml_gtx_kernel_launch_desc_t base = {};
    if (!gtx_build_kernel_launch_desc(node, base)) {
        return false;
    }

    // The L2-only kernel emits one packed dot-product sequence per output.
    // Keep that command stream at or below the largest focused shape already
    // verified on Spike.  This is a host macro tile: firmware still only maps
    // raw views, and the kernel retains its own SPU-level output partitioning.
    constexpr uint64_t max_outputs_per_tile = 256u * 1024u;
    const uint64_t outputs_per_channel =
        (uint64_t)base.ne[0] * base.ne[1] * base.ne[3];
    if (outputs_per_channel == 0u || outputs_per_channel > max_outputs_per_tile) {
        return false;
    }
    const uint32_t max_channels = (uint32_t)std::max<uint64_t>(
        1u, max_outputs_per_tile / outputs_per_channel);

    out_descs.clear();
    for (uint32_t c0 = 0; c0 < base.ne[2]; c0 += max_channels) {
        const uint32_t channels = std::min<uint32_t>(max_channels, base.ne[2] - c0);
        ggml_gtx_kernel_launch_desc_t tile = base;
        tile.dst_addr += (uint64_t)c0 * base.nb[2];
        tile.ne[2] = channels;
        tile.src_addr[0] += (uint64_t)c0 * base.src_nb[0][2];
        tile.src_ne[0][2] = channels;
        out_descs.push_back(tile);
    }
    return !out_descs.empty();
}

static bool gtx_scheduler_lower_spatial_scalar_rhs_node(
        const ggml_tensor * node,
        std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (node == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        (gtx_kernel_id_from_node(node) != GGML_GTX_KERNEL_ADD_F16 &&
         gtx_kernel_id_from_node(node) != GGML_GTX_KERNEL_MUL_F16) ||
        !gtx_l2_rhs_broadcast_shape_ok(node, true) ||
        node->src[1]->ne[0] != 1 || node->ne[3] != 1 ||
        node->src[1]->ne[1] != 1 || node->src[1]->ne[3] != 1) {
        return false;
    }

    ggml_gtx_kernel_launch_desc_t base = {};
    if (!gtx_build_kernel_launch_desc(node, base)) {
        return false;
    }
    const uint64_t data_budget = gtx_stash_data_budget(node);
    const uint64_t slot_budget = (data_budget / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint32_t width = base.ne[0];
    const uint32_t height = base.ne[1];
    const uint32_t channels_total = base.ne[2];
    const auto aligned = [](uint64_t bytes) { return (bytes + 31u) & ~31ull; };
    const auto fits = [&](uint32_t rows, uint32_t channels) {
        const uint64_t data_bytes = aligned((uint64_t)width * rows * channels * sizeof(uint16_t));
        const uint64_t rhs_channels = base.src_ne[1][2] == 1u ? 1u : channels;
        const uint64_t rhs_bytes = aligned(rhs_channels * sizeof(uint16_t));
        return data_bytes > 0 && data_bytes <= slot_budget && rhs_bytes <= slot_budget &&
               data_bytes * 2u + rhs_bytes <= data_budget;
    };

    out_descs.clear();
    for (uint32_t c0 = 0; c0 < channels_total;) {
        uint32_t channels = channels_total - c0;
        while (channels > 0 && !fits(1u, channels)) {
            channels = channels == 1u ? 0u : (channels + 1u) / 2u;
        }
        if (channels == 0u) {
            out_descs.clear();
            return false;
        }
        for (uint32_t y0 = 0; y0 < height;) {
            uint32_t rows = height - y0;
            while (rows > 0 && !fits(rows, channels)) {
                rows = rows == 1u ? 0u : (rows + 1u) / 2u;
            }
            if (rows == 0u) {
                out_descs.clear();
                return false;
            }
            ggml_gtx_kernel_launch_desc_t tile = base;
            const uint64_t data_off = (uint64_t)y0 * base.nb[1] + (uint64_t)c0 * base.nb[2];
            tile.dst_addr += data_off;
            tile.src_addr[0] += data_off;
            tile.ne[1] = rows;
            tile.ne[2] = channels;
            tile.src_ne[0][1] = rows;
            tile.src_ne[0][2] = channels;
            tile.src_ne[1][1] = 1u;
            if (base.src_ne[1][2] != 1u) {
                tile.src_addr[1] += (uint64_t)c0 * base.src_nb[1][2];
                tile.src_ne[1][2] = channels;
            }
            out_descs.push_back(tile);
            y0 += rows;
        }
        c0 += channels;
    }
    return !out_descs.empty();
}

static bool gtx_scheduler_lower_spatial_dense_node(
        const ggml_tensor * node,
        std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (node == nullptr || node->src[0] == nullptr || node->ne[3] != 1 ||
        !gtx_tensor_remote_f16_dense(node) ||
        !gtx_tensor_same_shape(node, node->src[0])) {
        return false;
    }
    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    const bool supported =
        kernel_id == GGML_GTX_KERNEL_ADD_F16 || kernel_id == GGML_GTX_KERNEL_MUL_F16 ||
        kernel_id == GGML_GTX_KERNEL_SCALE_F16 || kernel_id == GGML_GTX_KERNEL_GELU_F16 ||
        kernel_id == GGML_GTX_KERNEL_ABS_F16 || kernel_id == GGML_GTX_KERNEL_NEG_F16 ||
        kernel_id == GGML_GTX_KERNEL_SGN_F16 || kernel_id == GGML_GTX_KERNEL_CEIL_F16 ||
        kernel_id == GGML_GTX_KERNEL_FLOOR_F16 || kernel_id == GGML_GTX_KERNEL_TANH_F16 ||
        kernel_id == GGML_GTX_KERNEL_SILU_F16 || kernel_id == GGML_GTX_KERNEL_RELU_F16 ||
        kernel_id == GGML_GTX_KERNEL_LEAKY_RELU_F16 || kernel_id == GGML_GTX_KERNEL_CLAMP_F16;
    if (!supported) {
        return false;
    }
    for (uint32_t i = 0; i < 4u; ++i) {
        const ggml_tensor * src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        if (!gtx_tensor_remote_f16_dense(src) || src->ne[3] != 1) {
            return false;
        }
        for (uint32_t d = 0; d < 3u; ++d) {
            if (src->ne[d] != 1 && src->ne[d] != node->ne[d]) {
                return false;
            }
        }
    }

    ggml_gtx_kernel_launch_desc_t base = {};
    if (!gtx_build_kernel_launch_desc(node, base)) {
        return false;
    }
    const uint64_t data_budget = gtx_stash_data_budget(node);
    const uint64_t slot_budget = (data_budget / GTX_STASH_SLOT_COUNT) & ~31ull;
    const uint32_t width = base.ne[0];
    const uint32_t height = base.ne[1];
    const uint32_t channels_total = base.ne[2];
    // Unary/scale uploaded kernels consume one flat vector through ne[0].
    // ADD/MUL are deliberately excluded: their kernels retain the 4-D shape
    // to implement per-dimension broadcasting.
    const bool flat_kernel = gtx_scheduler_row_tile_kernel_is_flat(kernel_id) &&
                             kernel_id != GGML_GTX_KERNEL_ADD_F16 &&
                             kernel_id != GGML_GTX_KERNEL_MUL_F16;
    const auto aligned = [](uint64_t bytes) { return (bytes + 31u) & ~31ull; };
    const auto tile_bytes = [&](const ggml_tensor * tensor, uint32_t rows, uint32_t channels) {
        const uint64_t w = tensor->ne[0] == 1 ? 1u : width;
        const uint64_t h = tensor->ne[1] == 1 ? 1u : rows;
        const uint64_t c = tensor->ne[2] == 1 ? 1u : channels;
        return aligned(w * h * c * sizeof(uint16_t));
    };
    const auto fits = [&](uint32_t rows, uint32_t channels) {
        const uint64_t logical_elems = (uint64_t)width * rows * channels;
        if (flat_kernel && logical_elems > GTX_FLAT_KERNEL_MAX_ELEMS) {
            return false;
        }
        uint64_t total = aligned((uint64_t)width * rows * channels * sizeof(uint16_t));
        if (total == 0 || total > slot_budget) {
            return false;
        }
        for (uint32_t i = 0; i < 4u; ++i) {
            if (node->src[i] == nullptr) {
                continue;
            }
            const uint64_t bytes = tile_bytes(node->src[i], rows, channels);
            if (bytes == 0 || bytes > slot_budget || total > UINT64_MAX - bytes) {
                return false;
            }
            total += bytes;
        }
        return total <= data_budget;
    };

    // 평평한 1-D 입력은 아래 공간 타일러가 쪼갤 행도 채널도 없다. `ne[0]` 을 직접
    // 잘라 두 뷰의 주소를 함께 전진시킨다.
    // ⚠️ **이 경로가 SCALE 에만 열려 있었다.** 같은 ABI 를 쓰는 relu·clamp·leaky_relu·
    //    neg 등은 1-D 가 커지면 그대로 한 번에 나가 조용히 틀렸다
    //    (실측: 32769 원소부터. `gtx_smoke --limits`). flat 커널 전체로 넓힌다.
    if (flat_kernel &&
        height == 1u && channels_total == 1u && width > GTX_FLAT_KERNEL_MAX_ELEMS) {
        constexpr uint32_t max_aligned_f16_elems = (uint32_t) GTX_FLAT_KERNEL_MAX_ELEMS & ~15u;
        out_descs.clear();
        for (uint32_t x0 = 0; x0 < width;) {
            uint32_t elems = std::min<uint32_t>(width - x0, max_aligned_f16_elems);
            while (elems > 0u) {
                const uint64_t bytes = aligned((uint64_t)elems * sizeof(uint16_t));
                if (bytes <= slot_budget && bytes <= data_budget / 2u) {
                    break;
                }
                elems /= 2u;
            }
            if (elems == 0u) {
                out_descs.clear();
                return false;
            }

            ggml_gtx_kernel_launch_desc_t tile = base;
            tile.dst_addr += (uint64_t)x0 * base.nb[0];
            tile.src_addr[0] += (uint64_t)x0 * base.src_nb[0][0];
            gtx_scheduler_make_dense_tile_shape(tile, elems, 1u);
            gtx_scheduler_make_dense_src_tile_shape(tile, 0u, elems, 1u);
            out_descs.push_back(tile);
            x0 += elems;
        }
        return !out_descs.empty();
    }

    out_descs.clear();
    for (uint32_t c0 = 0; c0 < channels_total;) {
        uint32_t channels = channels_total - c0;
        if (flat_kernel) {
            const uint64_t plane_elems = (uint64_t)width * height;
            // A flattened descriptor must also describe a physically
            // contiguous raw DDR view.  Use complete channel planes when a
            // plane fits the flat-kernel ABI; otherwise restrict the tile to
            // one channel and split that plane by rows below.
            if (plane_elems <= GTX_FLAT_KERNEL_MAX_ELEMS) {
                channels = std::min<uint32_t>(
                    channels, (uint32_t)(UINT16_MAX / plane_elems));
            } else {
                channels = 1u;
            }
        }
        while (channels > 0 && !fits(1u, channels)) {
            channels = channels == 1u ? 0u : (channels + 1u) / 2u;
        }
        if (channels == 0u) {
            out_descs.clear();
            return false;
        }
        for (uint32_t y0 = 0; y0 < height;) {
            uint32_t rows = height - y0;
            while (rows > 0 && !fits(rows, channels)) {
                rows = rows == 1u ? 0u : (rows + 1u) / 2u;
            }
            if (rows == 0u) {
                out_descs.clear();
                return false;
            }
            ggml_gtx_kernel_launch_desc_t tile = base;
            tile.dst_addr += (uint64_t)y0 * base.nb[1] + (uint64_t)c0 * base.nb[2];
            tile.ne[1] = rows;
            tile.ne[2] = channels;
            for (uint32_t i = 0; i < 4u; ++i) {
                const ggml_tensor * src = node->src[i];
                if (src == nullptr) {
                    continue;
                }
                if (src->ne[1] != 1) {
                    tile.src_addr[i] += (uint64_t)y0 * base.src_nb[i][1];
                    tile.src_ne[i][1] = rows;
                }
                if (src->ne[2] != 1) {
                    tile.src_addr[i] += (uint64_t)c0 * base.src_nb[i][2];
                    tile.src_ne[i][2] = channels;
                }
            }
            if (flat_kernel) {
                const uint64_t logical_elems = (uint64_t)width * rows * channels;
                if (logical_elems == 0u || logical_elems > UINT16_MAX) {
                    out_descs.clear();
                    return false;
                }
                gtx_scheduler_make_dense_tile_shape(tile, (uint32_t)logical_elems, 1u);
                for (uint32_t i = 0; i < 4u; ++i) {
                    if (node->src[i] != nullptr && tile.src_addr[i] != 0u) {
                        gtx_scheduler_make_dense_src_tile_shape(
                            tile, i, (uint32_t)logical_elems, 1u);
                    }
                }
            }
            out_descs.push_back(tile);
            y0 += rows;
        }
        c0 += channels;
    }
    return !out_descs.empty();
}

/* Lower a single L2-only kernel without relying on firmware-side identity.
 * This is also used when the multi-node scheduler is disabled: the host still
 * has to provide a raw L2 plan for an uploaded L2-only executable. */
static bool gtx_scheduler_lower_single_l2_node(const ggml_tensor * node,
                                               std::vector<ggml_gtx_kernel_launch_desc_t> & out_descs) {
    if (node == nullptr) {
        return false;
    }

    const uint32_t kernel_id = gtx_kernel_id_from_node(node);
    if ((kernel_id == GGML_GTX_KERNEL_CONV_2D_F16 ||
         kernel_id == GGML_GTX_KERNEL_CONV_2D_DW_F16) &&
        gtx_scheduler_lower_conv_l2_node(node, out_descs)) {
        return true;
    }
    if (kernel_id == GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16 &&
        gtx_scheduler_lower_conv_transpose_l2_node(node, out_descs)) {
        return true;
    }
    if (gtx_scheduler_lower_spatial_dense_node(node, out_descs)) {
        return true;
    }
    if (gtx_scheduler_lower_spatial_scalar_rhs_node(node, out_descs)) {
        return true;
    }

    std::vector<ggml_tensor *> one = { const_cast<ggml_tensor *>(node) };
    if (gtx_scheduler_tileable_row_node(node) &&
        gtx_scheduler_lower_row_island(one, out_descs)) {
        return true;
    }

    // PAD has only two simultaneously-live tensor views (src0 and dst).
    // Do not infer a fixed per-slot capacity from the generic three-view
    // budget: the host does not own the firmware stash allocator or know its
    // current residency/fragmentation.  Admit the whole raw view when its
    // aggregate live bytes fit; firmware remains authoritative for mapping,
    // eviction, and reclaim/reset retry.
    if (kernel_id == GGML_GTX_KERNEL_PAD_F16 && node->src[0] != nullptr) {
        const uint64_t data_budget = gtx_stash_data_budget(node);
        const uint64_t dst_bytes = gtx_remote_size_bytes(node);
        const uint64_t src_bytes = gtx_remote_size_bytes(node->src[0]);
        const uint64_t scratch_bytes = gtx_kernel_l2_scratch_size(node);
        ggml_gtx_kernel_launch_desc_t desc = {};
        if (dst_bytes > 0u && src_bytes > 0u &&
            dst_bytes <= data_budget && src_bytes <= data_budget &&
            scratch_bytes <= data_budget &&
            dst_bytes <= UINT64_MAX - src_bytes &&
            dst_bytes + src_bytes <= UINT64_MAX - scratch_bytes &&
            dst_bytes + src_bytes + scratch_bytes <= data_budget &&
            gtx_build_kernel_launch_desc(node, desc)) {
            out_descs = { desc };
            return true;
        }
        return false;
    }

    // Binary broadcast is a kernel-side ABI feature.  Keep host lowering
    // generic: submit one raw macro view and preserve src1's smaller shape;
    // stash packs it and the uploaded ADD/MUL kernel selects the broadcast
    // index from src_ne[1], without host-side row launches.
    const uint64_t slot_budget = (gtx_stash_data_budget(node) / GTX_STASH_SLOT_COUNT) & ~31ull;
    if ((kernel_id == GGML_GTX_KERNEL_REPEAT_F16 ||
         kernel_id == GGML_GTX_KERNEL_CONCAT_F16 ||
         kernel_id == GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16 ||
         kernel_id == GGML_GTX_KERNEL_CONV_2D_F16 ||
         kernel_id == GGML_GTX_KERNEL_CONV_2D_DW_F16 ||
         kernel_id == GGML_GTX_KERNEL_SIN_F16 ||
         kernel_id == GGML_GTX_KERNEL_COS_F16)) {
        const uint64_t dst_bytes = gtx_remote_size_bytes(node);
        uint64_t total_bytes = dst_bytes;
        bool fits = dst_bytes > 0u && dst_bytes <= slot_budget;
        for (uint32_t i = 0; i < 4u && fits; ++i) {
            if (node->src[i] == nullptr) {
                continue;
            }
            const uint64_t src_bytes = gtx_remote_size_bytes(node->src[i]);
            if (src_bytes == 0u || src_bytes > slot_budget ||
                total_bytes > UINT64_MAX - src_bytes) {
                fits = false;
                break;
            }
            total_bytes += src_bytes;
        }
        ggml_gtx_kernel_launch_desc_t desc = {};
        if (fits && total_bytes <= gtx_stash_data_budget(node) &&
            gtx_build_kernel_launch_desc(node, desc)) {
            out_descs = { desc };
            return true;
        }
        return false;
    }
    if ((kernel_id == GGML_GTX_KERNEL_ADD_F16 ||
         kernel_id == GGML_GTX_KERNEL_MUL_F16) &&
        node->src[0] != nullptr && node->src[1] != nullptr &&
        gtx_tensor_remote_f16_vector_contiguous(node) &&
        gtx_tensor_remote_f16_vector_contiguous(node->src[0]) &&
        gtx_tensor_remote_f16_vector_contiguous(node->src[1]) &&
        gtx_tensor_same_shape(node, node->src[0]) &&
        (node->src[1]->ne[0] == node->ne[0] || node->src[1]->ne[0] == 1) &&
        (uint64_t) gtx_remote_size_bytes(node) <= slot_budget &&
        (uint64_t) gtx_remote_size_bytes(node->src[0]) <= slot_budget &&
        (uint64_t) gtx_remote_size_bytes(node->src[1]) <= slot_budget) {
        bool broadcastable = true;
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            broadcastable = broadcastable &&
                (node->src[1]->ne[d] == 1 || node->src[1]->ne[d] == node->ne[d]);
        }
        ggml_gtx_kernel_launch_desc_t desc = {};
        if (broadcastable && gtx_build_kernel_launch_desc(node, desc)) {
            out_descs = { desc };
            return true;
        }
    }

    if (!gtx_scheduler_tileable_col_node(node)) {
        return false;
    }
    if (kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16) {
        return gtx_scheduler_lower_col_island(one, out_descs);
    }
    const int64_t cols64 = node->ne[1] * node->ne[2] * node->ne[3];
    if (cols64 <= 0 || cols64 > UINT32_MAX) {
        return false;
    }

    const uint32_t total_cols = (uint32_t)cols64;
    // Standalone ROPE used to force one descriptor per flattened column
    // (effectively one launch per head and token).  Coalesce columns within
    // the existing ne1/layout and stash-budget limits.  Keep the legacy
    // single-column behavior for the other standalone column kernels.
    const uint32_t max_cols_per_tile =
        kernel_id == GGML_GTX_KERNEL_ROPE_F16 ?
            gtx_scheduler_max_cols_per_tile(one, total_cols) : 1u;
    if (max_cols_per_tile == 0u) {
        return false;
    }

    out_descs.clear();
    for (uint32_t col = 0; col < total_cols;) {
        uint32_t cols = std::min<uint32_t>(max_cols_per_tile, total_cols - col);
        while (cols > 1u &&
               !gtx_scheduler_col_tile_contiguous_for_node(node, col, cols)) {
            --cols;
        }
        ggml_gtx_kernel_launch_desc_t desc = {};
        if (!gtx_scheduler_col_tile_contiguous_for_node(node, col, cols) ||
            !gtx_scheduler_build_col_tile_desc(node, col, cols, desc)) {
            out_descs.clear();
            return false;
        }
        out_descs.push_back(desc);
        col += cols;
    }
    return !out_descs.empty();
}

static enum ggml_status gtx_compute_node_host_driven(ggml_backend_gtx_context * ctx,
                                                     ggml_tensor * node,
                                                     int node_index,
                                                     bool request_stash_chain = false) {
    if (node == nullptr || gtx_is_metadata_op(node->op)) {
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_nelements(node) == 0) {
        return GGML_STATUS_SUCCESS;
    }

    gtx_log_op_shapes(node);
    if (node->op == GGML_OP_SOFT_MAX) {
        gtx_log_softmax_tensor_stats(ctx, node->src[0], "pre-src0");
        gtx_log_softmax_tensor_stats(ctx, node->src[1], "pre-mask");
    }

    if (node->op == GGML_OP_CPY && !ctx->use_uart) {
        if (node->src[0] == nullptr || !gtx_copy_tensor_rpc(ctx->conn.get(), node->src[0], node)) {
            ctx->last_error = "COPY_TENSOR failed";
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    ggml_gtx_kernel_launch_desc_t desc = {};
    if (gtx_build_kernel_launch_desc(node, desc)) {
        const gtx_l2_chainability chain = gtx_l2_chainability_from_node(node);
        if (chain.kind == gtx_l2_chain_kind::native ||
            desc.kernel_id == GGML_GTX_KERNEL_MUL_MAT_F16 ||
            desc.kernel_id == GGML_GTX_KERNEL_CONV_TRANSPOSE_2D_F16) {
            std::vector<ggml_gtx_kernel_launch_desc_t> l2_descs;
            if (gtx_scheduler_lower_single_l2_node(node, l2_descs)) {
                for (ggml_gtx_kernel_launch_desc_t & l2_desc : l2_descs) {
                    l2_desc.flags &= ~GGML_GTX_DESC_FLAG_DDR_ALLOWED;
                    l2_desc.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN |
                                     GGML_GTX_DESC_FLAG_STASH_HOST_TILED |
                                     GGML_GTX_DESC_FLAG_L2_REQUIRED;
                }
                gtx_dump_op_sample_before_launch(ctx, node);
                const gtx_op_check_pre_state check_pre = gtx_check_ops_before_launch(ctx, node);
                if (gtx_submit_launch_descs(ctx, l2_descs)) {
                    if (!gtx_diagnostic_materialize_stash(ctx, node)) {
                        ctx->last_error = "diagnostic stash flush/reset failed";
                        return GGML_STATUS_FAILED;
                    }
                    gtx_diagnostic_report_first_nonfinite(ctx, node);
                    gtx_check_mul_mat_after_launch(ctx, node);
                    gtx_check_ops_after_launch(ctx, node, check_pre);
                    gtx_dump_op_sample_after_launch(ctx, node);
                    if (gtx_cpu_ref_exists(node_index) && ctx->use_uart && ctx->uart != nullptr) {
                        if (!ctx->uart->stash_flush_reset()) {
                            ctx->last_error = "CPU reference stash materialization failed";
                            return GGML_STATUS_FAILED;
                        }
                        (void)gtx_compare_cpu_reference_after_launch(ctx, node, node_index);
                    }
                    return GGML_STATUS_SUCCESS;
                }
                ctx->last_error = "L2-only descriptor stream launch failed";
                return GGML_STATUS_FAILED;
            }
            GTX_LOG_ERROR("host cannot lower L2-only kernel into raw L2 views: %s "
                          "dst=[%lld,%lld,%lld,%lld] src0=[%lld,%lld,%lld,%lld] "
                          "src1=[%lld,%lld,%lld,%lld]\n",
                          gtx_tensor_op_display_name(node),
                          (long long) node->ne[0], (long long) node->ne[1],
                          (long long) node->ne[2], (long long) node->ne[3],
                          node->src[0] ? (long long) node->src[0]->ne[0] : 0,
                          node->src[0] ? (long long) node->src[0]->ne[1] : 0,
                          node->src[0] ? (long long) node->src[0]->ne[2] : 0,
                          node->src[0] ? (long long) node->src[0]->ne[3] : 0,
                          node->src[1] ? (long long) node->src[1]->ne[0] : 0,
                          node->src[1] ? (long long) node->src[1]->ne[1] : 0,
                          node->src[1] ? (long long) node->src[1]->ne[2] : 0,
                          node->src[1] ? (long long) node->src[1]->ne[3] : 0);
            ctx->last_error = "L2-only kernel lacks host raw-view lowering";
            return GGML_STATUS_FAILED;
        }
        if (request_stash_chain) {
            desc.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN;
        }
        gtx_dump_op_sample_before_launch(ctx, node);
        const gtx_op_check_pre_state check_pre = gtx_check_ops_before_launch(ctx, node);
        if (ctx->use_uart ? gtx_uart_launch_desc(*ctx->uart, desc) : gtx_submit_launch_desc(ctx, desc)) {
            if (!gtx_diagnostic_materialize_stash(ctx, node)) {
                ctx->last_error = "diagnostic stash flush/reset failed";
                return GGML_STATUS_FAILED;
            }
            gtx_diagnostic_report_first_nonfinite(ctx, node);
            gtx_check_mul_mat_after_launch(ctx, node);
            gtx_check_ops_after_launch(ctx, node, check_pre);
            if (node->op == GGML_OP_SOFT_MAX) {
                gtx_log_softmax_tensor_stats(ctx, node, "post-dst");
            }
            gtx_dump_op_sample_after_launch(ctx, node);
            if (gtx_cpu_ref_exists(node_index) && ctx->use_uart && ctx->uart != nullptr) {
                if (!ctx->uart->stash_flush_reset()) {
                    ctx->last_error = "CPU reference stash materialization failed";
                    return GGML_STATUS_FAILED;
                }
                (void)gtx_compare_cpu_reference_after_launch(ctx, node, node_index);
            }
            return GGML_STATUS_SUCCESS;
        }
    }

    GTX_LOG_ERROR("host-driven graph failed op: %s (%d)\n",
                  gtx_tensor_op_display_name(node), (int) node->op);
    ctx->last_error = "Graph op launch failed";
    return GGML_STATUS_FAILED;
}

static enum ggml_status gtx_compute_graph_tiling_scheduler(ggml_backend_gtx_context * ctx,
                                                           ggml_cgraph * cgraph) {
    if (ctx == nullptr || cgraph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes;) {
        ggml_tensor * node = cgraph->nodes[i];
        if (gtx_scheduler_node_skippable(node)) {
            ++i;
            continue;
        }

        int island_end = i;
        std::vector<ggml_tensor *> island_nodes;
        std::vector<ggml_gtx_kernel_launch_desc_t> descs;
        if (gtx_scheduler_find_col_island(cgraph, i, island_end, island_nodes) &&
            gtx_scheduler_lower_col_island(island_nodes, descs)) {
            for (ggml_gtx_kernel_launch_desc_t & desc : descs) {
                // The host has already lowered the matrix view into one
                // contiguous L2 tile with its raw shape/stride metadata.
                // The kernel, not firmware, owns the ensuing L1/SPU split.
                desc.flags &= ~GGML_GTX_DESC_FLAG_DDR_ALLOWED;
                desc.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN |
                              GGML_GTX_DESC_FLAG_STASH_HOST_TILED |
                              GGML_GTX_DESC_FLAG_L2_REQUIRED;
            }
            if (gtx_log_chain_ops_enabled()) {
                const long long total_cols =
                    (long long)island_nodes[0]->ne[1] *
                    (long long)island_nodes[0]->ne[2] *
                    (long long)island_nodes[0]->ne[3];
                const size_t groups = island_nodes.empty() ? 0 : descs.size() / island_nodes.size();
                const size_t rem = island_nodes.empty() ? 0 : descs.size() % island_nodes.size();
                fprintf(stderr,
                        "[GTX TILING] col-island nodes=%zu descs=%zu groups=%zu rem=%zu cols=%lld start=%d end=%d\n",
                        island_nodes.size(),
                        descs.size(),
                        groups,
                        rem,
                        total_cols,
                        i,
                        island_end);
            }
            if (!gtx_submit_launch_descs(ctx, descs)) {
                ctx->last_error = "Tiled descriptor stream launch failed";
                return GGML_STATUS_FAILED;
            }
            bool has_reference = false;
            for (int node_index = i; node_index < island_end; ++node_index) {
                has_reference = has_reference || gtx_cpu_ref_exists(node_index);
            }
            if (has_reference && ctx->use_uart && ctx->uart != nullptr) {
                if (!ctx->uart->stash_flush_reset()) {
                    ctx->last_error = "CPU reference island materialization failed";
                    return GGML_STATUS_FAILED;
                }
                for (int node_index = i; node_index < island_end; ++node_index) {
                    if (gtx_cpu_ref_exists(node_index)) {
                        (void)gtx_compare_cpu_reference_after_launch(
                            ctx, cgraph->nodes[node_index], node_index);
                    }
                }
            }
            i = island_end;
            continue;
        }

        if (gtx_scheduler_find_row_island(cgraph, i, island_end, island_nodes) &&
            gtx_scheduler_lower_row_island(island_nodes, descs)) {
            for (ggml_gtx_kernel_launch_desc_t & desc : descs) {
                desc.flags &= ~GGML_GTX_DESC_FLAG_DDR_ALLOWED;
                desc.flags |= GGML_GTX_DESC_FLAG_STASH_CHAIN |
                              GGML_GTX_DESC_FLAG_STASH_HOST_TILED |
                              GGML_GTX_DESC_FLAG_L2_REQUIRED;
            }
            if (gtx_log_chain_ops_enabled()) {
                fprintf(stderr,
                        "[GTX TILING] row-island nodes=%zu descs=%zu rows=%lld ne0=%lld start=%d end=%d\n",
                        island_nodes.size(),
                        descs.size(),
                        (long long)island_nodes[0]->ne[1],
                        (long long)island_nodes[0]->ne[0],
                        i,
                        island_end);
            }
            if (!gtx_submit_launch_descs(ctx, descs)) {
                ctx->last_error = "Tiled descriptor stream launch failed";
                return GGML_STATUS_FAILED;
            }
            bool has_reference = false;
            for (int node_index = i; node_index < island_end; ++node_index) {
                has_reference = has_reference || gtx_cpu_ref_exists(node_index);
            }
            if (has_reference && ctx->use_uart && ctx->uart != nullptr) {
                if (!ctx->uart->stash_flush_reset()) {
                    ctx->last_error = "CPU reference island materialization failed";
                    return GGML_STATUS_FAILED;
                }
                for (int node_index = i; node_index < island_end; ++node_index) {
                    if (gtx_cpu_ref_exists(node_index)) {
                        (void)gtx_compare_cpu_reference_after_launch(
                            ctx, cgraph->nodes[node_index], node_index);
                    }
                }
            }
            i = island_end;
            continue;
        }

        enum ggml_status status = gtx_compute_node_host_driven(ctx, node, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }

        ++i;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_buffer_type_i ggml_backend_gtx_buffer_type_interface = {
    /* .get_name      = */ ggml_backend_gtx_buffer_type_get_name,
    /* .alloc_buffer  = */ ggml_backend_gtx_buffer_type_alloc_buffer,
    /* .get_alignment = */ ggml_backend_gtx_buffer_type_get_alignment,
    /* .get_max_size  = */ ggml_backend_gtx_buffer_type_get_max_size,
    /* .get_alloc_size= */ ggml_backend_gtx_buffer_type_get_alloc_size,
    /* .is_host       = */ ggml_backend_gtx_buffer_type_is_host,
};

/*=============================================================================
 * Buffer Implementation
 *============================================================================*/

static void ggml_backend_gtx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;

    if (ctx->conn && ctx->remote_ptr != 0) {
        gtx_msg_free_buffer_req req = { ctx->remote_ptr };
        ctx->conn->send_cmd(GTX_CMD_FREE_BUFFER, &req, sizeof(req), nullptr, 0);
    }
    if (ctx->backend_ctx != nullptr && ctx->remote_ptr != 0) {
        ctx->backend_ctx->allocated_buffers.erase(ctx->remote_ptr);
    }

    delete ctx;
}

static void* ggml_backend_gtx_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;

    /* Return remote pointer as base (will be used as offset) */
    return (void*)(uintptr_t)ctx->remote_ptr;
}

static void ggml_backend_gtx_buffer_memset_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
    uint8_t value, size_t offset, size_t size) {

    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;
    GGML_UNUSED(size);

    if (ctx->backend_ctx != nullptr && ctx->backend_ctx->use_uart) {
        std::vector<uint8_t> zeros(gtx_remote_byte_offset(tensor, size), value);
        ctx->backend_ctx->uart->upload_bytes(gtx_remote_data_ptr(tensor) + gtx_remote_byte_offset(tensor, offset),
                                             zeros.data(), zeros.size());
    } else if (ctx->conn) {
        gtx_msg_buffer_clear_req req;
        req.remote_ptr = (uint64_t)(uintptr_t)tensor->data + offset;
        req.value = value;
        ctx->conn->send_cmd(GTX_CMD_BUFFER_CLEAR, &req, sizeof(req), nullptr, 0);
    }
}

static void ggml_backend_gtx_buffer_set_tensor(
    ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
    const void* data, size_t offset, size_t size) {

    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;

    GTX_LOG("set_tensor: %s, offset=%zu, size=%zu\n", tensor->name, offset, size);

    if (ctx->backend_ctx != nullptr && ctx->backend_ctx->use_uart) {
        const size_t chunk_size = gtx_upload_chunk_size();
        size_t sent = 0;

        while (sent < size) {
            size_t chunk = gtx_upload_local_chunk_size(tensor, size - sent, chunk_size);
            std::vector<uint8_t> packed = gtx_pack_tensor_chunk(tensor, (const uint8_t *) data + sent, chunk);
            uint64_t remote = gtx_remote_data_ptr(tensor) + gtx_remote_byte_offset(tensor, offset + sent);
            if (!ctx->backend_ctx->uart->upload_bytes(remote, packed.data(), packed.size())) {
                GTX_LOG_ERROR("Failed to upload tensor over UART\n");
                return;
            }
            sent += chunk;
        }

        if (offset + size <= ctx->shadow_buffer.size()) {
            memcpy(ctx->shadow_buffer.data() + offset, data, size);
        }
    } else if (ctx->conn) {
        const size_t chunk_size = gtx_upload_chunk_size() - sizeof(gtx_rpc_tensor);
        size_t sent = 0;

        while (sent < size) {
            size_t chunk = gtx_upload_local_chunk_size(tensor, size - sent, chunk_size);
            gtx_rpc_tensor rpc_tensor = gtx_serialize_tensor(tensor);
            rpc_tensor.data = gtx_remote_data_ptr(tensor) + gtx_remote_byte_offset(tensor, offset + sent);

            std::vector<uint8_t> packed = gtx_pack_tensor_chunk(tensor, (const uint8_t *) data + sent, chunk);

            std::vector<uint8_t> msg_data(sizeof(gtx_rpc_tensor) + packed.size());
            memcpy(msg_data.data(), &rpc_tensor, sizeof(gtx_rpc_tensor));
            memcpy(msg_data.data() + sizeof(gtx_rpc_tensor), packed.data(), packed.size());

            if (!ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, msg_data.data(), msg_data.size()) ||
                !ctx->conn->send_cmd_staged(GTX_CMD_SET_TENSOR,
                                            GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                            (uint32_t) msg_data.size(), nullptr, 0)) {
                GTX_LOG_ERROR("Failed to stage tensor upload\n");
                return;
            }

            sent += chunk;
        }

        if (offset + size <= ctx->shadow_buffer.size()) {
            memcpy(ctx->shadow_buffer.data() + offset, data, size);
        }
    }
}

static void ggml_backend_gtx_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor,
    void* data, size_t offset, size_t size) {

    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;

    GTX_LOG("get_tensor: %s, offset=%zu, size=%zu\n", tensor->name, offset, size);

    if (ctx->backend_ctx != nullptr && ctx->backend_ctx->use_uart) {
        const size_t chunk_size = 16ULL * 1024ULL;
        size_t received = 0;

        while (received < size) {
            size_t chunk = std::min(chunk_size, size - received);
            const size_t remote_size = (size_t) gtx_remote_byte_offset(tensor, chunk);
            std::vector<uint8_t> packed(remote_size);
            uint64_t remote = gtx_remote_data_ptr(tensor) + gtx_remote_byte_offset(tensor, offset + received);
            if (!ctx->backend_ctx->uart->download_bytes(remote, packed.data(), packed.size())) {
                GTX_LOG_ERROR("Failed to download tensor over UART\n");
                return;
            }
            gtx_unpack_tensor_chunk(tensor, packed.data(), (uint8_t *) data + received, chunk);
            received += chunk;
        }

        if (offset + size <= ctx->shadow_buffer.size()) {
            memcpy(ctx->shadow_buffer.data() + offset, data, size);
        }
    } else if (ctx->conn) {
        const size_t chunk_size = GTX_BAR2_DOWNLOAD_CHUNK;
        size_t received = 0;

        while (received < size) {
            size_t chunk = std::min(chunk_size, size - received);
            gtx_msg_get_tensor_req req = {};
            req.tensor = gtx_serialize_tensor(tensor);
            req.offset = gtx_remote_byte_offset(tensor, offset + received);
            req.size = gtx_remote_byte_offset(tensor, chunk);

            std::vector<uint8_t> packed(req.size);

            if (!ctx->conn->send_data(GTX_BAR2_STAGING_OFFSET, &req, sizeof(req)) ||
                !ctx->conn->send_cmd_staged(GTX_CMD_GET_TENSOR,
                                            GTX_BAR2_DEVICE_BASE + GTX_BAR2_STAGING_OFFSET,
                                            (uint32_t) sizeof(req), nullptr, 0) ||
                !ctx->conn->recv_data(GTX_BAR2_STAGING_OFFSET,
                                      packed.data(), packed.size())) {
                GTX_LOG_ERROR("Failed to stage tensor download\n");
                return;
            }

            gtx_unpack_tensor_chunk(tensor, packed.data(), (uint8_t *) data + received, chunk);

            received += chunk;
        }

        if (offset + size <= ctx->shadow_buffer.size()) {
            memcpy(ctx->shadow_buffer.data() + offset, data, size);
        }
    }
}

static void ggml_backend_gtx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_gtx_buffer_context* ctx = (ggml_backend_gtx_buffer_context*)buffer->context;

    if (ctx->backend_ctx != nullptr && ctx->backend_ctx->use_uart) {
        std::vector<uint8_t> zeros(ctx->size, value);
        ctx->backend_ctx->uart->upload_bytes(ctx->remote_ptr, zeros.data(), zeros.size());
    } else if (ctx->conn) {
        gtx_msg_buffer_clear_req req;
        req.remote_ptr = ctx->remote_ptr;
        req.value = value;
        ctx->conn->send_cmd(GTX_CMD_BUFFER_CLEAR, &req, sizeof(req), nullptr, 0);
    }

    /* Clear shadow buffer too */
    memset(ctx->shadow_buffer.data(), value, ctx->shadow_buffer.size());
}

static bool ggml_backend_gtx_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_gtx_buffer_context * ctx =
        (ggml_backend_gtx_buffer_context *) buffer->context;

    if (ctx == nullptr) {
        return false;
    }

    if (ctx->backend_ctx != nullptr && ctx->backend_ctx->use_uart) {
        return ctx->backend_ctx->uart != nullptr &&
               ctx->backend_ctx->uart->copy_tensor(src, dst);
    }

    if (ctx->conn == nullptr) {
        return false;
    }

    return gtx_copy_tensor_rpc(ctx->conn, src, dst);
}

static ggml_backend_buffer_i ggml_backend_gtx_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_gtx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_gtx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_gtx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_gtx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_gtx_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_gtx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_gtx_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_gtx_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {

    ggml_backend_gtx_context* backend_ctx = (ggml_backend_gtx_context*)buft->context;

    if (backend_ctx->use_uart) {
        if (!backend_ctx->uart->open()) {
            GTX_LOG_ERROR("Failed to open %s stream connection for buffer allocation\n",
                          gtx_selected_transport());
            return nullptr;
        }
        const size_t aligned_offset = gtx_align_up(backend_ctx->uart_arena_next, 64);
        const size_t aligned_size = gtx_align_up(size, 64);
        const uint64_t arena_size = gtx_fixed_tensor_arena_size();
        if (aligned_offset > arena_size ||
            aligned_size > arena_size - aligned_offset) {
            GTX_LOG_ERROR("UART VP fixed tensor arena too small: requested=%zu used=%zu arena=%llu\n",
                          size,
                          aligned_offset,
                          (unsigned long long) arena_size);
            return nullptr;
        }
        const uint64_t remote_ptr = gtx_fixed_tensor_arena_addr() + aligned_offset;
        backend_ctx->uart_arena_next = aligned_offset + aligned_size;
        backend_ctx->allocated_buffers[remote_ptr] = size;

        if (gtx_log_ops_enabled()) {
            fprintf(stderr, "[GTX ALLOC] uart buffer addr=0x%llx size=%zu next=%zu arena=%llu\n",
                    (unsigned long long) remote_ptr,
                    size,
                    backend_ctx->uart_arena_next,
                    (unsigned long long) gtx_fixed_tensor_arena_size());
        }

        ggml_backend_gtx_buffer_context* ctx = new ggml_backend_gtx_buffer_context();
        ctx->device_id = backend_ctx->device_id;
        ctx->remote_ptr = remote_ptr;
        ctx->size = size;
        ctx->conn = nullptr;
        ctx->backend_ctx = backend_ctx;
        ctx->shadow_buffer.resize(size);
        return ggml_backend_buffer_init(buft, ggml_backend_gtx_buffer_interface, ctx, size);
    }

    if (!ggml_backend_gtx_open_device_connection(backend_ctx->conn.get(), backend_ctx->device_id)) {
        GTX_LOG_ERROR("Failed to open GTX connection for buffer allocation\n");
        return nullptr;
    }

    /* Allocate buffer on device */
    gtx_msg_alloc_buffer_req req;
    req.device = backend_ctx->device_id;
    req.size = size;

    gtx_msg_alloc_buffer_rsp rsp = {};

    GTX_LOG("ALLOC_BUFFER request: device=%d size=%zu\n", backend_ctx->device_id, size);

    if (!backend_ctx->conn->send_cmd(GTX_CMD_ALLOC_BUFFER, &req, sizeof(req), &rsp, sizeof(rsp))) {
        GTX_LOG_ERROR("Failed to allocate buffer: size=%zu\n", size);
        return nullptr;
    }

    GTX_LOG("ALLOC_BUFFER response: remote_ptr=0x%llx remote_size=%llu\n",
            (unsigned long long) rsp.remote_ptr,
            (unsigned long long) rsp.remote_size);

    if (rsp.remote_ptr == 0) {
        GTX_LOG_ERROR("Device returned null pointer for buffer allocation: size=%zu remote_size=%llu\n",
                      size,
                      (unsigned long long) rsp.remote_size);
        return nullptr;
    }

    GTX_LOG("Allocated buffer: ptr=0x%llx, size=%llu\n",
            (unsigned long long)rsp.remote_ptr,
            (unsigned long long)rsp.remote_size);

    /* Create buffer context */
    ggml_backend_gtx_buffer_context* ctx = new ggml_backend_gtx_buffer_context();
    ctx->device_id = backend_ctx->device_id;
    ctx->remote_ptr = rsp.remote_ptr;
    ctx->size = rsp.remote_size;
    ctx->conn = backend_ctx->conn.get();
    ctx->backend_ctx = backend_ctx;
    ctx->shadow_buffer.resize(size);

    return ggml_backend_buffer_init(buft, ggml_backend_gtx_buffer_interface, ctx, size);
}

/*=============================================================================
 * Backend Implementation
 *============================================================================*/

static const char* ggml_backend_gtx_get_name(ggml_backend_t backend) {
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    static char name[64];
    snprintf(name, sizeof(name), "GTX%d", ctx->device_id);
    return name;
}

static void ggml_backend_gtx_free(ggml_backend_t backend) {
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    delete ctx;
    delete backend;
}

void ggml_backend_gtx_synchronize(ggml_backend_t backend) {
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;

    /* Wait for all pending operations using IRQ if available */
    gtx_pcie_irq_ctx* irq_ctx = ctx->get_irq_ctx();
    if (irq_ctx) {
        /* Poll and process any pending MSIs */
        gtx_pcie_irq_process(irq_ctx);
    }
}

static enum ggml_status ggml_backend_gtx_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    gtx_cpu_ref_current_graph = gtx_cpu_ref_graph_sequence++;

    GTX_LOG("graph_compute: %d nodes\n", cgraph->n_nodes);
    gtx_dump_graph_node_types(cgraph);
    gtx_log_graph_chainability(cgraph);

    auto start = std::chrono::steady_clock::now();

    /* Host interprets ggml graph nodes and submits descriptor/copy RPCs. */
    if (ctx->use_uart ? ctx->uart->is_open() : ctx->conn->is_open()) {
        if (gtx_tiling_scheduler_enabled() && !gtx_tiling_scheduler_diagnostics_active()) {
            enum ggml_status status = gtx_compute_graph_tiling_scheduler(ctx, cgraph);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        } else {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                enum ggml_status status = gtx_compute_node_host_driven(ctx, cgraph->nodes[i], i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    } else {
        GTX_LOG_ERROR("GTX connection not open\n");
        ctx->last_error = "GTX connection not open";
        return GGML_STATUS_FAILED;
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    ctx->stats.graph_computes++;
    ctx->stats.total_compute_us += elapsed;

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_gtx_interface = {
    /* .get_name            = */ ggml_backend_gtx_get_name,
    /* .free                = */ ggml_backend_gtx_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ ggml_backend_gtx_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_gtx_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

/*=============================================================================
 * GUID
 *============================================================================*/

static ggml_guid_t ggml_backend_gtx_guid() {
    static ggml_guid guid = {
        0x47, 0x54, 0x58, 0x00, /* 'GTX\0' */
        0x4e, 0x50, 0x55, 0x00, /* 'NPU\0' */
        0x42, 0x41, 0x43, 0x4b, /* 'BACK' */
        0x45, 0x4e, 0x44, 0x00  /* 'END\0' */
    };
    return &guid;
}

/*=============================================================================
 * Device Implementation
 *============================================================================*/

struct ggml_backend_gtx_device_context {
    int device_id;
    std::string description;
    std::shared_ptr<gtx_pcie_connection> conn;
    uint8_t firmware_version[3];

    ggml_backend_gtx_device_context(int id) : device_id(id) {
        char desc[128];
        snprintf(desc, sizeof(desc), "GTX NPU Device %d", id);
        description = desc;
        conn = ggml_backend_gtx_shared_connection(id);
        memset(firmware_version, 0, sizeof(firmware_version));
    }
};

static const char * ggml_backend_gtx_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_gtx_device_context * ctx = (ggml_backend_gtx_device_context *)dev->context;
    static char name[64];
    snprintf(name, sizeof(name), "GTX%d", ctx->device_id);
    return name;
}

static const char * ggml_backend_gtx_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_gtx_device_context * ctx = (ggml_backend_gtx_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_gtx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    if (gtx_use_xdma_stream_transport()) {
        *total = GTX_FPGA_TENSOR_ARENA_SIZE;
        *free  = GTX_FPGA_TENSOR_ARENA_SIZE;
    } else {
        *total = 16ULL * 1024 * 1024 * 1024;
        *free  = 15ULL * 1024 * 1024 * 1024;
    }
}

static enum ggml_backend_dev_type ggml_backend_gtx_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_gtx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gtx_device_get_name(dev);
    props->description = ggml_backend_gtx_device_get_description(dev);
    props->type        = ggml_backend_gtx_device_get_type(dev);
    ggml_backend_gtx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_gtx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)params;
    ggml_backend_gtx_device_context * dev_ctx = (ggml_backend_gtx_device_context *)dev->context;
    int device_id = dev_ctx->device_id;

    /* Create backend context */
    ggml_backend_gtx_context* ctx = new ggml_backend_gtx_context(device_id);
    fprintf(stderr, "[GTX] selected transport=%s stream_path=%s\n",
            gtx_selected_transport(), ctx->use_uart ? "yes" : "no");
    /* Share PCIe connection from device context */
    ctx->conn = dev_ctx->conn;

    /* Open PCIe connection */
    char device_path[64];
#ifdef _WIN32
    snprintf(device_path, sizeof(device_path), "\\\\.\\gtx%d", device_id);
#else
    snprintf(device_path, sizeof(device_path), "/dev/gtx%d", device_id);
#endif

    if (ctx->use_uart) {
        if (!ctx->uart->open()) {
            GTX_LOG_ERROR("Failed to open %s stream transport\n",
                          gtx_selected_transport());
        } else {
            GTX_LOG("%s stream transport opened successfully\n",
                    gtx_selected_transport());
        }
    } else if (!ctx->conn->open(device_path)) {
        GTX_LOG_ERROR("Failed to open PCIe connection\n");
        /* Continue anyway - may work in simulation mode */
    } else {
        GTX_LOG("PCIe connection opened successfully\n");
    }

    /* Send HELLO to verify connection */
    gtx_msg_hello_rsp hello_rsp = {};
    if (!ctx->use_uart && ctx->conn->is_open()) {
        if (ctx->conn->send_cmd(GTX_CMD_HELLO, nullptr, 0, &hello_rsp, sizeof(hello_rsp))) {
            ctx->firmware_version[0] = hello_rsp.major;
            ctx->firmware_version[1] = hello_rsp.minor;
            ctx->firmware_version[2] = hello_rsp.patch;
            GTX_LOG("Connected to firmware v%d.%d.%d\n",
                    hello_rsp.major, hello_rsp.minor, hello_rsp.patch);
        }
    }

    /* Create backend */
    ggml_backend_t backend = new ggml_backend();
    backend->guid = ggml_backend_gtx_guid();
    backend->iface = ggml_backend_gtx_interface;
    backend->device = dev;
    backend->context = ctx;

    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_gtx_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_gtx_device_context * ctx = (ggml_backend_gtx_device_context *)dev->context;
    return ggml_backend_gtx_buffer_type(ctx->device_id);
}

static bool ggml_backend_gtx_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;

    if (op == nullptr) {
        return false;
    }

    auto finish = [&](bool supported, const char * reason) {
        gtx_log_supports_op_decision(op, supported, reason);
        return supported;
    };

    const uint32_t normalized_op = gtx_normalize_host_op(op);
    const char * op_name = gtx_normalized_op_name(normalized_op);

    if (gtx_env_list_contains("GTX_DISABLE_OPS", op_name) ||
        gtx_env_substr_matches("GTX_DISABLE_NAME_SUBSTR", op->name)) {
        return finish(false, "disabled_by_env");
    }
    if (normalized_op == (uint32_t) GGML_OP_ADD &&
        gtx_env_substr_matches("GTX_DISABLE_ADD_NAME_SUBSTR", op->name)) {
        return finish(false, "disabled_add_by_name_env");
    }
    if (gtx_is_add_f16_f32_vector_to_f16(op)) {
        return finish(true, "add_f16_f32_vector_to_f16");
    }
    if (gtx_is_add_f16_bias_vector_basic(op)) {
        return finish(true, "add_f16_bias_vector_basic");
    }
    if (normalized_op == (uint32_t) GGML_OP_CONT &&
        !gtx_is_cont_same_shape(op)) {
        return finish(false, "cont_shape_change_cpu_fallback");
    }
    if (normalized_op == (uint32_t) GGML_OP_GLU) {
        return finish(gtx_is_swiglu_node(op), "glu_swiglu_only");
    }

    if (op->op == GGML_OP_SCALE) {
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)                               &&
                               // ⚠️ **타일러가 4차원 flat 커널을 아직 못 쪼갠다.**
                               //    1차원 경로는 2026-09-09 에 고쳤고 `gtx_smoke --limits`
                               //    가 131072 까지 통과한다. 그러나 모델의 4차원 relu
                               //    (`[112,112,64,1]` 등)는 다른 lowering 경로를 타고
                               //    여전히 한 번에 나간다 — ResNet-18 argmax 282 → 912.
                               //    그 경로를 고칠 때까지 큰 것은 CPU 로 보낸다.
                               ggml_nelements(op) <= 32768;
        return finish(supported, supported ? "scale_f32_f16" : "scale_type_guard");
    }

    if (op->op == GGML_OP_NORM) {
        const uint64_t rows64 = (uint64_t) op->ne[1] * (uint64_t) op->ne[2] * (uint64_t) op->ne[3];
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               op->ne[0] > 0 &&
                               op->ne[0] <= UINT16_MAX &&
                               rows64 > 0 &&
                               rows64 <= UINT16_MAX &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]);
        return finish(supported, supported ? "norm_f32_f16_dense" : "norm_type_or_layout_guard");
    }

    if (gtx_is_gelu_node(op)) {
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]);
        return finish(supported, supported ? "gelu_f32_f16_dense" : "gelu_type_or_layout_guard");
    }

    if (op->op == GGML_OP_SIN) {
        const uint64_t elems64 = (uint64_t) op->ne[0] * (uint64_t) op->ne[1] *
                                 (uint64_t) op->ne[2] * (uint64_t) op->ne[3];
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               elems64 > 0 &&
                               (elems64 % 16u) == 0 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]);
        return finish(supported, supported ? "sin_f16_reference" : "sin_reference_guard");
    }

    if (op->op == GGML_OP_COS) {
        const uint64_t elems64 = (uint64_t) op->ne[0] * (uint64_t) op->ne[1] *
                                 (uint64_t) op->ne[2] * (uint64_t) op->ne[3];
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               elems64 > 0 &&
                               (elems64 % 16u) == 0 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]);
        return finish(supported, supported ? "cos_f16_l2" : "cos_l2_guard");
    }

    if (op->op == GGML_OP_UNARY) {
        const ggml_unary_op uop = (ggml_unary_op) op->op_params[0];
        const bool p2_unary =
            uop == GGML_UNARY_OP_ABS ||
            uop == GGML_UNARY_OP_NEG ||
            uop == GGML_UNARY_OP_SGN ||
            uop == GGML_UNARY_OP_CEIL ||
            uop == GGML_UNARY_OP_FLOOR ||
            uop == GGML_UNARY_OP_TANH ||
            uop == GGML_UNARY_OP_SILU ||
            uop == GGML_UNARY_OP_RELU;
        const bool supported = p2_unary &&
                               op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0])                               &&
                               // ⚠️ **타일러가 4차원 flat 커널을 아직 못 쪼갠다.**
                               //    1차원 경로는 2026-09-09 에 고쳤고 `gtx_smoke --limits`
                               //    가 131072 까지 통과한다. 그러나 모델의 4차원 relu
                               //    (`[112,112,64,1]` 등)는 다른 lowering 경로를 타고
                               //    여전히 한 번에 나간다 — ResNet-18 argmax 282 → 912.
                               //    그 경로를 고칠 때까지 큰 것은 CPU 로 보낸다.
                               ggml_nelements(op) <= 32768;
        return finish(supported, supported ? "unary_p2_f16_reference" : "unary_p2_reference_guard");
    }

    if (op->op == GGML_OP_LEAKY_RELU) {
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0])                               &&
                               // ⚠️ **타일러가 4차원 flat 커널을 아직 못 쪼갠다.**
                               //    1차원 경로는 2026-09-09 에 고쳤고 `gtx_smoke --limits`
                               //    가 131072 까지 통과한다. 그러나 모델의 4차원 relu
                               //    (`[112,112,64,1]` 등)는 다른 lowering 경로를 타고
                               //    여전히 한 번에 나간다 — ResNet-18 argmax 282 → 912.
                               //    그 경로를 고칠 때까지 큰 것은 CPU 로 보낸다.
                               ggml_nelements(op) <= 32768;
        return finish(supported, supported ? "leaky_relu_f16_reference" : "leaky_relu_reference_guard");
    }

    if (normalized_op == (uint32_t) GGML_OP_CLAMP) {
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0])                               &&
                               // ⚠️ **타일러가 4차원 flat 커널을 아직 못 쪼갠다.**
                               //    1차원 경로는 2026-09-09 에 고쳤고 `gtx_smoke --limits`
                               //    가 131072 까지 통과한다. 그러나 모델의 4차원 relu
                               //    (`[112,112,64,1]` 등)는 다른 lowering 경로를 타고
                               //    여전히 한 번에 나간다 — ResNet-18 argmax 282 → 912.
                               //    그 경로를 고칠 때까지 큰 것은 CPU 로 보낸다.
                               ggml_nelements(op) <= 32768;
        return finish(supported, supported ? "clamp_f16_reference" : "clamp_reference_guard");
    }

    if (op->op == GGML_OP_REPEAT) {
        constexpr uint64_t gtx_repeat_max_bytes = 4ull * 1024ull * 1024ull;
        bool multiples = op->src[0] != nullptr;
        for (int i = 0; multiples && i < GGML_MAX_DIMS; ++i) {
            multiples = op->src[0]->ne[i] > 0 &&
                        op->ne[i] >= op->src[0]->ne[i] &&
                        (op->ne[i] % op->src[0]->ne[i]) == 0 &&
                        op->ne[i] <= UINT16_MAX &&
                        op->src[0]->ne[i] <= UINT16_MAX;
        }
        const uint64_t bytes = (uint64_t) ggml_nbytes(op);
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               multiples &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]) &&
                               bytes > 0 &&
                               bytes <= gtx_repeat_max_bytes;
        return finish(supported, supported ? "repeat_f32_f16_dense" : "repeat_type_or_shape_guard");
    }

    if (op->op == GGML_OP_CONCAT) {
        constexpr uint64_t gtx_concat_max_bytes = 4ull * 1024ull * 1024ull;
        const int32_t dim = ((const int32_t *) op->op_params)[0];
        bool shape_ok = op->src[0] != nullptr && op->src[1] != nullptr &&
                        dim >= 0 && dim < GGML_MAX_DIMS;
        for (int i = 0; shape_ok && i < GGML_MAX_DIMS; ++i) {
            const int64_t expected = i == dim ?
                op->src[0]->ne[i] + op->src[1]->ne[i] : op->src[0]->ne[i];
            shape_ok = op->src[0]->ne[i] <= UINT16_MAX &&
                       op->src[1]->ne[i] <= UINT16_MAX &&
                       op->ne[i] <= UINT16_MAX &&
                       (i == dim || op->src[0]->ne[i] == op->src[1]->ne[i]) &&
                       op->ne[i] == expected;
        }
        const uint64_t bytes = (uint64_t) ggml_nbytes(op);
        const bool supported = op->src[0] != nullptr &&
                               op->src[1] != nullptr &&
                               op->src[0]->type == op->src[1]->type &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[1]) == GGML_TYPE_F16 &&
                               shape_ok &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]) &&
                               ggml_is_contiguous(op->src[1]) &&
                               bytes > 0 &&
                               bytes <= gtx_concat_max_bytes;
        return finish(supported, supported ? "concat_f32_f16_dense" : "concat_type_or_shape_guard");
    }

    if (op->op == GGML_OP_IM2COL) {
        const int32_t * params = (const int32_t *) op->op_params;
        const bool is_2D = params[6] == 1;
        constexpr int64_t gtx_im2col_1d_max_output_bytes = 512 * 1024;
        constexpr int64_t gtx_im2col_2d_max_output_bytes = 4 * 1024 * 1024;
        const int64_t iw = op->src[1] != nullptr ? op->src[1]->ne[0] : 0;
        const int64_t ih = op->src[1] != nullptr ? op->src[1]->ne[1] : 0;
        const int64_t ic = op->src[1] != nullptr ? op->src[1]->ne[2] : 0;
        const int64_t n = op->src[1] != nullptr ? op->src[1]->ne[3] : 0;
        const int64_t kw = op->src[0] != nullptr ? op->src[0]->ne[0] : 0;
        const int64_t kh = op->src[0] != nullptr ? op->src[0]->ne[1] : 0;
        const int64_t w_in_pad = iw + (int64_t) params[2] * 2;
        const int64_t h_in_pad = ih + (int64_t) params[3] * 2;
        const int64_t k_eff_w = (kw - 1) * (int64_t) params[4] + 1;
        const int64_t k_eff_h = (kh - 1) * (int64_t) params[5] + 1;
        const int64_t w_out = (params[0] > 0 && w_in_pad >= k_eff_w) ?
            (w_in_pad - k_eff_w) / params[0] + 1 : 0;
        const int64_t h_out = (is_2D && params[1] > 0 && h_in_pad >= k_eff_h) ?
            (h_in_pad - k_eff_h) / params[1] + 1 : 1;
        const int64_t output_bytes = w_out * h_out * ic * kw * kh * n * (int64_t) ggml_type_size(op->type);

        bool supported = false;
        const char * reason = "im2col_guard";
        if (op->src[0] != nullptr && op->src[1] != nullptr &&
            op->src[1]->type == GGML_TYPE_F32 &&
            op->type == GGML_TYPE_F16 &&
            params[0] > 0 &&
            params[2] >= 0 && params[2] <= 255 &&
            params[4] > 0 &&
            output_bytes > 0) {
            if (!is_2D) {
                supported = op->src[0]->ne[1] == op->src[1]->ne[1] &&
                            op->src[1]->ne[3] == 1 &&
                            kw <= 255 &&
                            output_bytes <= gtx_im2col_1d_max_output_bytes;
                reason = supported ? "im2col_1d" : "im2col_1d_guard";
            } else {
                supported = op->src[0]->ne[2] == op->src[1]->ne[2] &&
                            kh == kw &&
                            params[1] == params[0] &&
                            params[3] == params[2] &&
                            params[5] == params[4] &&
                            kw <= 255 &&
                            output_bytes <= gtx_im2col_2d_max_output_bytes;
                reason = supported ? "im2col_2d_square" : "im2col_2d_guard";
            }
        }
        return finish(supported, reason);
    }

    if (op->op == GGML_OP_CONV_2D) {
        const bool supported = op->src[0] != nullptr &&
                               op->src[1] != nullptr &&
                               op->src[0]->ne[2] == op->src[1]->ne[2] &&
                               op->src[0]->ne[0] > 0 && op->src[0]->ne[1] > 0 &&
                               op->src[0]->ne[2] > 0 && op->src[0]->ne[3] > 0 &&
                               op->src[1]->ne[0] > 0 && op->src[1]->ne[1] > 0 &&
                               op->src[1]->ne[3] > 0 &&
                               op->src[1]->ne[3] == 1 &&
                               op->ne[3] == 1 &&
                               op->src[0]->ne[0] == op->src[0]->ne[1] &&
                               op->op_params[0] > 0 && op->op_params[1] > 0 &&
                               op->op_params[4] > 0 && op->op_params[5] > 0 &&
                               op->op_params[0] == op->op_params[1] &&
                               op->op_params[4] == op->op_params[5] &&
                               op->op_params[2] >= 0 && op->op_params[3] >= 0 &&
                               op->src[0]->ne[0] <= UINT16_MAX &&
                               op->src[0]->ne[1] <= UINT16_MAX &&
                               op->src[0]->ne[2] <= UINT16_MAX &&
                               op->src[0]->ne[3] <= UINT16_MAX &&
                               op->src[1]->ne[0] <= UINT16_MAX &&
                               op->src[1]->ne[1] <= UINT16_MAX &&
                               op->src[1]->ne[2] <= UINT16_MAX &&
                               op->src[1]->ne[3] <= UINT16_MAX &&
                               op->ne[0] <= UINT16_MAX &&
                               op->ne[1] <= UINT16_MAX &&
                               op->ne[2] <= UINT16_MAX &&
                               op->ne[3] <= UINT16_MAX &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[1]) == GGML_TYPE_F16 &&
                               gtx_l2_conv_layout_ok(op) &&
                               ggml_nbytes(op) > 0;
        return finish(supported, supported ? "conv_2d_f16_reference" : "conv_2d_reference_guard");
    }

    if (op->op == GGML_OP_CONV_2D_DW) {
        const bool supported = op->src[0] != nullptr &&
                               op->src[1] != nullptr &&
                               op->src[0]->ne[2] == 1 &&
                               op->src[0]->ne[3] == op->src[1]->ne[2] &&
                               op->ne[2] == op->src[1]->ne[2] &&
                               op->ne[3] == op->src[1]->ne[3] &&
                               op->src[1]->ne[3] == 1 &&
                               op->ne[3] == 1 &&
                               op->src[0]->ne[0] == op->src[0]->ne[1] &&
                               op->op_params[0] > 0 && op->op_params[1] > 0 &&
                               op->op_params[4] > 0 && op->op_params[5] > 0 &&
                               op->op_params[0] == op->op_params[1] &&
                               op->op_params[4] == op->op_params[5] &&
                               op->op_params[2] >= 0 && op->op_params[3] >= 0 &&
                               op->src[0]->ne[0] <= UINT16_MAX &&
                               op->src[0]->ne[1] <= UINT16_MAX &&
                               op->src[0]->ne[3] <= UINT16_MAX &&
                               op->src[1]->ne[0] <= UINT16_MAX &&
                               op->src[1]->ne[1] <= UINT16_MAX &&
                               op->src[1]->ne[2] <= UINT16_MAX &&
                               op->src[1]->ne[3] <= UINT16_MAX &&
                               op->ne[0] <= UINT16_MAX &&
                               op->ne[1] <= UINT16_MAX &&
                               op->ne[2] <= UINT16_MAX &&
                               op->ne[3] <= UINT16_MAX &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[1]) == GGML_TYPE_F16 &&
                               gtx_l2_conv_layout_ok(op) &&
                               ggml_nbytes(op) > 0;
        return finish(supported, supported ? "conv_2d_dw_f16_reference" : "conv_2d_dw_reference_guard");
    }

    if (normalized_op == (uint32_t) GGML_OP_CONV_TRANSPOSE_2D) {
        const int64_t dst_bytes = (int64_t) gtx_remote_size_bytes(op);
        const int64_t input_bytes = op->src[1] != nullptr ? (int64_t) gtx_remote_size_bytes(op->src[1]) : 0;
        const int64_t kernel_bytes = op->src[0] != nullptr ? (int64_t) gtx_remote_size_bytes(op->src[0]) : 0;
        const int32_t stride = op->op_params[0];
        const int64_t kw = op->src[0] != nullptr ? op->src[0]->ne[0] : 0;
        const int64_t kh = op->src[0] != nullptr ? op->src[0]->ne[1] : 0;
        const int64_t oc = op->src[0] != nullptr ? op->src[0]->ne[2] : 0;
        const int64_t ic = op->src[0] != nullptr ? op->src[0]->ne[3] : 0;
        const int64_t iw = op->src[1] != nullptr ? op->src[1]->ne[0] : 0;
        const int64_t ih = op->src[1] != nullptr ? op->src[1]->ne[1] : 0;
        const int64_t batch = op->src[1] != nullptr ? op->src[1]->ne[3] : 0;
        const bool shape_ok = op->src[0] != nullptr &&
                              op->src[1] != nullptr &&
                              stride > 0 &&
                              kw > 0 && kh > 0 && oc > 0 && ic > 0 &&
                              iw > 0 && ih > 0 && batch > 0 &&
                              op->src[1]->ne[2] == ic &&
                              op->ne[0] == (iw - 1) * stride + kw &&
                              op->ne[1] == (ih - 1) * stride + kh &&
                              op->ne[2] == oc &&
                              op->ne[3] == batch &&
                              kw * kh * ic <= 32768 &&
                              op->ne[0] <= UINT16_MAX &&
                              op->src[1]->ne[0] <= UINT16_MAX &&
                              op->src[0]->ne[0] <= UINT16_MAX &&
                              op->ne[1] * op->ne[2] * op->ne[3] <= UINT16_MAX &&
                              op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3] <= UINT16_MAX &&
                              op->src[0]->ne[1] * op->src[0]->ne[2] * op->src[0]->ne[3] <= UINT16_MAX;
        const bool l2_ok = input_bytes > 0 &&
                           kernel_bytes > 0 &&
                           dst_bytes > 0 &&
                           kernel_bytes <= 2 * 1024 * 1024 &&
                           input_bytes <= 6 * 1024 * 1024 &&
                           dst_bytes <= 8 * 1024 * 1024;
        const bool supported = op->src[0] != nullptr &&
                               op->src[1] != nullptr &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[1]) == GGML_TYPE_F16 &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]) &&
                               ggml_is_contiguous(op->src[1]) &&
                               shape_ok &&
                               l2_ok;
        return finish(supported, supported ? "conv_transpose_2d_f16_reference" : "conv_transpose_2d_reference_guard");
    }

    if (normalized_op == (uint32_t) GGML_OP_UPSCALE) {
        const uint32_t mode_flags = (uint32_t) op->op_params[0];
        const uint32_t mode = mode_flags & 0xffu;
        const bool scale_mode_ok = mode == GGML_SCALE_MODE_NEAREST ||
                                   mode == GGML_SCALE_MODE_BILINEAR ||
                                   mode == GGML_SCALE_MODE_BICUBIC;
        const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;
        const int64_t dst_bytes = (int64_t) gtx_remote_size_bytes(op);
        const bool supported = op->src[0] != nullptr &&
                               scale_mode_ok &&
                               !antialias &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               op->src[0]->ne[0] > 0 &&
                               op->src[0]->ne[1] > 0 &&
                               op->src[0]->ne[2] > 0 &&
                               op->src[0]->ne[3] > 0 &&
                               op->ne[0] > 0 &&
                               op->ne[1] > 0 &&
                               op->ne[2] > 0 &&
                               op->ne[3] > 0 &&
                               op->ne[2] == op->src[0]->ne[2] &&
                               op->ne[3] == op->src[0]->ne[3] &&
                               op->src[0]->ne[0] <= UINT16_MAX &&
                               op->src[0]->ne[1] <= UINT16_MAX &&
                               op->src[0]->ne[2] <= UINT16_MAX &&
                               op->src[0]->ne[3] <= UINT16_MAX &&
                               op->ne[0] <= UINT16_MAX &&
                               op->ne[1] <= UINT16_MAX &&
                               op->ne[2] <= UINT16_MAX &&
                               op->ne[3] <= UINT16_MAX &&
                               dst_bytes > 0 &&
                               (uint64_t) dst_bytes <= GTX_STASH_L2_BYTES;
        return finish(supported, supported ? "upscale_f16_resize" : "upscale_resize_guard");
    }

    if (normalized_op == (uint32_t) GGML_OP_PAD) {
        const int32_t * p = (const int32_t *) op->op_params;
        const int64_t remote_bytes = (int64_t) gtx_remote_size_bytes(op);
        const bool shape_ok = op->src[0] != nullptr &&
                              p[0] == 0 && p[2] == 0 && p[4] == 0 && p[6] == 0 &&
                              p[1] >= 0 && p[3] >= 0 && p[5] >= 0 && p[7] == 0 &&
                              p[8] == 0 &&
                              op->ne[0] == op->src[0]->ne[0] + p[1] &&
                              op->ne[1] == op->src[0]->ne[1] + p[3] &&
                              op->ne[2] == op->src[0]->ne[2] + p[5] &&
                              op->ne[3] == op->src[0]->ne[3] &&
                              op->ne[0] <= UINT16_MAX &&
                              op->ne[1] <= UINT16_MAX &&
                              op->src[0]->ne[0] <= UINT16_MAX &&
                              op->src[0]->ne[1] <= UINT16_MAX;
        const bool supported = op->src[0] != nullptr &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                               gtx_remote_tensor_type(op) == GGML_TYPE_F16 &&
                               gtx_remote_tensor_type(op->src[0]) == GGML_TYPE_F16 &&
                               shape_ok &&
                               ggml_is_contiguous(op) &&
                               ggml_is_contiguous(op->src[0]) &&
                               remote_bytes > 0 &&
                               remote_bytes <= 14 * 1024 * 1024;
        return finish(supported, supported ? "pad_f16_reference" : "pad_reference_guard");
    }

    switch (normalized_op) {
        case GGML_OP_NONE:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_MUL_MAT:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_CLAMP:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_ROPE:
            return finish(true, "supported_descriptor_or_metadata");
        default:
            return finish(false, "unsupported_op");
    }
}

static bool ggml_backend_gtx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_gtx_device_context * ctx = (ggml_backend_gtx_device_context *)dev->context;
    return buft == ggml_backend_gtx_buffer_type(ctx->device_id);
}

static bool ggml_backend_gtx_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!gtx_env_truthy("GTX_ENABLE_OFFLOAD_OP")) {
        return false;
    }
    return ggml_backend_gtx_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_gtx_device_i = {
    /* .get_name            = */ ggml_backend_gtx_device_get_name,
    /* .get_description     = */ ggml_backend_gtx_device_get_description,
    /* .get_memory          = */ ggml_backend_gtx_device_get_memory,
    /* .get_type            = */ ggml_backend_gtx_device_get_type,
    /* .get_props           = */ ggml_backend_gtx_device_get_props,
    /* .init_backend        = */ ggml_backend_gtx_device_init_backend,
    /* .get_buffer_type     = */ ggml_backend_gtx_device_get_buffer_type,
    /* .get_host_buffer_type= */ nullptr,
    /* .buffer_from_host_ptr= */ nullptr,
    /* .supports_op         = */ ggml_backend_gtx_device_supports_op,
    /* .supports_buft       = */ ggml_backend_gtx_device_supports_buft,
    /* .offload_op          = */ ggml_backend_gtx_device_offload_op,
    /* .event_new           = */ nullptr,
    /* .event_free          = */ nullptr,
    /* .event_synchronize   = */ nullptr,
};

/*=============================================================================
 * Registry Implementation
 *============================================================================*/

static const char * ggml_backend_gtx_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "GTX";
}

static size_t ggml_backend_gtx_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1; /* TODO: Enumerate actual devices */
}

static ggml_backend_dev_t ggml_backend_gtx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static std::vector<std::unique_ptr<ggml_backend_gtx_device_context>> device_contexts;
    static std::vector<ggml_backend_device> devices;
    static bool initialized = false;

    if (!initialized) {
        size_t count = ggml_backend_gtx_reg_get_device_count(reg);

        /* First, create all device contexts */
        for (size_t i = 0; i < count; i++) {
            device_contexts.push_back(std::make_unique<ggml_backend_gtx_device_context>((int)i));
        }

        /* Then, create devices with stable pointers */
        devices.reserve(count);
        for (size_t i = 0; i < count; i++) {
            devices.push_back({
                /* .iface   = */ ggml_backend_gtx_device_i,
                /* .reg     = */ reg,
                /* .context = */ device_contexts[i].get(),
            });
        }

        initialized = true;
    }

    if (index >= devices.size()) {
        return nullptr;
    }
    return &devices[index];
}

static ggml_backend_feature * ggml_backend_gtx_get_features(ggml_backend_reg_t reg) {
    (void)reg;
    static std::vector<ggml_backend_feature> features = {
        { "GTX_DESCRIPTOR_GRAPH", "1" },
        { "GTX_STASH_L2_CHAINING", "1" },
        { "GTX_STASH_L2_CHAIN_NATIVE_OPS",
          "ADD,MUL,SCALE,MUL_MAT,ROPE,SOFT_MAX,RMS_NORM,SWIGLU,GELU,RELU,LEAKY_RELU,CLAMP,ABS,NEG,SGN,CEIL,FLOOR,TANH,SILU" },
        { "GTX_STASH_L2_CHAIN_PRODUCER_OPS", "host-tiled contiguous descriptors" },
        { "GTX_STASH_L2_CHAIN_BOUNDARY_OPS",
          "CONT,CPY,GET_ROWS,SET_ROWS,NORM,ADD_F16_F32_TO_F16,SIN,COS" },
        { nullptr, nullptr },
    };
    return features.data();
}

static void * ggml_backend_gtx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    if (name != nullptr && strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_gtx_get_features;
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_gtx_reg_i = {
    /* .get_name         = */ ggml_backend_gtx_reg_get_name,
    /* .get_device_count = */ ggml_backend_gtx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gtx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_gtx_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_gtx_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gtx_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

/*=============================================================================
 * Public API Implementation
 *============================================================================*/

extern "C" {

int ggml_gtx_get_device_count(void) {
    /* TODO: Enumerate PCIe devices */
    /* For now, return 1 simulated device */
    return 1;
}

bool ggml_gtx_get_device_info(int device, ggml_gtx_device_info_t* info) {
    if (info == nullptr || device < 0 || device >= ggml_gtx_get_device_count()) {
        return false;
    }

    memset(info, 0, sizeof(*info));
    info->device_id = device;
    snprintf(info->name, sizeof(info->name), "GTX%d", device);
    snprintf(info->description, sizeof(info->description), "GTX NPU Device %d", device);
    if (gtx_use_xdma_stream_transport()) {
        info->total_memory = GTX_FPGA_TENSOR_ARENA_SIZE;
        info->free_memory = GTX_FPGA_TENSOR_ARENA_SIZE;
    } else {
        info->total_memory = 16ULL * 1024 * 1024 * 1024;
        info->free_memory = 15ULL * 1024 * 1024 * 1024;
    }
    info->num_nests = 1;
    info->num_spus = 1;
    info->pcie_gen = 4;
    info->pcie_width = 4;
    info->firmware_version[0] = 0;
    info->firmware_version[1] = 1;
    info->firmware_version[2] = 0;

    return true;
}

ggml_backend_t ggml_backend_gtx_init(int device) {
    if (device < 0 || device >= ggml_gtx_get_device_count()) {
        GTX_LOG_ERROR("Invalid device: %d\n", device);
        return nullptr;
    }

    /* Get device from registry and use device init */
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_gtx_reg(), device);
    if (!dev) {
        GTX_LOG_ERROR("Failed to get device %d\n", device);
        return nullptr;
    }

    return ggml_backend_dev_init(dev, nullptr);
}

bool ggml_backend_is_gtx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_gtx_guid());
}

int ggml_backend_gtx_get_device(ggml_backend_t backend) {
    if (!ggml_backend_is_gtx(backend)) {
        return -1;
    }
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    return ctx->device_id;
}

ggml_backend_buffer_type_t ggml_backend_gtx_buffer_type(int device) {
    if (device < 0 || device >= ggml_gtx_get_device_count()) {
        return nullptr;
    }

    /* Lazy initialization of buffer types */
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_gtx_context>> buffer_type_contexts;
    static bool initializing = false;
    static bool initialized = false;

    if (!initialized && !initializing) {
        initializing = true;

        int count = ggml_gtx_get_device_count();

        /* First pass: create contexts and buffer types without device reference */
        for (int i = 0; i < count; i++) {
            /* Create a context for buffer type operations */
            auto ctx = std::make_unique<ggml_backend_gtx_context>(i);

            auto buft = std::make_unique<ggml_backend_buffer_type>();
            buft->iface = ggml_backend_gtx_buffer_type_interface;
            buft->device = nullptr;  /* Will be set later to avoid circular reference */
            buft->context = ctx.get();

            buffer_type_contexts.push_back(std::move(ctx));
            buffer_types.push_back(std::move(buft));
        }

        /* Second pass: set device references after all buffer types are created */
        for (int i = 0; i < count; i++) {
            buffer_types[i]->device = ggml_backend_reg_dev_get(ggml_backend_gtx_reg(), i);
        }

        initialized = true;
        initializing = false;
    }

    return buffer_types[device].get();
}

ggml_backend_buffer_type_t ggml_backend_gtx_host_buffer_type(int device) {
    (void)device;
    return nullptr; /* Not supported yet */
}

void ggml_backend_gtx_get_stats(ggml_backend_t backend, ggml_gtx_stats_t* stats) {
    if (!ggml_backend_is_gtx(backend) || stats == nullptr) {
        return;
    }
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    *stats = ctx->stats;
}

void ggml_backend_gtx_reset_stats(ggml_backend_t backend) {
    if (!ggml_backend_is_gtx(backend)) {
        return;
    }
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    memset(&ctx->stats, 0, sizeof(ctx->stats));
}

void ggml_gtx_set_debug(bool enable) {
    g_gtx_debug = enable;
}

bool ggml_backend_gtx_is_ready(ggml_backend_t backend) {
    if (!ggml_backend_is_gtx(backend)) {
        return false;
    }
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    return ctx->use_uart ? ctx->uart->is_open() : ctx->conn->is_open();
}

const char* ggml_backend_gtx_get_error(ggml_backend_t backend) {
    if (!ggml_backend_is_gtx(backend)) {
        return nullptr;
    }
    ggml_backend_gtx_context* ctx = (ggml_backend_gtx_context*)backend->context;
    return ctx->last_error.empty() ? nullptr : ctx->last_error.c_str();
}

} /* extern "C" */

GGML_BACKEND_DL_IMPL(ggml_backend_gtx_reg)
