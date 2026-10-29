//==================================================================
// GGML Baremetal - Minimal GGML for RISC-V Baremetal Environment
//
// This is a stripped-down version of GGML designed to run on
// baremetal RISC-V systems without OS support.
//
// Key differences from standard GGML:
// - No dynamic memory allocation (malloc/free)
// - No file I/O
// - No threading
// - Static memory pools
// - Hardware-specific optimizations via GTX intrinsics
//==================================================================

#ifndef GGML_BAREMETAL_H
#define GGML_BAREMETAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

//==================================================================
// Configuration
//==================================================================

#define GGML_GTX_MAX_DIMS        4
#define GGML_GTX_MAX_SRC         10
#define GGML_GTX_MAX_NAME        32
#define GGML_GTX_MAX_OP_PARAMS   64
#define GGML_GTX_MAX_NODES       4096
#define GGML_GTX_MAX_TENSORS     8192

// Memory alignment for tensors
#define GGML_GTX_MEM_ALIGN       64

// Default memory pool sizes (can be overridden)
#ifndef GGML_GTX_POOL_SIZE
#define GGML_GTX_POOL_SIZE       (64 * 1024 * 1024)  // 64 MB
#endif

#ifndef GGML_GTX_SCRATCH_SIZE
#define GGML_GTX_SCRATCH_SIZE    (10 * 1024 * 1024)  // n1s1 L2 capacity
#endif

//==================================================================
// Utility Macros
//==================================================================

#define GGML_GTX_UNUSED(x)       ((void)(x))
#define GGML_GTX_PAD(x, n)       (((x) + (n) - 1) & ~((n) - 1))
#define GGML_GTX_MIN(a, b)       ((a) < (b) ? (a) : (b))
#define GGML_GTX_MAX(a, b)       ((a) > (b) ? (a) : (b))

// Assertion macro - can be customized for baremetal
#ifndef GGML_GTX_ASSERT
    #ifdef GGML_GTX_DEBUG
        #define GGML_GTX_ASSERT(x) do { if (!(x)) { ggml_gtx_abort(__FILE__, __LINE__, #x); } } while(0)
    #else
        #define GGML_GTX_ASSERT(x) ((void)0)
    #endif
#endif

//==================================================================
// Data Types
//==================================================================

typedef enum {
    GGML_GTX_TYPE_F32  = 0,
    GGML_GTX_TYPE_F16  = 1,
    GGML_GTX_TYPE_Q4_0 = 2,
    GGML_GTX_TYPE_Q4_1 = 3,
    GGML_GTX_TYPE_Q5_0 = 6,
    GGML_GTX_TYPE_Q5_1 = 7,
    GGML_GTX_TYPE_Q8_0 = 8,
    GGML_GTX_TYPE_Q8_1 = 9,
    GGML_GTX_TYPE_I8   = 24,
    GGML_GTX_TYPE_I16  = 25,
    GGML_GTX_TYPE_I32  = 26,
    GGML_GTX_TYPE_I64  = 27,
    GGML_GTX_TYPE_F64  = 28,
    GGML_GTX_TYPE_BF16 = 30,
    GGML_GTX_TYPE_COUNT,
} ggml_gtx_type;

//==================================================================
// Operations
//==================================================================

// 장치로 보내는 연산 번호다. **숫자를 고정해 둔다** — ggml enum(GGML_OP_*) 에서 파생하지 않는다.
// depend/llama 는 upstream 을 따라가고 upstream 은 새 연산을 enum 중간에 끼워 넣기 때문이다.
// 값은 2026-10-26 에 ISS 에서 검증한 판의 ggml 번호를 그대로 굳힌 것이고, 오른쪽 주석이 짝이 되는
// ggml 연산이다. ggml 연산 → 이 번호 변환은 ggml-gtx.cpp 의 gtx_wire_op_from_ggml_op() 가 이름으로 한다.
// 이미 있는 값을 바꾸지 마라. 새 연산은 맨 뒤에 붙인다.
typedef enum {
    GGML_GTX_OP_NONE              = 0,   // GGML_OP_NONE
    GGML_GTX_OP_DUP               = 1,   // GGML_OP_DUP
    GGML_GTX_OP_ADD               = 2,   // GGML_OP_ADD
    GGML_GTX_OP_ADD_ID            = 3,   // GGML_OP_ADD_ID
    GGML_GTX_OP_ADD1              = 4,   // GGML_OP_ADD1
    GGML_GTX_OP_ACC               = 5,   // GGML_OP_ACC
    GGML_GTX_OP_SUB               = 6,   // GGML_OP_SUB
    GGML_GTX_OP_MUL               = 7,   // GGML_OP_MUL
    GGML_GTX_OP_DIV               = 8,   // GGML_OP_DIV
    GGML_GTX_OP_SQR               = 9,   // GGML_OP_SQR
    GGML_GTX_OP_SQRT              = 10,  // GGML_OP_SQRT
    GGML_GTX_OP_LOG               = 11,  // GGML_OP_LOG
    GGML_GTX_OP_SIN               = 12,  // GGML_OP_SIN
    GGML_GTX_OP_COS               = 13,  // GGML_OP_COS
    GGML_GTX_OP_SUM               = 14,  // GGML_OP_SUM
    GGML_GTX_OP_SUM_ROWS          = 15,  // GGML_OP_SUM_ROWS
    GGML_GTX_OP_CUMSUM            = 16,  // GGML_OP_CUMSUM
    GGML_GTX_OP_MEAN              = 17,  // GGML_OP_MEAN
    GGML_GTX_OP_REPEAT            = 20,  // GGML_OP_REPEAT
    GGML_GTX_OP_CONCAT            = 22,  // GGML_OP_CONCAT
    GGML_GTX_OP_NORM              = 24,  // GGML_OP_NORM
    GGML_GTX_OP_RMS_NORM          = 25,  // GGML_OP_RMS_NORM
    GGML_GTX_OP_GROUP_NORM        = 27,  // GGML_OP_GROUP_NORM
    GGML_GTX_OP_L2_NORM           = 28,  // GGML_OP_L2_NORM
    GGML_GTX_OP_MUL_MAT           = 29,  // GGML_OP_MUL_MAT
    GGML_GTX_OP_MUL_MAT_ID        = 30,  // GGML_OP_MUL_MAT_ID
    GGML_GTX_OP_OUT_PROD          = 31,  // GGML_OP_OUT_PROD
    GGML_GTX_OP_SCALE             = 32,  // GGML_OP_SCALE
    GGML_GTX_OP_SET               = 33,  // GGML_OP_SET
    GGML_GTX_OP_CPY               = 34,  // GGML_OP_CPY
    GGML_GTX_OP_CONT              = 35,  // GGML_OP_CONT
    GGML_GTX_OP_RESHAPE           = 36,  // GGML_OP_RESHAPE
    GGML_GTX_OP_VIEW              = 37,  // GGML_OP_VIEW
    GGML_GTX_OP_PERMUTE           = 38,  // GGML_OP_PERMUTE
    GGML_GTX_OP_TRANSPOSE         = 39,  // GGML_OP_TRANSPOSE
    GGML_GTX_OP_GET_ROWS          = 40,  // GGML_OP_GET_ROWS
    GGML_GTX_OP_SET_ROWS          = 42,  // GGML_OP_SET_ROWS
    GGML_GTX_OP_DIAG              = 43,  // GGML_OP_DIAG
    GGML_GTX_OP_DIAG_MASK_INF     = 44,  // GGML_OP_DIAG_MASK_INF
    GGML_GTX_OP_DIAG_MASK_ZERO    = 45,  // GGML_OP_DIAG_MASK_ZERO
    GGML_GTX_OP_SOFTMAX           = 46,  // GGML_OP_SOFT_MAX
    GGML_GTX_OP_ROPE              = 48,  // GGML_OP_ROPE
    GGML_GTX_OP_CLAMP             = 50,  // GGML_OP_CLAMP
    GGML_GTX_OP_CONV_TRANSPOSE_1D = 51,  // GGML_OP_CONV_TRANSPOSE_1D
    GGML_GTX_OP_IM2COL            = 52,  // GGML_OP_IM2COL
    GGML_GTX_OP_IM2COL_3D         = 54,  // GGML_OP_IM2COL_3D
    GGML_GTX_OP_CONV_2D           = 55,  // GGML_OP_CONV_2D
    GGML_GTX_OP_CONV_3D           = 56,  // GGML_OP_CONV_3D
    GGML_GTX_OP_CONV_2D_DW        = 57,  // GGML_OP_CONV_2D_DW
    GGML_GTX_OP_CONV_TRANSPOSE_2D = 58,  // GGML_OP_CONV_TRANSPOSE_2D
    GGML_GTX_OP_POOL_1D           = 59,  // GGML_OP_POOL_1D
    GGML_GTX_OP_POOL_2D           = 60,  // GGML_OP_POOL_2D
    GGML_GTX_OP_UPSCALE           = 62,  // GGML_OP_UPSCALE
    GGML_GTX_OP_PAD               = 63,  // GGML_OP_PAD
    GGML_GTX_OP_PAD_REFLECT_1D    = 64,  // GGML_OP_PAD_REFLECT_1D
    GGML_GTX_OP_ROLL              = 65,  // GGML_OP_ROLL
    GGML_GTX_OP_ARANGE            = 66,  // GGML_OP_ARANGE
    GGML_GTX_OP_TIMESTEP_EMBEDDING = 67,  // GGML_OP_TIMESTEP_EMBEDDING
    GGML_GTX_OP_ARGSORT           = 68,  // GGML_OP_ARGSORT
    GGML_GTX_OP_TOP_K             = 69,  // GGML_OP_TOP_K
    GGML_GTX_OP_LEAKY_RELU        = 70,  // GGML_OP_LEAKY_RELU
    GGML_GTX_OP_TRI               = 71,  // GGML_OP_TRI
    GGML_GTX_OP_FILL              = 72,  // GGML_OP_FILL
    GGML_GTX_OP_FLASH_ATTN_EXT    = 73,  // GGML_OP_FLASH_ATTN_EXT
    GGML_GTX_OP_SSM_CONV          = 75,  // GGML_OP_SSM_CONV
    GGML_GTX_OP_SSM_SCAN          = 76,  // GGML_OP_SSM_SCAN
    GGML_GTX_OP_WIN_PART          = 77,  // GGML_OP_WIN_PART
    GGML_GTX_OP_WIN_UNPART        = 78,  // GGML_OP_WIN_UNPART
    GGML_GTX_OP_GET_REL_POS       = 79,  // GGML_OP_GET_REL_POS
    GGML_GTX_OP_ADD_REL_POS       = 80,  // GGML_OP_ADD_REL_POS
    GGML_GTX_OP_GATED_LINEAR_ATTN = 82,  // GGML_OP_GATED_LINEAR_ATTN
    GGML_GTX_OP_SOLVE_TRI         = 84,  // GGML_OP_SOLVE_TRI

    // GTX extensions / ggml unary specializations
    GGML_GTX_OP_NEG               = 97,   // GGML_OP_COUNT when frozen
    GGML_GTX_OP_ABS,
    GGML_GTX_OP_EXP,
    GGML_GTX_OP_MAX,
    GGML_GTX_OP_MIN,
    GGML_GTX_OP_LAYER_NORM,
    GGML_GTX_OP_RELU,
    GGML_GTX_OP_GELU,
    GGML_GTX_OP_SILU,
    GGML_GTX_OP_TANH,
    GGML_GTX_OP_SIGMOID,
    GGML_GTX_OP_ELU,
    GGML_GTX_OP_HARDSWISH,
    GGML_GTX_OP_HARDSIGMOID,
    GGML_GTX_OP_SOFTPLUS,
    GGML_GTX_OP_FLASH_ATTN,
    GGML_GTX_OP_SGN,
    GGML_GTX_OP_STEP,
    GGML_GTX_OP_CEIL,
    GGML_GTX_OP_FLOOR,
    GGML_GTX_OP_TRUNC,
    GGML_GTX_OP_ROUND,
    GGML_GTX_OP_GELU_QUICK,
    GGML_GTX_OP_GLU,
    GGML_GTX_OP_SWIGLU,
    GGML_GTX_OP_GEGLU,
    GGML_GTX_OP_REGLU,
    GGML_GTX_OP_GELU_ERF,
    GGML_GTX_OP_XIELU,
    GGML_GTX_OP_SWIGLU_OAI,
    GGML_GTX_OP_GEGLU_ERF,
    GGML_GTX_OP_GEGLU_QUICK,
    GGML_GTX_OP_EXPM1,
    GGML_GTX_OP_ARGMAX_OP,
    GGML_GTX_OP_COUNT_EQUAL,
    GGML_GTX_OP_RWKV_WKV6,
    GGML_GTX_OP_RWKV_WKV7,

    GGML_GTX_OP_COUNT,
    GGML_GTX_OP_CONV1D,
    GGML_GTX_OP_CONV2D,
    GGML_GTX_OP_MAXPOOL1D,
    GGML_GTX_OP_MAXPOOL2D,
    GGML_GTX_OP_AVG_POOL_1D,
    GGML_GTX_OP_AVG_POOL_2D,
    GGML_GTX_OP_ARGMAX,
    GGML_GTX_OP_ARGMIN,
    GGML_GTX_OP_FLATTEN,
    GGML_GTX_OP_BATCH_NORM,
    GGML_GTX_OP_SELF_ATTENTION,
    GGML_GTX_OP_MULTI_HEAD_SELF_ATTENTION,



} ggml_gtx_op;

//==================================================================
// Status Codes
//==================================================================

typedef enum {
    GGML_GTX_STATUS_SUCCESS = 0,
    GGML_GTX_STATUS_ALLOC_FAILED,
    GGML_GTX_STATUS_INVALID_PARAM,
    GGML_GTX_STATUS_OP_NOT_SUPPORTED,
    GGML_GTX_STATUS_COMPUTE_FAILED,
} ggml_gtx_status;

//==================================================================
// Tensor Structure
//==================================================================

typedef struct ggml_gtx_tensor {
    ggml_gtx_type type;
    ggml_gtx_op   op;

    int64_t ne[GGML_GTX_MAX_DIMS];  // number of elements
    size_t  nb[GGML_GTX_MAX_DIMS];  // stride in bytes

    void * data;

    char name[GGML_GTX_MAX_NAME];

    int32_t op_params[GGML_GTX_MAX_OP_PARAMS / sizeof(int32_t)];

    struct ggml_gtx_tensor * src[GGML_GTX_MAX_SRC];
    struct ggml_gtx_tensor * view_src;
    size_t view_offs;

    // Flags
    bool is_param;
    bool requires_grad;
} ggml_gtx_tensor;

//==================================================================
// Computation Graph
//==================================================================

typedef struct {
    int n_nodes;
    int n_leafs;

    ggml_gtx_tensor * nodes[GGML_GTX_MAX_NODES];
    ggml_gtx_tensor * leafs[GGML_GTX_MAX_NODES];
    ggml_gtx_tensor * grads[GGML_GTX_MAX_NODES];
} ggml_gtx_cgraph;

//==================================================================
// Context (Static Memory Pool Based)
//==================================================================

typedef struct {
    void *   mem_buffer;      // Memory pool start
    size_t   mem_size;        // Total pool size
    size_t   mem_used;        // Currently used

    int      n_tensors;       // Number of allocated tensors
    ggml_gtx_tensor tensors[GGML_GTX_MAX_TENSORS];

    // Scratch buffer for intermediate computations
    void *   scratch_buffer;
    size_t   scratch_size;
    size_t   scratch_used;
} ggml_gtx_context;

//==================================================================
// Memory Pool Configuration
//==================================================================

typedef struct {
    void *   mem_buffer;      // Pre-allocated memory buffer
    size_t   mem_size;        // Size of buffer
    void *   scratch_buffer;  // Pre-allocated scratch buffer
    size_t   scratch_size;    // Size of scratch buffer
} ggml_gtx_init_params;

//==================================================================
// Core API
//==================================================================

// Initialization
ggml_gtx_context * ggml_gtx_init(ggml_gtx_init_params params);
void              ggml_gtx_reset(ggml_gtx_context * ctx);  // Reset without freeing
size_t            ggml_gtx_used_mem(const ggml_gtx_context * ctx);

// Type utilities
size_t            ggml_gtx_type_size(ggml_gtx_type type);
const char *      ggml_gtx_type_name(ggml_gtx_type type);
const char *      ggml_gtx_op_name(ggml_gtx_op op);
bool              ggml_gtx_is_quantized(ggml_gtx_type type);

// Tensor creation (from static pool)
ggml_gtx_tensor *  ggml_gtx_new_tensor(ggml_gtx_context * ctx, ggml_gtx_type type, int n_dims, const int64_t * ne);
ggml_gtx_tensor *  ggml_gtx_new_tensor_1d(ggml_gtx_context * ctx, ggml_gtx_type type, int64_t ne0);
ggml_gtx_tensor *  ggml_gtx_new_tensor_2d(ggml_gtx_context * ctx, ggml_gtx_type type, int64_t ne0, int64_t ne1);
ggml_gtx_tensor *  ggml_gtx_new_tensor_3d(ggml_gtx_context * ctx, ggml_gtx_type type, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_gtx_tensor *  ggml_gtx_new_tensor_4d(ggml_gtx_context * ctx, ggml_gtx_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// Tensor utilities
int64_t           ggml_gtx_nelements(const ggml_gtx_tensor * tensor);
size_t            ggml_gtx_nbytes(const ggml_gtx_tensor * tensor);
bool              ggml_gtx_is_contiguous(const ggml_gtx_tensor * tensor);
void              ggml_gtx_set_name(ggml_gtx_tensor * tensor, const char * name);

// Data access
void              ggml_gtx_set_f32(ggml_gtx_tensor * tensor, float value);
float             ggml_gtx_get_f32_1d(const ggml_gtx_tensor * tensor, int64_t i);
void              ggml_gtx_set_f32_1d(ggml_gtx_tensor * tensor, int64_t i, float value);

//==================================================================
// Tensor Operations
//==================================================================

// View operations (no data copy)
ggml_gtx_tensor * ggml_gtx_view_tensor(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_view_1d(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int64_t ne0, size_t offset);
ggml_gtx_tensor * ggml_gtx_view_2d(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
ggml_gtx_tensor * ggml_gtx_reshape(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_reshape_2d(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int64_t ne0, int64_t ne1);
ggml_gtx_tensor * ggml_gtx_reshape_3d(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_gtx_tensor * ggml_gtx_transpose(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_permute(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int axis0, int axis1, int axis2, int axis3);
ggml_gtx_tensor * ggml_gtx_cont(ggml_gtx_context * ctx, ggml_gtx_tensor * a);

// Arithmetic operations
ggml_gtx_tensor * ggml_gtx_add(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_sub(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_mul(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_div(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_neg(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_abs(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_sqrt(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_exp(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_log(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_scale(ggml_gtx_context * ctx, ggml_gtx_tensor * a, float s);

// Reduction operations
ggml_gtx_tensor * ggml_gtx_sum(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_sum_rows(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_mean(ggml_gtx_context * ctx, ggml_gtx_tensor * a);

// Matrix operations
ggml_gtx_tensor * ggml_gtx_mul_mat(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);

// Normalization
ggml_gtx_tensor * ggml_gtx_norm(ggml_gtx_context * ctx, ggml_gtx_tensor * a, float eps);
ggml_gtx_tensor * ggml_gtx_rms_norm(ggml_gtx_context * ctx, ggml_gtx_tensor * a, float eps);

// Activation functions
ggml_gtx_tensor * ggml_gtx_relu(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_gelu(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_silu(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_tanh(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
ggml_gtx_tensor * ggml_gtx_softmax(ggml_gtx_context * ctx, ggml_gtx_tensor * a);

// Position encoding
ggml_gtx_tensor * ggml_gtx_rope(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b,
                              int n_dims, int mode, int n_ctx_orig,
                              float freq_base, float freq_scale,
                              float ext_factor, float attn_factor,
                              float beta_fast, float beta_slow);

// Attention
ggml_gtx_tensor * ggml_gtx_flash_attn(ggml_gtx_context * ctx,
                                    ggml_gtx_tensor * q,
                                    ggml_gtx_tensor * k,
                                    ggml_gtx_tensor * v,
                                    bool masked);

// Misc
ggml_gtx_tensor * ggml_gtx_get_rows(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b);
ggml_gtx_tensor * ggml_gtx_concat(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * b, int dim);
ggml_gtx_tensor * ggml_gtx_clamp(ggml_gtx_context * ctx, ggml_gtx_tensor * a, float min, float max);


// Convolution
ggml_gtx_tensor * ggml_gtx_conv_1d(ggml_gtx_context * ctx,
                                 ggml_gtx_tensor * x,
                                 ggml_gtx_tensor * w,
                                 int64_t stride,
                                 int64_t padding);

ggml_gtx_tensor * ggml_gtx_conv_2d(ggml_gtx_context * ctx,
                                 ggml_gtx_tensor * x,
                                 ggml_gtx_tensor * w,
                                 int64_t stride,
                                 int64_t padding);        

// Max Pooling 1D
ggml_gtx_tensor * ggml_gtx_max_pool_1d(ggml_gtx_context * ctx,
                                      ggml_gtx_tensor * x,
                                      int64_t pool_size,
                                      int64_t stride,
                                      int64_t padding);
// Max Pooling 2D
ggml_gtx_tensor * ggml_gtx_max_pool_2d(ggml_gtx_context * ctx, 
                                        ggml_gtx_tensor * x,int64_t pool_size_h, 
                                        int64_t pool_size_w, int64_t stride_h, int64_t stride_w, 
                                        int64_t padding_h, int64_t padding_w);
// Average Pooling 1D
ggml_gtx_tensor * ggml_gtx_avg_pool_1d(ggml_gtx_context * ctx, 
                                        ggml_gtx_tensor * x, 
                                        int64_t pool_size, 
                                        int64_t stride, 
                                        int64_t padding);
// Average Pooling 2D
ggml_gtx_tensor * ggml_gtx_avg_pool_2d(ggml_gtx_context * ctx, 
                                        ggml_gtx_tensor * x,int64_t pool_size_h, 
                                        int64_t pool_size_w, int64_t stride_h, int64_t stride_w, 
                                        int64_t padding_h, int64_t padding_w);

// Leaky ReLU
ggml_gtx_tensor * ggml_gtx_leaky_relu(ggml_gtx_context * ctx,
                                      ggml_gtx_tensor * a,
                                      float alpha);

// argmax
ggml_gtx_tensor * ggml_gtx_argmax(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
// argmin
ggml_gtx_tensor * ggml_gtx_argmin(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
// flatten
ggml_gtx_tensor * ggml_gtx_flatten(ggml_gtx_context * ctx, ggml_gtx_tensor * a);
// upscale
ggml_gtx_tensor * ggml_gtx_upscale(ggml_gtx_context * ctx, ggml_gtx_tensor * a, int scale_factor);
// batchnorm
ggml_gtx_tensor * ggml_gtx_batch_norm(ggml_gtx_context * ctx, ggml_gtx_tensor * a, ggml_gtx_tensor * mean, ggml_gtx_tensor * var, float eps);
// self-attention
ggml_gtx_tensor * ggml_gtx_self_attention(ggml_gtx_context * ctx, ggml_gtx_tensor * q, ggml_gtx_tensor * k, ggml_gtx_tensor * v);
// multi-head self-attention
ggml_gtx_tensor * ggml_gtx_multi_head_self_attention(ggml_gtx_context * ctx, ggml_gtx_tensor * q, ggml_gtx_tensor * k, ggml_gtx_tensor * v, int num_heads);



//==================================================================
// Computation Graph API
//==================================================================

void              ggml_gtx_graph_init(ggml_gtx_cgraph * graph);
void              ggml_gtx_build_forward(ggml_gtx_cgraph * graph, ggml_gtx_tensor * tensor);
ggml_gtx_status    ggml_gtx_graph_compute(ggml_gtx_context * ctx, ggml_gtx_cgraph * graph);

//==================================================================
// Debug / Utility
//==================================================================

void              ggml_gtx_print_tensor(const ggml_gtx_tensor * tensor);
void              ggml_gtx_abort(const char * file, int line, const char * msg);

#ifdef __cplusplus
}
#endif

#endif // GGML_BAREMETAL_H
