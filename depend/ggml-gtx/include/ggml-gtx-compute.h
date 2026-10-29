//==================================================================
// GGML Baremetal - GTX NPU Compute API
//
// Native GTX NPU computation interface following the NEST/SPU
// parallel execution model from reference_usage.c
//
// Execution Model:
//   __split()
//     __start_plan(nest_id)
//       __start_shared()           // DDR <-> L2 DMA (TMU)
//         __credit_ld()            // Signal to threads
//         __load() / __mcast_g2s() // DDR -> L2
//         __credit_chk()           // Wait for threads
//         __store()                // L2 -> DDR
//         __credit_st()            // Signal completion
//       __end_shared()
//       __start_thread(spu_id)     // L2 <-> L1 + Compute (SPU)
//         __credit_chk()           // Wait for shared
//         __load()                 // L2 -> L1
//         // compute (__mm, __add_vv, etc.)
//         __store()                // L1 -> L2
//         __credit_ld/st()         // Signal completion
//       __end_thread(spu_id)
//     __end_plan(nest_id)
//   __join()
//
// Author: Based on reference_usage.c patterns
//==================================================================

#ifndef GGML_GTX_COMPUTE_H
#define GGML_GTX_COMPUTE_H

#include "ggml-gtx-baremetal.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//==================================================================
// Hardware Configuration
//==================================================================

#ifndef GGML_GTX_NEST_NUM
#define GGML_GTX_NEST_NUM            1
#endif

#ifndef GGML_GTX_SPU_NUM_PER_NEST
#define GGML_GTX_SPU_NUM_PER_NEST    1
#endif

#ifndef GGML_GTX_DTYPE_SIZE
#define GGML_GTX_DTYPE_SIZE          2       // FP16
#endif

#ifndef GGML_GTX_L1_BANK_SIZE
#define GGML_GTX_L1_BANK_SIZE        (128 * 1024 * 2 + 2 * 64 * 1024)  // (128KB)*2 + 64KB*2=384 KB
#endif

//==================================================================
// Memory Addresses (from reference_usage.c)
//==================================================================

// DDR Base Addresses
#define GGML_GTX_DDR_INPUT           0x50000000ULL
#define GGML_GTX_DDR_A               0x50000000ULL
#define GGML_GTX_DDR_B               0x51000000ULL
#define GGML_GTX_DDR_C               0x52000000ULL
#define GGML_GTX_DDR_RESULT          0x60000000ULL

#ifndef GGML_GTX_FIXED_DDR_BASE
#define GGML_GTX_FIXED_DDR_BASE      0x370000000ULL
#endif
#define GGML_GTX_FIXED_ARENA_ADDR    (GGML_GTX_FIXED_DDR_BASE + 0x01100000ULL)
#define GGML_GTX_FIXED_ARENA_SIZE    0x7EA00000ULL
#define GGML_GTX_DDR_TEMP_0          (GGML_GTX_FIXED_ARENA_ADDR + GGML_GTX_FIXED_ARENA_SIZE)
#define GGML_GTX_DDR_TEMP_1          (GGML_GTX_DDR_TEMP_0 + 0x00200000ULL)

// L2 SPM Base Addresses
#define GGML_GTX_L2_INPUT            0x000000
#define GGML_GTX_L2_A                0x000000
#define GGML_GTX_L2_B                0x050000
#define GGML_GTX_L2_C                0x080000
#define GGML_GTX_L2_RESULT           0x0A0000
#define GGML_GTX_L2_BANK_OFFSET      0x100000

// L2 SPM for Transpose operations
#define GGML_GTX_L2_A_T              0x000000
#define GGML_GTX_L2_B_T              0x500000
#define GGML_GTX_L2_RESULT_T         0xA00000

// L1 SPM Bank Address Offsets
// These are the canonical address definitions.
// ggml-gtx-config.h also defines these (as both macros and enum values),
// so we guard with #ifndef to avoid redefinition.
#ifndef GGML_GTX_L1_BANK_A_OFFSET
#define GGML_GTX_L1_BANK_A_OFFSET       0x00000
#endif
#ifndef GGML_GTX_L1_BANK_B_OFFSET
#define GGML_GTX_L1_BANK_B_OFFSET       0x20000
#endif
#ifndef GGML_GTX_L1_BANK_C_OFFSET
#define GGML_GTX_L1_BANK_C_OFFSET       0x30000
#endif
#ifndef GGML_GTX_L1_BANK_R_OFFSET
#define GGML_GTX_L1_BANK_R_OFFSET       0x50000
#endif

//==================================================================
// Utility Macros
//==================================================================

#define GGML_GTX_CEIL_DIV(a, b)      (((a) + (b) - 1) / (b))
#define GGML_GTX_LAST_TILE(x, tile)  (((x) % (tile)) == 0 ? (tile) : ((x) % (tile)))
#define GGML_GTX_EFF_MASK_64(n)      ((n) == 64 ? UINT64_MAX : ((1ULL << (n)) - 1))
#define GGML_GTX_EFF_MASK_16(n)      ((n) == 16 ? UINT16_MAX : ((1ULL << (n)) - 1))

//==================================================================
// Tiling Context
//==================================================================

// Per-thread tiling information
typedef struct {
    uint32_t cal_num;           // Number of calculations for this thread
    uint32_t iteration;         // Number of full iterations
    uint32_t remain;            // Remaining calculations after iterations
    uint64_t offset;            // Byte offset in DDR
} ggml_gtx_thread_tile_t;

// Per-nest tiling information
typedef struct {
    uint32_t cal_num;           // Number of calculations for this nest
    uint8_t  eff_thread_int;    // Effective number of threads
    uint16_t eff_thread_hex;    // Bitmask of effective threads
    ggml_gtx_thread_tile_t threads[GGML_GTX_SPU_NUM_PER_NEST];
} ggml_gtx_nest_tile_t;

// Complete tiling plan for an operation
typedef struct {
    // Input dimensions
    uint16_t i_row;
    uint16_t i_col;

    // Tiling parameters
    uint32_t unit_per_thread;       // Bytes per calculation unit
    uint32_t max_cal_per_thread;    // Max calculations that fit in L1

    // Effective hardware
    uint8_t  eff_nest_int;          // Number of nests to use
    uint64_t eff_nest_hex;          // Bitmask of effective nests

    // Per-nest information
    ggml_gtx_nest_tile_t nests[GGML_GTX_NEST_NUM];
} ggml_gtx_tile_plan_t;

// Matrix multiplication tiling (more complex)
typedef struct {
    // Matrix dimensions: C[A_ROW, B_ROW] = A[A_ROW, H_COL] x B^T[B_ROW, H_COL]
    uint16_t a_row;
    uint16_t h_col;
    uint16_t b_row;
    uint16_t h_col_origin;      // Original stride for A
    uint16_t res_col_origin;    // Original stride for result

    // Tile sizes
    uint16_t tile_a_row;
    uint16_t tile_h_col;
    uint16_t tile_b_row;

    // Tile counts
    uint16_t num_tile_row_a;
    uint16_t num_tile_col_h;
    uint16_t num_tile_row_b;

    // Last tile sizes
    uint16_t last_row_a;
    uint16_t last_col_h;
    uint16_t last_row_b;

    // Effective hardware
    uint8_t  eff_nest_int;
    uint64_t eff_nest_hex;
    uint8_t  eff_thread_int[GGML_GTX_NEST_NUM];
    uint16_t eff_thread_hex[GGML_GTX_NEST_NUM];

    // Per-nest/thread tile distribution
    uint16_t num_row_a_per_spread[GGML_GTX_NEST_NUM];
    uint16_t num_row_b_per_nest[GGML_GTX_NEST_NUM];
    uint16_t num_row_a_per_thread[GGML_GTX_NEST_NUM][GGML_GTX_SPU_NUM_PER_NEST];
} ggml_gtx_matmul_tile_t;

//==================================================================
// Compute Context
//==================================================================

typedef struct {
    // Hardware configuration
    uint8_t  nest_num;
    uint8_t  spu_num_per_nest;

    // Statistics
    uint64_t total_ops;
    uint64_t dma_bytes_in;
    uint64_t dma_bytes_out;
} ggml_gtx_compute_ctx_t;

//==================================================================
// Tiling API
//==================================================================

// Initialize tiling plan for element-wise operations (add, mul, div, gelu, etc.)
void ggml_gtx_tile_plan_init(
    ggml_gtx_tile_plan_t * plan,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Initialize tiling plan for matrix multiplication
void ggml_gtx_matmul_tile_init(
    ggml_gtx_matmul_tile_t * plan,
    uint16_t a_row,
    uint16_t h_col,
    uint16_t b_row,
    uint16_t h_col_origin,
    uint16_t res_col_origin,
    uint16_t tile_a_row,
    uint16_t tile_h_col,
    uint16_t tile_b_row,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Element-wise Operations (Tensor x Tensor)
//==================================================================

// Add two tensors: R = A + B
void ggml_gtx_compute_add_tt(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Subtract two tensors: R = A - B
void ggml_gtx_compute_sub_tt(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Multiply two tensors: R = A * B
void ggml_gtx_compute_mul_tt(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Divide two tensors: R = A / B
void ggml_gtx_compute_div_tt(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Element-wise Operations (Tensor x Scalar)
//==================================================================

// Divide tensor by scalar: R = A / scalar
void ggml_gtx_compute_div_ts(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t scalar,        // FP16 format
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Multiply tensor by scalar: R = A * scalar
void ggml_gtx_compute_mul_ts(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t scalar,        // FP16 format
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Add scalar to tensor: R = A + scalar
void ggml_gtx_compute_add_ts(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t scalar,        // FP16 format
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Scale tensor by scalar and add bias: R = A * scale + bias
void ggml_gtx_compute_scale(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t scale,         // FP16 format
    uint16_t bias,          // FP16 format
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Element-wise Operations (Tensor x Vector - broadcast)
//==================================================================

// Add vector to each row: R[i] = A[i] + B
void ggml_gtx_compute_add_tv(
    uint64_t ddr_a,
    uint64_t ddr_b,         // 1D vector of size i_col
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Multiply vector with each row: R[i] = A[i] * B
void ggml_gtx_compute_mul_tv(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Activation Functions
//==================================================================

// GELU activation: R = gelu(A)
void ggml_gtx_compute_gelu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ReLU activation: R = relu(A)
void ggml_gtx_compute_relu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SiLU activation: R = silu(A) = A * sigmoid(A)
void ggml_gtx_compute_silu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Tanh activation: R = tanh(A)
void ggml_gtx_compute_tanh(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Sigmoid activation: R = sigmoid(A)
void ggml_gtx_compute_sigmoid(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SQR: R = A * A
void ggml_gtx_compute_sqr(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SQRT: R = sqrt(A)
void ggml_gtx_compute_sqrt(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// NEG: R = -A
void ggml_gtx_compute_neg(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ABS: R = abs(A)
void ggml_gtx_compute_abs(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// EXP: R = exp(A)
void ggml_gtx_compute_exp(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// LOG: R = log(A)
void ggml_gtx_compute_log(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// CLAMP: R = clamp(A, min, max)
void ggml_gtx_compute_clamp(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t min_val,
    uint16_t max_val,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SGN: R = sign(A) — returns -1, 0, or 1
void ggml_gtx_compute_sgn(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// STEP: R = step(A) — returns 0 if A < 0, 1 otherwise (Heaviside)
void ggml_gtx_compute_step(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// CEIL: R = ceil(A)
void ggml_gtx_compute_ceil(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// FLOOR: R = floor(A)
void ggml_gtx_compute_floor(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// TRUNC: R = trunc(A) — truncate towards zero
void ggml_gtx_compute_trunc(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ROUND: R = round(A) — round to nearest even
void ggml_gtx_compute_round(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SUM: R = sum(A)
void ggml_gtx_compute_sum(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SUM_ROWS: R = sum_rows(A)
void ggml_gtx_compute_sum_rows(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// MEAN: R = mean(A)
void ggml_gtx_compute_mean(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// MAX: Per-row max reduction using __max_vs (FP16 GTX ISA)
void ggml_gtx_compute_max(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// MIN: Per-row min reduction using __min_vs (FP16 GTX ISA)
void ggml_gtx_compute_min(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ARANGE: Generate sequence [start, start+step, start+2*step, ...]
// No input tensor — output only. All params as FP16 bit patterns.
void ggml_gtx_compute_arange(
    uint64_t ddr_result,
    uint16_t n_elements,
    uint16_t start_fp16,    // FP16 bit pattern
    uint16_t step_fp16,     // FP16 bit pattern
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Normalization
//==================================================================

// Layer normalization: R = (A - mean) / sqrt(var + epsilon)
void ggml_gtx_compute_layernorm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,       // FP16 format
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Layer normalization with affine: R = scale * ((A - mean) / sqrt(var + eps)) + shift
void ggml_gtx_compute_layernorm_affine(
    uint64_t ddr_input,
    uint64_t ddr_scale,
    uint64_t ddr_shift,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// RMS normalization: R = A / sqrt(mean(A^2) + epsilon)
void ggml_gtx_compute_rmsnorm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// RMS normalization over compact L2-resident rows.
void ggml_gtx_compute_rmsnorm_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// RMS normalization with scale: R = (A / sqrt(mean(A^2) + epsilon)) * scale
void ggml_gtx_compute_rmsnorm_scale(
    uint64_t ddr_input,
    uint64_t ddr_scale,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t epsilon,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Softmax
//==================================================================

// Row-wise softmax: R[i] = softmax(A[i])
void ggml_gtx_compute_softmax(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Row-wise softmax over compact L2-resident rows.
void ggml_gtx_compute_softmax_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Row-wise softmax for ggml_soft_max_ext: R = softmax(A*scale + mask)
void ggml_gtx_compute_softmax_ext(
    uint64_t ddr_input,
    uint64_t ddr_mask,
    uint64_t ddr_result,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint32_t dst_nb1,
    uint32_t dst_nb2,
    uint32_t dst_nb3,
    uint16_t mask_ne2,
    uint16_t mask_ne3,
    uint32_t mask_nb1,
    uint32_t mask_nb2,
    uint32_t mask_nb3,
    float scale,
    float max_bias
);

//==================================================================
// Matrix Multiplication
//==================================================================

// Matrix multiplication: C = A x B^T
// A: [a_row, h_col], B: [b_row, h_col], C: [a_row, b_row]
void ggml_gtx_compute_matmul(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_result,
    uint16_t a_row,
    uint16_t h_col,
    uint16_t b_row,
    uint32_t h_col_origin,      // Actual row stride of A in memory
    uint32_t res_col_origin,    // Actual row stride of result in memory
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Matrix multiplication with bias: C = A x B^T + bias
void ggml_gtx_compute_matmul_bias(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_bias,
    uint64_t ddr_result,
    uint16_t a_row,
    uint16_t h_col,
    uint16_t b_row,
    uint16_t h_col_origin,
    uint16_t res_col_origin,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Memory Operations
//==================================================================

// Copy tensor: R = A
void ggml_gtx_compute_copy(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// CONT: make a tensor contiguous using explicit byte strides from the source view.
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
    uint16_t elem_size
);

// Transpose: R = A^T (for 2D tensor)
void ggml_gtx_compute_transpose(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t depth,         // For 3D transpose
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// SAM (Segment Anything Model) Operations
//==================================================================

// GET_REL_POS: Extract relative position embeddings
// src0: [d, 2*L-1] (FP16 lookup table)
// dst:  [d, w, w]   (FP16 output)
// For each (i2, i1): pos = (w - i1 - 1) + i2; dst[i2][i1][:] = src0[pos][:]
void ggml_gtx_compute_get_rel_pos(
    uint64_t ddr_src,       // DDR address of input lookup table
    uint64_t ddr_dst,       // DDR address of output tensor
    uint16_t d,             // Embedding dimension (ne0)
    uint16_t w,             // Window size (ne1 = ne2)
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// WIN_PART: Partition 2D spatial feature map into windows
// src0: [d, w_in, h_in]    (FP16, 2D spatial)
// dst:  [d, w, w, npx*npy] (FP16, windowed output, must be pre-zeroed)
// npx = ceil(w_in/w), npy = ceil(h_in/w)
// Padding positions (outside src bounds) are left as zero.
void ggml_gtx_compute_win_part(
    uint64_t ddr_src,       // DDR address of input tensor
    uint64_t ddr_dst,       // DDR address of output tensor (pre-zeroed)
    uint16_t d,             // Embedding dimension (ne0)
    uint16_t w_in,          // Input width (ne1 of src)
    uint16_t h_in,          // Input height (ne2 of src)
    uint16_t w,             // Window size
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// WIN_UNPART: Reconstruct 2D spatial feature map from windowed partitions
// src0: [d, w, w, npx*npy] (FP16, windowed partitions)
// dst:  [d, w0, h0]        (FP16, reconstructed spatial)
// npx = ceil(w0/w)
// Reverse of WIN_PART: gathers rows from windows back to spatial positions.
void ggml_gtx_compute_win_unpart(
    uint64_t ddr_src,       // DDR address of input windowed tensor
    uint64_t ddr_dst,       // DDR address of output spatial tensor
    uint16_t d,             // Embedding dimension (ne0)
    uint16_t w0,            // Output width
    uint16_t h0,            // Output height
    uint16_t w,             // Window size
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ADD_REL_POS: Add decomposed 2D relative position bias to attention matrix
// src0 (a):  [ne0, ne1, ne2]  F32 attention matrix (ne0 = w*w)
// src1 (pw): [w, w, nh, np]   F32 width relative position bias
// src2 (ph): [w, w, nh, np]   F32 height relative position bias
// dst:       [ne0, ne1, ne2]  F32 output = src0 + bias(src1, src2)
// Constraints: ne10*ne10 == ne0, ne11*ne12 == ne1, ne13 == ne2
// Uses direct DDR F32 access (no GTX ISA — ISA supports FP16 only).
void ggml_gtx_compute_add_rel_pos(
    uint64_t ddr_src0,      // DDR address of attention matrix (F32)
    uint64_t ddr_src1,      // DDR address of width position bias (F32)
    uint64_t ddr_src2,      // DDR address of height position bias (F32)
    uint64_t ddr_dst,       // DDR address of output (F32)
    uint16_t ne10,          // Window size w (ne10 = ne11 = w)
    uint16_t ne11,          // Same as ne10 (w)
    uint16_t ne12,          // Number of heads
    uint16_t ne13,          // Number of patches
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Diffusion / Mamba Operations
//==================================================================

// TIMESTEP_EMBEDDING: Sinusoidal timestep embeddings for diffusion models
// src0: [n_timesteps]       (F32 timestep values)
// dst:  [dim, n_timesteps]  (F32 embeddings)
// For j in [0, dim/2): embed[j] = cos(t*freq), embed[j+half] = sin(t*freq)
// where freq = exp(-log(max_period) * j / half)
void ggml_gtx_compute_timestep_embedding(
    uint64_t ddr_src,       // DDR address of timestep values (F32)
    uint64_t ddr_dst,       // DDR address of output embeddings (F32)
    uint16_t n_timesteps,   // Number of timesteps
    uint16_t dim,           // Embedding dimension
    uint32_t max_period,    // Max period (typically 10000)
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SSM_CONV: 1D depthwise convolution for Mamba SSM
// src0 (conv_x): [ncs, nr, n_s]  F32 sliding window input
// src1 (weight):  [nc, nr]       F32 conv1d weight
// dst:            [nr, n_t, n_s] F32 output
// nc = d_conv, ncs = d_conv-1+n_t, nr = d_inner
void ggml_gtx_compute_ssm_conv(
    uint64_t ddr_src0,      // DDR address of conv_x (F32)
    uint64_t ddr_src1,      // DDR address of weight (F32)
    uint64_t ddr_dst,       // DDR address of output (F32)
    uint16_t nc,            // d_conv (kernel size)
    uint16_t ncs,           // d_conv - 1 + n_t (input width)
    uint16_t nr,            // d_inner
    uint16_t n_t,           // tokens per sequence
    uint16_t n_s,           // number of sequences
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SSM_SCAN: Selective State Space Model scan (Mamba)
// 7 source tensors: s, x, dt, A, B, C, ids
// dst = [y_output, updated_state]
// Supports both Mamba-1 (per-state A) and Mamba-2 (scalar A).
void ggml_gtx_compute_ssm_scan(
    uint64_t ddr_s,         // DDR: previous state [nc, nr, nh, ns+] F32
    uint64_t ddr_x,         // DDR: input [nr, nh, nt, ns] F32
    uint64_t ddr_dt,        // DDR: timestep [nh, nt, ns] F32
    uint64_t ddr_A,         // DDR: decay [1,nh] or [nc,nh] F32
    uint64_t ddr_B,         // DDR: input gate [nc, ng, nt, ns] F32
    uint64_t ddr_C,         // DDR: output gate [nc, ng, nt, ns] F32
    uint64_t ddr_ids,       // DDR: sequence ids [ns] int32
    uint64_t ddr_dst,       // DDR: output [y + s_new] F32
    uint16_t nc,            // d_state
    uint16_t nr,            // dim
    uint16_t nh,            // n_head
    uint16_t ng,            // n_group
    uint16_t nt,            // tokens per sequence
    uint16_t ns,            // number of sequences
    uint16_t a_ne0,         // A.ne[0]: 1 for Mamba-2, nc for Mamba-1
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// RoPE: Rotary Position Embedding
// src0: [ne0, ne1, ne2, ne3]  F32 input tensor
// src1: [ne2]                 int32 position indices
// dst:  [ne0, ne1, ne2, ne3]  F32 output
// Supports NORMAL (mode=0) and NEOX (mode=2) modes.
void ggml_gtx_compute_rope(
    uint64_t ddr_src0,      // DDR address of input tensor (F32)
    uint64_t ddr_src1,      // DDR address of position indices (int32)
    uint64_t ddr_dst,       // DDR address of output (F32)
    uint16_t ne0,           // Inner dimension
    uint16_t ne1,           // Number of heads
    uint16_t ne2,           // Sequence length
    uint16_t ne3,           // Batch size
    uint16_t n_dims,        // Number of rotary dimensions
    uint8_t  mode,          // 0=NORMAL, 2=NEOX
    uint32_t freq_base_b,   // Base frequency as F32 bits (default 0x461C4000 = 10000.0)
    uint32_t freq_scale_b,  // Frequency scale as F32 bits (default 0x3F800000 = 1.0)
    uint32_t attn_factor_b, // Attention factor as F32 bits (default 0x3F800000 = 1.0)
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// CONV_TRANSPOSE_1D: 1D transposed convolution (deconvolution)
// src0 (kernel): [ne00, ne01, ne02] = [K, Cout, Cin]  F32
// src1 (input):  [ne10, ne11]       = [L, Cin]         F32
// dst:           [L_out, Cout]      F32, L_out = (L-1)*s0 + K
// Uses DDR_TEMP_0/1 for workspace (kernel/input rearrangement).
void ggml_gtx_compute_conv_transpose_1d(
    uint64_t ddr_src0,      // DDR address of kernel (F32)
    uint64_t ddr_src1,      // DDR address of input (F32)
    uint64_t ddr_dst,       // DDR address of output (F32)
    uint16_t ne00,          // Kernel size (K)
    uint16_t ne01,          // Output channels (Cout)
    uint16_t ne02,          // Input channels (Cin)
    uint16_t ne10,          // Input length (L)
    uint16_t ne11,          // Input channels (Cin, == ne02)
    uint16_t s0,            // Stride
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// High-Level Graph Compute
//==================================================================

// Compute a graph using GTX NPU
ggml_gtx_status ggml_gtx_compute_graph_gtx(
    ggml_gtx_context * ctx,
    ggml_gtx_cgraph * graph
);

// Compute a single tensor operation
ggml_gtx_status ggml_gtx_compute_tensor_gtx(
    ggml_gtx_context * ctx,
    ggml_gtx_tensor * tensor
);

//==================================================================
// Masking Operations
//==================================================================

// DIAG_MASK_INF: Set elements above the shifted diagonal to -INF
// Input/output: F32 tensor [nc, nr, nz]. n_past shifts the diagonal.
// Uses CPU volatile pointer writes (no GTX ISA needed). Can be in-place.
void ggml_gtx_compute_diag_mask_inf(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t nc,            // number of columns (ne0)
    uint16_t nr,            // number of rows per head (ne1)
    uint16_t nz,            // number of heads * batch (ne2 * ne3)
    int32_t  n_past,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// DIAG_MASK_ZERO: Same as DIAG_MASK_INF but fills with 0.0f
void ggml_gtx_compute_diag_mask_zero(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t nc,
    uint16_t nr,
    uint16_t nz,
    int32_t  n_past,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Embedding / Gather Operations
//==================================================================

// GET_ROWS: Indexed row gather from embedding table.
// data[n_embd, n_vocab, n_channels, n_batches] -> dst[n_embd, n_rows, n_channels, n_batches]
// indices[n_rows, n_channels, n_batches] can be I32 or I64.
void ggml_gtx_compute_get_rows(
    uint64_t ddr_data,      // Embedding table [n_embd, n_vocab] FP16
    uint64_t ddr_indices,   // Row indices [n_rows] I32
    uint64_t ddr_dst,       // Output [n_embd, n_rows] FP16
    uint16_t n_embd,        // Embedding dimension (ne0)
    uint16_t n_vocab,
    uint16_t n_rows,
    uint16_t n_channels,
    uint16_t n_batches,
    uint16_t idx_size,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Concatenation / Repeat Operations
//==================================================================

// CONCAT: Concatenate two tensors along a specified dimension
// Both inputs same type. Output ne[dim] = a_ne[dim] + b_ne[dim].
// Uses CPU volatile pointer copies.
void ggml_gtx_compute_concat(
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_dst,
    uint16_t ne0,           // Output ne0
    uint16_t ne1,           // Output ne1
    uint16_t ne2,           // Output ne2
    uint16_t ne3,           // Output ne3
    uint16_t a_ne_dim,      // A's size along concat dim
    int32_t  dim,           // Concat dimension (0-3)
    uint16_t elem_size,     // Element size in bytes (2=FP16, 4=F32)
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// REPEAT: Tile/broadcast tensor to fill output shape
// Each ne[d] must be integer multiple of ne0[d].
// Uses CPU volatile pointer copies.
void ggml_gtx_compute_repeat(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne00,          // Source dimensions
    uint16_t ne01,
    uint16_t ne02,
    uint16_t ne03,
    uint16_t ne0,           // Output dimensions
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t elem_size,     // Element size (2=FP16, 4=F32)
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Compound Activation Operations
//==================================================================

// GELU_QUICK: x * sigmoid(1.702 * x)
// 3-step GTX ISA: __mul_vs(1.702) → __sigm → __mul_vv with bank remapping
void ggml_gtx_compute_gelu_quick(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SWIGLU: swish(x) * gate.
// If ddr_gate is 0, each input row is split into [x | gate] and output width is n_cols / 2.
// If ddr_gate is non-zero, x and gate are provided as separate tensors and output width is n_cols.
void ggml_gtx_compute_swiglu(
    uint64_t ddr_input,
    uint64_t ddr_gate,
    uint64_t ddr_result,
    uint16_t n_rows,
    uint16_t n_cols,
    uint8_t  swapped,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// Packed SWIGLU over compact L2-resident rows. Input row width is 2 * output width.
void ggml_gtx_compute_swiglu_l2(
    uint32_t l2_input,
    uint32_t l2_result,
    uint16_t n_rows,
    uint16_t n_cols,
    uint8_t  swapped,
    uint8_t  nest_id,
    uint8_t  spu_num_per_nest
);

// GLU variants: split row in half → activation(first_half) * second_half
// glu_type: 0=SWIGLU(SiLU), 1=GEGLU(GELU), 2=REGLU(ReLU)
// swapped: if true, swap gate and input halves
void ggml_gtx_compute_glu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t n_rows,        // Number of rows
    uint16_t n_cols,        // Full row width (will be split in half)
    uint8_t  glu_type,      // 0=SWIGLU, 1=GEGLU, 2=REGLU
    uint8_t  swapped,       // 0=normal, 1=swapped
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// FP16 Conversion Helpers
//==================================================================

// Convert F32 raw bits (uint32_t) to FP16 — ISS-safe, no float type used
uint16_t ggml_gtx_fp32_bits_to_fp16(uint32_t f32);

// Convert FP16 to F32 raw bits (uint32_t) — ISS-safe, no float type used
uint32_t ggml_gtx_fp16_to_fp32_bits(uint16_t value);

// Legacy wrappers using float type — WARNING: fmv.x.w returns 0 on ISS v1.1.4
uint16_t ggml_gtx_fp32_to_fp16(float value);
float ggml_gtx_fp16_to_fp32(uint16_t value);

//==================================================================
// Software F32 Arithmetic (uint32_t IEEE-754 bit manipulation)
//
// ISS v1.1.4 cannot convert float<->uint32_t (fmv.x.w returns 0).
// All pre-compute F32 math is done on raw IEEE-754 bits as uint32_t.
//==================================================================

// Well-known F32 bit constants
#define GGML_GTX_F32B_ZERO      0x00000000u  // 0.0f
#define GGML_GTX_F32B_ONE       0x3F800000u  // 1.0f
#define GGML_GTX_F32B_NEG_ONE   0xBF800000u  // -1.0f
#define GGML_GTX_F32B_TWO       0x40000000u  // 2.0f
#define GGML_GTX_F32B_HALF      0x3F000000u  // 0.5f

// Basic F32 bit arithmetic
uint32_t ggml_gtx_f32b_neg(uint32_t a);
uint32_t ggml_gtx_f32b_abs(uint32_t a);
int32_t  ggml_gtx_f32b_gt(uint32_t a, uint32_t b);
uint32_t ggml_gtx_f32b_from_int(int32_t n);
int32_t  ggml_gtx_f32b_to_int(uint32_t a);
uint32_t ggml_gtx_f32b_add(uint32_t a, uint32_t b);
uint32_t ggml_gtx_f32b_sub(uint32_t a, uint32_t b);
uint32_t ggml_gtx_f32b_mul(uint32_t a, uint32_t b);
uint32_t ggml_gtx_f32b_div(uint32_t a, uint32_t b);

// Transcendental functions (F32 bits in/out)
uint32_t ggml_gtx_f32b_exp(uint32_t x);       // exp(x)
uint32_t ggml_gtx_f32b_log(uint32_t x);       // ln(x)
uint32_t ggml_gtx_f32b_softplus(uint32_t x);  // log(1 + exp(x))

//==================================================================
// FP16 Trigonometric Functions (GTX ISA)
//==================================================================

// Compute sin(x) for n FP16 elements in L1 Bank A → Bank R
// Uses Taylor/Horner approximation. Must be called within thread scope.
// Bank B, C used as scratch. Input should be in [-π, π].
void ggml_gtx_sin_v(uint32_t n);

// Compute cos(x) for n FP16 elements in L1 Bank A → Bank R
// Uses Taylor/Horner approximation. Must be called within thread scope.
// Bank B used as scratch. Input should be in [-π, π].
void ggml_gtx_cos_v(uint32_t n);

//==================================================================
// Data Operations (CPU-based, no GTX ISA)
//==================================================================

// PAD: Zero-pad a tensor on all 4 dimensions. F32 data.
void ggml_gtx_compute_pad(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t src_ne0,
    uint16_t src_ne1,
    uint16_t src_ne2,
    uint16_t src_ne3,
    uint16_t dst_ne0,
    uint16_t dst_ne1,
    uint16_t dst_ne2,
    uint16_t dst_ne3,
    uint16_t lp0,
    uint16_t lp1,
    uint16_t lp2,
    uint16_t lp3
);

// SET: Embed src1 into a sub-view of src0. F32 data.
void ggml_gtx_compute_set(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t src0_ne0,
    uint16_t src0_ne1,
    uint16_t src0_ne2,
    uint16_t src0_ne3,
    uint16_t src1_ne0,
    uint16_t src1_ne1,
    uint16_t src1_ne2,
    uint16_t src1_ne3,
    uint32_t nb1,
    uint32_t nb2,
    uint32_t nb3,
    uint32_t offset,
    bool     inplace
);

// CPY: Copy a source tensor into destination strides.
void ggml_gtx_compute_cpy(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint32_t ne0,
    uint32_t ne1,
    uint32_t ne2,
    uint32_t ne3,
    uint32_t src_ne0,
    uint32_t src_ne1,
    uint32_t src_ne2,
    uint32_t src_ne3,
    uint32_t src_nb0,
    uint32_t src_nb1,
    uint32_t src_nb2,
    uint32_t src_nb3,
    uint32_t dst_nb0,
    uint32_t dst_nb1,
    uint32_t dst_nb2,
    uint32_t dst_nb3,
    uint16_t elem_size
);

// IM2COL: Rearrange input for conv-as-matmul. F32 input, FP16 output.
void ggml_gtx_compute_im2col(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t IW,
    uint16_t IH,
    uint16_t IC,
    uint16_t N,
    uint16_t KW,
    uint16_t KH,
    int16_t  s0,
    int16_t  s1,
    int16_t  p0,
    int16_t  p1,
    int16_t  d0,
    int16_t  d1,
    bool     is_2D
);

// POOL_2D: 2D max or average pooling. F32 data.
// pool_type: 0 = MAX, 1 = AVG
void ggml_gtx_compute_pool_2d(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t W,
    uint16_t H,
    uint16_t C,
    uint16_t N,
    uint16_t OW,
    uint16_t OH,
    uint16_t k0,
    uint16_t k1,
    int16_t  s0,
    int16_t  s1,
    int16_t  p0,
    int16_t  p1,
    uint8_t  pool_type
);

// GROUP_NORM: Group normalization (Stable Diffusion UNet). F32 data.
// y = (x - mean) / sqrt(var + eps), no learnable gamma/beta.
void ggml_gtx_compute_group_norm(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    int32_t  n_groups,
    uint32_t eps_bits
);

// CONV_2D: Direct 2D convolution. F32 data.
// kernel[KW,KH,IC,OC] × input[W,H,C,N] → output[OW,OH,OC,N]
void ggml_gtx_compute_conv_2d(
    uint64_t ddr_kernel,
    uint64_t ddr_input,
    uint64_t ddr_dst,
    uint16_t W,
    uint16_t H,
    uint16_t IC,
    uint16_t N,
    uint16_t KW,
    uint16_t KH,
    uint16_t OC,
    uint16_t OW,
    uint16_t OH,
    int16_t  s0,
    int16_t  s1,
    int16_t  p0,
    int16_t  p1,
    int16_t  d0,
    int16_t  d1
);

// CONV_2D_DW: Depthwise 2D convolution. F32 data.
// kernel[KW,KH,1,C] × input[W,H,C,N] → output[OW,OH,C,N]
void ggml_gtx_compute_conv_2d_dw(
    uint64_t ddr_kernel,
    uint64_t ddr_input,
    uint64_t ddr_dst,
    uint16_t W,
    uint16_t H,
    uint16_t C,
    uint16_t N,
    uint16_t KW,
    uint16_t KH,
    uint16_t OW,
    uint16_t OH,
    int16_t  s0,
    int16_t  s1,
    int16_t  p0,
    int16_t  p1,
    int16_t  d0,
    int16_t  d1
);

// FLASH_ATTN_EXT: Fused flash attention (online softmax). F32 Q/K/V/mask.
// q[DK, N, n_head, ne3], k[DK, n_kv, n_head_kv, ne3],
// v[DV, n_kv, n_head_kv, ne3], mask[n_kv, N, ...] (FP16 or NULL)
// dst[DV, n_head, N, ne3] (permuted 0,2,1,3)
// op_params[0]=scale, op_params[1]=max_bias, op_params[2]=logit_softcap
void ggml_gtx_compute_flash_attn_ext(
    uint64_t ddr_q,
    uint64_t ddr_k,
    uint64_t ddr_v,
    uint64_t ddr_mask,
    uint64_t ddr_dst,
    uint16_t DK,
    uint16_t DV,
    uint16_t N,
    uint16_t n_kv,
    uint16_t n_head,
    uint16_t n_head_kv,
    uint16_t ne3,
    uint32_t scale_bits,
    uint32_t max_bias_bits,
    uint32_t logit_softcap_bits
);

// MUL_MAT_ID: MoE expert-indexed matrix multiply. F32 data.
// as[cols, rows, n_expert] × b[cols, n_expert_used, n_tokens] → dst[rows, n_expert_used, n_tokens]
// ids[n_expert_used, n_tokens] (I32): expert selection indices
void ggml_gtx_compute_mul_mat_id(
    uint64_t ddr_as,
    uint64_t ddr_b,
    uint64_t ddr_ids,
    uint64_t ddr_dst,
    uint16_t cols,
    uint16_t rows,
    uint16_t n_expert,
    uint16_t n_expert_used,
    uint16_t n_tokens
);

// GATED_LINEAR_ATTN: Gated linear attention (RWKV-style).
// k[S,H,T], v[S,H,T], q[S,H,T], g[S,H,T] (gate), state[S*S*H, n_seqs]
// Output: [S*H, T + S*n_seqs] (concat of output tokens and updated state)
void ggml_gtx_compute_gated_linear_attn(
    uint64_t ddr_k,
    uint64_t ddr_v,
    uint64_t ddr_q,
    uint64_t ddr_g,
    uint64_t ddr_state,
    uint64_t ddr_dst,
    uint16_t S,
    uint16_t H,
    uint16_t T,
    uint16_t n_seqs,
    uint32_t scale_bits
);

// LEAKY_RELU: y = x > 0 ? x : negative_slope * x
// GTX ISA: __lrelu (fixed 0.01) or composition for arbitrary slope.
void ggml_gtx_compute_leaky_relu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest,
    uint32_t negative_slope_bits   // op_params[0], F32 bits
);

// ELU: y = x > 0 ? x : exp(x) - 1
// GTX ISA: Two-plan — Plan 1: exp(x)-1 via __exp_v+__add_vs → TEMP,
//          Plan 2: element-wise select max(x, TEMP).
void ggml_gtx_compute_elu(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// HARDSIGMOID: y = clamp((x + 3) / 6, 0, 1)
// GTX ISA: __add_vs(3) → __mul_vs(1/6) → __clamp_min(0) → __clamp_max(1).
void ggml_gtx_compute_hardsigmoid(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// HARDSWISH: y = x * hardsigmoid(x)
// GTX ISA: Two-plan — Plan 1: hardsigmoid → TEMP, Plan 2: mul_tt(x, TEMP).
void ggml_gtx_compute_hardswish(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SOFTPLUS: y = log(1 + exp(x))
// GTX ISA: __exp_v → __add_vs(1.0) → __ln_v chain.
void ggml_gtx_compute_softplus(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// SIN: element-wise sin(x). GTX ISA FP16 tile plan with range reduction.
void ggml_gtx_compute_sin(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// COS: element-wise cos(x). GTX ISA FP16 tile plan with range reduction.
void ggml_gtx_compute_cos(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// L2_NORM: per-row L2 normalization. y = x / max(||x||_2, eps).
// GTX ISA: __mul_vv (sqr) + __accum (sum) + sqrt + __div_vs.
// Row-based: i_row = ne1*ne2*ne3, i_col = ne0.
void ggml_gtx_compute_l2_norm(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest,
    uint32_t eps_bits
);

// CUMSUM: per-row cumulative sum. y[i] = sum(x[0..i]). F32 CPU.
void ggml_gtx_compute_cumsum(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3
);

// UPSCALE: Nearest-neighbor upscaling. F32 CPU.
// dst[i] = src[floor(i * src_ne / dst_ne)] for each dim.
void ggml_gtx_compute_upscale(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t src_ne0,
    uint16_t src_ne1,
    uint16_t src_ne2,
    uint16_t src_ne3,
    uint16_t dst_ne0,
    uint16_t dst_ne1,
    uint16_t dst_ne2,
    uint16_t dst_ne3
);

// DIAG: Vector → diagonal matrix. F32 CPU.
// src[N,1,ne2,ne3] → dst[N,N,ne2,ne3], zero off-diagonal.
void ggml_gtx_compute_diag(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t n,
    uint16_t ne2,
    uint16_t ne3
);

// ROLL: Circular shift on all 4 dims. F32 CPU.
// dst[i] = src[(i - shift) mod ne] per dim.
void ggml_gtx_compute_roll(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    int32_t  s0,
    int32_t  s1,
    int32_t  s2,
    int32_t  s3
);

// PAD_REFLECT_1D: Reflect-pad dim 0. F32 CPU.
// Output ne0 = ne0 + p0 + p1. Reflects interior elements.
void ggml_gtx_compute_pad_reflect_1d(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t p0,
    uint16_t p1
);

// SET_ROWS: Scatter rows by index (inverse of GET_ROWS).
// data[nc,nr,ne2,ne3] + idx[nr,ne_idx1,ne_idx2] + src_base[nc,n_vocab,ne2,ne3]
//   -> dst[nc,n_vocab,ne2,ne3]
// indices can be I32 or I64.
void ggml_gtx_compute_set_rows(
    uint64_t ddr_data,
    uint64_t ddr_idx,
    uint64_t ddr_src_base,
    uint64_t ddr_dst,
    uint16_t nc,
    uint16_t nr,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t n_vocab,
    uint16_t ne_idx1,
    uint16_t ne_idx2,
    uint16_t idx_size
);

// OUT_PROD: Outer product. dst = src0 @ src1^T. F32 CPU.
// src0[m,n,q0,r0], src1[p,n,q1,r1] → dst[m,p,q1,r1]
// Broadcasting: src0 broadcastable to src1 on dims 2,3.
void ggml_gtx_compute_out_prod(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t ne00,          // m
    uint16_t ne01,          // n (contracted)
    uint16_t ne02,          // src0 dim 2
    uint16_t ne03,          // src0 dim 3
    uint16_t ne10,          // p
    uint16_t ne2,           // dst dim 2
    uint16_t ne3            // dst dim 3
);

// POOL_1D: 1D pooling (max or avg). F32 CPU.
// src[IW,C,ne2,ne3] → dst[OW,C,ne2,ne3]
// pool_type: 0=MAX, 1=AVG
void ggml_gtx_compute_pool_1d(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t IW,
    uint16_t C,
    uint16_t ne2,
    uint16_t ne3,
    uint16_t OW,
    uint16_t k0,
    int16_t  s0,
    int16_t  p0,
    uint8_t  pool_type
);

// CONV_TRANSPOSE_2D: Transposed 2D convolution. F32 CPU.
// kernel[KW,KH,Cout,Cin], input[SW,SH,Cin,N] → dst[OW,OH,Cout,N]
// OW = (SW-1)*stride + KW, OH = (SH-1)*stride + KH
void ggml_gtx_compute_conv_transpose_2d(
    uint64_t ddr_kernel,
    uint64_t ddr_input,
    uint64_t ddr_dst,
    uint16_t KW,
    uint16_t KH,
    uint16_t Cout,
    uint16_t Cin,
    uint16_t SW,
    uint16_t SH,
    uint16_t N,
    uint16_t OW,
    uint16_t OH,
    int16_t  stride
);

// GELU_ERF: Exact GELU via error function.
// GTX ISA: __gelu activation intrinsic (R→A direction).
void ggml_gtx_compute_gelu_erf(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// XIELU: Xi Exponential Linear Unit (parameterized). F32 CPU.
// Params are pre-computed by graph constructor.
void ggml_gtx_compute_xielu(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint32_t n_total,
    float alpha_n,
    float alpha_p,
    float beta,
    float eps
);

// SWIGLU_OAI: OpenAI SwiGLU variant. F32 CPU.
// out_glu = min(x,limit) * sigmoid(alpha * min(x,limit))
// dst = out_glu * (clamp(g, -limit, limit) + 1)
// Supports single-tensor (row split in half) or split-tensor (src0=x, src1=g).
// For single-tensor: n_cols = full row width, split into two halves.
// For split-tensor: pass src1_addr != 0, n_cols = row width (no halving).
void ggml_gtx_compute_swiglu_oai(
    uint64_t ddr_src0,
    uint64_t ddr_src1,      // 0 if single-tensor (split in half)
    uint64_t ddr_dst,
    uint16_t n_rows,
    uint16_t n_cols,        // Full row width (if single-tensor) or per-tensor width (if split)
    uint8_t  swapped,       // 0=normal, 1=swapped (only for single-tensor)
    float    alpha,         // default 1.702
    float    limit          // default 7.0
);

// GEGLU_ERF: Gated GELU with erf activation. F32 CPU.
// dst = (0.5 * x * (1 + erf(x / sqrt(2)))) * g
// Supports single-tensor (row split in half) or split-tensor.
void ggml_gtx_compute_geglu_erf(
    uint64_t ddr_src0,
    uint64_t ddr_src1,      // 0 if single-tensor (split in half)
    uint64_t ddr_dst,
    uint16_t n_rows,
    uint16_t n_cols,        // Full row width (if single-tensor) or per-tensor width (if split)
    uint8_t  swapped        // 0=normal, 1=swapped (only for single-tensor)
);

// GEGLU_QUICK: Gated GELU-quick with ISA. FP16 GTX ISA.
// activation(x) = x * sigmoid(1.702 * x)
// dst = activation(x) * g
// Uses __mul_vs + __sigm + __mul_vv ISA composition.
// Single-tensor only (row split in half, FP16).
void ggml_gtx_compute_geglu_quick(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t n_rows,
    uint16_t n_cols,        // Full row width (will be split in half)
    uint8_t  swapped,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

//==================================================================
// Phase 5 Batch 1: Utility / Postprocess Ops
//==================================================================

// FILL: Fill every element with a constant FP16 value.
// GTX ISA: __fill DMA intrinsic.
void ggml_gtx_compute_fill(
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest,
    uint32_t value_bits       // op_params[0], F32 bits of fill constant
);

// EXPM1: y = exp(x) - 1.
// GTX ISA: __exp_v(A→R) + __add_vs(-1.0) composition.
void ggml_gtx_compute_expm1(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest
);

// ARGMAX: For each row, find the index of the maximum element.
// src0: F32 [ne0, ne1], dst: I32 [ne1] (one index per row).
void ggml_gtx_compute_argmax(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t ne0,             // row width
    uint16_t ne1              // number of rows
);

// COUNT_EQUAL: Count elements where src0[i] == src1[i].
// src0: I32, src1: I32 (same shape). dst: single I64 scalar.
void ggml_gtx_compute_count_equal(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint32_t total_elements
);

// TRI: Triangular matrix mask.
// src0: F32 [N, N, ...], dst: same shape.
// tri_type: 0=UPPER_DIAG, 1=UPPER, 2=LOWER_DIAG, 3=LOWER
// Copies elements within the triangle; zeros outside.
void ggml_gtx_compute_tri(
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t n,               // matrix size (N, must be square ne0==ne1)
    uint16_t ne2,
    uint16_t ne3,
    int32_t  tri_type
);

// ACC: dst = copy(src0); view(dst, offset, strides) += src1
// op_params: [nb1, nb2, nb3, offset, inplace]
void ggml_gtx_compute_acc(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t ne00,
    uint16_t ne01,
    uint16_t ne02,
    uint16_t ne03,
    uint16_t ne10,
    uint16_t ne11,
    uint16_t ne12,
    uint16_t ne13,
    uint32_t nb1,
    uint32_t nb2,
    uint32_t nb3,
    uint32_t offset,
    bool     inplace
);

// ADD_ID: dst[row] = src0[row] + src1[ids[row], :]
// Used for MoE routing. src2 = I32 index tensor.
void ggml_gtx_compute_add_id(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_ids,
    uint64_t ddr_dst,
    uint16_t ne0,
    uint16_t ne1,
    uint16_t ne2,
    uint16_t ne3
);

// ARGSORT: Per-row sort returning I32 indices
// sort_order: 0=ASC, 1=DESC
void ggml_gtx_compute_argsort(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    int32_t  sort_order
);

// TOP_K: Per-row top-k selection returning I32 indices
// k = number of top elements per row
void ggml_gtx_compute_top_k(
    uint64_t ddr_input,
    uint64_t ddr_result,
    uint16_t i_row,
    uint16_t i_col,
    uint16_t k
);

// CONV_3D: 3D convolution (direct, no scratch buffer)
// kernel[OC*IC, KD, KH, KW], src[N*IC, ID, IH, IW]
// dst[N*OC, OD, OH, OW]
// op_params: [s0,s1,s2, p0,p1,p2, d0,d1,d2, IC, N, OC]
void ggml_gtx_compute_conv_3d(
    uint64_t ddr_kernel,
    uint64_t ddr_src,
    uint64_t ddr_dst,
    uint16_t IW, uint16_t IH, uint16_t ID,
    uint16_t KW, uint16_t KH, uint16_t KD,
    uint16_t OW, uint16_t OH, uint16_t OD,
    int16_t  s0, int16_t  s1, int16_t  s2,
    int16_t  p0, int16_t  p1, int16_t  p2,
    int16_t  d0, int16_t  d1, int16_t  d2,
    uint16_t IC, uint16_t N, uint16_t OC
);

// RWKV_WKV6: RWKV v6 WKV attention mechanism
// 6 inputs: k[T,C], v[T,C], r[T,C], time_faaaa[C], time_decay[T,C], state[head_size,C,n_seqs]
// Output: [T,C] concatenated with new state [head_size,C,n_seqs]
void ggml_gtx_compute_rwkv_wkv6(
    uint64_t ddr_k,
    uint64_t ddr_v,
    uint64_t ddr_r,
    uint64_t ddr_time_faaaa,
    uint64_t ddr_time_decay,
    uint64_t ddr_state,
    uint64_t ddr_dst,
    uint16_t T,
    uint16_t C,
    uint16_t HEADS,
    uint16_t n_seqs
);

// RWKV_WKV7: RWKV v7 WKV attention mechanism
// 7 inputs: r[T,C], w[T,C], k[T,C], v[T,C], a[T,C], b[T,C], state[head_size,C,n_seqs]
// Output: [T,C] concatenated with new state [head_size,C,n_seqs]
void ggml_gtx_compute_rwkv_wkv7(
    uint64_t ddr_r,
    uint64_t ddr_w,
    uint64_t ddr_k,
    uint64_t ddr_v,
    uint64_t ddr_a,
    uint64_t ddr_b,
    uint64_t ddr_state,
    uint64_t ddr_dst,
    uint16_t T,
    uint16_t C,
    uint16_t HEADS,
    uint16_t n_seqs
);

// IM2COL_3D: 3D volume unfold for conv-as-GEMM
// src0 = kernel [OC*IC, KD, KH, KW] (shape only)
// src1 = input [N*IC, ID, IH, IW]
// dst  = [N*OD, OH, OW, IC*KD*KH*KW]
// op_params: [s0,s1,s2, p0,p1,p2, d0,d1,d2, IC]
void ggml_gtx_compute_im2col_3d(
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t IW, uint16_t IH, uint16_t ID,
    uint16_t KW, uint16_t KH, uint16_t KD,
    uint16_t OW, uint16_t OH, uint16_t OD,
    int16_t  s0, int16_t  s1, int16_t  s2,
    int16_t  p0, int16_t  p1, int16_t  p2,
    int16_t  d0, int16_t  d1, int16_t  d2,
    uint16_t IC, uint16_t N
);

// SOLVE_TRI: Forward substitution for lower-triangular system Ax=B
// src0 = A [n, n, B1, B2] (lower triangular), src1 = B [k, n, B1, B2]
// dst = X [k, n, B1, B2] where AX = B
void ggml_gtx_compute_solve_tri(
    uint64_t ddr_src0,
    uint64_t ddr_src1,
    uint64_t ddr_dst,
    uint16_t n,
    uint16_t k,
    uint16_t ne02,
    uint16_t ne03
);

#ifdef __cplusplus
}
#endif

#endif // GGML_GTX_COMPUTE_H
