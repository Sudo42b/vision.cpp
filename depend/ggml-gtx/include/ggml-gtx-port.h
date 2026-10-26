//==================================================================
// GGML Baremetal - Platform Abstraction Layer
//
// This file provides platform-specific implementations that must
// be customized for your target hardware.
//
// Users should implement these functions for their specific
// baremetal environment.
//==================================================================

#ifndef GGML_GTX_PORT_H
#define GGML_GTX_PORT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//==================================================================
// Memory Configuration
//
// Define these in your linker script or platform header:
//   - DDR_BASE: Start address of DDR memory
//   - DDR_SIZE: Total DDR size
//   - L2_BASE: Start address of L2 scratchpad (if available)
//   - L2_SIZE: L2 scratchpad size
//   - L1_BASE: Start address of L1 scratchpad (if available)
//   - L1_SIZE: L1 scratchpad size
//==================================================================

#ifndef DDR_BASE
#define DDR_BASE        0x80000000ULL
#endif

#ifndef DDR_SIZE
#define DDR_SIZE        (116ULL * 1024 * 1024 * 1024)  // 128 GB
#endif

#ifndef L2_BASE
#define L2_BASE         0x20000000ULL
#endif

#ifndef L2_SIZE
#define L2_SIZE         (10 * 1024 * 1024)  // 10 MB on n1s1 FPGA
#endif

#ifndef L1_BASE
#define L1_BASE         0x10000000ULL
#endif

#ifndef L1_SIZE
#define L1_SIZE         (384 * 1024)  // 384 KB
#endif

//==================================================================
// Memory Management (Static Pool Based)
//
// These must be implemented by the user.
// Default implementations use simple bump allocators.
//==================================================================

// Memory pool structure
#define GGML_GTX_MAX_POOL_BLOCKS 256

typedef struct {
    size_t offset;
    size_t size;
    bool   in_use;
} ggml_gtx_pool_block;

typedef struct {
    uint8_t *            base;
    size_t               size;
    size_t               used;
    size_t               alignment;
    size_t               n_blocks;
    ggml_gtx_pool_block  blocks[GGML_GTX_MAX_POOL_BLOCKS];
} ggml_gtx_mem_pool;

// Initialize a memory pool at the given address
void ggml_gtx_pool_init(ggml_gtx_mem_pool * pool, void * base, size_t size, size_t alignment);

// Allocate from pool (returns NULL if no space)
void * ggml_gtx_pool_alloc(ggml_gtx_mem_pool * pool, size_t size);

// Free an allocation from pool (returns false for invalid or double free)
bool ggml_gtx_pool_free(ggml_gtx_mem_pool * pool, void * ptr);

// Reset pool (mark all memory as free)
void ggml_gtx_pool_reset(ggml_gtx_mem_pool * pool);

// Get remaining space
size_t ggml_gtx_pool_remaining(const ggml_gtx_mem_pool * pool);

// Get the largest contiguous free payload that can satisfy an aligned alloc
size_t ggml_gtx_pool_largest_free(const ggml_gtx_mem_pool * pool);

//==================================================================
// Memory Copy / Set Functions
//
// Optimized implementations for your platform.
// Default implementations use simple loops.
//==================================================================

void * ggml_gtx_memcpy(void * dst, const void * src, size_t n);
void * ggml_gtx_memset(void * dst, int c, size_t n);
void * ggml_gtx_memset_f32(float * dst, float value, size_t count);
void * ggml_gtx_memset_f16(uint16_t * dst, uint16_t value, size_t count);

//==================================================================
// DMA Operations (Optional)
//
// For hardware with DMA engines, implement these for faster
// memory transfers.
//==================================================================

typedef enum {
    GGML_GTX_DMA_DDR_TO_L2,
    GGML_GTX_DMA_L2_TO_DDR,
    GGML_GTX_DMA_DDR_TO_L1,
    GGML_GTX_DMA_L1_TO_DDR,
    GGML_GTX_DMA_L2_TO_L1,
    GGML_GTX_DMA_L1_TO_L2,
} ggml_gtx_dma_dir;

// Start async DMA transfer
int ggml_gtx_dma_start(void * dst, const void * src, size_t size, ggml_gtx_dma_dir dir);

// Wait for DMA completion
void ggml_gtx_dma_wait(int dma_handle);

// Synchronous DMA (start + wait)
void ggml_gtx_dma_copy(void * dst, const void * src, size_t size, ggml_gtx_dma_dir dir);

//==================================================================
// Timing / Performance Counters
//==================================================================

// Get current cycle count (for performance measurement)
uint64_t ggml_gtx_get_cycles(void);

// Get time in microseconds (if RTC available)
uint64_t ggml_gtx_get_time_us(void);

//==================================================================
// Debug / Logging
//==================================================================

// Print a string (must be implemented for your UART/console)
void ggml_gtx_print(const char * str);

// Printf-like function (optional, can be stubbed out)
void ggml_gtx_printf(const char * fmt, ...);

// Abort execution (for fatal errors)
void ggml_gtx_halt(void);

//==================================================================
// FP16 <-> FP32 Conversion
//
// If your hardware has native FP16 support, optimize these.
//==================================================================

uint16_t ggml_gtx_fp32_to_fp16(float f);
float    ggml_gtx_fp16_to_fp32(uint16_t h);

// Vectorized versions (for bulk conversion)
void ggml_gtx_fp32_to_fp16_row(const float * src, uint16_t * dst, size_t n);
void ggml_gtx_fp16_to_fp32_row(const uint16_t * src, float * dst, size_t n);

//==================================================================
// Hardware-Specific Accelerator Interface
//
// For NPU/accelerator integration (GTX in your case)
//==================================================================

#ifdef GGML_GTX_USE_GTX

// Include GTX intrinsics (from local intrinsics folder)
#include "intrinsics/intrin.h"
#include "intrinsics/gtx_utils.h"

// GTX-accelerated matrix multiplication
void ggml_gtx_gtx_mul_mat(
    const void * A, const void * B, void * C,
    int M, int K, int N,
    int type_a, int type_b, int type_c);

// GTX-accelerated softmax
void ggml_gtx_gtx_softmax(const void * src, void * dst, int n);

// GTX-accelerated activation functions
void ggml_gtx_gtx_gelu(const void * src, void * dst, int n);
void ggml_gtx_gtx_silu(const void * src, void * dst, int n);
void ggml_gtx_gtx_relu(const void * src, void * dst, int n);

// GTX-accelerated normalization
void ggml_gtx_gtx_rms_norm(const void * src, void * dst, int n, float eps);

// GTX synchronization
void ggml_gtx_gtx_sync(void);

#endif // GGML_GTX_USE_GTX

#ifdef __cplusplus
}
#endif

#endif // GGML_GTX_PORT_H
