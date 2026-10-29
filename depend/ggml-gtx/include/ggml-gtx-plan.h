/**
 * @file ggml_gtx_plan.h
 * @brief GGML Baremetal Plan Execution API
 *
 * High-level API for NEST/SPU parallel execution.
 *
 * Execution Model:
 *   __split()
 *   __start_plan(nest_id)
 *      __start_shared(nest_id)
 *          __credit_ld()   // Signal TMU: DDR->L2 load starting
 *          __load()        // DDR -> L2
 *          __credit_chk()  // Wait for TMU completion
 *          __store()       // L2 -> DDR
 *          __credit_st()   // Signal SMU: L2->DDR store starting
 *      __end_shared(nest_id)
 *      __start_thread(spu_id)
 *          __credit_chk()  // Wait for SMU ready
 *          __load()        // L2 -> L1
 *          __credit_ld()   // Signal TMU: L1->L2 load starting
 *          // compute
 *          __store()       // L1 -> L2
 *          __credit_st()   // Signal SMU: L2->L1 store starting
 *      __end_thread(spu_id)
 *   __end_plan(nest_id)
 *   __join()
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GGML_GTX_PLAN_H
#define GGML_GTX_PLAN_H

#include <stdint.h>
#include <stdbool.h>
#include "ggml-gtx-config.h"
#include "intrinsics/intrin.h"

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Plan Context
 *============================================================================*/

/**
 * @brief Plan execution context
 */
typedef struct ggml_gtx_plan_ctx {
    uint32_t nest_mask;         /* Active NESTs bitmask */
    uint32_t spu_mask[GGML_GTX_NUM_NESTS];
    uint32_t current_nest;      /* Current NEST being configured */
    uint32_t current_spu;       /* Current SPU being configured */
    bool     in_shared;         /* Inside shared region */
    bool     in_thread;         /* Inside thread region */
} ggml_gtx_plan_ctx_t;

/**
 * @brief Tile descriptor for DMA operations
 */
typedef struct ggml_gtx_tile {
    uint64_t ddr_addr;          /* DDR address */
    uint32_t l2_offset;         /* L2 offset within NEST */
    uint32_t l1_offset;         /* L1 offset within SPU */
    uint16_t width;             /* Tile width in elements */
    uint16_t height;            /* Tile height in rows */
    uint32_t ddr_stride;        /* DDR row stride in bytes */
    uint32_t l2_stride;         /* L2 row stride in bytes */
    uint32_t l1_stride;         /* L1 row stride in bytes */
} ggml_gtx_tile_t;

/*=============================================================================
 * Plan Initialization
 *============================================================================*/

/**
 * @brief Initialize plan context with all NESTs and SPUs
 */
static inline void ggml_gtx_plan_init_all(ggml_gtx_plan_ctx_t *ctx) {
    const uint32_t nest_mask = (1u << GGML_GTX_NUM_NESTS) - 1u;
    const uint32_t spu_mask = (1u << GGML_GTX_SPUS_PER_NEST) - 1u;
    ctx->nest_mask = nest_mask;
    for (uint32_t i = 0; i < GGML_GTX_NUM_NESTS; i++) {
        ctx->spu_mask[i] = spu_mask;
    }
    ctx->current_nest = 0;
    ctx->current_spu = 0;
    ctx->in_shared = false;
    ctx->in_thread = false;
}

/**
 * @brief Initialize plan context with specific NESTs
 */
static inline void ggml_gtx_plan_init(ggml_gtx_plan_ctx_t *ctx, uint32_t nest_mask) {
    const uint32_t valid_nests = (1u << GGML_GTX_NUM_NESTS) - 1u;
    const uint32_t valid_spus = (1u << GGML_GTX_SPUS_PER_NEST) - 1u;
    ctx->nest_mask = nest_mask & valid_nests;
    for (uint32_t i = 0; i < GGML_GTX_NUM_NESTS; i++) {
        ctx->spu_mask[i] = (ctx->nest_mask & (1u << i)) ? valid_spus : 0u;
    }
    ctx->current_nest = 0;
    ctx->current_spu = 0;
    ctx->in_shared = false;
    ctx->in_thread = false;
}

/**
 * @brief Set active SPUs for a specific NEST
 */
static inline void ggml_gtx_plan_set_spus(ggml_gtx_plan_ctx_t *ctx,
                                          uint32_t nest_id,
                                          uint32_t spu_mask) {
    if (nest_id < GGML_GTX_NUM_NESTS) {
        const uint32_t valid_spus = (1u << GGML_GTX_SPUS_PER_NEST) - 1u;
        ctx->spu_mask[nest_id] = spu_mask & valid_spus;
    }
}

/*=============================================================================
 * Plan Execution Macros (Wrappers around intrinsics)
 *============================================================================*/

/**
 * @brief Start parallel execution across NESTs
 * Call before __start_plan() to fork execution
 */
#define ggml_gtx_split() __split()

/**
 * @brief End parallel execution and synchronize
 * Call after __end_plan() to join all NESTs
 */
#define ggml_gtx_join() __join()

/**
 * @brief Start plan execution on a NEST
 * @param nest_id NEST ID (0-3)
 */
#define ggml_gtx_start_plan(nest_id) __start_p(nest_id)

/**
 * @brief End plan execution on a NEST
 * @param nest_id NEST ID (0-3)
 */
#define ggml_gtx_end_plan(nest_id) __end_p(nest_id)

/**
 * @brief Start shared memory region (SMU context)
 * Operations in this region use L2 shared memory
 */
#define ggml_gtx_start_shared() __start_s()

/**
 * @brief End shared memory region
 */
#define ggml_gtx_end_shared() __end_s()

/**
 * @brief Start thread execution on an SPU
 * @param spu_id Local SPU ID within NEST (0-15)
 */
#define ggml_gtx_start_thread(spu_id) __start_t(spu_id)

/**
 * @brief End thread execution on an SPU
 * @param spu_id Local SPU ID within NEST (0-15)
 */
#define ggml_gtx_end_thread(spu_id) __end_t(spu_id)

/*=============================================================================
 * Credit-based Synchronization
 *============================================================================*/

/**
 * @brief Signal load credit (producer ready)
 * Called before DMA load to notify consumers not to access the target
 * @param target_spu SPU bitmask (for TMU notification)
 * @param target_nest NEST bitmask (for inter-NEST notification)
 */
#define ggml_gtx_credit_load(target_spu, target_nest) \
    __credit_ld(target_spu, target_nest)

/**
 * @brief Signal store credit (consumer ready)
 * Called before DMA store to notify producers not to access the source
 * @param target_spu SPU bitmask
 */
#define ggml_gtx_credit_store(target_spu) \
    __credit_st(target_spu)

/**
 * @brief Wait for credit (synchronization point)
 * Blocks until the specified SPUs have completed their operations
 * @param target_spu SPU bitmask to wait for
 */
#define ggml_gtx_credit_check(target_spu) \
    __credit_chk(target_spu)

/*=============================================================================
 * DMA Operations with Credit Management
 *============================================================================*/

/**
 * @brief Load tile from DDR to L2 with credit management
 * For use in shared region (__start_shared / __end_shared)
 */
static inline void ggml_gtx_tile_ddr_to_l2(const ggml_gtx_tile_t *tile,
                                           uint64_t target_spu_mask) {
    /* Signal threads: DDR->L2 load starting */
    __credit_ld(target_spu_mask, 0);

    /* DMA: DDR -> L2 */
    __load(tile->ddr_addr,
           GGML_GTX_L2_BASE + tile->l2_offset,
           tile->ddr_stride,
           tile->width * sizeof(uint16_t),  /* Assume FP16 */
           tile->height,
           tile->l2_stride);

    /* Memory barrier */
    __msync();
}

/**
 * @brief Store tile from L2 to DDR with credit management
 * For use in shared region
 */
static inline void ggml_gtx_tile_l2_to_ddr(const ggml_gtx_tile_t *tile,
                                           uint64_t target_spu_mask) {
    /* Wait for threads to complete their L2 writes */
    __credit_chk(target_spu_mask);

    /* DMA: L2 -> DDR */
    __store(GGML_GTX_L2_BASE + tile->l2_offset,
            tile->ddr_addr,
            tile->ddr_stride,
            tile->width * sizeof(uint16_t),
            tile->height,
            tile->l2_stride);

    /* Signal threads: L2->DDR store complete */
    __credit_st(target_spu_mask);

    /* Memory barrier */
    __msync();
}

/**
 * @brief Load tile from L2 to L1 with credit management
 * For use in thread region (__start_thread / __end_thread)
 */
static inline void ggml_gtx_tile_l2_to_l1(const ggml_gtx_tile_t *tile,
                                          uint64_t notify_mask) {
    /* Wait for SMU to complete L2 preparation */
    __credit_chk(notify_mask);

    /* DMA: L2 -> L1 (internal copy) */
    __copy(GGML_GTX_L2_BASE + tile->l2_offset,
           GGML_GTX_L1_BASE + tile->l1_offset,
           tile->l2_stride,
           tile->width * sizeof(uint16_t),
           tile->height,
           tile->l1_stride);

    /* Signal SMU: L2->L1 load starting */
    __credit_ld(notify_mask, 0);

    /* Memory barrier */
    __msync();
}

/**
 * @brief Store tile from L1 to L2 with credit management
 * For use in thread region
 */
static inline void ggml_gtx_tile_l1_to_l2(const ggml_gtx_tile_t *tile,
                                          uint64_t notify_mask) {
    /* DMA: L1 -> L2 (internal copy) */
    __copy(GGML_GTX_L1_BASE + tile->l1_offset,
           GGML_GTX_L2_BASE + tile->l2_offset,
           tile->l1_stride,
           tile->width * sizeof(uint16_t),
           tile->height,
           tile->l2_stride);

    /* Signal SMU: L1->L2 store complete */
    __credit_st(notify_mask);

    /* Memory barrier */
    __msync();
}

/*=============================================================================
 * Double-Buffered Execution Template
 *============================================================================*/

/**
 * @brief Double buffer state
 */
typedef struct ggml_gtx_double_buffer {
    uint32_t l2_buf[2];         /* L2 buffer offsets (ping-pong) */
    uint32_t l1_buf[2];         /* L1 buffer offsets (ping-pong) */
    uint32_t buf_size;          /* Buffer size in bytes */
    uint32_t current;           /* Current buffer index (0 or 1) */
} ggml_gtx_double_buffer_t;

/**
 * @brief Initialize double buffer
 */
static inline void ggml_gtx_double_buffer_init(ggml_gtx_double_buffer_t *db,
                                               uint32_t l2_base_offset,
                                               uint32_t l1_base_offset,
                                               uint32_t buf_size) {
    db->l2_buf[0] = l2_base_offset;
    db->l2_buf[1] = l2_base_offset + buf_size;
    db->l1_buf[0] = l1_base_offset;
    db->l1_buf[1] = l1_base_offset + buf_size;
    db->buf_size = buf_size;
    db->current = 0;
}

/**
 * @brief Swap double buffer
 */
static inline void ggml_gtx_double_buffer_swap(ggml_gtx_double_buffer_t *db) {
    db->current ^= 1;
}

/**
 * @brief Get current L2 buffer offset
 */
static inline uint32_t ggml_gtx_double_buffer_l2(const ggml_gtx_double_buffer_t *db) {
    return db->l2_buf[db->current];
}

/**
 * @brief Get current L1 buffer offset
 */
static inline uint32_t ggml_gtx_double_buffer_l1(const ggml_gtx_double_buffer_t *db) {
    return db->l1_buf[db->current];
}

/*=============================================================================
 * High-Level Plan Execution Helpers
 *============================================================================*/

/**
 * @brief Execute a function on all active NESTs in parallel
 * @param ctx Plan context
 * @param func Function to execute (receives nest_id)
 */
#define ggml_gtx_foreach_nest(ctx, func) do {            \
    ggml_gtx_split();                                    \
    for (uint32_t _n = 0; _n < GGML_GTX_NUM_NESTS; _n++) { \
        if ((ctx)->nest_mask & (1 << _n)) {             \
            ggml_gtx_start_plan(_n);                     \
            func(_n);                                   \
            ggml_gtx_end_plan(_n);                       \
        }                                               \
    }                                                   \
    ggml_gtx_join();                                     \
} while(0)

/**
 * @brief Execute a function on all active SPUs within current NEST
 * @param ctx Plan context
 * @param nest_id Current NEST ID
 * @param func Function to execute (receives spu_id)
 */
#define ggml_gtx_foreach_spu(ctx, nest_id, func) do {    \
    for (uint32_t _s = 0; _s < GGML_GTX_SPUS_PER_NEST; _s++) { \
        if ((ctx)->spu_mask[nest_id] & (1 << _s)) {     \
            ggml_gtx_start_thread(_s);                   \
            func(_s);                                   \
            ggml_gtx_end_thread(_s);                     \
        }                                               \
    }                                                   \
} while(0)

/*=============================================================================
 * Synchronization Barriers
 *============================================================================*/

/**
 * @brief Global barrier across all active NESTs
 */
static inline void ggml_gtx_barrier_global(void) {
    __bar();
}

/**
 * @brief Flush all pending operations
 */
static inline void ggml_gtx_flush(void) {
    __flush();
}

/**
 * @brief Memory synchronization barrier
 */
static inline void ggml_gtx_msync(void) {
    __msync();
}

/**
 * @brief Wait for specified clock cycles
 */
static inline void ggml_gtx_wait_cycles(uint32_t cycles) {
    __wait(cycles);
}

#ifdef __cplusplus
}
#endif

#endif /* GGML_GTX_PLAN_H */
