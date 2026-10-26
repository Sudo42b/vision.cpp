/**
 * @file ggml_gtx_gtx_config.h
 * @brief GTX NPU Hardware Configuration
 *
 * Defines memory map, NEST/SPU configuration, and peripheral addresses
 * for the GTX NPU baremetal environment.
 *
 * GTX NPU Architecture:
 *   - 1 NEST (Neural Engine SubsysTem)
 *   - 1 SPU (Scalar Processing Unit) per NEST
 *
 * Memory Hierarchy:
 *   - DDR: Up to 112GB (4 channels)
 *   - L2: 10MB
 *   - L1: 384KB
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GGML_GTX_GTX_CONFIG_H
#define GGML_GTX_GTX_CONFIG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * System Clock Configuration
 *============================================================================*/

#define GGML_GTX_XIN_HZ              25000000UL      /* 25 MHz external clock */
#define GGML_GTX_CORE_HZ             800000000UL     /* 800 MHz core clock */
#define GGML_GTX_DDR_HZ              250000000UL     /* 250 MHz DDR clock */
#define GGML_GTX_UART_HZ             50000000UL      /* 50 MHz UART clock */
#define GGML_GTX_TIMER_HZ            10000000UL      /* 10 MHz timer clock */

/*=============================================================================
 * NPU Architecture Configuration
 *============================================================================*/

#define GGML_GTX_NUM_NESTS           1               /* Number of NESTs */
#define GGML_GTX_SPUS_PER_NEST       1               /* SPUs per NEST */
#define GGML_GTX_TOTAL_SPUS          1               /* Total SPUs */

/*=============================================================================
 * L2 Memory Configuration (Per-NEST Scratchpad)
 *
 * The n1s1 FPGA exposes one 10MB shared scratchpad.
 * All NESTs see the SAME address (0x2000_0000).
 * Hardware routes accesses based on NEST context set by __start_plan(nest_id).
 *============================================================================*/

#define GGML_GTX_L2_BASE             0x20000000UL            /* Same for all NESTs */
#define GGML_GTX_L2_SIZE_PER_NEST    (10UL * 1024 * 1024)    /* 10MB visible */
#define GGML_GTX_L2_TOTAL_PHYSICAL   GGML_GTX_L2_SIZE_PER_NEST

/* Get L2 base address (same for all NESTs) */
static inline uintptr_t ggml_gtx_l2_base(unsigned nest) {
    (void)nest;  /* NEST selection is via __start_plan(), not address */
    return GGML_GTX_L2_BASE;
}

/*=============================================================================
 * L1 Memory Configuration (Per-SPU Scratchpad)
 *
 * L1 is a 384KB private scratchpad per SPU, totaling 24MB physical.
 * All SPUs see the SAME address (0x1000_0000).
 * Hardware routes accesses based on SPU context set by __start_plan().
 *
 * Bank Layout (per SPU, all SPUs see same addresses):
 *   Bank A: 0x1000_0000 - 0x1001_FFFF (128KB) - Input/Weight data
 *   Bank B: 0x1002_0000 - 0x1002_FFFF (64KB)  - Additional input
 *   Bank C: 0x1003_0000 - 0x1004_FFFF (128KB) - Output data
 *   Bank R: 0x1005_0000 - 0x1005_FFFF (64KB)  - Accumulator/Scratch
 *============================================================================*/

#define GGML_GTX_L1_BASE             0x10000000UL            /* Same for all SPUs */
#define GGML_GTX_L1_SIZE_PER_SPU     (384UL * 1024)          /* 384KB visible per SPU */
#define GGML_GTX_L1_TOTAL_PHYSICAL   GGML_GTX_L1_SIZE_PER_SPU

/* L1 Bank Offsets (within each SPU's 384KB) */
#define GGML_GTX_L1_BANK_A_OFFSET    0x00000     /* Bank A: 128KB */
#define GGML_GTX_L1_BANK_A_SIZE      (128UL * 1024)
#define GGML_GTX_L1_BANK_B_OFFSET    0x20000     /* Bank B: 64KB */
#define GGML_GTX_L1_BANK_B_SIZE      (64UL * 1024)
#define GGML_GTX_L1_BANK_C_OFFSET    0x30000     /* Bank C: 128KB */
#define GGML_GTX_L1_BANK_C_SIZE      (128UL * 1024)
#define GGML_GTX_L1_BANK_R_OFFSET    0x50000     /* Bank R: 64KB */
#define GGML_GTX_L1_BANK_R_SIZE      (64UL * 1024)

/* Bank identifiers */
typedef enum {
    GGML_GTX_L1_BANK_A = 0,
    GGML_GTX_L1_BANK_B = 1,
    GGML_GTX_L1_BANK_C = 2,
    GGML_GTX_L1_BANK_R = 3,
} ggml_gtx_l1_bank_t;

/* Get L1 bank offset */
static inline uint32_t ggml_gtx_l1_bank_offset(ggml_gtx_l1_bank_t bank) {
    static const uint32_t offsets[] = {
        GGML_GTX_L1_BANK_A_OFFSET,
        GGML_GTX_L1_BANK_B_OFFSET,
        GGML_GTX_L1_BANK_C_OFFSET,
        GGML_GTX_L1_BANK_R_OFFSET
    };
    return (bank < 4) ? offsets[bank] : 0;
}

/* Get L1 bank size */
static inline size_t ggml_gtx_l1_bank_size(ggml_gtx_l1_bank_t bank) {
    static const size_t sizes[] = {
        GGML_GTX_L1_BANK_A_SIZE,
        GGML_GTX_L1_BANK_B_SIZE,
        GGML_GTX_L1_BANK_C_SIZE,
        GGML_GTX_L1_BANK_R_SIZE
    };
    return (bank < 4) ? sizes[bank] : 0;
}

/*=============================================================================
 * DDR Memory Configuration
 *
 * GTX NPU supports up to 4 DDR channels with up to 32GB each (112GB total).
 * The simulator uses a unified memory model at 0x80000000.
 *============================================================================*/

/* DDR Channel Addresses (64-bit) */
#define GGML_GTX_DDR0_BASE           0x0400000000ULL     /* Channel 0: 16GB */
#define GGML_GTX_DDR1_BASE           0x0800000000ULL     /* Channel 1: 32GB */
#define GGML_GTX_DDR2_BASE           0x1000000000ULL     /* Channel 2: 32GB */
#define GGML_GTX_DDR3_BASE           0x1800000000ULL     /* Channel 3: 32GB */

#define GGML_GTX_DDR_CHANNEL0_SIZE   (16ULL * 1024 * 1024 * 1024)   /* 16GB */
#define GGML_GTX_DDR_CHANNEL1_SIZE   (32ULL * 1024 * 1024 * 1024)   /* 32GB */
#define GGML_GTX_DDR_CHANNEL2_SIZE   (32ULL * 1024 * 1024 * 1024)   /* 32GB */
#define GGML_GTX_DDR_CHANNEL3_SIZE   (32ULL * 1024 * 1024 * 1024)   /* 32GB */
#define GGML_GTX_DDR_TOTAL_SIZE      (112ULL * 1024 * 1024 * 1024)  /* 112GB */

/* Unified memory for simulator (alias at 0x80000000) */
#define GGML_GTX_UNIFIED_BASE        0x80000000UL
#define GGML_GTX_UNIFIED_SIZE        (4ULL * 1024 * 1024 * 1024)    /* 4GB */

/* Override default memory configuration for ggml_gtx_memory.h */
#undef GGML_GTX_DDR_BASE
#undef GGML_GTX_DDR_SIZE
#undef GGML_GTX_L2_SIZE
#undef GGML_GTX_L1_SIZE

#define GGML_GTX_DDR_BASE            GGML_GTX_UNIFIED_BASE
#define GGML_GTX_DDR_SIZE            GGML_GTX_UNIFIED_SIZE
#define GGML_GTX_L2_SIZE             GGML_GTX_L2_SIZE_PER_NEST   /* 10MB visible */
#define GGML_GTX_L1_SIZE             GGML_GTX_L1_SIZE_PER_SPU    /* 384KB visible */

/*=============================================================================
 * Peripheral Base Addresses
 *============================================================================*/

/* On-chip Memory */
#define GGML_GTX_ROM_RAM0_BASE       0x00000000UL
#define GGML_GTX_ROM_RAM0_SIZE       (512UL * 1024 * 1024)   /* 512MB */
#define GGML_GTX_SRAM1_BASE          0x20000000UL
#define GGML_GTX_SRAM1_SIZE          (1UL * 1024 * 1024)     /* 1MB */

/* Peripheral Region (64KB each) */
#define GGML_GTX_QSPI_BASE           0x30000000UL
#define GGML_GTX_UART_BASE           0x30010000UL
#define GGML_GTX_HOST_IF_BASE        0x30020000UL
#define GGML_GTX_HOST_UART_BASE      0x30030000UL
#define GGML_GTX_HOST_SPI_BASE       0x30040000UL
#define GGML_GTX_TIMER0_BASE         0x30050000UL
#define GGML_GTX_TIMER1_BASE         0x30060000UL
#define GGML_GTX_GPIO_BASE           0x30070000UL
#define GGML_GTX_SYS_CFG_BASE        0x30080000UL
#define GGML_GTX_WATCHDOG_BASE       0x30090000UL
#define GGML_GTX_DDR_MONITOR_BASE    0x300A0000UL
#define GGML_GTX_TEMPSENSOR_BASE     0x300B0000UL
#define GGML_GTX_DMA_CTRL_BASE       0x300C0000UL
#define GGML_GTX_I2C_MASTER_BASE     0x300D0000UL

/* DDR Controller Addresses */
#define GGML_GTX_DDR_CTRL0_BASE      0x34000000UL
#define GGML_GTX_DDR_CTRL1_BASE      0x36000000UL
#define GGML_GTX_DDR_CTRL2_BASE      0x38000000UL
#define GGML_GTX_DDR_CTRL3_BASE      0x3A000000UL
#define GGML_GTX_DDR_PHY_OFFSET      0x01000000UL

/* PCIe Addresses */
#define GGML_GTX_PCIE_DBI_BASE       0x40000000UL
#define GGML_GTX_PCIE_PHY_BASE       0x40800000UL
#define GGML_GTX_PCIE_CFG_BASE       0x40A00000UL
#define GGML_GTX_GSF_CSR_BASE        0x50000000UL
#define GGML_GTX_PCIE_SLAVE_BASE     0x60000000UL

/* SNOR (Serial NOR Flash) */
#define GGML_GTX_SNOR_BASE           0x70000000UL
#define GGML_GTX_SNOR_SIZE           (256UL * 1024 * 1024)   /* 256MB */

/*=============================================================================
 * RISC-V Timer Addresses (mtime compatible)
 *============================================================================*/

#define GGML_GTX_MTIME_BASE          0x00600000UL
#define GGML_GTX_MTIME_CTRL          (GGML_GTX_MTIME_BASE + 0x00)
#define GGML_GTX_MTIME_DIV           (GGML_GTX_MTIME_BASE + 0x04)
#define GGML_GTX_MTIME_LO            (GGML_GTX_MTIME_BASE + 0x08)
#define GGML_GTX_MTIME_HI            (GGML_GTX_MTIME_BASE + 0x0C)
#define GGML_GTX_MTIMECMP_LO         (GGML_GTX_MTIME_BASE + 0x10)
#define GGML_GTX_MTIMECMP_HI         (GGML_GTX_MTIME_BASE + 0x14)

/*=============================================================================
 * IRQ Numbers
 *============================================================================*/

typedef enum {
    /* System Peripherals (0-18) */
    GGML_GTX_IRQ_UART            = 0,
    GGML_GTX_IRQ_HOST_IF         = 1,
    GGML_GTX_IRQ_HOST_UART       = 2,
    GGML_GTX_IRQ_HOST_SPI0       = 3,
    GGML_GTX_IRQ_HOST_SPI1       = 4,
    GGML_GTX_IRQ_QSPI            = 5,
    GGML_GTX_IRQ_WATCHDOG        = 6,
    GGML_GTX_IRQ_TIMER0          = 7,
    GGML_GTX_IRQ_TIMER1          = 8,
    GGML_GTX_IRQ_TEMPSENSOR0     = 9,
    GGML_GTX_IRQ_TEMPSENSOR1     = 10,
    GGML_GTX_IRQ_TEMPSENSOR2     = 11,
    GGML_GTX_IRQ_TEMPSENSOR3     = 12,
    GGML_GTX_IRQ_DMA             = 13,
    GGML_GTX_IRQ_GPIO0           = 14,
    GGML_GTX_IRQ_GPIO1           = 15,
    GGML_GTX_IRQ_GPIO2           = 16,
    GGML_GTX_IRQ_GPIO3           = 17,
    GGML_GTX_IRQ_DDR_MONITOR     = 18,

    /* DDR Channel IRQs (19-62) */
    GGML_GTX_IRQ_DDR0_BASE       = 19,   /* 11 IRQs per channel */
    GGML_GTX_IRQ_DDR1_BASE       = 30,
    GGML_GTX_IRQ_DDR2_BASE       = 41,
    GGML_GTX_IRQ_DDR3_BASE       = 52,

    /* PCIe and Others (63-69) */
    GGML_GTX_IRQ_PCIE0           = 63,
    GGML_GTX_IRQ_PCIE1           = 64,
    GGML_GTX_IRQ_PCIE2           = 65,
    GGML_GTX_IRQ_PCIE3           = 66,
    GGML_GTX_IRQ_I2CM            = 67,
    GGML_GTX_IRQ_PMU             = 68,
    GGML_GTX_IRQ_GTX             = 69,

    GGML_GTX_IRQ_COUNT           = 70,
} ggml_gtx_irq_t;

#define GGML_GTX_DDR_IRQS_PER_CHANNEL    11

/*=============================================================================
 * NEST/SPU Utility Functions
 *============================================================================*/

/* Get NEST ID from SPU ID */
static inline unsigned ggml_gtx_spu_to_nest(unsigned spu_id) {
    return spu_id / GGML_GTX_SPUS_PER_NEST;
}

/* Get local SPU ID within NEST */
static inline unsigned ggml_gtx_spu_local_id(unsigned spu_id) {
    return spu_id % GGML_GTX_SPUS_PER_NEST;
}

/* Get global SPU ID from NEST and local ID */
static inline unsigned ggml_gtx_spu_global_id(unsigned nest, unsigned local_id) {
    return nest * GGML_GTX_SPUS_PER_NEST + local_id;
}

/* Get DDR controller base for a channel */
static inline uintptr_t ggml_gtx_ddr_ctrl_base(unsigned channel) {
    static const uintptr_t bases[] = {
        GGML_GTX_DDR_CTRL0_BASE,
        GGML_GTX_DDR_CTRL1_BASE,
        GGML_GTX_DDR_CTRL2_BASE,
        GGML_GTX_DDR_CTRL3_BASE
    };
    return (channel < 4) ? bases[channel] : 0;
}

/* Get DDR PHY base for a channel */
static inline uintptr_t ggml_gtx_ddr_phy_base(unsigned channel) {
    return ggml_gtx_ddr_ctrl_base(channel) + GGML_GTX_DDR_PHY_OFFSET;
}

/*=============================================================================
 * Memory Allocation Budgets
 *
 * Recommended memory allocation for GGML operations.
 *============================================================================*/

/* Per-NEST budgets */
#define GGML_GTX_L2_COMPUTE_BUDGET   (8UL * 1024 * 1024)     /* 8MB for compute */
#define GGML_GTX_L2_BUFFER_BUDGET    (2UL * 1024 * 1024)     /* 2MB for buffering */

/* Per-SPU budgets */
#define GGML_GTX_L1_INPUT_BUDGET     (192UL * 1024)          /* Banks A+B: 192KB */
#define GGML_GTX_L1_OUTPUT_BUDGET    (128UL * 1024)          /* Bank C: 128KB */
#define GGML_GTX_L1_SCRATCH_BUDGET   (64UL * 1024)           /* Bank R: 64KB */

/* Double buffering tile sizes */
#define GGML_GTX_L2_TILE_SIZE        (2UL * 1024 * 1024)     /* 2MB per tile */
#define GGML_GTX_L1_TILE_SIZE        (64UL * 1024)           /* 64KB per tile */

#ifdef __cplusplus
}
#endif

#endif /* GGML_GTX_GTX_CONFIG_H */
