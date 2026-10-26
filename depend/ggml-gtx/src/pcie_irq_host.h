/**
 * @file pcie_irq_host.h
 * @brief GTX PCIe Host-side Interrupt Handling
 *        Supports both vfio-user (simulator) and real hardware
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ## Interrupt Architecture
 *
 * ### Host -> Device (ELBI Doorbell)
 * - Write to BAR0 + 0x0E40 sets doorbell bits
 * - Device RISC-V receives external interrupt
 * - Device reads status, handles, clears
 *
 * ### Device -> Host (MSI/MSI-X)
 * - Device writes to DBI MSI_GEN register
 * - PCIe controller generates MSI TLP
 * - Host receives via eventfd (vfio-user) or IRQ (hardware)
 *
 * ## Usage (vfio-user mode)
 *
 * ```cpp
 * #include "pcie_irq_host.h"
 *
 * gtx_pcie_irq_ctx ctx;
 * gtx_pcie_irq_init(&ctx, "$HOME/.gtx-vfio.sock");
 *
 * // Register MSI handler
 * gtx_pcie_irq_register_msi(&ctx, 0, on_infer_done, user_data);
 *
 * // Send doorbell to device
 * gtx_pcie_irq_send_doorbell(&ctx, GTX_DOORBELL_CMD_READY);
 *
 * // Wait for MSI (blocking)
 * gtx_pcie_irq_wait_msi(&ctx, 0, 5000);  // 5 second timeout
 *
 * gtx_pcie_irq_cleanup(&ctx);
 * ```
 */

#ifndef GTX_PCIE_IRQ_HOST_H
#define GTX_PCIE_IRQ_HOST_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Constants
 *============================================================================*/

/* BAR0 offsets */
#define GTX_BAR0_ELBI_DOORBELL      0x0E40
#define GTX_BAR0_ELBI_MASK          0x0E44
#define GTX_BAR0_ELBI_CLEAR         0x0E48

/* Doorbell sources (Host -> Device) */
#define GTX_DOORBELL_CMD_READY      0
#define GTX_DOORBELL_DMA_DONE       1
#define GTX_DOORBELL_CONFIG_UPDATE  2
#define GTX_DOORBELL_ABORT          3
#define GTX_DOORBELL_SYNC           4
#define GTX_DOORBELL_MSG_AVAILABLE  5
#define GTX_DOORBELL_MAX            32

/* MSI vectors (Device -> Host) - matches GGML_GTX_PCIE_MSI_* in pcie-elbi-regs.h */
#define GTX_MSI_CMD_ACK             0
#define GTX_MSI_INFER_DONE          1
#define GTX_MSI_ERROR               2
#define GTX_MSI_DMA_COMPLETE        3
#define GTX_MSI_MSG_SENT            4
#define GTX_MSI_STATUS_CHANGE       5
#define GTX_MSI_HEARTBEAT           6
#define GTX_MSI_ABORT_ACK           7
#define GTX_MSI_MAX                 32

/*=============================================================================
 * Types
 *============================================================================*/

/* MSI handler callback */
typedef void (*gtx_msi_handler_t)(int vector, void* user_data);

/* IRQ context (opaque) */
typedef struct gtx_pcie_irq_ctx gtx_pcie_irq_ctx;

/*=============================================================================
 * Initialization
 *============================================================================*/

/**
 * @brief Create IRQ context for vfio-user mode
 * @param socket_path vfio-user socket path (e.g., "$HOME/.gtx-vfio.sock")
 * @return IRQ context, or NULL on error
 *
 * Connects to the GTX_ISS simulator via vfio-user socket.
 */
gtx_pcie_irq_ctx* gtx_pcie_irq_create_vfio(const char* socket_path);

/**
 * @brief Create IRQ context for real hardware
 * @param device_path Device path (e.g., "/dev/vfio/42")
 * @param group_id VFIO group ID
 * @return IRQ context, or NULL on error
 */
gtx_pcie_irq_ctx* gtx_pcie_irq_create_hw(const char* device_path, int group_id);

/**
 * @brief Create IRQ context with existing BAR mapping
 * @param bar0_base BAR0 mapped address
 * @param bar0_size BAR0 size
 * @return IRQ context, or NULL on error
 */
gtx_pcie_irq_ctx* gtx_pcie_irq_create_mapped(volatile void* bar0_base, size_t bar0_size);

/**
 * @brief Destroy IRQ context
 * @param ctx IRQ context
 */
void gtx_pcie_irq_destroy(gtx_pcie_irq_ctx* ctx);

/*=============================================================================
 * MSI Handling (Device -> Host)
 *============================================================================*/

/**
 * @brief Register MSI handler
 * @param ctx IRQ context
 * @param vector MSI vector (0-31)
 * @param handler Callback function
 * @param user_data User data passed to callback
 * @return 0 on success, -1 on error
 */
int gtx_pcie_irq_register_msi(gtx_pcie_irq_ctx* ctx, int vector,
                               gtx_msi_handler_t handler, void* user_data);

/**
 * @brief Unregister MSI handler
 * @param ctx IRQ context
 * @param vector MSI vector
 */
void gtx_pcie_irq_unregister_msi(gtx_pcie_irq_ctx* ctx, int vector);

/**
 * @brief Wait for MSI (blocking)
 * @param ctx IRQ context
 * @param vector MSI vector to wait for (-1 for any)
 * @param timeout_ms Timeout in milliseconds (-1 for infinite)
 * @return Vector number received, or -1 on timeout/error
 */
int gtx_pcie_irq_wait_msi(gtx_pcie_irq_ctx* ctx, int vector, int timeout_ms);

/**
 * @brief Poll for MSI (non-blocking)
 * @param ctx IRQ context
 * @return Vector number if MSI pending, or -1 if none
 */
int gtx_pcie_irq_poll_msi(gtx_pcie_irq_ctx* ctx);

/**
 * @brief Get pending MSI mask
 * @param ctx IRQ context
 * @return Bitmask of pending MSI vectors
 */
uint32_t gtx_pcie_irq_get_pending_msi(gtx_pcie_irq_ctx* ctx);

/**
 * @brief Clear pending MSI
 * @param ctx IRQ context
 * @param vector MSI vector to clear
 */
void gtx_pcie_irq_clear_msi(gtx_pcie_irq_ctx* ctx, int vector);

/*=============================================================================
 * Doorbell (Host -> Device)
 *============================================================================*/

/**
 * @brief Send doorbell to device
 * @param ctx IRQ context
 * @param source Doorbell source (0-31)
 * @return 0 on success, -1 on error
 *
 * Writes (1 << source) to BAR0 + 0x0E40.
 */
int gtx_pcie_irq_send_doorbell(gtx_pcie_irq_ctx* ctx, int source);

/**
 * @brief Send multiple doorbells
 * @param ctx IRQ context
 * @param sources Bitmask of sources
 * @return 0 on success, -1 on error
 */
int gtx_pcie_irq_send_doorbell_mask(gtx_pcie_irq_ctx* ctx, uint32_t sources);

/*=============================================================================
 * Convenience Functions
 *============================================================================*/

/**
 * @brief Notify device: command ready
 */
static inline int gtx_notify_cmd_ready(gtx_pcie_irq_ctx* ctx) {
    return gtx_pcie_irq_send_doorbell(ctx, GTX_DOORBELL_CMD_READY);
}

/**
 * @brief Notify device: DMA done
 */
static inline int gtx_notify_dma_done(gtx_pcie_irq_ctx* ctx) {
    return gtx_pcie_irq_send_doorbell(ctx, GTX_DOORBELL_DMA_DONE);
}

/**
 * @brief Notify device: abort request
 */
static inline int gtx_notify_abort(gtx_pcie_irq_ctx* ctx) {
    return gtx_pcie_irq_send_doorbell(ctx, GTX_DOORBELL_ABORT);
}

/**
 * @brief Wait for inference done MSI
 */
static inline int gtx_wait_infer_done(gtx_pcie_irq_ctx* ctx, int timeout_ms) {
    return gtx_pcie_irq_wait_msi(ctx, GTX_MSI_INFER_DONE, timeout_ms);
}

/**
 * @brief Wait for command ACK MSI
 */
static inline int gtx_wait_cmd_ack(gtx_pcie_irq_ctx* ctx, int timeout_ms) {
    return gtx_pcie_irq_wait_msi(ctx, GTX_MSI_CMD_ACK, timeout_ms);
}

/*=============================================================================
 * vfio-user RPC Communication
 *============================================================================*/

/**
 * @brief Write data to device BAR region via vfio-user
 * @param ctx IRQ context
 * @param region BAR region index (0 for BAR0, 2 for BAR2)
 * @param offset Offset within BAR
 * @param data Data to write
 * @param size Data size in bytes
 * @return 0 on success, -1 on error
 */
int gtx_pcie_irq_region_write(gtx_pcie_irq_ctx* ctx, uint32_t region,
                               uint64_t offset, const void* data, size_t size);

/**
 * @brief Read data from device BAR region via vfio-user
 * @param ctx IRQ context
 * @param region BAR region index (0 for BAR0, 2 for BAR2)
 * @param offset Offset within BAR
 * @param data Buffer to receive data
 * @param size Data size in bytes
 * @return 0 on success, -1 on error
 */
int gtx_pcie_irq_region_read(gtx_pcie_irq_ctx* ctx, uint32_t region,
                              uint64_t offset, void* data, size_t size);

/**
 * @brief Get vfio-user socket fd (for advanced use)
 * @param ctx IRQ context
 * @return Socket file descriptor, or -1 if not in vfio-user mode
 */
int gtx_pcie_irq_get_socket(gtx_pcie_irq_ctx* ctx);

/*=============================================================================
 * Event Loop Integration
 *============================================================================*/

/**
 * @brief Get eventfd for MSI notifications
 * @param ctx IRQ context
 * @param vector MSI vector
 * @return eventfd file descriptor, or -1 if not supported
 *
 * Allows integration with epoll/select event loops.
 */
int gtx_pcie_irq_get_eventfd(gtx_pcie_irq_ctx* ctx, int vector);

/**
 * @brief Process pending interrupts
 * @param ctx IRQ context
 * @return Number of interrupts processed
 *
 * Calls registered handlers for pending MSIs.
 */
int gtx_pcie_irq_process(gtx_pcie_irq_ctx* ctx);

/*=============================================================================
 * Debug
 *============================================================================*/

/**
 * @brief Print IRQ status
 * @param ctx IRQ context
 */
void gtx_pcie_irq_print_status(gtx_pcie_irq_ctx* ctx);

/**
 * @brief Enable/disable debug logging for IRQ subsystem
 * @param enable true to enable debug output, false to disable
 */
void gtx_pcie_irq_set_debug(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* GTX_PCIE_IRQ_HOST_H */
