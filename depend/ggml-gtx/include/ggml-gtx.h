/**
 * @file ggml-gtx.h
 * @brief GGML GTX NPU Backend Header
 *
 * Host-side GGML backend for GTX NPU accelerator.
 * Communicates with the NPU firmware via PCIe.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_GTX_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_GTX_BUILD
#            define GGML_GTX_API __declspec(dllexport) extern
#        else
#            define GGML_GTX_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_GTX_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_GTX_API extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * GTX Device Info
 *============================================================================*/

/**
 * @brief GTX device information
 */
typedef struct ggml_gtx_device_info {
    int      device_id;           /* Device index */
    char     name[64];            /* Device name */
    char     description[128];    /* Device description */
    uint64_t total_memory;        /* Total device memory in bytes */
    uint64_t free_memory;         /* Free device memory in bytes */
    int      num_nests;           /* Number of NESTs */
    int      num_spus;            /* Total number of SPUs */
    int      pcie_gen;            /* PCIe generation (1-5) */
    int      pcie_width;          /* PCIe lane width (x1-x16) */
    uint8_t  firmware_version[3]; /* Firmware version (major, minor, patch) */
} ggml_gtx_device_info_t;

/*=============================================================================
 * Backend Registration
 *============================================================================*/

/**
 * @brief Get the number of available GTX devices
 * @return Number of GTX devices
 */
GGML_GTX_API int ggml_gtx_get_device_count(void);

/**
 * @brief Get device information
 * @param device Device index
 * @param info Pointer to store device info
 * @return true on success
 */
GGML_GTX_API bool ggml_gtx_get_device_info(int device, ggml_gtx_device_info_t * info);

/**
 * @brief Initialize a GTX backend for a specific device
 * @param device Device index (0-based)
 * @return Backend handle, or NULL on failure
 */
GGML_GTX_API ggml_backend_t ggml_backend_gtx_init(int device);

/**
 * @brief Check if a backend is a GTX backend
 * @param backend Backend handle
 * @return true if GTX backend
 */
GGML_GTX_API bool ggml_backend_is_gtx(ggml_backend_t backend);

/**
 * @brief Get the device index for a GTX backend
 * @param backend Backend handle
 * @return Device index, or -1 on error
 */
GGML_GTX_API int ggml_backend_gtx_get_device(ggml_backend_t backend);

/*=============================================================================
 * Buffer Type
 *============================================================================*/

/**
 * @brief Get the buffer type for GTX device memory
 * @param device Device index
 * @return Buffer type handle
 */
GGML_GTX_API ggml_backend_buffer_type_t ggml_backend_gtx_buffer_type(int device);

/**
 * @brief Get the host buffer type for GTX (pinned memory for faster transfers)
 * @param device Device index
 * @return Buffer type handle, or NULL if not supported
 */
GGML_GTX_API ggml_backend_buffer_type_t ggml_backend_gtx_host_buffer_type(int device);

/*=============================================================================
 * Device Registration (for backend registry)
 *============================================================================*/

/**
 * @brief Get GTX backend registration
 * @return Backend registration handle
 */
GGML_GTX_API ggml_backend_reg_t ggml_backend_gtx_reg(void);

/*=============================================================================
 * PCIe Configuration
 *============================================================================*/

/**
 * @brief PCIe device path (e.g., "/dev/gtx0" or "\\.\gtx0")
 */
typedef struct ggml_gtx_pcie_config {
    const char * device_path;     /* PCIe device path */
    int          bar_index;       /* BAR index for memory mapping (default: 0) */
    size_t       msg_queue_size;  /* Message queue size (default: 64KB) */
    int          timeout_ms;      /* Operation timeout in ms (default: 5000) */
} ggml_gtx_pcie_config_t;

/**
 * @brief Initialize GTX with custom PCIe configuration
 * @param device Device index
 * @param config PCIe configuration
 * @return Backend handle, or NULL on failure
 */
GGML_GTX_API ggml_backend_t ggml_backend_gtx_init_with_config(
    int device,
    const ggml_gtx_pcie_config_t * config
);

/*=============================================================================
 * Synchronization and Status
 *============================================================================*/

/**
 * @brief Synchronize with GTX device (wait for all pending operations)
 * @param backend Backend handle
 */
GGML_GTX_API void ggml_backend_gtx_synchronize(ggml_backend_t backend);

/**
 * @brief Check if device is ready
 * @param backend Backend handle
 * @return true if ready
 */
GGML_GTX_API bool ggml_backend_gtx_is_ready(ggml_backend_t backend);

/**
 * @brief Get last error message
 * @param backend Backend handle
 * @return Error message, or NULL if no error
 */
GGML_GTX_API const char * ggml_backend_gtx_get_error(ggml_backend_t backend);

/*=============================================================================
 * Debug and Statistics
 *============================================================================*/

/**
 * @brief GTX statistics
 */
typedef struct ggml_gtx_stats {
    uint64_t tensors_set;         /* Number of tensor uploads */
    uint64_t tensors_get;         /* Number of tensor downloads */
    uint64_t bytes_sent;          /* Total bytes sent to device */
    uint64_t bytes_received;      /* Total bytes received from device */
    uint64_t graph_computes;      /* Number of graph computations */
    uint64_t total_compute_us;    /* Total compute time in microseconds */
    uint64_t total_transfer_us;   /* Total transfer time in microseconds */
} ggml_gtx_stats_t;

/**
 * @brief Get statistics for a GTX backend
 * @param backend Backend handle
 * @param stats Pointer to store statistics
 */
GGML_GTX_API void ggml_backend_gtx_get_stats(ggml_backend_t backend, ggml_gtx_stats_t * stats);

/**
 * @brief Reset statistics
 * @param backend Backend handle
 */
GGML_GTX_API void ggml_backend_gtx_reset_stats(ggml_backend_t backend);

/**
 * @brief Enable/disable debug logging
 * @param enable true to enable
 */
GGML_GTX_API void ggml_gtx_set_debug(bool enable);

#ifdef __cplusplus
}
#endif
