/**
 * @file pcie_irq_host.cpp
 * @brief GTX PCIe Host-side Interrupt Implementation
 *
 * Implements vfio-user binary protocol client for communicating with
 * GTX_ISS simulator running with USE_VFIO_USER=1.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pcie_irq_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/vfio.h>
/* Memory barrier for GCC/Clang */
#define MEMORY_BARRIER() __sync_synchronize()
#else
/* Memory barrier for Windows (placeholder) */
#define MEMORY_BARRIER() do {} while(0)
#endif

/*=============================================================================
 * vfio-user Protocol Definitions (Linux only)
 *============================================================================*/

#ifdef __linux__

/* vfio-user protocol version */
#define LIB_VFIO_USER_MAJOR     0
#define LIB_VFIO_USER_MINOR     1

/* vfio-user commands */
enum vfio_user_command {
    VFIO_USER_VERSION                   = 1,
    VFIO_USER_DMA_MAP                   = 2,
    VFIO_USER_DMA_UNMAP                 = 3,
    VFIO_USER_DEVICE_GET_INFO           = 4,
    VFIO_USER_DEVICE_GET_REGION_INFO    = 5,
    VFIO_USER_DEVICE_GET_REGION_IO_FDS  = 6,
    VFIO_USER_DEVICE_GET_IRQ_INFO       = 7,
    VFIO_USER_DEVICE_SET_IRQS           = 8,
    VFIO_USER_REGION_READ               = 9,
    VFIO_USER_REGION_WRITE              = 10,
    VFIO_USER_DMA_READ                  = 11,
    VFIO_USER_DMA_WRITE                 = 12,
    VFIO_USER_DEVICE_RESET              = 13,
    VFIO_USER_MAX,
};

/* vfio-user message header flags */
#define VFIO_USER_F_TYPE_COMMAND    (0x0)
#define VFIO_USER_F_TYPE_REPLY      (0x1)
#define VFIO_USER_F_NO_REPLY        (0x10)
#define VFIO_USER_F_ERROR           (0x20)

/* vfio-user message header */
struct vfio_user_header {
    uint16_t    msg_id;
    uint16_t    cmd;
    uint32_t    msg_size;
    uint32_t    flags;
    uint32_t    error_no;
} __attribute__((packed));

/* vfio-user version */
struct vfio_user_version {
    uint16_t    major;
    uint16_t    minor;
    uint8_t     data[];
} __attribute__((packed));

/* vfio-user device info */
struct vfio_user_device_info {
    uint32_t    argsz;
    uint32_t    flags;
    uint32_t    num_regions;
    uint32_t    num_irqs;
} __attribute__((packed));

/* vfio-user region access */
struct vfio_user_region_access {
    uint64_t    offset;
    uint32_t    region;
    uint32_t    count;
    uint8_t     data[];
} __attribute__((packed));

/* PCIe region indices */
#define VFU_PCI_DEV_BAR0_REGION_IDX     0
#define VFU_PCI_DEV_BAR1_REGION_IDX     1
#define VFU_PCI_DEV_BAR2_REGION_IDX     2
#define VFU_PCI_DEV_BAR3_REGION_IDX     3
#define VFU_PCI_DEV_BAR4_REGION_IDX     4
#define VFU_PCI_DEV_BAR5_REGION_IDX     5
#define VFU_PCI_DEV_ROM_REGION_IDX      6
#define VFU_PCI_DEV_CFG_REGION_IDX      7
#define VFU_PCI_DEV_VGA_REGION_IDX      8
#define VFU_PCI_DEV_NUM_REGIONS         9

#endif /* __linux__ */

/*=============================================================================
 * Debug Logging
 *============================================================================*/

static bool g_irq_debug = false;
static std::atomic<bool> g_irq_connection_broken(false);

#define IRQ_LOG(...) \
    do { if (g_irq_debug) fprintf(stderr, "[GTX-IRQ] " __VA_ARGS__); } while(0)

#define IRQ_LOG_ERROR(...) \
    fprintf(stderr, "[GTX-IRQ ERROR] " __VA_ARGS__)

static void irq_log_connection_broken_once(const char * msg) {
    bool expected = false;
    if (g_irq_connection_broken.compare_exchange_strong(expected, true)) {
        IRQ_LOG_ERROR("%s\n", msg);
    }
}

static bool irq_is_connection_broken_errno(int err) {
    return err == EPIPE || err == ECONNRESET || err == ENOTCONN;
}

/*=============================================================================
 * IRQ Context Structure
 *============================================================================*/

struct gtx_pcie_irq_ctx {
    /* Mode */
    enum {
        MODE_VFIO_USER,
        MODE_HARDWARE,
        MODE_MAPPED
    } mode;

    /* BAR0 mapping */
    volatile uint8_t* bar0_base;
    size_t bar0_size;
    bool bar0_owned;  /* true if we should unmap on destroy */

    /* vfio-user socket (for simulator) */
    int vfio_socket;

    /* MSI handlers */
    struct {
        gtx_msi_handler_t handler;
        void* user_data;
    } msi_handlers[GTX_MSI_MAX];

    /* MSI eventfds (one per vector for epoll integration) */
    int msi_eventfds[GTX_MSI_MAX];

    /* Pending MSI mask (atomic for thread safety) */
    std::atomic<uint32_t> pending_msi;

    /* MSI polling thread */
    std::thread msi_thread;
    std::atomic<bool> msi_thread_running;

    /* Mutex for handler registration */
    std::mutex handler_mutex;

    gtx_pcie_irq_ctx() {
        mode = MODE_MAPPED;
        bar0_base = nullptr;
        bar0_size = 0;
        bar0_owned = false;
        vfio_socket = -1;
        pending_msi = 0;
        msi_thread_running = false;

        memset(msi_handlers, 0, sizeof(msi_handlers));
        for (int i = 0; i < GTX_MSI_MAX; i++) {
            msi_eventfds[i] = -1;
        }
    }

    ~gtx_pcie_irq_ctx() {
        /* Stop MSI thread */
        if (msi_thread_running) {
            msi_thread_running = false;
            if (msi_thread.joinable()) {
                msi_thread.join();
            }
        }

        /* Close eventfds */
        for (int i = 0; i < GTX_MSI_MAX; i++) {
            if (msi_eventfds[i] >= 0) {
#ifdef __linux__
                close(msi_eventfds[i]);
#endif
                msi_eventfds[i] = -1;
            }
        }

        /* Close vfio socket */
        if (vfio_socket >= 0) {
#ifdef __linux__
            close(vfio_socket);
#endif
            vfio_socket = -1;
        }

        /* Unmap BAR0 if owned */
        if (bar0_owned && bar0_base) {
#ifdef __linux__
            munmap(const_cast<uint8_t*>(bar0_base), bar0_size);
#endif
            bar0_base = nullptr;
        }
    }
};

/*=============================================================================
 * vfio-user Binary Protocol Helpers
 *============================================================================*/

#ifdef __linux__

static std::atomic<uint16_t> g_msg_id(1);

/**
 * Send vfio-user message with iovec
 */
static int vfio_send_iovec(int sock, uint16_t msg_id, bool is_reply,
                           enum vfio_user_command cmd,
                           struct iovec *iovecs, size_t nr_iovecs,
                           int *fds, int fd_count) {
    struct vfio_user_header hdr;
    struct msghdr msg;
    size_t i;

    memset(&hdr, 0, sizeof(hdr));
    hdr.msg_id = msg_id;
    hdr.cmd = cmd;
    hdr.flags = is_reply ? VFIO_USER_F_TYPE_REPLY : VFIO_USER_F_TYPE_COMMAND;

    /* Calculate total message size */
    hdr.msg_size = sizeof(hdr);
    for (i = 1; i < nr_iovecs; i++) {
        hdr.msg_size += iovecs[i].iov_len;
    }

    /* Set header as first iovec */
    iovecs[0].iov_base = &hdr;
    iovecs[0].iov_len = sizeof(hdr);

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iovecs;
    msg.msg_iovlen = nr_iovecs;

    /* Add file descriptors if provided */
    char cmsg_buf[CMSG_SPACE(sizeof(int) * 32)];
    if (fds != NULL && fd_count > 0) {
        msg.msg_control = cmsg_buf;
        msg.msg_controllen = CMSG_SPACE(sizeof(int) * fd_count);

        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int) * fd_count);
        memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * fd_count);
    }

    ssize_t ret = sendmsg(sock, &msg, MSG_NOSIGNAL);
    if (ret == -1) {
        if (irq_is_connection_broken_errno(errno)) {
            irq_log_connection_broken_once("vfio-user connection closed");
        } else {
            IRQ_LOG_ERROR("sendmsg failed: %s\n", strerror(errno));
        }
        return -1;
    }
    if ((size_t)ret < hdr.msg_size) {
        IRQ_LOG_ERROR("short write: expected=%u, actual=%zd\n", hdr.msg_size, ret);
        return -1;
    }

    return 0;
}

/**
 * Send vfio-user message with data buffer
 */
static int vfio_send(int sock, uint16_t msg_id, bool is_reply,
                     enum vfio_user_command cmd,
                     void *data, size_t data_len) {
    struct iovec iovecs[2] = {
        { nullptr, 0 },  /* Header placeholder */
        { data, data_len }
    };
    return vfio_send_iovec(sock, msg_id, is_reply, cmd, iovecs, 2, nullptr, 0);
}

/**
 * Receive vfio-user message header and optional data
 */
static int vfio_recv(int sock, struct vfio_user_header *hdr, bool is_reply,
                     uint16_t *expected_msg_id, void *data, size_t *len,
                     int *fds, size_t *nr_fds) {
    struct iovec iov = { hdr, sizeof(*hdr) };
    struct msghdr msg;
    char cmsg_buf[CMSG_SPACE(sizeof(int) * 32)];

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    if (nr_fds != NULL && *nr_fds > 0) {
        msg.msg_control = cmsg_buf;
        msg.msg_controllen = sizeof(cmsg_buf);
    }

    ssize_t ret = recvmsg(sock, &msg, 0);
    if (ret < 0) {
        if (irq_is_connection_broken_errno(errno)) {
            irq_log_connection_broken_once("vfio-user connection closed");
        } else {
            IRQ_LOG_ERROR("recvmsg failed: %s\n", strerror(errno));
        }
        return -1;
    }
    if (ret == 0) {
        irq_log_connection_broken_once("vfio-user connection closed");
        return -1;
    }
    if ((size_t)ret < sizeof(*hdr)) {
        IRQ_LOG_ERROR("short header read: %zd\n", ret);
        return -1;
    }

    /* Parse file descriptors */
    if (nr_fds != NULL) {
        *nr_fds = 0;
        for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
             cmsg != NULL;
             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
                size_t fd_size = cmsg->cmsg_len - CMSG_LEN(0);
                *nr_fds = fd_size / sizeof(int);
                if (fds != NULL) {
                    memcpy(fds, CMSG_DATA(cmsg), fd_size);
                }
                break;
            }
        }
    }

    /* Validate reply */
    if (is_reply) {
        if (expected_msg_id != NULL && hdr->msg_id != *expected_msg_id) {
            IRQ_LOG_ERROR("msg_id mismatch: expected=%u, got=%u\n",
                         *expected_msg_id, hdr->msg_id);
            return -1;
        }
        if ((hdr->flags & 0xf) != VFIO_USER_F_TYPE_REPLY) {
            IRQ_LOG_ERROR("expected reply, got command\n");
            return -1;
        }
        if (hdr->flags & VFIO_USER_F_ERROR) {
            IRQ_LOG_ERROR("server error: %u\n", hdr->error_no);
            errno = hdr->error_no;
            return -1;
        }
    }

    /* Read data if present */
    if (len != NULL && *len > 0 && hdr->msg_size > sizeof(*hdr)) {
        size_t data_size = hdr->msg_size - sizeof(*hdr);
        if (data_size > *len) {
            data_size = *len;
        }
        ret = recv(sock, data, data_size, MSG_WAITALL);
        if (ret < 0) {
            IRQ_LOG_ERROR("recv data failed: %s\n", strerror(errno));
            return -1;
        }
        *len = ret;
    }

    return 0;
}

/**
 * Send and receive vfio-user message (synchronous)
 */
static int vfio_msg(int sock, uint16_t msg_id, enum vfio_user_command cmd,
                    void *send_data, size_t send_len,
                    void *recv_data, size_t recv_len) {
    struct iovec iovecs[2] = {
        { nullptr, 0 },
        { send_data, send_len }
    };

    int ret = vfio_send_iovec(sock, msg_id, false, cmd, iovecs, 2, nullptr, 0);
    if (ret < 0) {
        return ret;
    }

    struct vfio_user_header hdr;
    size_t len = recv_len;
    ret = vfio_recv(sock, &hdr, true, &msg_id, recv_data, &len, nullptr, nullptr);
    if (ret < 0) {
        return ret;
    }
    return (int) len;
}

/**
 * Negotiate vfio-user protocol version
 */
static int vfio_negotiate(int sock) {
    /* Send version */
    struct vfio_user_version ver = { LIB_VFIO_USER_MAJOR, LIB_VFIO_USER_MINOR };
    const char *caps = "{\"capabilities\":{\"max_msg_fds\":8,\"max_data_xfer_size\":1048576}}";
    size_t caps_len = strlen(caps) + 1;

    struct iovec iovecs[3] = {
        { nullptr, 0 },
        { &ver, sizeof(ver) },
        { (void*)caps, caps_len }
    };

    uint16_t msg_id = g_msg_id++;
    int ret = vfio_send_iovec(sock, msg_id, false, VFIO_USER_VERSION,
                              iovecs, 3, nullptr, 0);
    if (ret < 0) {
        IRQ_LOG_ERROR("Failed to send version\n");
        return -1;
    }

    /* Receive version response */
    struct vfio_user_header hdr;
    char recv_buf[1024];
    size_t recv_len = sizeof(recv_buf);
    ret = vfio_recv(sock, &hdr, true, &msg_id, recv_buf, &recv_len, nullptr, nullptr);
    if (ret < 0) {
        IRQ_LOG_ERROR("Failed to receive version\n");
        return -1;
    }

    IRQ_LOG("vfio-user version negotiated\n");
    return 0;
}

/**
 * Get device info
 */
static int vfio_get_device_info(int sock, struct vfio_user_device_info *info) {
    info->argsz = sizeof(*info);

    uint16_t msg_id = g_msg_id++;
    int ret = vfio_msg(sock, msg_id, VFIO_USER_DEVICE_GET_INFO,
                       info, sizeof(*info), info, sizeof(*info));
    if (ret < 0) {
        IRQ_LOG_ERROR("Failed to get device info\n");
        return -1;
    }

    IRQ_LOG("Device info: flags=%u, regions=%u, irqs=%u\n",
           info->flags, info->num_regions, info->num_irqs);
    return 0;
}

/**
 * Write to device region (for doorbell)
 */
static int vfio_region_write(int sock, uint32_t region, uint64_t offset,
                             void *data, size_t count) {
    size_t msg_size = sizeof(struct vfio_user_region_access) + count;
    uint8_t *buf = (uint8_t*)alloca(msg_size);

    struct vfio_user_region_access *access = (struct vfio_user_region_access *)buf;
    access->offset = offset;
    access->region = region;
    access->count = count;
    memcpy(access->data, data, count);

    struct vfio_user_region_access reply;
    uint16_t msg_id = g_msg_id++;
    int ret = vfio_msg(sock, msg_id, VFIO_USER_REGION_WRITE,
                       buf, msg_size, &reply, sizeof(reply));
    if (ret < 0) {
        if (!g_irq_connection_broken.load()) {
            IRQ_LOG_ERROR("Region write failed: msg_id=%u region=%u offset=0x%lx count=%zu msg_size=%zu\n",
                          msg_id, region, (unsigned long) offset, count, msg_size);
        }
        return -1;
    }

    IRQ_LOG("Region write: region=%u offset=0x%lx count=%zu\n",
           region, (unsigned long)offset, count);
    return 0;
}

/**
 * Read from device region
 */
static int vfio_region_read(int sock, uint32_t region, uint64_t offset,
                            void *data, size_t count) {
    struct vfio_user_region_access req;
    req.offset = offset;
    req.region = region;
    req.count = count;

    size_t reply_size = sizeof(struct vfio_user_region_access) + count;
    uint8_t *reply_buf = (uint8_t*)alloca(reply_size);

    uint16_t msg_id = g_msg_id++;
    int ret = vfio_msg(sock, msg_id, VFIO_USER_REGION_READ,
                       &req, sizeof(req), reply_buf, reply_size);
    if (ret < 0) {
        if (!g_irq_connection_broken.load()) {
            IRQ_LOG_ERROR("Region read failed\n");
        }
        return -1;
    }

    if (g_irq_debug && region == VFU_PCI_DEV_BAR2_REGION_IDX && offset == 0x8000) {
        const uint32_t *rw = (const uint32_t *) reply_buf;
        IRQ_LOG("BAR2 slot reply ret=%d raw32=[%08x %08x %08x %08x %08x %08x %08x %08x]\n",
               ret, rw[0], rw[1], rw[2], rw[3], rw[4], rw[5], rw[6], rw[7]);
    }

    /* libvfio-user replies with raw region bytes, while the in-tree custom
     * vfio path can reply with vfio_user_region_access + payload.
     * Accept either encoding.
     */
    if ((size_t)ret >= count && (size_t)ret < sizeof(struct vfio_user_region_access) + count) {
        memcpy(data, reply_buf, count);
    } else {
        struct vfio_user_region_access *reply = (struct vfio_user_region_access *)reply_buf;
        memcpy(data, reply->data, count);
    }

    if (g_irq_debug && region == VFU_PCI_DEV_BAR2_REGION_IDX && offset == 0x8000 && count >= 16) {
        const uint32_t *w = (const uint32_t *) data;
        IRQ_LOG("BAR2 slot read raw ret=%d words=[%08x %08x %08x %08x]\n",
               ret, w[0], w[1], w[2], w[3]);
    }

    IRQ_LOG("Region read: region=%u offset=0x%lx count=%zu\n",
           region, (unsigned long)offset, count);
    return 0;
}

/**
 * Configure IRQ eventfds
 */
static int vfio_set_irqs(int sock, uint32_t index, uint32_t start, uint32_t count,
                         int *fds) {
    struct vfio_irq_set irq_set;
    irq_set.argsz = sizeof(irq_set);
    irq_set.flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
    irq_set.index = index;
    irq_set.start = start;
    irq_set.count = count;

    struct iovec iovecs[2] = {
        { nullptr, 0 },
        { &irq_set, sizeof(irq_set) }
    };

    uint16_t msg_id = g_msg_id++;
    int ret = vfio_send_iovec(sock, msg_id, false, VFIO_USER_DEVICE_SET_IRQS,
                              iovecs, 2, fds, count);
    if (ret < 0) {
        IRQ_LOG_ERROR("Failed to send SET_IRQS\n");
        return -1;
    }

    struct vfio_user_header hdr;
    ret = vfio_recv(sock, &hdr, true, &msg_id, nullptr, nullptr, nullptr, nullptr);
    if (ret < 0) {
        IRQ_LOG_ERROR("Failed to receive SET_IRQS response\n");
        return -1;
    }

    IRQ_LOG("IRQs configured: index=%u start=%u count=%u\n", index, start, count);
    return 0;
}

#endif /* __linux__ */

/*=============================================================================
 * MSI Polling Thread (for vfio-user mode)
 *============================================================================*/

#ifdef __linux__

/* PCIe addresses in simulator memory map */
#define PCIE_SLAVE_BASE     0x60000000
#define PCIE_CFG_BASE       0x40A00000
#define PCIE_DBI_MSI        0x40000000 + 0x200D70

/**
 * MSI eventfd monitoring thread
 * Polls all registered MSI eventfds and calls handlers when triggered
 */
static void msi_poll_thread(gtx_pcie_irq_ctx* ctx) {
    IRQ_LOG("MSI poll thread started\n");

    struct pollfd pfds[GTX_MSI_MAX];
    int nfds = 0;

    while (ctx->msi_thread_running) {
        /* Build poll array from active eventfds */
        nfds = 0;
        for (int i = 0; i < GTX_MSI_MAX; i++) {
            if (ctx->msi_eventfds[i] >= 0) {
                pfds[nfds].fd = ctx->msi_eventfds[i];
                pfds[nfds].events = POLLIN;
                pfds[nfds].revents = 0;
                nfds++;
            }
        }

        if (nfds == 0) {
            /* No eventfds registered, just sleep */
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        /* Poll with timeout */
        int ret = poll(pfds, nfds, 100);  /* 100ms timeout */
        if (ret < 0) {
            if (errno == EINTR) continue;
            IRQ_LOG_ERROR("poll failed: %s\n", strerror(errno));
            break;
        }

        if (ret == 0) {
            /* Timeout, continue */
            continue;
        }

        /* Check which eventfds were triggered */
        int pfd_idx = 0;
        for (int i = 0; i < GTX_MSI_MAX && pfd_idx < nfds; i++) {
            if (ctx->msi_eventfds[i] < 0) continue;

            if (pfds[pfd_idx].revents & POLLIN) {
                /* Read and clear eventfd */
                uint64_t val;
                (void)read(ctx->msi_eventfds[i], &val, sizeof(val));

                /* Set pending bit */
                ctx->pending_msi.fetch_or(1U << i);

                /* Call handler if registered */
                std::lock_guard<std::mutex> lock(ctx->handler_mutex);
                if (ctx->msi_handlers[i].handler) {
                    IRQ_LOG("MSI %d triggered, calling handler\n", i);
                    ctx->msi_handlers[i].handler(i, ctx->msi_handlers[i].user_data);
                }
            }
            pfd_idx++;
        }
    }

    IRQ_LOG("MSI poll thread stopped\n");
}

#endif /* __linux__ */

/*=============================================================================
 * Initialization
 *============================================================================*/

extern "C" {

gtx_pcie_irq_ctx* gtx_pcie_irq_create_vfio(const char* socket_path) {
#ifndef __linux__
    IRQ_LOG_ERROR("vfio-user mode only supported on Linux\n");
    return nullptr;
#else
    g_irq_connection_broken.store(false);

    if (!socket_path) {
        IRQ_LOG_ERROR("socket_path is NULL\n");
        return nullptr;
    }

    /* Check if socket exists */
    struct stat st;
    if (stat(socket_path, &st) != 0) {
        IRQ_LOG_ERROR("Socket not found: %s\n", socket_path);
        return nullptr;
    }

    /* Create socket and connect */
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        IRQ_LOG_ERROR("Failed to create socket: %s\n", strerror(errno));
        return nullptr;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        IRQ_LOG_ERROR("Failed to connect to %s: %s\n", socket_path, strerror(errno));
        close(sock);
        return nullptr;
    }

    IRQ_LOG("Connected to vfio-user socket: %s\n", socket_path);

    /* Negotiate vfio-user protocol version */
    if (vfio_negotiate(sock) < 0) {
        IRQ_LOG_ERROR("Protocol negotiation failed\n");
        close(sock);
        return nullptr;
    }

    /* Get device info */
    struct vfio_user_device_info dev_info;
    memset(&dev_info, 0, sizeof(dev_info));
    if (vfio_get_device_info(sock, &dev_info) < 0) {
        IRQ_LOG_ERROR("Failed to get device info\n");
        close(sock);
        return nullptr;
    }

    /* Create context */
    gtx_pcie_irq_ctx* ctx = new gtx_pcie_irq_ctx();
    ctx->mode = gtx_pcie_irq_ctx::MODE_VFIO_USER;
    ctx->vfio_socket = sock;

    /* Create eventfds for MSI vectors */
    for (int i = 0; i < GTX_MSI_MAX; i++) {
        ctx->msi_eventfds[i] = eventfd(0, EFD_NONBLOCK);
        if (ctx->msi_eventfds[i] < 0) {
            IRQ_LOG_ERROR("Failed to create eventfd %d: %s\n", i, strerror(errno));
        }
    }

    /* Configure MSI IRQs with eventfds */
    /* VFU_DEV_MSI_IRQ = 1 (MSI interrupt type, supports up to 32 vectors per CLAUDE.md) */
    #define VFU_DEV_MSI_IRQ 1
    if (dev_info.num_irqs > 1) {
        /* Register MSI vectors with eventfds (up to 32 as per CLAUDE.md) */
        int msi_fds[GTX_MSI_MAX];
        int num_msi = GTX_MSI_MAX;
        for (int i = 0; i < num_msi; i++) {
            msi_fds[i] = ctx->msi_eventfds[i];
        }

        if (vfio_set_irqs(sock, VFU_DEV_MSI_IRQ, 0, num_msi, msi_fds) < 0) {
            IRQ_LOG("Failed to configure MSI IRQs (may not be supported)\n");
            /* Not a fatal error - device may not support MSI */
        }
    }

    /* Start MSI poll thread */
    ctx->msi_thread_running = true;
    ctx->msi_thread = std::thread(msi_poll_thread, ctx);

    IRQ_LOG("vfio-user context created successfully\n");

    return ctx;
#endif
}

gtx_pcie_irq_ctx* gtx_pcie_irq_create_hw(const char* device_path, int group_id) {
    (void)device_path;
    (void)group_id;

    /* TODO: Implement real VFIO hardware support */
    IRQ_LOG_ERROR("Hardware VFIO mode not implemented\n");
    return nullptr;
}

gtx_pcie_irq_ctx* gtx_pcie_irq_create_mapped(volatile void* bar0_base, size_t bar0_size) {
    if (!bar0_base || bar0_size == 0) {
        IRQ_LOG_ERROR("Invalid BAR0 mapping\n");
        return nullptr;
    }

    gtx_pcie_irq_ctx* ctx = new gtx_pcie_irq_ctx();
    ctx->mode = gtx_pcie_irq_ctx::MODE_MAPPED;
    ctx->bar0_base = (volatile uint8_t*)bar0_base;
    ctx->bar0_size = bar0_size;
    ctx->bar0_owned = false;

    IRQ_LOG("Created mapped IRQ context: bar0=%p size=%zu\n", bar0_base, bar0_size);

    return ctx;
}

void gtx_pcie_irq_destroy(gtx_pcie_irq_ctx* ctx) {
    if (ctx) {
        delete ctx;
    }
}

/*=============================================================================
 * MSI Handling
 *============================================================================*/

int gtx_pcie_irq_register_msi(gtx_pcie_irq_ctx* ctx, int vector,
                               gtx_msi_handler_t handler, void* user_data) {
    if (!ctx || vector < 0 || vector >= GTX_MSI_MAX) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx->handler_mutex);
    ctx->msi_handlers[vector].handler = handler;
    ctx->msi_handlers[vector].user_data = user_data;

    IRQ_LOG("Registered MSI handler for vector %d\n", vector);
    return 0;
}

void gtx_pcie_irq_unregister_msi(gtx_pcie_irq_ctx* ctx, int vector) {
    if (!ctx || vector < 0 || vector >= GTX_MSI_MAX) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->handler_mutex);
    ctx->msi_handlers[vector].handler = nullptr;
    ctx->msi_handlers[vector].user_data = nullptr;
}

int gtx_pcie_irq_wait_msi(gtx_pcie_irq_ctx* ctx, int vector, int timeout_ms) {
    if (!ctx) {
        return -1;
    }

    const uint32_t mask = (vector >= 0) ? (1U << vector) : 0xFFFFFFFF;

    auto consume_pending = [&]() -> int {
        uint32_t pending = ctx->pending_msi.load() & mask;
        if (!pending) {
            return -1;
        }

        int found = __builtin_ctz(pending);
        ctx->pending_msi.fetch_and(~(1U << found));
        return found;
    };

    int found = consume_pending();
    if (found >= 0) {
        return found;
    }

#ifdef __linux__
    /* Use eventfd if available */
    if (vector >= 0 && vector < GTX_MSI_MAX && ctx->msi_eventfds[vector] >= 0) {
        struct pollfd pfd = { ctx->msi_eventfds[vector], POLLIN, 0 };
        int ret = poll(&pfd, 1, timeout_ms);
        if (ret > 0 && (pfd.revents & POLLIN)) {
            uint64_t val;
            (void)read(ctx->msi_eventfds[vector], &val, sizeof(val));
            return vector;
        }

        return consume_pending();
    }
#endif

    /* Fallback: poll pending_msi */
    auto start = std::chrono::steady_clock::now();
    while (true) {
        int found = consume_pending();
        if (found >= 0) {
            return found;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (timeout_ms >= 0 && elapsed >= timeout_ms) {
            return -1;  /* Timeout */
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int gtx_pcie_irq_poll_msi(gtx_pcie_irq_ctx* ctx) {
    if (!ctx) {
        return -1;
    }

    uint32_t pending = ctx->pending_msi.load();
    if (pending) {
        int found = __builtin_ctz(pending);
        ctx->pending_msi.fetch_and(~(1U << found));
        return found;
    }

    return -1;
}

uint32_t gtx_pcie_irq_get_pending_msi(gtx_pcie_irq_ctx* ctx) {
    return ctx ? ctx->pending_msi.load() : 0;
}

void gtx_pcie_irq_clear_msi(gtx_pcie_irq_ctx* ctx, int vector) {
    if (ctx && vector >= 0 && vector < GTX_MSI_MAX) {
        ctx->pending_msi.fetch_and(~(1U << vector));
    }
}

/*=============================================================================
 * Doorbell (Host -> Device)
 *============================================================================*/

int gtx_pcie_irq_send_doorbell(gtx_pcie_irq_ctx* ctx, int source) {
    if (!ctx || source < 0 || source >= GTX_DOORBELL_MAX) {
        return -1;
    }

    return gtx_pcie_irq_send_doorbell_mask(ctx, 1U << source);
}

int gtx_pcie_irq_send_doorbell_mask(gtx_pcie_irq_ctx* ctx, uint32_t sources) {
    if (!ctx) {
        return -1;
    }

    switch (ctx->mode) {
        case gtx_pcie_irq_ctx::MODE_VFIO_USER:
#ifdef __linux__
            if (ctx->vfio_socket >= 0) {
                /* Write to ELBI doorbell via vfio-user REGION_WRITE */
                int ret = vfio_region_write(ctx->vfio_socket,
                                           VFU_PCI_DEV_BAR0_REGION_IDX,
                                           GTX_BAR0_ELBI_DOORBELL,
                                           &sources, sizeof(sources));
                if (ret < 0) {
                    IRQ_LOG_ERROR("Failed to send doorbell via vfio-user\n");
                    return -1;
                }

                IRQ_LOG("Doorbell sent via vfio-user: 0x%08x\n", sources);
                return 0;
            }
#endif
            IRQ_LOG_ERROR("vfio socket not connected\n");
            return -1;

        case gtx_pcie_irq_ctx::MODE_MAPPED:
            if (ctx->bar0_base && GTX_BAR0_ELBI_DOORBELL < ctx->bar0_size) {
                volatile uint32_t* doorbell =
                    (volatile uint32_t*)(ctx->bar0_base + GTX_BAR0_ELBI_DOORBELL);
                *doorbell = sources;
                __sync_synchronize();
                IRQ_LOG("Doorbell sent (mapped): 0x%08x\n", sources);
                return 0;
            }
            IRQ_LOG_ERROR("BAR0 not mapped or offset out of range\n");
            return -1;

        case gtx_pcie_irq_ctx::MODE_HARDWARE:
            /* TODO: Use real VFIO write */
            IRQ_LOG_ERROR("Hardware mode not implemented\n");
            return -1;
    }

    return -1;
}

/*=============================================================================
 * vfio-user RPC Communication
 *============================================================================*/

int gtx_pcie_irq_region_write(gtx_pcie_irq_ctx* ctx, uint32_t region,
                               uint64_t offset, const void* data, size_t size) {
    if (!ctx || !data || size == 0) {
        return -1;
    }

#ifdef __linux__
    if (ctx->mode == gtx_pcie_irq_ctx::MODE_VFIO_USER && ctx->vfio_socket >= 0) {
        return vfio_region_write(ctx->vfio_socket, region, offset,
                                 const_cast<void*>(data), size);
    }
#endif

    /* Fallback to mapped mode */
    if (ctx->mode == gtx_pcie_irq_ctx::MODE_MAPPED && ctx->bar0_base && region == 0) {
        if (offset + size <= ctx->bar0_size) {
            memcpy(const_cast<uint8_t*>(ctx->bar0_base) + offset, data, size);
            MEMORY_BARRIER();
            return 0;
        }
    }

    IRQ_LOG_ERROR("region_write: unsupported mode or invalid region\n");
    return -1;
}

int gtx_pcie_irq_region_read(gtx_pcie_irq_ctx* ctx, uint32_t region,
                              uint64_t offset, void* data, size_t size) {
    if (!ctx || !data || size == 0) {
        return -1;
    }

#ifdef __linux__
    if (ctx->mode == gtx_pcie_irq_ctx::MODE_VFIO_USER && ctx->vfio_socket >= 0) {
        return vfio_region_read(ctx->vfio_socket, region, offset, data, size);
    }
#endif

    /* Fallback to mapped mode */
    if (ctx->mode == gtx_pcie_irq_ctx::MODE_MAPPED && ctx->bar0_base && region == 0) {
        if (offset + size <= ctx->bar0_size) {
            memcpy(data, const_cast<uint8_t*>(ctx->bar0_base) + offset, size);
            return 0;
        }
    }

    IRQ_LOG_ERROR("region_read: unsupported mode or invalid region\n");
    return -1;
}

int gtx_pcie_irq_get_socket(gtx_pcie_irq_ctx* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->vfio_socket;
}

/*=============================================================================
 * Event Loop Integration
 *============================================================================*/

int gtx_pcie_irq_get_eventfd(gtx_pcie_irq_ctx* ctx, int vector) {
    if (!ctx || vector < 0 || vector >= GTX_MSI_MAX) {
        return -1;
    }
    return ctx->msi_eventfds[vector];
}

int gtx_pcie_irq_process(gtx_pcie_irq_ctx* ctx) {
    if (!ctx) {
        return 0;
    }

    int count = 0;
    uint32_t pending = ctx->pending_msi.exchange(0);

    if (pending) {
        std::lock_guard<std::mutex> lock(ctx->handler_mutex);
        for (int i = 0; i < GTX_MSI_MAX && pending; i++) {
            if (pending & (1U << i)) {
                if (ctx->msi_handlers[i].handler) {
                    ctx->msi_handlers[i].handler(i, ctx->msi_handlers[i].user_data);
                    count++;
                }
                pending &= ~(1U << i);
            }
        }
    }

    return count;
}

/*=============================================================================
 * Debug
 *============================================================================*/

void gtx_pcie_irq_print_status(gtx_pcie_irq_ctx* ctx) {
    if (!ctx) {
        fprintf(stderr, "GTX IRQ: context is NULL\n");
        return;
    }

    const char* mode_str = "unknown";
    switch (ctx->mode) {
        case gtx_pcie_irq_ctx::MODE_VFIO_USER: mode_str = "vfio-user"; break;
        case gtx_pcie_irq_ctx::MODE_HARDWARE: mode_str = "hardware"; break;
        case gtx_pcie_irq_ctx::MODE_MAPPED: mode_str = "mapped"; break;
    }

    fprintf(stderr, "GTX IRQ Status:\n");
    fprintf(stderr, "  Mode: %s\n", mode_str);
    fprintf(stderr, "  BAR0: %p (size=%zu)\n", ctx->bar0_base, ctx->bar0_size);
    fprintf(stderr, "  vfio socket: %d\n", ctx->vfio_socket);
    fprintf(stderr, "  Pending MSI: 0x%08x\n", ctx->pending_msi.load());
    fprintf(stderr, "  MSI thread: %s\n", ctx->msi_thread_running ? "running" : "stopped");

    fprintf(stderr, "  Registered handlers:\n");
    for (int i = 0; i < GTX_MSI_MAX; i++) {
        if (ctx->msi_handlers[i].handler) {
            fprintf(stderr, "    Vector %d: handler=%p\n", i, (void*)ctx->msi_handlers[i].handler);
        }
    }
}

/* Enable debug logging */
void gtx_pcie_irq_set_debug(bool enable) {
    g_irq_debug = enable;
}

} /* extern "C" */
