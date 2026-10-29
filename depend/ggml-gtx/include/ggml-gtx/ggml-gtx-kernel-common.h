#ifndef GGML_GTX_KERNEL_COMMON_H
#define GGML_GTX_KERNEL_COMMON_H

#include "ggml-gtx-compute.h"
#include "ggml-gtx-port.h"
#include "ggml-gtx/intrinsics/gtx_csr.h"
#include "ggml-gtx/intrinsics/intrin.h"

#ifdef GGML_GTX_KERNEL_DEBUG
static inline void ggml_gtx_debug_dump_fp16_head(const char * tag, uint64_t addr) {
    volatile uint16_t * p = (volatile uint16_t *)(uintptr_t)addr;
    ggml_gtx_printf("[%s] addr=%p head=[%04x %04x %04x %04x]\n",
                   tag,
                   (void *)(uintptr_t)addr,
                   p[0], p[1], p[2], p[3]);
}
#else
#define ggml_gtx_debug_dump_fp16_head(tag, addr) \
    do { (void) (tag); (void) (addr); } while (0)
#endif

#endif
