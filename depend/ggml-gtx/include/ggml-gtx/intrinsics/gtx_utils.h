#ifndef GTX_UTILS_H
#define GTX_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>
#include <stdarg.h>
// #include "uart.hpp"
#include "ggml-gtx/sc_print.h"

#ifdef __cplusplus
extern "C" {
#endif

// CSR 읽기 (RISC-V 표준 명령어 기반)
static inline uint32_t read_csr(uint32_t addr) {
    uint32_t val;
    __asm__ volatile("csrr %0, %1" : "=r"(val) : "i"(addr));
    return val;
}

void __capture_l1_nest(uint16_t nest, uintptr_t dst);

void __capture_l1(uintptr_t dst);

#ifdef __cplusplus
}
#endif

#endif