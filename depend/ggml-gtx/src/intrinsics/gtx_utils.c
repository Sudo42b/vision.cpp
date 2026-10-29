#include "ggml-gtx/intrinsics/intrin.h"


#define NUM_NESTS        4
#define NUM_SPUS        16

#define SHARED_DBG      0xf00000    // 0x00f0_0000 ~ 0x00ff_ffff

#define L1_SPRAM_BYTES  (384u * 1024u)


void __capture_l1_nest(uint16_t nest, uintptr_t dst)
{
    const uint16_t maskall = UINT16_C(0xFFFF);
    
    __start_plan(nest);
        
        __start_shared();
            __credit_chk(maskall);
            __store(SHARED_DBG, (uint64_t)dst, 0x6000UL, 0x6000UL, 16 * 16, 0x6000UL);
        __end_shared();

        for (uint32_t spu = 0; spu < NUM_SPUS; ++spu) {
            const uint16_t mask = UINT16_C(1) << spu;
            __start_thread(spu);
                __store(0x00000, SHARED_DBG + 0x60000UL * spu, 0x6000UL, 0x6000UL, 16, 0x6000UL);
                __credit_st(mask);
            __end_thread(spu);
        }

    __end_plan(nest);
}

void __capture_l1(uintptr_t dst)
{
    const uint16_t maskall = UINT16_C(0xFFFF);
    
    for (uint32_t nest = 0; nest < NUM_NESTS; ++nest) {

        __start_plan(nest);
            
            __start_shared();
                __credit_chk(maskall);
                __store(SHARED_DBG, (uint64_t)dst + nest * NUM_SPUS * L1_SPRAM_BYTES, 0x6000UL, 0x6000UL, 16 * 16, 0x6000UL);
            __end_shared();

            for (uint32_t spu = 0; spu < NUM_SPUS; ++spu) {
                const uint16_t mask = UINT16_C(1) << spu;
                __start_thread(spu);
                    __store(0x00000, SHARED_DBG + 0x60000UL * spu, 0x6000UL, 0x6000UL, 16, 0x6000UL);
                    __credit_st(mask);
                __end_thread(spu);
            }

        __end_plan(nest);
    }
}
