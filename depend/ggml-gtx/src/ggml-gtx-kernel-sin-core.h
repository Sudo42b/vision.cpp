//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Test        : timestep_embedding
// Description : GTX timestep_embedding kernel
// Author      : kyumlee ( NPU Div - NPU Core Design Team )    
// Last Update : 2026/02/26
//==================================================================


// Intrinsic header file
#include "gtx/intrin.h"
#include "gtx/print.h"
#include "gtx_verify.h"
#include "gtx_test.h"
//#include "gtx_ckernel_param.h"
#include <string.h>

#define NEST_TARGET_ID      0               // Target nest ID
#define SPU_NUM_PER_NEST    16              // SPU num per nest
#define SINCOS_DTYPE        2               // FP16

#define SINCOS_LEGACY_L2_A  0x20000
#define SINCOS_LEGACY_L2_B  0x40000

#define SINCOS_BANK_A      0X00000
#define SINCOS_BANK_B      0X20000
#define SINCOS_BANK_C      0X30000
#define SINCOS_BANK_R      0X50000

#define SINCOS_SVR_ADDR            0x800

static void sincos_vec_impl(
        uint8_t NEST_NUM,
        uint8_t SPU_NUM,
        uint64_t DDR_SRC_ADDR,
        uint64_t DDR_DST_ADDR,
        uint8_t SINCOS_SEL,
        uint32_t SINCOS_L2_A,
        uint32_t SINCOS_L2_B,
        uint8_t L2_ONLY
){
    /*
    if(SINCOS_SEL < 0 || SINCOS_SEL > 1){
    printf("\n[SINCOS] SEL ERROR\n");
    return 0;
    }
    */
    uint8_t nest_id = NEST_NUM;
    uint8_t thread_id = SPU_NUM;
    uint16_t vec_size = 16 * SINCOS_DTYPE;

    __start_plan(nest_id);

        __start_shared();
            if (L2_ONLY) {
                __copy((uint32_t)DDR_SRC_ADDR, SINCOS_L2_A,
                       vec_size, vec_size, 1, vec_size);
            } else {
                __load(DDR_SRC_ADDR, SINCOS_L2_A, vec_size, vec_size, 1, vec_size);
            }
        __end_shared();

    __end_plan(nest_id);

    __start_plan(nest_id);

        __start_thread(thread_id);
            __load(SINCOS_L2_A, SINCOS_BANK_A, vec_size, vec_size, 1, vec_size);

            //x to svr, m = round(x/2pi)
            __load_svr(SINCOS_BANK_A,0);
            __mul_is(0,0x3118,0,0);
            __rne_i(0,0);
            
            //make two vector[x,m,m,m]. [1,-2pi_1, -2pi_2, -2pi_3]            
            __wrspr(SINCOS_SVR_ADDR+4, 0,
                    0x012BC248C2493C00, 0);
            __store_svr(SINCOS_BANK_A+vec_size, 0);
            __store_svr(SINCOS_BANK_B, 1);

            __store(SINCOS_BANK_A, SINCOS_L2_A,
                    vec_size,
                    vec_size,
                    1,
                    vec_size);
            for(int i=0; i<3; i++){
                __store(SINCOS_BANK_A + vec_size, SINCOS_L2_A + (i+1) * vec_size,
                        vec_size,
                        vec_size,
                        1,
                        vec_size);
            }
        __end_thread(thread_id);
    __end_plan(nest_id);
    
    __start_plan(nest_id);
        __start_shared();
            __tpose(SINCOS_L2_A, SINCOS_L2_B,
                    16, 4, 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(SINCOS_L2_B, SINCOS_BANK_C, vec_size, vec_size, 4, vec_size);
                    
            //matmul two vector (= x0)
            __set_spm_addr(SINCOS_BANK_R, SINCOS_BANK_A, SINCOS_BANK_B, SINCOS_BANK_C);
            __mm(16, 4, 1);
            
            //x0 to svr, k = round(x0/pi)
            __load_svr(SINCOS_BANK_R, 0);
            __mul_is(0,0x3518, 0,0);
            __rne_i(0,0);

            //make two vector[x0,k,k,k]. [1,-pi_1, -pi_2, -pi_3]
            __wrspr(SINCOS_SVR_ADDR+4, 0,
                    0x0095BE48BE493C00, 0);                    
            __store_svr(SINCOS_BANK_R+vec_size, 0);
            __store_svr(SINCOS_BANK_B, 1);

            __store(SINCOS_BANK_R, SINCOS_L2_A,
                    vec_size,
                    vec_size,
                    1,
                    vec_size);
            for(int i=0; i<3; i++){
                __store(SINCOS_BANK_R + vec_size, SINCOS_L2_A + (i+1) * vec_size,
                        vec_size,
                        vec_size,
                        1,
                        vec_size);
            }

        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            __tpose(SINCOS_L2_A, SINCOS_L2_B,
                    16, 4, 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_thread(thread_id);
            __load(SINCOS_L2_B, SINCOS_BANK_C, vec_size, vec_size, 4, vec_size);

            //matmul two vector (= r)
            __mm(16, 4, 1);
            
            //k transform to -1(odd) or 1(even)
            __mul_is(0, 0x3800, 1, 0);
            __ceil_i(1,1);
            __mul_is(1, 0x4000, 1, 0);
            __sub_ii(0,1,1);
            __add_is(1, 0x3C00, 2, 0);
            __add_ii(1,2,0);
            __store_svr(SINCOS_BANK_C, 0);
            
            //maclaurin
            __load_svr(SINCOS_BANK_R, 1);
            __mvsvr(1,2,0);
            __mul_ii(1,2,2);

            if(SINCOS_SEL){
                 __wrspr(SINCOS_SVR_ADDR+4, 0, 0x3C003C003C003C00, 0);
                 __cpsvr(1,3);
                 __store_svr(SINCOS_BANK_R,1);
            }

            for(int i=0; i<3+SINCOS_SEL; i++){
                __mul_ii(1,2,1);
                __store_svr(SINCOS_BANK_R+(i+1)*vec_size,1);
            }

            __store(SINCOS_BANK_R, SINCOS_L2_A,
                    vec_size,
                    vec_size,
                    4+SINCOS_SEL,
                    vec_size);

        __end_thread(thread_id);
    __end_plan(nest_id);

    __start_plan(nest_id);
        __start_shared();
            __tpose(SINCOS_L2_A, SINCOS_L2_B,
                    16, 4+SINCOS_SEL, 4, 1, 2, 0, 0b00, 1);
        __end_shared();
    __end_plan(nest_id);

    __start_plan(nest_id);
            __start_thread(thread_id);
            __load(SINCOS_L2_B, SINCOS_BANK_A, vec_size, vec_size, 4+SINCOS_SEL, vec_size);

            if(SINCOS_SEL){
                __copy(SINCOS_BANK_R, SINCOS_BANK_B,
                    SINCOS_DTYPE,SINCOS_DTYPE,1,SINCOS_DTYPE);
                __wrspr(SINCOS_SVR_ADDR+4, 0, 0x01A095B02955B800,0);
                __store_svr(SINCOS_BANK_B+vec_size,1);
                __copy(SINCOS_BANK_B+vec_size, SINCOS_BANK_B+SINCOS_DTYPE,
                        4*SINCOS_DTYPE, 4*SINCOS_DTYPE, 1, SINCOS_DTYPE);
            }
            else {
                __wrspr(SINCOS_SVR_ADDR+4, 0, 0x8A802044B1553C00,0);
                __store_svr(SINCOS_BANK_B,1);
            }

            __set_spm_addr(SINCOS_BANK_R,SINCOS_BANK_C,SINCOS_BANK_B,SINCOS_BANK_A);
            __mm(16,4+SINCOS_SEL,1);
            
            // k transform * maclaurin
            __set_spm_addr(SINCOS_BANK_A,SINCOS_BANK_B,SINCOS_BANK_C,SINCOS_BANK_R);
            __mul_vv(16);
            
            __store(SINCOS_BANK_A, SINCOS_L2_A, vec_size, vec_size, 1, vec_size);
        __end_thread(thread_id);

    __end_plan(nest_id);
    
    __start_plan(nest_id);

        __start_shared();
            if (L2_ONLY) {
                __copy(SINCOS_L2_A, (uint32_t)DDR_DST_ADDR,
                       vec_size, vec_size, 1, vec_size);
            } else {
                __store(SINCOS_L2_A, DDR_DST_ADDR, vec_size, vec_size, 1, vec_size);
            }
        __end_shared();
        
    __end_plan(nest_id);

    return;
}

void sincos_vec_l2(
        uint8_t NEST_NUM,
        uint8_t SPU_NUM,
        uint32_t L2_SRC_ADDR,
        uint32_t L2_DST_ADDR,
        uint8_t SINCOS_SEL,
        uint32_t SINCOS_L2_A,
        uint32_t SINCOS_L2_B
){
    sincos_vec_impl(NEST_NUM, SPU_NUM, L2_SRC_ADDR, L2_DST_ADDR, SINCOS_SEL,
                    SINCOS_L2_A, SINCOS_L2_B, 1);
}

void sincos_vec(
        uint8_t NEST_NUM,
        uint8_t SPU_NUM,
        uint64_t DDR_SRC_ADDR,
        uint64_t DDR_DST_ADDR,
        uint8_t SINCOS_SEL
){
    sincos_vec_impl(NEST_NUM, SPU_NUM, DDR_SRC_ADDR, DDR_DST_ADDR, SINCOS_SEL,
                    SINCOS_LEGACY_L2_A, SINCOS_LEGACY_L2_B, 0);
}
