//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic function description (level 2)
// 				 convenience intrinsic function for programming
// Author      : mh.kim ( NPU Div - NPU Core Team )    
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_LEVEL2_C
#define INTRIN_LEVEL2_C

#include "ggml-gtx/intrinsics/intrin_level1.h"
#include "ggml-gtx/intrinsics/intrin_level2.h"
#include "ggml-gtx/intrinsics/gtx_csr.h"


//=================================
// Auto setting
//=================================
// [init L1SPM bank addr]
// 4bit input: en_restore_R[3], en_restore_C[2], en_restore_B[1], en_restore_A[0]  
// initial spm_addr_A - 0x00000
// initial spm_addr_B - 0x20000
// initial spm_addr_C - 0x30000
// initial spm_addr_R - 0x50000
__attribute__((noinline)) void __init_spm_addr(uint8_t target_spm_bank)
{
    uint64_t rs1, rs2;
    
    if (target_spm_bank & 0x1) {
        rs1 = ((uint64_t) SPM_ADDR_A);
        rs2 = ((uint64_t) 0x00000);
	    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
    }

    if (target_spm_bank & 0x2) {
        rs1 = ((uint64_t) SPM_ADDR_B);
        rs2 = ((uint64_t) 0x20000);
	    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
    }

    if (target_spm_bank & 0x4) {
        rs1 = ((uint64_t) SPM_ADDR_C);
        rs2 = ((uint64_t) 0x30000);
        __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
    }

    if (target_spm_bank & 0x8) {
        rs1 = ((uint64_t) SPM_ADDR_R);
        rs2 = ((uint64_t) 0x50000);
	    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
    }
}


// [set L1SPM bank addr]
// spm_addr_A - 24bit
// spm_addr_B - 24bit
// spm_addr_C - 24bit
// spm_addr_R - 24bit
__attribute__((noinline)) void __set_spm_addr(uint32_t spm_addr_R, uint32_t spm_addr_C, uint32_t spm_addr_B, uint32_t spm_addr_A)
{
    uint64_t rs1, rs2;

    rs1 = ((uint64_t) SPM_ADDR_A);
    rs2 = ((uint64_t) spm_addr_A);
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

    rs1 = ((uint64_t) SPM_ADDR_B);
    rs2 = ((uint64_t) spm_addr_B);
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

    rs1 = ((uint64_t) SPM_ADDR_C);
    rs2 = ((uint64_t) spm_addr_C);
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

    rs1 = ((uint64_t) SPM_ADDR_R);
    rs2 = ((uint64_t) spm_addr_R);
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [set L1SPM bank addr A]
// spm_addr_A - 24bit
__attribute__((noinline)) void __set_spm_addr_A(uint32_t spm_addr_A)
{
    uint64_t rs1 = ((uint64_t) SPM_ADDR_A);
    uint64_t rs2 = ((uint64_t) spm_addr_A);
    
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [set L1SPM bank addr B]
// spm_addr_B - 24bit
__attribute__((noinline)) void __set_spm_addr_B(uint32_t spm_addr_B)
{
    uint64_t rs1 = ((uint64_t) SPM_ADDR_B);
    uint64_t rs2 = ((uint64_t) spm_addr_B);
    
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [set L1SPM bank addr C]
// spm_addr_C - 24bit
__attribute__((noinline)) void __set_spm_addr_C(uint32_t spm_addr_C)
{
    uint64_t rs1 = ((uint64_t) SPM_ADDR_C);
    uint64_t rs2 = ((uint64_t) spm_addr_C);
    
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [set L1SPM bank addr R]
// spm_addr_R - 24bit
__attribute__((noinline)) void __set_spm_addr_R(uint32_t spm_addr_R)
{
    uint64_t rs1 = ((uint64_t) SPM_ADDR_R);
    uint64_t rs2 = ((uint64_t) spm_addr_R);
    
    __asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// DMA with credit
//=================================
// [load with credit]
// load rs1   = src_addr[63:27], dst_addr[26:0]
// load rs2   = heigth[63:48], length[47:32], read_stride[31:0]
// load rs3   = write_stride[31:0]
// additional = auto_credit(1bit), target_spu(64bit), target_nest(64bit)
// auto_credit 1: 'load' with 'credit_ld' 
//		 	   0: 'load'
__attribute__((noinline)) void __load_cr(uint64_t src_addr   , uint32_t dst_addr  , uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride, 
                                    	 uint8_t  auto_credit, uint64_t target_spu, uint64_t target_nest)
{	
	src_addr = src_addr & 0x0000001FFFFFFFFF;
	dst_addr = dst_addr & 0x07FFFFFF;

	uint64_t rs1 = ((uint64_t) src_addr << 27) | ((uint64_t) dst_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = ((uint64_t) write_stride);

	__opset(0, rs3);
	__asm__ volatile ("load %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
    
    if (auto_credit & 0x1) {
        __credit_ld(target_spu, target_nest);
    }   
}


// [store with credit]
// store rs1  = dst_addr[63:27], src_addr[26:0]
// store rs2  = heigth[63:48], length[47:32], write_stride[31:0]
// store rs3  = read_stride[31:0]
// additional = auto_credit(1bit), target_spu(64bit)
// auto_credit 1: 'store' with 'credit_st' 
//			   0: 'store' 
__attribute__((noinline)) void __store_cr(uint32_t src_addr   , uint64_t dst_addr  , uint32_t write_stride, uint16_t length, uint16_t height, uint32_t read_stride, 
										  uint8_t  auto_credit, uint64_t target_spu)
{	
	src_addr = src_addr & 0x07FFFFFF;
	dst_addr = dst_addr & 0x0000001FFFFFFFFF;

	uint64_t rs1 = ((uint64_t) dst_addr << 27) | ((uint64_t) src_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) write_stride);
	uint64_t rs3 = ((uint64_t) read_stride);

	__opset(0, rs3);
	__asm__ volatile ("store %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

    if (auto_credit & 0x1) {
        __credit_st(target_spu);
    }   
}



//=================================
// Renaming intrinsic for programming (same operation)
//=================================
// [sum]
// same as __mm_o
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sum(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mm.o %[r1]\n" : : [r1] "r"(rs1));
}


// [sum_acc]
// same as __mmc_o
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sum_acc(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mmc.o %[r1]\n" : : [r1] "r"(rs1));
}


// [dot_product]
// same as __mm_v
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __dot_product(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mm.v %[r1]\n" : : [r1] "r"(rs1));
}


// [dot_product_acc]
// same as __mmc_v
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __dot_product_acc(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mmc.v %[r1]\n" : : [r1] "r"(rs1));
}



//=================================
// Usable function (not consisting of GTX-ISA)
//=================================
// [int16_to_fp16] 
// integer 16 format convert to floating point 16 format (RNE role)
__attribute__((noinline)) uint16_t __int16_to_fp16(int16_t integer)
{
    uint16_t exp;
    uint16_t mantissa;
    uint16_t sign = 0;
    uint16_t abs_val;
    uint16_t integer_msb = 15;
    uint8_t  guard_bit; 
    uint8_t  round_bit; 
    uint8_t  sticky_bit; 

    if (integer == 0) {
        return 0;
    }

    if (integer < 0) {
        sign = 1;
        abs_val = (uint16_t)(-integer);
    } else {
        abs_val = (uint16_t)integer;
    }

    while ((abs_val & (1 << integer_msb)) == 0) {
        integer_msb--;
    }

    exp = integer_msb + 15;

    if (integer_msb <= 10) {
        mantissa = (abs_val << (10 - integer_msb)) & 0x3FF;
    } else {
        uint8_t shift = integer_msb - 10;

        mantissa = (abs_val >> shift) & 0x3FF;

        guard_bit  = (abs_val >> (shift - 1)) & 1;
        round_bit  = (abs_val >> (shift - 2)) & 1;
        sticky_bit = (shift > 2) ? ((abs_val & ((1U << (shift - 2)) - 1)) != 0) : 0;

        if ((guard_bit && (round_bit || sticky_bit)) || 
            (guard_bit && !round_bit && !sticky_bit && (mantissa & 1))) {
            mantissa++;
        }

        if (mantissa == 0x400) {
            mantissa = 0;
            exp++;
        }
    }

    if (exp >= 31) {
        return (sign << 15) | 0x7C00;
    }

    return (sign << 15) | ((exp & 0x1F) << 10) | (mantissa & 0x3FF);
}



#endif