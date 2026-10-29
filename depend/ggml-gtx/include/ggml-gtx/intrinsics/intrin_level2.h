//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic definition (level 2)
// 				 convenience intrinsic for programming
// Author      : mh.kim ( NPU Div - NPU Core Team )
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_LEVEL2_H
#define INTRIN_LEVEL2_H

#include <stdint.h>

#ifndef __gtx_giss_unmask_threads
#ifdef GGML_GTX_GISS_MASK_COMPAT
#define __gtx_giss_unmask_threads() \
	do { \
		uint64_t __gtx_giss_zero = 0; \
		uint64_t __gtx_giss_thread = 0x400; \
		__asm__ volatile ("opset %[r1], %[r2]\n" : : [r1] "r"(__gtx_giss_zero), [r2] "r"(__gtx_giss_zero)); \
		__asm__ volatile ("wrspr %[r1], %[r2]\n" : : [r1] "r"(__gtx_giss_thread), [r2] "r"(__gtx_giss_zero)); \
	} while (0)
#else
#define __gtx_giss_unmask_threads() do { } while (0)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

//=================================
// Auto setting
//=================================
// [init L1SPM bank addr]
// 4bit input: en_restore_R[3], en_restore_C[2], en_restore_B[1], en_restore_A[0]  
// initial spm_addr_A - 0x00000
// initial spm_addr_B - 0x20000
// initial spm_addr_C - 0x30000
// initial spm_addr_R - 0x50000
__attribute__((noinline)) void __init_spm_addr(uint8_t target_spm_bank);


// [set L1SPM bank addr]
// spm_addr_A - 24bit
// spm_addr_B - 24bit
// spm_addr_C - 24bit
// spm_addr_R - 24bit
__attribute__((noinline)) void __set_spm_addr(uint32_t spm_addr_R, uint32_t spm_addr_C, uint32_t spm_addr_B, uint32_t spm_addr_A);


// [set L1SPM bank addr A]
// spm_addr_A - 24bit
__attribute__((noinline)) void __set_spm_addr_A(uint32_t spm_addr_A);


// [set L1SPM bank addr B]
// spm_addr_B - 24bit
__attribute__((noinline)) void __set_spm_addr_B(uint32_t spm_addr_B);


// [set L1SPM bank addr C]
// spm_addr_C - 24bit
__attribute__((noinline)) void __set_spm_addr_C(uint32_t spm_addr_C);


// [set L1SPM bank addr R]
// spm_addr_R - 24bit
__attribute__((noinline)) void __set_spm_addr_R(uint32_t spm_addr_R);



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
                                    	 uint8_t  auto_credit, uint64_t target_spu, uint64_t target_nest);


// [store with credit]
// store rs1  = dst_addr[63:27], src_addr[26:0]
// store rs2  = heigth[63:48], length[47:32], write_stride[31:0]
// store rs3  = read_stride[31:0]
// additional = auto_credit(1bit), target_spu(64bit)
// auto_credit 1: 'store' with 'credit_st' 
//			   0: 'store' 
__attribute__((noinline)) void __store_cr(uint32_t src_addr   , uint64_t dst_addr  , uint32_t write_stride, uint16_t length, uint16_t height, uint32_t read_stride, 
										  uint8_t  auto_credit, uint64_t target_spu);



//=================================
// Renaming intrinsic for programming (same operation)
//=================================
// [sum]
// same as __mm_o
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sum(uint16_t col_A_size, uint8_t result_svr_addr);


// [sum_acc]
// same as __mmc_o
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sum_acc(uint16_t col_A_size, uint8_t result_svr_addr);


// [dot_product]
// same as __mm_v
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __dot_product(uint16_t col_A_size, uint8_t result_svr_addr);


// [dot_product_acc]
// same as __mmc_v
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __dot_product_acc(uint16_t col_A_size, uint8_t result_svr_addr);


// [start thread] 
// same as __start_t
#define __start_thread(spu_id)\
	__asm__ volatile("start.t %[r1], %[imm]\n" : : [r1] "r"(spu_id), [imm] "i"(0x000));


// [start thread immediate] 
// same as __start_ti
#define __start_threadi(spu_id)\
	__asm__ volatile("start.t %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(spu_id | 0x400));


// [end thread] 
// same as __end_t
#define __end_thread(spu_id)\
	__asm__ volatile("end.t %[r1], %[imm]\n" : : [r1] "r"(spu_id), [imm] "i"(0x000));


// [end thread immediate] 
// same as __end_ti
#define __end_threadi(spu_id)\
	__asm__ volatile("end.t %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(spu_id | 0x400));


// [start shared] 
// same as __start_s
#define __start_shared()\
	__asm__ volatile("start.s\n" : : );

 
// [end shared] 
// same as __end_s
#define __end_shared()\
	__asm__ volatile("end.s\n" : : );


// [start plan] 
// same as __start_p
#define __start_plan(nest_id)\
	do { \
		__asm__ volatile("start.p %[r1], %[imm]\n" : : [r1] "r"(nest_id), [imm] "i"(0x000)); \
		__gtx_giss_unmask_threads(); \
	} while (0)


// [start plan immediate]
// same as __start_pi
#define __start_plani(nest_id)\
	do { \
		__asm__ volatile("start.p %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(nest_id | 0x400)); \
		__gtx_giss_unmask_threads(); \
	} while (0)


// [end plan] 
// same as __end_p
#define __end_plan(nest_id)\
	__asm__ volatile("end.p %[r1], %[imm]\n" : : [r1] "r"(nest_id), [imm] "i"(0x000));


// [end plan immediate] 
// same as __end_pi
#define __end_plani(nest_id)\
	__asm__ volatile("end.p %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(nest_id | 0x400));



//=================================
// Usable function (not consisting of GTX-ISA)
//=================================
// [int16_to_fp16]
// integer 16 format convert to floating point 16 format (RNE role)
__attribute__((noinline)) uint16_t __int16_to_fp16(int16_t integer);

#ifdef __cplusplus
}
#endif

#endif
