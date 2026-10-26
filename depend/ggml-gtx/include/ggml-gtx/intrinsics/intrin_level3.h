//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic definition (level 3)
//				 complex operation intrinsic consisting of compound instructions
// Author      : mh.kim ( NPU Div - NPU Core Team )
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_LEVEL3_H
#define INTRIN_LEVEL3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=================================
// Special function - memory
//=================================							
// [pad]
// fill 0 to dst_addr & DMA to dst_addr per channel
__attribute__((noinline)) void __pad(uint64_t src_addr      , uint64_t dst_addr   , 
 									 uint32_t fill_stride	, uint16_t length	  , uint16_t height, 
									 uint8_t  ystr		  	, uint8_t  yend	   	  , uint8_t  xstr  , uint8_t xend  , uint64_t fill_pattern,
									 uint32_t channel_stride, uint16_t channel_num, uint8_t  rw    , uint8_t d_type);


// [load_3d]
// 3D load from L2SPM to L1SPM 
// src_addr_L2SPM(28bit) / dst_addr_L1SPM(24bit)
// heigth(16bit) / length(16bit) / read_stride(32bit) / depth(16bit) / depth_stride(32bit)
__attribute__((noinline)) void __load_3d(uint32_t src_addr_L2SPM, uint32_t dst_addr_L1SPM, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t depth_stride, uint16_t depth);


// [store_3d]
// 3D store from L1SPM to L2SPM 
// src_addr_L1SPM(24bit) / dst_addr_L2SPM(28bit)
// heigth(16bit) / length(16bit) / write_stride(32bit) / depth(16bit) / depth_stride(32bit)
__attribute__((noinline)) void __store_3d(uint32_t src_addr_L1SPM, uint32_t dst_addr_L2SPM, uint32_t write_stride, uint16_t length, uint16_t height, uint32_t depth_stride, uint16_t depth);



//=================================
// Special function - activation
//=================================
// [relu] 
// relu(x) = (x < 0) ? 0 : x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __relu(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_R_addr);


// [relu6] 
// relu6(x) = (x < 0) ? 0 : 
//			  (x > 6) ? 6 : x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __relu6(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_R_addr);


// [lrelu] 
// lrelu(x) = (x > 0) ? x : 0.01 * x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __lrelu(uint32_t vector_size);


// [silu] 
// silu(x) = x * sigmoid(x)
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __silu(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr);



//=================================
// Special function - math
//=================================
// [var] 
// variance(x) = sum(((x-mean(x))/sqrt(x_num))^2)
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : SVR_1
// result: SVR_2
__attribute__((noinline)) void __var(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr);



//=================================
// Special function - normalization
//=================================
// [layernorm] 
// layernorm(x) = (x-mean(x))/sqrt(var(x)+epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __layernorm(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon);


// [layernorm_aff] 
// layernorm_aff(x) = (scale*((x-mean(x))/var(x)+epsilon)) + shift
// input : L1SPM_A_BANK (l1_A_addr)
// scale : L1SPM_B_BANK (l1_B_addr)
// shift : L1SPM_C_BANK (l1_C_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __layernorm_aff(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_C_addr, uint32_t l1_R_addr, uint16_t epsilon);


// [rmsnorm] 
// rmsnorm(x) = x / R(MS(x)+epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __rmsnorm(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon);


// [rmsnorm_scale] 
// rmsnorm_scale(x) = (x/R(MS(x)+epsilon))* scale
// input : L1SPM_A_BANK (l1_A_addr)
// scale : L1SPM_B_BANK (l1_B_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __rmsnorm_scale(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon);


// [batchnorm] 
// batchnorm(x) = (x-mean)/sqrt(var_epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : fp16 value
// var 	 : fp16 value
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __batchnorm(uint32_t vector_size, uint16_t mean, uint16_t var, uint32_t l1_A_addr, uint32_t l1_R_addr, uint16_t epsilon);


// [batchnorm_aff] 
// batchnorm_aff(x) = (x-mean)/sqrt(var+epsilon)*scale + shift
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : fp16 value
// var 	 : fp16 value
// scale : fp16 value
// shift : fp16 value
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __batchnorm_aff(uint32_t vector_size, uint16_t mean, uint16_t var, uint16_t scale, uint16_t shift, uint32_t l1_A_addr, uint32_t l1_R_addr, uint16_t epsilon);

#ifdef __cplusplus
}
#endif

#endif