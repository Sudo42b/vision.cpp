//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic function description (level 3)
//				 complex operation intrinsic function consisting of compound instructions
// Author      : mh.kim ( NPU Div - NPU Core Team )    
// Last Update : 2025/12/15
//==================================================================

#ifndef INTRIN_LEVEL3_C
#define INTRIN_LEVEL3_C

#include "ggml-gtx/intrinsics/intrin_level1.h"
#include "ggml-gtx/intrinsics/intrin_level2.h"
#include "ggml-gtx/intrinsics/intrin_level3.h"
#include "ggml-gtx/intrinsics/gtx_csr.h"


//=================================
// Special function - memory
//=================================	
// [pad]
// padding: fill 0 to dst_addr & DMA to dst_addr per channel
__attribute__((noinline)) void __pad(uint64_t src_addr      , uint64_t dst_addr   , 
 									 uint32_t fill_stride	, uint16_t length	  , uint16_t height, 
									 uint8_t  ystr		  	, uint8_t  yend	   	  , uint8_t  xstr  , uint8_t xend  , uint64_t fill_pattern,
									 uint32_t channel_stride, uint16_t channel_num, uint8_t  rw    , uint8_t d_type)
{
	volatile int i;

	uint64_t rs1, rs2, rs3;

	xstr = xstr << d_type;
	xend = xend << d_type; 

	uint16_t out_l    = (length + xstr + xend);
	uint16_t out_h    = (height + ystr + yend);
	uint16_t fill_h   = out_h * channel_num;
	uint32_t img_size = out_l * out_h;

    // [01:00] : 11 = x, 10 = read DDR write L2SPM , 01 = read L2SPM write DDR , 00 = read L2SPM write L2SPM
	uint8_t fill_dir =0;
	switch (rw) {
		case 0:
			src_addr = src_addr & 0x0000000007FFFFFF;
			dst_addr = dst_addr & 0x0000000007FFFFFF;
			fill_dir = 0;
			break;
		case 1:
			src_addr = src_addr & 0x0000000007FFFFFF;
			dst_addr = dst_addr & 0x0000001FFFFFFFFF;
			fill_dir = 1;
			break;
		case 2:
			src_addr = src_addr & 0x0000001FFFFFFFFF;
			dst_addr = dst_addr & 0x0000000007FFFFFF;
			fill_dir = 0;
			break;
		case 3:
			break;
		default:
			break;
	}
		
	// fill destination
	rs1 = ((uint64_t) fill_dir << 48) | dst_addr; 
	rs2 = ((uint64_t) fill_h << 48) | ((uint64_t) out_l << 32) | ((uint64_t) out_l);
	rs3 = fill_pattern;
	__opset(0, rs3);
	__asm__ volatile("fill %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

	// move source data  
    uint32_t write_stride = out_l;
	dst_addr = dst_addr + (out_l * ystr) + xstr;
	
	for(i = 0 ; i < channel_num ; i++) {
		
		// [01:00] : 11 = x, 10 = read DDR write L2SPM , 01 = read L2SPM write DDR , 00 = read L2SPM write L2SPM
		switch (rw) {
			case 0:
				// L2 -> L2: copy
				rs1 = ((uint64_t) dst_addr << 32) | ((uint64_t) src_addr);
				rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) fill_stride); 
				rs3 = ((uint64_t) write_stride); 
				__opset(0, rs3);
				__asm__ volatile("copy %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
				break;
			case 1: 
				// L2 -> DDR: store
				rs1 = ((uint64_t) dst_addr << 27) | ((uint64_t) src_addr);
				rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) write_stride); 
				rs3 = ((uint64_t) fill_stride); 
				__opset(0, rs3);
				__asm__ volatile("store %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
				break;
			case 2:
				// DDR -> l2: load
				rs1 = ((uint64_t) src_addr << 27) | ((uint64_t) dst_addr);
				rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) fill_stride); 
				rs3 = ((uint64_t) write_stride); 
				__opset(0, rs3);
				__asm__ volatile("load %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
				break;
			case 3:
				break;
			default:
				break;
		}

		src_addr += channel_stride;
		dst_addr += img_size;
	}
}


// [load_3d]
// 3D load from L2SPM to L1SPM 
// src_addr_L2SPM(28bit) / dst_addr_L1SPM(24bit)
// heigth(16bit) / length(16bit) / read_stride(32bit) / depth(16bit) / depth_stride(32bit)
__attribute__((noinline)) void __load_3d(uint32_t src_addr_L2SPM, uint32_t dst_addr_L1SPM, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t depth_stride, uint16_t depth)
{
	volatile int i;

	uint64_t rs1, rs2;
	uint32_t l1spm_offset = length * height;

	for(i = 0; i < depth; i++) {
		rs1 = ((uint64_t) src_addr_L2SPM << 27) | ((uint64_t) dst_addr_L1SPM);
		rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
		__asm__ volatile ("load %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

		src_addr_L2SPM += depth_stride;
		dst_addr_L1SPM += l1spm_offset;
	}
}


// [store_3d]
// 3D store from L1SPM to L2SPM 
// src_addr_L1SPM(24bit) / dst_addr_L2SPM(28bit)
// heigth(16bit) / length(16bit) / write_stride(32bit) / depth(16bit) / depth_stride(32bit)
__attribute__((noinline)) void __store_3d(uint32_t src_addr_L1SPM, uint32_t dst_addr_L2SPM, uint32_t write_stride, uint16_t length, uint16_t height, uint32_t depth_stride, uint16_t depth)
{	
	volatile int i;

	uint64_t rs1, rs2;
	uint32_t l1spm_offset = length * height;

	for(i = 0; i < depth; i++) {
		rs1 = ((uint64_t) dst_addr_L2SPM << 27) | ((uint64_t) src_addr_L1SPM);
		rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) write_stride);
		__asm__ volatile ("store %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

		dst_addr_L2SPM += depth_stride;
		src_addr_L1SPM += l1spm_offset;
	}
}



//=================================
// Special function - activation
//=================================
// [relu] 
// relu(x) = (x < 0) ? 0 : x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __relu(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_R_addr)
{	
	// relu
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) 0x0);
	__set_spm_addr_A(l1_R_addr);
	__set_spm_addr_R(l1_A_addr);
	__asm__ volatile("clamp.min %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
	
	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
	__set_spm_addr_R(l1_R_addr);
}


// [relu6] 
// relu6(x) = (x < 0) ? 0 : 
//			  (x > 6) ? 6 : x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __relu6(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_R_addr)
{	
	uint64_t rs1, rs2;

	// relu6 (min clamp)
	rs1 = ((uint64_t) vector_size);
	rs2 = ((uint64_t) 0x0);
	__set_spm_addr_A(l1_R_addr);
	__set_spm_addr_R(l1_A_addr);
	__asm__ volatile("clamp.min %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

	// relu6 (max clamp)
	rs2 = ((uint64_t) 0x4600);
	__set_spm_addr_A(l1_A_addr);
	__asm__ volatile("clamp.max %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));

	// Restore L1SPM addr
	__set_spm_addr_R(l1_R_addr);
}


// [lrelu] 
// lrelu(x) = (x < 0) ? 0 : 0.01 * x
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __lrelu(uint32_t vector_size)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) 0x211E); // fp16_hex(0.01)
	__asm__ volatile ("prelu %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [silu] 
// silu(x) = x * sigmoid(x)
// input : L1SPM_R_BANK (l1_R_addr)
// result: L1SPM_A_BANK (l1_A_addr)
__attribute__((noinline)) void __silu(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr)
{	
	// sigmoid(x) 
	uint64_t rs1 = ((uint64_t) vector_size);
	__asm__ volatile ("sigm %[r1]\n" : : [r1] "r"(rs1)); // sigmoid(x) result: A bank

	// silu(x) = x * sigmoid(x)
	__set_spm_addr_B(l1_R_addr);
	__set_spm_addr_R(l1_A_addr);
	__asm__ volatile ("mul.vv %[r1]\n" : : [r1] "r"(rs1)); // silu(x) result: A bank

	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
	__set_spm_addr_B(l1_B_addr);
	__set_spm_addr_R(l1_R_addr);
}



//=================================
// Special function - math
//=================================
// [var] 
// variance(x) = sum(((x-mean(x)/sqrt(x_num))^2)
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : SVR_1
// result: SVR_2
__attribute__((noinline)) void __var(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr)
{
	uint64_t rs1, rs2, rs3, rs4;

	// sum(x)
	rs1 = ((uint64_t) vector_size << 16);
	__asm__ volatile("mm.o %[r1]\n" : : [r1] "r"(rs1)); // sum(x): SVR_0
	
	// mean(x) = sum(x) / x_num
	rs1 = ((uint64_t) 0x0);
	rs2 = ((uint64_t) __int16_to_fp16(vector_size));
	rs3 = ((uint64_t) 0x1);
	__opset(0, rs3);
	__asm__ volatile("div.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // mean(x): SVR_1

	// x - mean(x)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x184);
	__opset(1, rs4);
	__asm__ volatile("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x - mean(x): R bank
	
	// store x_num to SVR
	rs1 = ((uint64_t) SGPR_0);
	rs2 = ((uint64_t) __int16_to_fp16(vector_size));
	__asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x_num store: SVR_0

	// sqrt(x_num)
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(x_num): SVR_0

	// (x - mean(x)) / sqrt(x_num)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__set_spm_addr_A(l1_R_addr);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // (x - mean(x)) / sqrt(x_num): R bank
																			
	// variance(x) = sum(((x - mean(x) / sqrt(x_num))^2)
	rs1 = ((uint64_t) vector_size << 16);
	rs3 = ((uint64_t) 0x2);
	__set_spm_addr_B(l1_R_addr);
	__opset(0, rs3);
	__asm__ volatile ("mm.v %[r1]\n" : : [r1] "r"(rs1)); // variance(x): SVR_2	

	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
	__set_spm_addr_B(l1_B_addr);	
}

	

//=================================
// Special function - normalization
//=================================
// [layernorm] 
// layernorm(x) = (x-mean(x))/sqrt(var(x)+epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __layernorm(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs3, rs4;
	
	// varience(x)
	// mean    : SVR_1
	// variance: SVR_2
	__var(vector_size, l1_A_addr, l1_B_addr, l1_R_addr);

	// variance(x) + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x2);
		rs2 = ((uint64_t) epsilon);
		rs3 = ((uint64_t) 0x2);
		__opset(0, rs3);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));  // variance + epsilon: SVR_2
	}

	// sqrt(variance(x) + epsilon)
	rs1 = ((uint64_t) 0x2);
	rs3 = ((uint64_t) 0x3);
	__opset(0, rs3);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(variance(x) + epsilon): SVR_3

	// x - mean(x)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x184);
	__opset(1, rs4);
	__asm__ volatile("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x - mean(x): R bank

	// layernorm_eps(x) = (x-mean(x))/sqrt(var(x)+epsilon)
	rs4 = ((uint64_t) 0x18C);
	__set_spm_addr_A(l1_R_addr);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1) ,[r2] "r"(rs2)); // layernorm_eps(x): R bank

	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
}


// [layernorm_aff] 
// layernorm_aff(x) = ((x-mean(x))/var(x)+epsilon) * scale + shift
// input : L1SPM_A_BANK (l1_A_addr)
// scale : L1SPM_B_BANK (l1_B_addr)
// shift : L1SPM_C_BANK (l1_C_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __layernorm_aff(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_C_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs3, rs4;
	
	// variance(x)
	// mean    : SVR_1
	// variance: SVR_2
	__var(vector_size, l1_A_addr, l1_B_addr, l1_R_addr);

	// variance(x) + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x2);
		rs2 = ((uint64_t) epsilon);
		rs3 = ((uint64_t) 0x2);
		__opset(0, rs3);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));  // variance + epsilon: SVR_2
	}

	// sqrt(variance(x) + epsilon)
	rs1 = ((uint64_t) 0x2);
	rs3 = ((uint64_t) 0x3);
	__opset(0, rs3);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(variance(x) + epsilon): SVR_3

	// x - mean(x)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x184);
	__opset(1, rs4);
	__asm__ volatile("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x - mean(x): R bank

	// layernorm_eps(x) = (x-mean(x))/sqrt(variance(x)+epsilon)
	rs4 = ((uint64_t) 0x18C);
	__set_spm_addr_A(l1_R_addr);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1) ,[r2] "r"(rs2)); // layernorm_eps(x): R bank

	// layernorm_eps(x) * scale
	__asm__ volatile("mul.vv %[r1]\n" : : [r1] "r"(rs1));  // layernorm(x) * scale: R bank

	// layernorm_eps_aff(x) = layernorm_eps(x) * scale + shift
	__set_spm_addr_B(l1_C_addr);
	__asm__ volatile("add.vv %[r1]\n" : : [r1] "r"(rs1));  // layernorm_aff(x): R bank

	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
	__set_spm_addr_B(l1_B_addr);
}


// [rmsnorm] 
// rmsnorm(x) = x / R(MS(x)+epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __rmsnorm(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs3, rs4;

	// store x_num to SVR
	rs1 = ((uint64_t) SGPR_0);
	rs2 = ((uint64_t) __int16_to_fp16(vector_size));
	__asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x_num: SVR_0

	// sqrt(x_num)
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(x_num): SVR_0

	// x/sqrt(x_num)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x/sqrt(x_num): R bank
	
	// MS(x) = mean((x/sqrt(x_num))^2)
	rs1 = ((uint64_t) vector_size << 16);
	rs3 = ((uint64_t) 0x1);
	__set_spm_addr_A(l1_R_addr);
	__set_spm_addr_B(l1_R_addr);
	__opset(0, rs3);
	__asm__ volatile ("mm.v %[r1]\n" : : [r1] "r"(rs1)); // mean((x/sqrt(x_num))^2): SVR_1

	// MS(x) + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x1);
		rs2 = ((uint64_t) epsilon);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // MS(x) + epsilon: SVR_0
	}

	// R(MS(x) + epsilon) = sqrt(MS(x) + epsilon) 
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // RMS(x): SVR_0

	// rmsnorm_eps(x) = x / R(MS(x) + epsilon)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__set_spm_addr_A(l1_A_addr);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" :  : [r1] "r"(rs1), [r2] "r"(rs2)); // rmsnorm_eps(x): R bank

	// Restore L1SPM addr
	__set_spm_addr_B(l1_B_addr);
}


// [rmsnorm_scale] 
// rmsnorm_scale(x) = (x/R(MS(x)+epsilon))* scale
// input : L1SPM_A_BANK (l1_A_addr)
// scale : L1SPM_B_BANK (l1_B_addr)
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __rmsnorm_scale(uint32_t vector_size, uint32_t l1_A_addr, uint32_t l1_B_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs3, rs4;

	// store x_num to SVR
	rs1 = ((uint64_t) SGPR_0);
	rs2 = ((uint64_t) __int16_to_fp16(vector_size));
	__asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x_num: SVR_0

	// sqrt(x_num)
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(x_num): SVR_0

	// x/sqrt(x_num)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x/sqrt(x_num): R bank
	
	// MS(x) = mean((x/sqrt(x_num))^2)
	rs1 = ((uint64_t) vector_size << 16);
	rs3 = ((uint64_t) 0x1);
	__set_spm_addr_A(l1_R_addr);
	__set_spm_addr_B(l1_R_addr);
	__opset(0, rs3);
	__asm__ volatile ("mm.v %[r1]\n" : : [r1] "r"(rs1)); // mean((x/sqrt(x_num))^2): SVR_1
	
	// MS(x) + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x1);
		rs2 = ((uint64_t) epsilon);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // MS(x) + epsilon: SVR_0
	}

	// R(MS(x) + epsilon) = sqrt(MS(x) + epsilon) 
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // RMS(x): SVR_0

	// rmsnorm_eps(x) = x / R(MS(x) + epsilon)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__set_spm_addr_A(l1_A_addr);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" :  : [r1] "r"(rs1), [r2] "r"(rs2)); // rmsnorm_eps(x): R bank

	// rmsnorm_eps_scale(x) = (x/R(MS(x) + epsilon))* scale
	__set_spm_addr_A(l1_R_addr);
	__set_spm_addr_B(l1_B_addr);
	__asm__ volatile("mul.vv %[r1]\n" : : [r1] "r"(rs1));  // rmsnorm_eps_scale(x): R Bank

	// Restore L1SPM addr
	__set_spm_addr_A(l1_A_addr);
}


// [batchnorm] 
// batchnorm(x) = (x-mean)/sqrt(var_epsilon)
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : fp16 value
// var 	 : fp16 value
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __batchnorm(uint32_t vector_size, uint16_t mean, uint16_t var, uint32_t l1_A_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs4;
	
	// x - mean
	rs1 = ((uint64_t) vector_size);
	rs2 = ((uint64_t) mean);
	__set_spm_addr_R(l1_A_addr); 
	__asm__ volatile("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x - mean result: A bank

	// store variacne to SVR
	rs1 = ((uint64_t) SGPR_0);
	rs2 = ((uint64_t) var);
	__asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // var: SVR_0

	// var + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x0);
		rs2 = ((uint64_t) epsilon);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // var + epsilon: SVR_0
	}
	
	// sqrt(var + epsilon)
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(var + epsilon): SVR_0

	// batchnorm_eps(x) = (x - mean) / sqrt(var + epsilon)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__set_spm_addr_R(l1_R_addr); 
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" :  : [r1] "r"(rs1), [r2] "r"(rs2)); // batchnorm_eps(x): R bank
}


// [batchnorm_aff] 
// batchnorm_aff(x) = (x-mean)/sqrt(var+epsilon)*scale + shift
// input : L1SPM_A_BANK (l1_A_addr)
// mean  : fp16 value
// var 	 : fp16 value
// scale : fp16 value
// shift : fp16 value
// result: L1SPM_R_BANK (l1_R_addr)
__attribute__((noinline)) void __batchnorm_aff(uint32_t vector_size, uint16_t mean, uint16_t var, uint16_t scale, uint16_t shift, uint32_t l1_A_addr, uint32_t l1_R_addr, uint16_t epsilon)
{
	uint64_t rs1, rs2, rs4;
	
	// x - mean
	rs1 = ((uint64_t) vector_size);
	rs2 = ((uint64_t) mean);
	__set_spm_addr_R(l1_A_addr); 
	__asm__ volatile("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // x - mean result: A bank

	// store variacne to SVR
	rs1 = ((uint64_t) SGPR_0);
	rs2 = ((uint64_t) var);
	__asm__ volatile("wrspr %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // var: SVR_0

	// var + epsilon
	if ((epsilon != 0x0000) || (epsilon != 0x8000)) {
		rs1 = ((uint64_t) 0x0);
		rs2 = ((uint64_t) epsilon);
		__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // var + epsilon: SVR_0
	}
	
	// sqrt(var + epsilon)
	rs1 = ((uint64_t) 0x0);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1)); // sqrt(var + epsilon): SVR_0

	// batchnorm_eps(x) = (x - mean) / sqrt(var + epsilon)
	rs1 = ((uint64_t) vector_size);
	rs4 = ((uint64_t) 0x180);
	__opset(1, rs4);
	__asm__ volatile("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2)); // batchnorm_eps(x): A bank

	// batchnorm_eps(x) * scale
	rs2 = ((uint64_t) scale); 
	__asm__ volatile("mul.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));  // batchnorm_eps(x) * scale: A Bank

	// batchnorm_eps_aff(x) = batchnorm_eps(x) * scale + shift
	rs2 = ((uint64_t) shift); 
	__set_spm_addr_R(l1_R_addr); 
	__asm__ volatile("add.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));  // batchnorm_eps_aff(x): R Bank
}



#endif
