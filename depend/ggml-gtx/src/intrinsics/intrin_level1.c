//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic function description (level 1)
// 				 base instruction intrinsic function
// Author      : mh.kim ( NPU Div - NPU Core Team )    
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_LEVEL1_C
#define INTRIN_LEVEL1_C

#include "ggml-gtx/intrinsics/intrin_level1.h"
#include "ggml-gtx/intrinsics/gtx_csr.h"


//=================================
// Matrix multiplication
//=================================
// [mm.s] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm_s(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
{
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);	
	
	__asm__ volatile ("mm.s %[r1]\n" : : [r1] "r"(rs1));
}


// [mm.o] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mm_o(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mm.o %[r1]\n" : : [r1] "r"(rs1));
}


// [mm.v] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mm_v(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mm.v %[r1]\n" : : [r1] "r"(rs1));
}


// [mm] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
{
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);
	
	__asm__ volatile ("mm %[r1]\n" : : [r1] "r"(rs1));
}


// [mm.t] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm_t(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size) 
{	
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);	
	
	__asm__ volatile ("mm.t %[r1]\n" : : [r1] "r"(rs1));
}


// [mmc.s] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc_s(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size) 
{	
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);		
	
	__asm__ volatile ("mmc.s %[r1]\n" : : [r1] "r"(rs1));
}


// [mmc.o] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mmc_o(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mmc.o %[r1]\n" : : [r1] "r"(rs1));
}


// [mmc.v] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mmc_v(uint16_t col_A_size, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile ("mmc.v %[r1]\n" : : [r1] "r"(rs1));
}


// [mmc] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size) 
{	
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);		
	
	__asm__ volatile ("mmc %[r1]\n" : : [r1] "r"(rs1));
}


// [mmc.t] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc_t(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size) 
{	
	uint64_t rs1 = ((uint64_t) col_B_size << 48) | ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);		
	
	__asm__ volatile ("mmc.t %[r1]\n" : : [r1] "r"(rs1));
}



//=================================
// Convolution
//=================================
// [im2col.n] rs1 rs2
// rs1 = col_A_size[31:16], row_A_size[15:0]
// rs2 = num_of_channel[47:32], stride[31:16], dilation[9:8], kernel_size[4:0]
__attribute__((noinline)) void __im2col_n(uint16_t row_A_size , uint16_t col_A_size, 
 										  uint8_t  kernel_size, uint8_t  dilation  , uint16_t stride, uint16_t num_of_channel)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);
	uint64_t rs2 = ((uint64_t) num_of_channel << 32) | ((uint64_t) stride << 16) | ((uint64_t) dilation << 8) | ((uint64_t) kernel_size);
	
	__asm__ volatile ("im2col.n %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [im2col.d] rs1 rs2
// rs1 = col_A_size[31:16], row_A_size[15:0]
// rs2 = num_of_channel[47:32], stride[31:16], dilation[9:8], kernel_size[4:0]
__attribute__((noinline)) void __im2col_d(uint16_t row_A_size , uint16_t col_A_size, 
 										  uint8_t  kernel_size, uint8_t  dilation  , uint16_t stride, uint16_t num_of_channel)
{	
	uint64_t rs1 = ((uint64_t) col_A_size << 16) | ((uint64_t) row_A_size);
	uint64_t rs2 = ((uint64_t) num_of_channel << 32) | ((uint64_t) stride << 16) | ((uint64_t) dilation << 8) | ((uint64_t) kernel_size);
	
	__asm__ volatile ("im2col.d %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Scalar calculation
//=================================
// [add.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __add_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(1, rs4);
	__asm__ volatile ("add.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [sub.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __sub_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(1, rs4);
	__asm__ volatile ("sub.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mul.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __mul_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(1, rs4);
	__asm__ volatile ("mul.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [div.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __div_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(1, rs4);
	__asm__ volatile ("div.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [fmadd.vss] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value1[31:16], scalar_value0[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __fmadd_vss(uint32_t vector_size, uint16_t scalar_value0, uint16_t scalar_value1, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value1 << 16) | ((uint64_t) scalar_value0);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile("fmadd.vss %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [max.vs] rs1 rs2 rs3 rs4
// rs1 = vector_size[23:0]
// rs2 = previous_max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __max_vs(uint32_t vector_size, uint16_t previous_max_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) previous_max_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("max.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [min.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = previous_min_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __min_vs(uint32_t vector_size, uint16_t previous_min_value, uint8_t result_svr_addr, uint16_t r2_sel)
{	
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) previous_min_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("min.vs %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [add.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __add_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("add.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [sub.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __sub_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("sub.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mul.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __mul_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("mul.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mul.is.f32] rs1 rs2 rs3
// rs1 = src_svr_addr[4:0]
// rs2 = FP32 scalar raw bits[31:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mul_is_f32(uint8_t src_svr_addr, float scalar_value, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	union {
		float f;
		uint32_t u;
	} scalar_bits = { scalar_value };
	uint64_t rs2 = ((uint64_t) scalar_bits.u);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile(".insn r 0x0B, 6, 0x12, x0, %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [div.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __div_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("div.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [fmadd.iss] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value1[31:16], scalar_value0[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __fmadd_iss(uint8_t src_svr_addr, uint16_t scalar_value0, uint16_t scalar_value1, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) scalar_value1 << 16) | ((uint64_t) scalar_value0);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("fmadd.iss %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [max.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = previous_max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __max_is(uint8_t src_svr_addr, uint16_t previous_max_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) previous_max_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("max.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));	
}


// [min.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = previous_min_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __min_is(uint8_t src_svr_addr, uint16_t previous_min_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr);
	uint64_t rs2 = ((uint64_t) previous_min_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("min.is %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Vector calculation
//=================================
// [clamp.min] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __clamp_min(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{	
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile("clamp.min %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [clamp.max] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __clamp_max(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel)
{	
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) scalar_value);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile("clamp.max %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [arange] rs1 rs2
// rs1 = vector_size[23:0]
// rs2 = step[31:16], start_value[15:0]
__attribute__((noinline)) void __arange(uint32_t vector_size, uint16_t start_value, uint16_t step)
{	
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) step << 16) | ((uint64_t) start_value);

	__asm__ volatile("arange %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [add.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __add_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{	
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("add.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [sub.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sub_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("sub.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mul.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mul_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("mul.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [div.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __div_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("div.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [fmadd.iii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_C[9:5], src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __fmadd_iii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t src_svr_addr_C, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_C << 5) | ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("fmadd.iii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [sqrt.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sqrt_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("sqrt.i %[r1]\n" : : [r1] "r"(rs1));
}


// [exp.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = mode[1:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __exp_i(uint8_t src_svr_addr_A, uint8_t mode, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) mode);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	// __asm__ volatile("exp.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



// [ln.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = mode[1:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __ln_i(uint8_t src_svr_addr_A, uint8_t mode, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) mode);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	// __asm__ volatile("ln.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [abs.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __abs_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("abs.i %[r1]\n" : : [r1] "r"(rs1));
}


// [neg.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __neg_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("neg.i %[r1]\n" : : [r1] "r"(rs1));
}


// [sign.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sign_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("sign.i %[r1]\n" : : [r1] "r"(rs1));
}


// [step.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __step_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("step.i %[r1]\n" : : [r1] "r"(rs1));
}


// [ceil.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __ceil_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("ceil.i %[r1]\n" : : [r1] "r"(rs1));
}


// [trunc.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __trunc_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("trunc.i %[r1]\n" : : [r1] "r"(rs1));
}


// [floor.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __floor_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("floor.i %[r1]\n" : : [r1] "r"(rs1));
}


// [rne.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __rne_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("rne.i %[r1]\n" : : [r1] "r"(rs1));
}


// [and.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __and_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("and.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [or.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __or_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) src_svr_addr_B);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("or.ii %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [not.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __not_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("not.i %[r1]\n" : : [r1] "r"(rs1));
}


// [shift.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = shift_mode[4], shift_num[3:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __shift_i(uint8_t src_svr_addr_A, uint8_t shift_num, uint8_t shift_mode, uint8_t result_svr_addr)

{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) shift_mode << 4) | ((uint64_t) shift_num);
	uint64_t rs3 = ((uint64_t) result_svr_addr);

	__opset(0, rs3);
	__asm__ volatile("shift.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Format conversion
//=================================
// [scvt.qh] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_qh(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) offset << 16) | ((uint64_t) scale);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("scvt.qh %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [scvt.hq] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hq(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) offset << 16) | ((uint64_t) scale);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("scvt.hq %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [scvt.ih] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_ih(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) offset << 16) | ((uint64_t) scale);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("scvt.ih %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [scvt.hi] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hi(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) offset << 16) | ((uint64_t) scale);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("scvt.hi %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [scvt.hn] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hn(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) offset << 16) | ((uint64_t) scale);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("scvt.hn %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Activation
//=================================
// [prelu] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = slop_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __prelu(uint32_t vector_size, uint16_t slop_value, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) slop_value);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("prelu %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [prelu.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = slop_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __prelu_i(uint8_t src_svr_addr_A, uint16_t slop_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) slop_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);
	
	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile("prelu.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [gelu.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __gelu_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile("gelu.i %[r1]\n" : : [r1] "r"(rs1));
}


// [tanh.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __tanh_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile("tanh.i %[r1]\n" : : [r1] "r"(rs1));
}


// [sigm.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sigm_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	
	__opset(0, rs3);
	__asm__ volatile("sigm.i %[r1]\n" : : [r1] "r"(rs1));
}



//=================================
// Softmax
//=================================
// [esum] rs1 rs2 rs3 rs4
// rs1 = vertor_size[23:0]
// rs2 = accumulated_data[31:16], max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __esum(uint32_t vector_size, uint16_t max_value, uint16_t accumulated_data, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) accumulated_data << 16) | ((uint64_t) max_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile ("esum %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [softmax] rs1 rs2 rs4
// rs1 = vertor_size[23:0]
// rs2 = esum_value[31:16], max_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __softmax(uint32_t vector_size, uint16_t max_value, uint16_t esum_value, uint16_t r2_sel)
{	
	uint64_t rs1 = ((uint64_t) vector_size);
	uint64_t rs2 = ((uint64_t) esum_value << 16) | ((uint64_t) max_value);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(1, rs4);
	__asm__ volatile ("softmax %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [esum.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = accumulated_data[31:16], max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __esum_i(uint8_t src_svr_addr_A, uint16_t max_value, uint16_t accumulated_data, uint8_t result_svr_addr, uint16_t r2_sel)
{	
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) accumulated_data << 16) | ((uint64_t) max_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(0, rs3);
	__opset(1, rs4);
	__asm__ volatile ("esum.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [softmax.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = esum_value[31:16], max_value[15:0] 
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __softmax_i(uint8_t src_svr_addr_A, uint16_t max_value, uint16_t esum_value, uint8_t result_svr_addr, uint16_t r2_sel)
{
	uint64_t rs1 = ((uint64_t) src_svr_addr_A);
	uint64_t rs2 = ((uint64_t) esum_value << 16) | ((uint64_t) max_value);
	uint64_t rs3 = ((uint64_t) result_svr_addr);
	uint64_t rs4 = ((uint64_t) r2_sel);

	__opset(0, rs3);
	__opset(1, rs4);	
	__asm__ volatile("softmax.i %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Pooling
//=================================
// [pool.m] rs1 rs2
// rs1 = col_OUT[63:48], row_OUT[47:32], col_IN[31:16], row_IN[15:0]
// rs2 = col_stride[31:24], row_stride[23:16], col_K[15:8], row_K[7:0]
__attribute__((noinline)) void __pool_m(uint16_t row_IN, uint16_t col_IN, uint16_t row_OUT   , uint16_t col_OUT   , 
 										uint8_t  row_K , uint8_t  col_K , uint8_t  row_stride, uint8_t  col_stride)
{
	uint64_t rs1 = ((uint64_t) col_OUT << 48) | ((uint64_t) row_OUT << 32) | ((uint64_t) col_IN << 16) | ((uint64_t) row_IN);
	uint64_t rs2 = ((uint64_t) col_stride << 24) | ((uint64_t) row_stride << 16) | ((uint64_t) col_K << 8) | ((uint64_t) row_K);
	
	__asm__ volatile ("pool.m %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [pool.a] rs1, rs2
// rs1 = col_OUT[63:48], row_OUT[47:32], col_IN[31:16], row_IN[15:0]
// rs2 = k_value[47:32]. col_stride[31:24], row_stride[23:16], col_K[15:8], row_K[7:0]
__attribute__((noinline)) void __pool_a(uint16_t row_IN, uint16_t col_IN, uint16_t row_OUT   , uint16_t col_OUT   , 
 										uint8_t  row_K , uint8_t  col_K , uint8_t  row_stride, uint8_t  col_stride, uint16_t k_value)
{	
	uint64_t rs1 = ((uint64_t) col_OUT << 48) | ((uint64_t) row_OUT << 32) | ((uint64_t) col_IN << 16) | ((uint64_t) row_IN);
	uint64_t rs2 = ((uint64_t) k_value << 32) | ((uint64_t) col_stride << 24) | ((uint64_t) row_stride << 16) | ((uint64_t) col_K << 8) | ((uint64_t) row_K);
	
	__asm__ volatile ("pool.a %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// Memory operation
//=================================
// [tpose] rs1 rs2 rs3
// rs1 = dtype[56], rw_dir[49:48], src_addr[36:0]
// rs2 = dim2_size[63:48], dim1_size[47:32], dim0_size[31:16], dim2[9:8], dim1[5:4], dim0[1:0]
// rs3 = dst_addr[36:0]
__attribute__((noinline)) void __tpose(uint64_t src_addr , uint64_t dst_addr , 
 									   uint16_t dim2_size, uint16_t dim1_size, uint16_t dim0_size, uint8_t dim2, uint8_t dim1, uint8_t dim0, 
									   uint8_t  rw_dir   , uint8_t  dtype    )
{
	src_addr = src_addr & 0x0000001FFFFFFFFF;
	
	uint64_t rs1 = ((uint64_t) dtype << 56) | ((uint64_t) rw_dir << 48) | ((uint64_t) src_addr);
	uint64_t rs2 = ((uint64_t) dim2_size << 48) | ((uint64_t) dim1_size << 32) | ((uint64_t) dim0_size << 16) | ((uint64_t) dim2 << 8) | ((uint64_t) dim1 << 4) | ((uint64_t) dim0);
	uint64_t rs3 = dst_addr;

	__opset(0, rs3);
	__asm__ volatile("tpose %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [fill] rs1 rs2 rs3
// rs1 = dir[48], dst_addr[36:0]
// rs2 = height[63:48], length[47:32], write_stride[31:0]
// rs3 = fill_pattern[63:0]
__attribute__((noinline)) void __fill(uint64_t dst_addr, uint32_t write_stride, uint16_t length, uint16_t height, uint64_t fill_pattern, uint8_t dir)
{	
	dst_addr = dst_addr & 0x0000001FFFFFFFFF;

	uint64_t rs1 = ((uint64_t) dir << 48) | ((uint64_t) dst_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) write_stride);
	uint64_t rs3 = fill_pattern;

	__opset(0, rs3);
	__asm__ volatile("fill %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// DMA
//=================================
// [load] rs1 rs2 rs3
// rs1 = src_addr[63:27], dst_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = write_stride[31:0]
__attribute__((noinline)) void __load(uint64_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride)
{
	src_addr = src_addr & 0x0000001FFFFFFFFF;
	dst_addr = dst_addr & 0x07FFFFFF;

	uint64_t rs1 = ((uint64_t) src_addr << 27) | ((uint64_t) dst_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = ((uint64_t) write_stride);

	__opset(0, rs3);
	__asm__ volatile ("load %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [store] rs1 rs2 rs3
// rs1 = dst_addr[63:27], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], write_stride[31:0]
// rs3 = read_stride[31:0]
__attribute__((noinline)) void __store(uint32_t src_addr, uint64_t dst_addr, uint32_t write_stride, uint16_t length, uint16_t height, uint32_t read_stride)
{	
	src_addr = src_addr & 0x07FFFFFF;
	dst_addr = dst_addr & 0x0000001FFFFFFFFF;

	uint64_t rs1 = ((uint64_t) dst_addr << 27) | ((uint64_t) src_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) write_stride);
	uint64_t rs3 = ((uint64_t) read_stride);

	__opset(0, rs3);
	__asm__ volatile ("store %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [copy] rs1 rs2 rs3
// rs1 = dst_addr[58:32], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = write_stride[31:0]
__attribute__((noinline)) void __copy(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride)
{
	uint64_t rs1 = ((uint64_t) dst_addr << 32) | ((uint64_t) src_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = ((uint64_t) write_stride);

	__opset(0, rs3);
	__asm__ volatile ("copy %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



// [mcast.s2l] rs1 rs2 rs3
// rs1 = src_addr[58:32], dst_addr[23:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_spu[63:0]
__attribute__((noinline)) void __mcast_s2l(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint64_t target_spu)
{
	uint64_t rs1 = ((uint64_t) src_addr << 32) | ((uint64_t) dst_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = target_spu;

	__opset(0, rs3);
	__asm__ volatile("mcast.s2l %[r1], %[r2]\n" :  : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mcast.g2s] rs1 rs2 rs3
// rs1 = src_addr[63:27], dst_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_nest[63:0]
__attribute__((noinline)) void __mcast_g2s(uint64_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint64_t target_nest)
{		
	src_addr = src_addr & 0x0000001FFFFFFFFF;

	uint64_t rs1 = ((uint64_t) src_addr << 27) | ((uint64_t) dst_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = target_nest;

	__opset(0, rs3);
	__asm__ volatile("mcast.g2s %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mcast.s2s] rs1 rs2 rs3
// rs1 = target_nest_sel[63], src_nest_id[61:56], dst_addr[55:27], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_nest[63:32], write_stride[31:0]
__attribute__((noinline)) void __mcast_s2s(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride, 
										   uint8_t src_nest_id, uint8_t target_nest_sel, uint32_t target_nest)
{
	src_addr = src_addr & 0x07FFFFFF;
	dst_addr = dst_addr & 0x07FFFFFF;

	uint64_t rs1 = ((uint64_t) target_nest_sel << 63) | ((uint64_t) src_nest_id << 56) | ((uint64_t) dst_addr << 27) | ((uint64_t) src_addr);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = ((uint64_t) target_nest << 32) | ((uint64_t) write_stride);
	
	__opset(0, rs3);
	__asm__ volatile("mcast.s2s %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [copy.mem] rs1 rs2 rs3
// rs1 = write_stride_L[63:48], src_addr_DDR[36:0]
// rs3 = heigth[63:48], length[47:32], read_stride[31:0]
// rs2 = write_stride_H[63:48], dst_addr_DDR[36:0]
__attribute__((noinline)) void __copy_mem(uint64_t src_addr_DDR, uint64_t dst_addr_DDR, uint32_t read_stride, uint16_t length, uint16_t height, uint16_t write_stride_L, uint16_t write_stride_H)
{
	src_addr_DDR = src_addr_DDR & 0x0000001FFFFFFFFF;
	dst_addr_DDR = dst_addr_DDR & 0x0000001FFFFFFFFF;

	uint64_t rs1 = ((uint64_t) write_stride_L << 48) | ((uint64_t) src_addr_DDR);
	uint64_t rs2 = ((uint64_t) height << 48) | ((uint64_t) length << 32) | ((uint64_t) read_stride);
	uint64_t rs3 = ((uint64_t) write_stride_H << 48) | ((uint64_t) dst_addr_DDR);

	__opset(0, rs3);
	__asm__ volatile("copy.mem %[r1], %[r2]\n" : : [r1] "r"(rs1), [r2] "r"(rs2));
}



//=================================
// SPR
//=================================
// [rdspr] rd rs1
// rd  = spr_data[63:0]
// rs1 = nest_id[29:24], spu_id[21:16], spr_addr[11:0]
__attribute__((noinline)) uint64_t __rdspr(uint16_t spr_addr, uint8_t nest_id, uint8_t spu_id)
{
	uint64_t res;
	uint64_t rs1 = ((uint64_t) nest_id << 24) | ((uint64_t) spu_id << 16) | ((uint64_t) spr_addr);
	
	__asm__ volatile("rdspr %[result], %[r1]\n" : [result] "=r"(res) : [r1] "r"(rs1));

	return res;
}


// [wrspr] rs1 rs2 rs3 rs4
// rs1 = wrstb_n[23:16], spr_addr[11:0]
// rs2 = spr_data[63:0]
// rs3 = target_mask[63:0]
__attribute__((noinline)) void __wrspr(uint16_t spr_addr, uint8_t wrstb_n, uint64_t spr_data, uint64_t target_mask)
{
	uint64_t rs1 = ((uint64_t) wrstb_n << 16) | ((uint64_t) spr_addr);
	uint64_t rs2 = spr_data;
	uint64_t rs3 = target_mask;

	__opset(0, rs3);
	__asm__ volatile("wrspr %[r1], %[r2]\n" :  : [r1] "r"(rs1), [r2] "r"(rs2));
}


// [mvsvr] rs1 rs2 rs3
// rs1 = src_svr_addr[4:0]
// rs2 = dst_svr_addr[4:0]
// rs3 = wrstrb_n[31:0]
__attribute__((noinline)) void __mvsvr(uint8_t src_svr_addr, uint8_t dst_svr_addr, uint32_t wrstrb_n)
{
	uint64_t rs3 = ((uint64_t) wrstrb_n);

	__opset(0, rs3);
	__asm__ volatile("mvsvr %[r1], %[r2]\n" :  : [r1] "r"(src_svr_addr), [r2] "r"(dst_svr_addr));
}



//=================================
// Credit
//=================================



//=================================
// Micro Code
//=================================



//=================================
// Sync
//=================================


//=================================
// Warp operation
//=================================



#endif