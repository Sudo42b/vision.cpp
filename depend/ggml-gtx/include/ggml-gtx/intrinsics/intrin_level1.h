//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic definition (level 1)
// 				 base instruction intrinsic
// Author      : mh.kim ( NPU Div - NPU Core Team )
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_LEVEL1_H
#define INTRIN_LEVEL1_H

#include <stdint.h>

#define GTX_VEC_DTYPE_F16_F32_FLAG (1u << 30)
#define GTX_VEC_DTYPE_F32_F32_TO_F16_FLAG (1u << 29)

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
// Matrix multiplication
//=================================
// [mm.s] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm_s(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);


// [mm.o] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mm_o(uint16_t col_A_size, uint8_t result_svr_addr);


// [mm.v] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mm_v(uint16_t col_A_size, uint8_t result_svr_addr);


// [mm] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);


// [mm.t] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mm_t(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);


// [mmc.s] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc_s(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);


// [mmc.o] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mmc_o(uint16_t col_A_size, uint8_t result_svr_addr);


// [mmc.v] rs1 rs3
// rs1 = col_A_size[31:16]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mmc_v(uint16_t col_A_size, uint8_t result_svr_addr);


// [mmc] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);


// [mmc.t] rs1
// rs1 = col_B_size[63:48], col_A_size[31:16], row_A_size[15:0]
__attribute__((noinline)) void __mmc_t(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);



//=================================
// Convolution
//=================================
// [im2col.n] rs1 rs2
// rs1 = col_A_size[31:16], row_A_size[15:0]
// rs2 = num_of_channel[47:32], stride[31:16], dilation[9:8], kernel_size[4:0]
__attribute__((noinline)) void __im2col_n(uint16_t row_A_size , uint16_t col_A_size, 
 										  uint8_t  kernel_size, uint8_t  dilation  , uint16_t stride, uint16_t num_of_channel);


// [im2col.d] rs1 rs2
// rs1 = col_A_size[31:16], row_A_size[15:0]
// rs2 = num_of_channel[47:32], stride[31:16], dilation[9:8], kernel_size[4:0]
__attribute__((noinline)) void __im2col_d(uint16_t row_A_size , uint16_t col_A_size, 
 										  uint8_t  kernel_size, uint8_t  dilation  , uint16_t stride, uint16_t num_of_channel);



//=================================
// Scalar calculation
//=================================
// [add.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __add_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [sub.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __sub_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [mul.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __mul_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [div.vs] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __div_vs(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [fmadd.vss] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value1[31:16], scalar_value0[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __fmadd_vss(uint32_t vector_size, uint16_t scalar_value0, uint16_t scalar_value1, uint16_t r2_sel);


// [max.vs] rs1 rs2 rs3 rs4
// rs1 = vector_size[23:0]
// rs2 = previous_max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __max_vs(uint32_t vector_size, uint16_t previous_max_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [min.vs] rs1 rs2 rs3 rs4
// rs1 = vector_size[23:0]
// rs2 = previous_min_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __min_vs(uint32_t vector_size, uint16_t previous_min_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [add.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __add_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [sub.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __sub_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [mul.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __mul_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [mul.is.f32] rs1 rs2 rs3
// rs1 = src_svr_addr[4:0]
// rs2 = FP32 scalar raw bits[31:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mul_is_f32(uint8_t src_svr_addr, float scalar_value, uint8_t result_svr_addr);


// [div.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __div_is(uint8_t src_svr_addr, uint16_t scalar_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [fmadd.iss] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = scalar_value1[31:16], scalar_value0[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __fmadd_iss(uint8_t src_svr_addr, uint16_t scalar_value0, uint16_t scalar_value1, uint8_t result_svr_addr, uint16_t r2_sel);


// [max.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = previous_max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __max_is(uint8_t src_svr_addr, uint16_t previous_max_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [min.is] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr[4:0]
// rs2 = previous_min_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __min_is(uint8_t src_svr_addr, uint16_t previous_min_value, uint8_t result_svr_addr, uint16_t r2_sel);



//=================================
// Vector calculation
//=================================
// [add.vv] rs1
// rs1 = vector_size[23:0]
#define __add_vv(vector_size)\
	__asm__ volatile("add.vv %[r1]\n" : : [r1] "r"(vector_size));


// [sub.vv] rs1
// rs1 = vector_size[23:0]
#define __sub_vv(vector_size)\
	__asm__ volatile("sub.vv %[r1]\n" : : [r1] "r"(vector_size));


// [mul.vv] rs1
// rs1 = vector_size[23:0]
#define __mul_vv(vector_size)\
	__asm__ volatile("mul.vv %[r1]\n" : : [r1] "r"(vector_size));


// [mul.vv F16xF32] rs1
// ADDRA is FP16, ADDRB is FP32, ADDRR is FP16.
// rs1 = F16_F32_FLAG[30] | vector_size[15:0]
#define __mul_vv_f16_f32(vector_size)\
	__asm__ volatile("mul.vv %[r1]\n" : : [r1] "r"(((uint32_t)(vector_size) & 0xFFFFu) | GTX_VEC_DTYPE_F16_F32_FLAG));


// [mul.vv F32xF32] rs1
// ADDRA is FP32, ADDRB is FP32, ADDRR is FP16.
// rs1 = F32_F32_TO_F16_FLAG[29] | vector_size[15:0]
#define __mul_vv_f32_f32(vector_size)\
	__asm__ volatile("mul.vv %[r1]\n" : : [r1] "r"(((uint32_t)(vector_size) & 0xFFFFu) | GTX_VEC_DTYPE_F32_F32_TO_F16_FLAG));


// [div.vv] rs1
// rs1 = vector_size[23:0]
#define __div_vv(vector_size)\
	__asm__ volatile("div.vv %[r1]\n" : : [r1] "r"(vector_size));


// [fmadd.vvv] rs1
// rs1 = vector_size[23:0]
#define __fmadd_vvv(vector_size)\
	__asm__ volatile("fmadd.vvv %[r1]\n" : : [r1] "r"(vector_size));


// [sqrt.v] rs1
// rs1 = vector_size[23:0]
#define __sqrt_v(vector_size)\
	__asm__ volatile("sqrt.v %[r1]\n" : : [r1] "r"(vector_size));


// [exp.v] rs1 rs2
// rs1 = vector_size[23:0]
// rs2 = mode[1:0]
#define __exp_v(vector_size, mode)\
	__asm__ volatile("exp.v %[r1]\n" : : [r1] "r"(vector_size));


// [ln.v] rs1 rs2
// rs1 = vector_size[23:0]
// rs2 = mode[1:0]
#define __ln_v(vector_size, mode)\
	__asm__ volatile("ln.v %[r1]\n" : : [r1] "r"(vector_size));


// [abs.v] rs1
// rs1 = vector_size[23:0]
#define __abs_v(vector_size)\
	__asm__ volatile("abs.v %[r1]\n" : : [r1] "r"(vector_size));


// [neg.v] rs1
// rs1 = vector_size[23:0]
#define __neg_v(vector_size)\
	__asm__ volatile("neg.v %[r1]\n" : : [r1] "r"(vector_size));


// [sign.v] rs1
// rs1 = vector_size[23:0]
#define __sign_v(vector_size)\
	__asm__ volatile("sign.v %[r1]\n" : : [r1] "r"(vector_size));


// [step.v] rs1
// rs1 = vector_size[23:0]
#define __step_v(vector_size)\
	__asm__ volatile("step.v %[r1]\n" : : [r1] "r"(vector_size));


// [ceil.v] rs1
// rs1 = vector_size[23:0]
#define __ceil_v(vector_size)\
	__asm__ volatile("ceil.v %[r1]\n" : : [r1] "r"(vector_size));


// [trunc.v] rs1
// rs1 = vector_size[23:0]
#define __trunc_v(vector_size)\
	__asm__ volatile("trunc.v %[r1]\n" : : [r1] "r"(vector_size));


// [floor.v] rs1
// rs1 = vector_size[23:0]
#define __floor_v(vector_size)\
	__asm__ volatile("floor.v %[r1]\n" : : [r1] "r"(vector_size));


// [rne.v] rs1
// rs1 = vector_size[23:0]
#define __rne_v(vector_size)\
	__asm__ volatile("rne.v %[r1]\n" : : [r1] "r"(vector_size));


// [clamp.min] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __clamp_min(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [clamp.max] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = scalar_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __clamp_max(uint32_t vector_size, uint16_t scalar_value, uint16_t r2_sel);


// [accum] rs1
// rs1 = vector_size[23:0]
#define __accum(vector_size)\
	__asm__ volatile("accum %[r1]\n" : : [r1] "r"(vector_size));


// [arange] rs1 rs2
// rs1 = vector_size[23:0]
// rs2 = step[31:16], start_value[15:0]
__attribute__((noinline)) void __arange(uint32_t vector_size, uint16_t start_value, uint16_t step);


// [add.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __add_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [sub.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sub_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [mul.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __mul_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [div.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __div_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [fmadd.iii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_C[9:5], src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __fmadd_iii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t src_svr_addr_C, uint8_t result_svr_addr);

 
// [sqrt.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sqrt_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [exp.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = mode[1:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __exp_i(uint8_t src_svr_addr_A, uint8_t mode, uint8_t result_svr_addr);


// [ln.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = mode[1:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __ln_i(uint8_t src_svr_addr_A, uint8_t mode, uint8_t result_svr_addr);


// [abs.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __abs_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [neg.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __neg_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [sign.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sign_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [step.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __step_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [ceil.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __ceil_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [trunc.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __trunc_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [floor.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __floor_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [rne.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __rne_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [and.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __and_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [or.ii] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = src_svr_addr_B[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __or_ii(uint8_t src_svr_addr_A, uint8_t src_svr_addr_B, uint8_t result_svr_addr);


// [not.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __not_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [shift.i] rs1 rs2 rs3
// rs1 = src_svr_addr_A[4:0]
// rs2 = shift_mode[4], shift_num[3:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __shift_i(uint8_t src_svr_addr_A, uint8_t shift_num, uint8_t shift_mode, uint8_t result_svr_addr);



//=================================
// Format conversion
//=================================
// [scvt.qh] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_qh(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel);


// [scvt.hq] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hq(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel);


// [scvt.ih] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_ih(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel);


// [scvt.hi] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hi(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel);


// [scvt.hn] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = offset[31:16], scale[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __scvt_hn(uint32_t vector_size, uint16_t scale, uint16_t offset, uint16_t r2_sel);


// [fcvt.sh] rd rs1
// rd  = result_data[63:0]
// rs1 = input_data[63:0]
#define __fcvt_sh(input_data)\
	({uint64_t res; __asm__ volatile("fcvt.sh %[rd], %[r1]\n" : [rd] "=r"(res) : [r1] "r"(input_data)); res;})


// [fcvt.hs] rd rs1
// rd  = result_data[63:0]
// rs1 = input_data[63:0]
#define __fcvt_hs(input_data)\
	({uint64_t res; __asm__ volatile("fcvt.hs %[rd], %[r1]\n" : [rd] "=r"(res) : [r1] "r"(input_data)); res;})


// [fcvt.dh] rd rs1
// rd  = result_data[63:0]
// rs1 = input_data[63:0]
#define __fcvt_dh(input_data)\
	({uint64_t res; __asm__ volatile("fcvt.dh %[rd], %[r1]\n" : [rd] "=r"(res) : [r1] "r"(input_data)); res;})


// [fcvt.hd] rd rs1
// rd  = result_data[63:0]
// rs1 = input_data[63:0]
#define __fcvt_hd(input_data)\
	({uint64_t res; __asm__ volatile("fcvt.hd %[rd], %[r1]\n" : [rd] "=r"(res) : [r1] "r"(input_data)); res;})



//=================================
// Activation
//=================================
// [prelu] rs1 rs2 rs4
// rs1 = vector_size[23:0]
// rs2 = slop_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __prelu(uint32_t vector_size, uint16_t slop_value, uint16_t r2_sel);


// [gelu] rs1
// rs1 = vector_size[23:0]
#define __gelu(vector_size)\
	__asm__ volatile("gelu %[r1]\n" : : [r1] "r"(vector_size));


// [tanh] rs1
// rs1 = vector_size[23:0]
#define __tanh(vector_size)\
	__asm__ volatile("tanh %[r1]\n" : : [r1] "r"(vector_size));


// [sigm] rs1
// rs1 = vector_size[23:0]
#define __sigm(vector_size)\
	__asm__ volatile("sigm %[r1]\n" : : [r1] "r"(vector_size));


// [prelu.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = slop_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __prelu_i(uint8_t src_svr_addr_A, uint16_t slop_value, uint8_t result_svr_addr, uint16_t r2_sel);


// [gelu.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __gelu_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [tanh.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __tanh_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);


// [sigm.i] rs1 rs3
// rs1 = src_svr_addr_A[4:0]
// rs3 = result_svr_addr[4:0]
__attribute__((noinline)) void __sigm_i(uint8_t src_svr_addr_A, uint8_t result_svr_addr);



//=================================
// Softmax
//=================================
// [esum] rs1 rs2 rs3 rs4
// rs1 = vertor_size[23:0]
// rs2 = accumulated_data[31:16], max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __esum(uint32_t vector_size, uint16_t max_value, uint16_t accumulated_data, uint8_t result_svr_addr, uint16_t r2_sel);


// [softmax] rs1 rs2 rs4
// rs1 = vertor_size[23:0]
// rs2 = esum_value[31:16], max_value[15:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __softmax(uint32_t vector_size, uint16_t max_value, uint16_t esum_value, uint16_t r2_sel);


// [esum.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = accumulated_data[31:16], max_value[15:0]
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __esum_i(uint8_t src_svr_addr_A, uint16_t max_value, uint16_t accumulated_data, uint8_t result_svr_addr, uint16_t r2_sel);


// [softmax.i] rs1 rs2 rs3 rs4
// rs1 = src_svr_addr_A[4:0]
// rs2 = esum_value[31:16], max_value[15:0] 
// rs3 = result_svr_addr[4:0]
// rs4 = r2_sel[8:0]
__attribute__((noinline)) void __softmax_i(uint8_t src_svr_addr_A, uint16_t max_value, uint16_t esum_value, uint8_t result_svr_addr, uint16_t r2_sel);



//=================================
// Pooling
//=================================
// [pool.m] rs1 rs2
// rs1 = col_OUT[63:48], row_OUT[47:32], col_IN[31:16], row_IN[15:0]
// rs2 = col_stride[31:24], row_stride[23:16], col_K[15:8], row_K[7:0]
__attribute__((noinline)) void __pool_m(uint16_t row_IN, uint16_t col_IN, uint16_t row_OUT   , uint16_t col_OUT   , 
 										uint8_t  row_K , uint8_t  col_K , uint8_t  row_stride, uint8_t  col_stride);


// [pool.a] rs1, rs2
// rs1 = col_OUT[63:48], row_OUT[47:32], col_IN[31:16], row_IN[15:0]
// rs2 = k_value[47:32]. col_stride[31:24], row_stride[23:16], col_K[15:8], row_K[7:0]
__attribute__((noinline)) void __pool_a(uint16_t row_IN, uint16_t col_IN, uint16_t row_OUT   , uint16_t col_OUT   , 
 										uint8_t  row_K , uint8_t  col_K , uint8_t  row_stride, uint8_t  col_stride, uint16_t k_value);



//=================================
// Memory operation
//=================================
// [tpose] rs1 rs2 rs3
// rs1 = dtype[56], rw_dir[49:48], src_addr[36:0]
// rs2 = dim2_size[63:48], dim1_size[47:32], dim0_size[31:16], dim2[9:8], dim1[5:4], dim0[1:0]
// rs3 = dst_addr[36:0]
__attribute__((noinline)) void __tpose(uint64_t src_addr , uint64_t dst_addr , 
 									   uint16_t dim2_size, uint16_t dim1_size, uint16_t dim0_size, uint8_t dim2, uint8_t dim1, uint8_t dim0, 
									   uint8_t  rw_dir   , uint8_t  dtype    );


// [fill] rs1 rs2 rs3
// rs1 = dir[48], dst_addr[36:0]
// rs2 = height[63:48], length[47:32], write_stride[31:0]
// rs3 = fill_pattern[63:0]
__attribute__((noinline)) void __fill(uint64_t dst_addr, uint32_t write_stride, uint16_t length, uint16_t height, uint64_t fill_pattern, uint8_t dir);



//=================================
// DMA
//=================================
// [load] rs1 rs2 rs3
// rs1 = src_addr[63:27], dst_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = write_stride[31:0]
__attribute__((noinline)) void __load(uint64_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride);


// [store] rs1 rs2 rs3
// rs1 = dst_addr[63:27], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], write_stride[31:0]
// rs3 = read_stride[31:0]
__attribute__((noinline)) void __store(uint32_t src_addr, uint64_t dst_addr, uint32_t write_stride, uint16_t length, uint16_t height, uint32_t read_stride);


// [copy] rs1 rs2 rs3
// rs1 = dst_addr[58:32], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = write_stride[31:0]
__attribute__((noinline)) void __copy(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride);


// [load.svr] rs1 rs2
// rs1 = addr_L1SPM[23:0]
// rs2 = svr_addr[4:0]
#define __load_svr(addr_L1SPM, svr_addr)\
	__asm__ volatile("load.svr %[r1], %[r2]\n" : : [r1] "r"(addr_L1SPM), [r2] "r"(svr_addr));


// [store.svr] rs1 rs2
// rs1 = addr_L1SPM[23:0]
// rs2 = svr_addr[4:0]
#define __store_svr(addr_L1SPM, svr_addr)\
	__asm__ volatile("store.svr %[r1], %[r2]\n" : : [r1] "r"(addr_L1SPM), [r2] "r"(svr_addr));


// [mcast.s2l] rs1 rs2 rs3
// rs1 = src_addr[58:32], dst_addr[23:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_spu[63:0]
__attribute__((noinline)) void __mcast_s2l(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint64_t target_spu);


// [mcast.g2s] rs1 rs2 rs3
// rs1 = src_addr[63:27], dst_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_nest[63:0]
__attribute__((noinline)) void __mcast_g2s(uint64_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint64_t target_nest);


// [mcast.s2s] rs1 rs2 rs3
// rs1 = target_nest_sel[63], src_nest_id[61:56], dst_addr[55:27], src_addr[26:0]
// rs2 = heigth[63:48], length[47:32], read_stride[31:0]
// rs3 = target_nest[63:32], write_stride[31:0]
__attribute__((noinline)) void __mcast_s2s(uint32_t src_addr, uint32_t dst_addr, uint32_t read_stride, uint16_t length, uint16_t height, uint32_t write_stride, 
										   uint8_t src_nest_id, uint8_t target_nest_sel, uint32_t target_nest);


// [copy.mem] rs1 rs2 rs3
// rs1 = write_stride_L[63:48], src_addr_DDR[36:0]
// rs3 = heigth[63:48], length[47:32], read_stride[31:0]
// rs2 = write_stride_H[63:48], dst_addr_DDR[36:0]
__attribute__((noinline)) void __copy_mem(uint64_t src_addr_DDR, uint64_t dst_addr_DDR, uint32_t read_stride, uint16_t length, uint16_t height, uint16_t write_stride_L, uint16_t write_stride_H);



//=================================
// SPR
//=================================
// [rdspr] rd rs1
// rd  = spr_data[63:0]
// rs1 = nest_id[29:24], spu_id[21:16], spr_addr[11:0]
__attribute__((noinline)) uint64_t __rdspr(uint16_t spr_addr, uint8_t nest_id, uint8_t spu_id);


// [wrspr] rs1 rs2 rs3 rs4
// rs1 = wrstb_n[23:16], spr_addr[11:0]
// rs2 = spr_data[63:0]
// rs3 = target_mask[63:0]
__attribute__((noinline)) void __wrspr(uint16_t spr_addr, uint8_t wrstb_n, uint64_t spr_data, uint64_t target_mask);


// [opset] rs1 rs2
// rs1 = target[0]
// rs2 = data[63:0]
#define __opset(target, data)\
	__asm__ volatile ("opset %[r1], %[r2]\n" : : [r1] "r"(target), [r2] "r"(data));


// [cpsvr] rs1 rs2
// rs1 = svr_addr[4:0]
// rs2 = byte_size[1:0]
#define __cpsvr(svr_addr, byte_size)\
	__asm__ volatile ("cpsvr %[r1], %[r2]\n" : : [r1] "r"(svr_addr), [r2] "r"(byte_size));


// [mvsvr] rs1 rs2 rs3
// rs1 = src_svr_addr[4:0]
// rs2 = dst_svr_addr[4:0]
// rs3 = wrstrb_n[31:0]
__attribute__((noinline)) void __mvsvr(uint8_t src_svr_addr, uint8_t dst_svr_addr, uint32_t wrstrb_n);



//=================================
// Credit
//=================================
// [credit.ld] rs1 rs2
// rs1 = target_spu[63:0]
// rs2 = target_nest[63:0]
#define __credit_ld(target_spu, target_nest)\
	__asm__ volatile ("credit.ld %[r1], %[r2]\n" : : [r1] "r"(target_spu), [r2] "r"(target_nest));


// [credit.st] rs1
// rs1 = target_spu[63:0]
#define __credit_st(target_spu)\
	__asm__ volatile ("credit.st %[r1]\n" : : [r1] "r"(target_spu));


// [credit.chk] rs1
// rs1 = target_spu[63:0]
#define __credit_chk(target_spu) \
	do { uint64_t __tmp = (uint64_t)(target_spu); \
	__asm__ volatile ("credit.chk %[r1]\n" : : [r1] "r"(__tmp)); } while(0)



//=================================
// Micro Code
//=================================
// [mexec] rd rs1
// rd  = status[0]
// rs1 = start_addr[36:0]
// rs2 = target_nest[63:0]
#define __mexec(start_addr, target_nest)\
	({uint64_t res; __asm__ volatile("mexec %[rd], %[r1], %[r2]\n" : [rd] "=r"(res) : [r1] "r"(start_addr), [r2] "r"(target_nest)); res;})


// [mbar]
#define __mbar()\
	__asm__ volatile("mbar\n" : : )


// [msync]
#define __msync()\
	__asm__ volatile("msync\n" : : )


// [eom]
#define __eom()\
	__asm__ volatile("eom\n" : : )



//=================================
// Sync
//=================================
// [bar] rd
// rd  = status[0]
#define __bar()\
	({uint64_t res; __asm__ volatile("bar %[rd]\n" : [rd] "=r"(res) : ); res;})


// [wait] rd rs1
// rd  = status[0]
// rs1 = wait_clk_count[31:0]
#define __wait(wait_clk_count)\
	({uint64_t res; __asm__ volatile("wait %[rd], %[r1]\n" : [rd] "=r"(res) : [r1] "r"(wait_clk_count)); res;})


// [intr] rs1
// rs1 = intr[63:0]
#define __intr(intr)\
	__asm__ volatile("intr %[r1]\n" : : [r1] "r"(intr))
	

// [flush]
#define __flush()\
	__asm__ volatile("flush\n" : : )
	

// [halt]
#define __halt()\
	__asm__ volatile("halt\n" : : )
	


//=================================
// Warp operation
//=================================
// [start.t] rs1 or imm12 
#define __start_t(spu_id)\
	__asm__ volatile("start.t %[r1], %[imm]\n" : : [r1] "r"(spu_id), [imm] "i"(0x000))
	

#define __start_ti(spu_id)\
	__asm__ volatile("start.t %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(spu_id | 0x400))
	

// [end.t] rs1 or imm12
#define __end_t(spu_id)\
	__asm__ volatile("end.t %[r1], %[imm]\n" : : [r1] "r"(spu_id), [imm] "i"(0x000))
	

#define __end_ti(spu_id)\
	__asm__ volatile("end.t %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(spu_id | 0x400))
	

// [start.s]
#define __start_s()\
	__asm__ volatile("start.s\n" : : )
	

// [end.s]
#define __end_s()\
	__asm__ volatile("end.s\n" : : )
	

// [start.p] rs1 or imm12 
#define __start_p(nest_id)\
	do { \
		__asm__ volatile("start.p %[r1], %[imm]\n" : : [r1] "r"(nest_id), [imm] "i"(0x000)); \
		__gtx_giss_unmask_threads(); \
	} while (0)
	

#define __start_pi(nest_id)\
	do { \
		__asm__ volatile("start.p %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(nest_id | 0x400)); \
		__gtx_giss_unmask_threads(); \
	} while (0)
	

// [end.p] rs1 or imm12 
#define __end_p(nest_id)\
	__asm__ volatile("end.p %[r1], %[imm]\n" : : [r1] "r"(nest_id), [imm] "i"(0x000))
	

#define __end_pi(nest_id)\
	__asm__ volatile("end.p %[r1], %[imm]\n" : : [r1] "r"(0x0), [imm] "i"(nest_id | 0x400))
	

// [split]/[join]
#if defined(GGML_GTX_KERNEL_UART_CHECKPOINTS) && GGML_GTX_KERNEL_UART_CHECKPOINTS
void ggml_gtx_kernel_uart_checkpoint(char phase, uint32_t source_line);

#define __split() \
	do { \
		ggml_gtx_kernel_uart_checkpoint('S', (uint32_t) __LINE__); \
		__asm__ volatile("split\n" : : ); \
	} while (0)

#define __join() \
	do { \
		__asm__ volatile("join\n" : : ); \
		ggml_gtx_kernel_uart_checkpoint('J', (uint32_t) __LINE__); \
	} while (0)
#else
#define __split()\
	__asm__ volatile("split\n" : : )

#define __join()\
	__asm__ volatile("join\n" : :)
#endif

#ifdef __cplusplus
}
#endif

#endif
