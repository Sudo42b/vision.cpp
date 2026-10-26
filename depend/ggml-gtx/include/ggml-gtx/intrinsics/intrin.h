//==================================================================
// Copyright   : (C) Supergate - All Rights Reserved
// Project     : GSF / VTS
// Description : GTX intrinsic include header file
// 				 level 1: base instruction intrinsic
// 				 level 2: convenience intrinsic for programming
// 				 level 3: complex operation intrinsic consisting of compound instructions
// Author      : mh.kim ( NPU Div - NPU Core Team )
// Last Update : 2026/01/23
//==================================================================

#ifndef INTRIN_H
#define INTRIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml-gtx/intrinsics/intrin_level1.h"
#include "ggml-gtx/intrinsics/intrin_level2.h"
#include "ggml-gtx/intrinsics/intrin_level3.h"
#include "ggml-gtx/intrinsics/gtx_compat.h"


//=============================================
// .insn register type
//=============================================
// [RISCV R type]
// +-------+-----+-----+-------+----+---------+
// | func7 | rs2 | rs1 | func3 | rd | opcode7 |
// +-------+-----+-----+-------+----+---------+
// 31     25    20    15      12    7         0
// 
// .insn r opcode7, func3, func7, rd, rs1, rs2
//  #define INSN_R_TYPE(opcode7 ,func7, func3, rd, rs1, rs2)
// 	        asm volatile (".insn r 0xB, 0x58, 0, %3, %4, %5\n"
// 					     : "=r" (rd)
// 					     : "i" (opcode7), "i"(func3), "i"(func7), "r"(rs1), "r"(rs2))

#ifdef __cplusplus
}
#endif

#endif