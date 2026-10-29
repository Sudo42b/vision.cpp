// ==============================================================================
// GTX compatibility header — workaround for assembler limitations
//
// The standard assembler doesn't recognize exp.v and ln.v with 2-operand form.
// This header redefines them using .insn r encoding.
//
// Include this AFTER intrin.h to override the macros.
// ==============================================================================

#ifndef GTX_COMPAT_H
#define GTX_COMPAT_H

// exp.v: funct7=0x1C, funct3=1, opcode=0x0B
// R-type: .insn r opcode, func3, func7, rd, rs1, rs2
#ifdef __exp_v
#undef __exp_v
#endif
#define __exp_v(vector_size, mode) \
    do { \
        register unsigned long _vs __asm__("a0") = (unsigned long)(vector_size); \
        register unsigned long _md __asm__("a1") = (unsigned long)(mode); \
        __asm__ volatile( \
            ".insn r 0x0b, 1, 0x1c, x0, %[r1], %[r2]\n" \
            : : [r1] "r"(_vs), [r2] "r"(_md) \
        ); \
    } while(0)

// ln.v: funct7=0x1C, funct3=2, opcode=0x0B
#ifdef __ln_v
#undef __ln_v
#endif
#define __ln_v(vector_size, mode) \
    do { \
        register unsigned long _vs __asm__("a0") = (unsigned long)(vector_size); \
        register unsigned long _md __asm__("a1") = (unsigned long)(mode); \
        __asm__ volatile( \
            ".insn r 0x0b, 2, 0x1c, x0, %[r1], %[r2]\n" \
            : : [r1] "r"(_vs), [r2] "r"(_md) \
        ); \
    } while(0)

#endif // GTX_COMPAT_H
