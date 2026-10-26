//==================================================================
// GGML GTX kernel common utilities
//==================================================================

#include "ggml-gtx/ggml-gtx-kernel-common.h"

#if defined(GGML_GTX_KERNEL_UART_CHECKPOINTS) && GGML_GTX_KERNEL_UART_CHECKPOINTS
#ifndef GGML_GTX_KERNEL_UART_BASE
#define GGML_GTX_KERNEL_UART_BASE  0x10020000ull
#endif
#define GTX_KERNEL_UART_BASE       ((uint64_t) GGML_GTX_KERNEL_UART_BASE)
#define GTX_KERNEL_UART_TXR_OFF    0x00ull
#define GTX_KERNEL_UART_SR_OFF     0x24ull
#define GTX_KERNEL_UART_TX_FULL    (1u << 1)
#define GTX_KERNEL_UART_TIMEOUT    1000000u

static volatile uint32_t * ggml_gtx_kernel_uart_reg(uint64_t offset) {
    return (volatile uint32_t *)(uintptr_t)(GTX_KERNEL_UART_BASE + offset);
}

static void ggml_gtx_kernel_uart_putc(char value) {
    uint32_t timeout = GTX_KERNEL_UART_TIMEOUT;
    while ((*ggml_gtx_kernel_uart_reg(GTX_KERNEL_UART_SR_OFF) &
            GTX_KERNEL_UART_TX_FULL) != 0u &&
           timeout-- != 0u) {
        __asm__ __volatile__("nop");
    }
    if (timeout != 0u) {
        *ggml_gtx_kernel_uart_reg(GTX_KERNEL_UART_TXR_OFF) = (uint8_t) value;
    }
}

static void ggml_gtx_kernel_uart_puts(const char * text) {
    while (text != 0 && *text != '\0') {
        if (*text == '\n') {
            ggml_gtx_kernel_uart_putc('\r');
        }
        ggml_gtx_kernel_uart_putc(*text++);
    }
}

void ggml_gtx_kernel_uart_checkpoint(char phase, uint32_t source_line) {
    static const char hex[] = "0123456789abcdef";
    ggml_gtx_kernel_uart_puts("[GTX-KERNEL] ");
    ggml_gtx_kernel_uart_putc(phase);
    ggml_gtx_kernel_uart_puts("@0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        ggml_gtx_kernel_uart_putc(hex[(source_line >> shift) & 0xfu]);
    }
    ggml_gtx_kernel_uart_putc('\n');
}

#endif

uint16_t ggml_gtx_fp32_bits_to_fp16(uint32_t f32) {
    // Software conversion (F32 bits -> FP16)
    // Takes raw F32 IEEE-754 bits as uint32_t, returns FP16 bits.
    // NOTE: __fcvt_sh returns 0 on ISS v1.1.4.
    // NOTE: ISS has float<->int store-forwarding issues, so callers
    //       should read F32 data as uint32_t directly from DDR.
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f32 >> 13) & 0x3FF;

    if (exp <= 0) {
        return sign;  // Underflow to zero
    } else if (exp >= 31) {
        return sign | 0x7C00;  // Overflow to infinity
    }
    return sign | ((uint16_t)exp << 10) | frac;
}

uint32_t ggml_gtx_fp16_to_fp32_bits(uint16_t value) {
    // Software conversion (FP16 -> F32 bits)
    // Takes FP16 bits, returns raw F32 IEEE-754 bits as uint32_t.
    // NOTE: __fcvt_hs returns 0 on ISS v1.1.4.
    // NOTE: ISS has float<->int store-forwarding issues, so callers
    //       should write F32 data as uint32_t directly to DDR.
    uint16_t sign = (value >> 15) & 1;
    int32_t exp = (value >> 10) & 0x1F;
    uint32_t frac = value & 0x3FF;

    uint32_t f32;
    if (exp == 0) {
        f32 = sign << 31;  // Zero
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (frac << 13);  // Inf/NaN
    } else {
        f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }

    return f32;
}

// Legacy wrappers (ggml_gtx_fp32_to_fp16 / ggml_gtx_fp16_to_fp32) are
// defined in ggml-gtx-port-riscv.c. Not duplicated here to avoid
// multiple-definition linker errors.

//==================================================================
// Software F32 Arithmetic (uint32_t IEEE-754 bit manipulation)
//
// ISS v1.1.4 cannot convert float<->uint32_t (fmv.x.w returns 0).
// All F32 math is done on raw IEEE-754 bits as uint32_t.
// These are NOT speed-optimized; they handle the scalar pre-compute
// before __split() and the F32 bias accumulation in add_rel_pos.
//==================================================================

// Well-known F32 bit constants
#define F32B_ZERO       0x00000000u  // 0.0f
#define F32B_ONE        0x3F800000u  // 1.0f
#define F32B_NEG_ONE    0xBF800000u  // -1.0f
#define F32B_TWO        0x40000000u  // 2.0f
#define F32B_HALF       0x3F000000u  // 0.5f
#define F32B_LN2        0x3F317218u  // ln(2) = 0.6931472
#define F32B_LOG2E      0x3FB8AA3Bu  // log2(e) = 1.442695
#define F32B_INF        0x7F800000u  // +infinity
#define F32B_SIGN_MASK  0x80000000u

static inline int32_t f32b_is_zero(uint32_t a) {
    return (a & ~F32B_SIGN_MASK) == 0;
}

static inline int32_t f32b_is_neg(uint32_t a) {
    return (a >> 31) & 1;
}

uint32_t ggml_gtx_f32b_neg(uint32_t a) {
    return a ^ F32B_SIGN_MASK;
}

uint32_t ggml_gtx_f32b_abs(uint32_t a) {
    return a & ~F32B_SIGN_MASK;
}

// Compare: returns 1 if a > b (as F32), 0 otherwise.
int32_t ggml_gtx_f32b_gt(uint32_t a, uint32_t b) {
    int32_t sa = f32b_is_neg(a);
    int32_t sb = f32b_is_neg(b);
    if (sa != sb) {
        // Different signs: negative < positive (unless both zero)
        if (f32b_is_zero(a) && f32b_is_zero(b)) return 0;
        return sb;  // a>b iff b is negative
    }
    // Same sign
    if (sa) {
        // Both negative: larger magnitude = smaller value
        return (a & ~F32B_SIGN_MASK) < (b & ~F32B_SIGN_MASK);
    } else {
        return (a & ~F32B_SIGN_MASK) > (b & ~F32B_SIGN_MASK);
    }
}

// Integer to F32 bits
uint32_t ggml_gtx_f32b_from_int(int32_t n) {
    if (n == 0) return F32B_ZERO;
    uint32_t sign = 0;
    uint32_t abs_n;
    if (n < 0) {
        sign = F32B_SIGN_MASK;
        abs_n = (uint32_t)(-n);
    } else {
        abs_n = (uint32_t)n;
    }
    // Find highest bit position
    int32_t shift = 0;
    uint32_t tmp = abs_n;
    while (tmp > 1) { tmp >>= 1; shift++; }
    // exp = 127 + shift, mantissa = (abs_n << (23 - shift)) & 0x7FFFFF
    uint32_t exp = (uint32_t)(127 + shift);
    uint32_t mant;
    if (shift >= 23) {
        mant = (abs_n >> (shift - 23)) & 0x7FFFFFu;
    } else {
        mant = (abs_n << (23 - shift)) & 0x7FFFFFu;
    }
    return sign | (exp << 23) | mant;
}

// F32 bits to integer (truncate toward zero)
int32_t ggml_gtx_f32b_to_int(uint32_t a) {
    if (f32b_is_zero(a)) return 0;
    int32_t exp = ((a >> 23) & 0xFF) - 127;
    if (exp < 0) return 0;
    uint32_t mant = (1u << 23) | (a & 0x7FFFFFu);  // implicit 1
    int32_t result;
    if (exp >= 23) {
        result = (int32_t)(mant << (exp - 23));
    } else {
        result = (int32_t)(mant >> (23 - exp));
    }
    return f32b_is_neg(a) ? -result : result;
}

// F32 addition: a + b  (raw IEEE-754 bits)
uint32_t ggml_gtx_f32b_add(uint32_t a, uint32_t b) {
    if (f32b_is_zero(a)) return b;
    if (f32b_is_zero(b)) return a;

    uint32_t sign_a = a & F32B_SIGN_MASK;
    uint32_t sign_b = b & F32B_SIGN_MASK;
    int32_t exp_a = (a >> 23) & 0xFF;
    int32_t exp_b = (b >> 23) & 0xFF;
    // Mantissa with implicit 1 bit, shifted left by 2 for guard bits
    uint32_t mant_a = ((1u << 23) | (a & 0x7FFFFFu)) << 2;
    uint32_t mant_b = ((1u << 23) | (b & 0x7FFFFFu)) << 2;

    // Align exponents
    if (exp_a > exp_b) {
        int32_t shift = exp_a - exp_b;
        if (shift >= 26) return a;  // b too small
        mant_b >>= shift;
    } else if (exp_b > exp_a) {
        int32_t shift = exp_b - exp_a;
        if (shift >= 26) return b;  // a too small
        mant_a >>= shift;
        exp_a = exp_b;
        sign_a = sign_b;  // will be corrected below
    }

    uint32_t result_mant;
    uint32_t result_sign;
    if (sign_a == sign_b) {
        // Same sign: add mantissas
        result_mant = mant_a + mant_b;
        result_sign = sign_a;
    } else {
        // Different signs: subtract mantissas
        uint32_t abs_a = (a & ~F32B_SIGN_MASK);
        uint32_t abs_b = (b & ~F32B_SIGN_MASK);
        if (mant_a >= mant_b) {
            result_mant = mant_a - mant_b;
            // sign follows the larger magnitude
            result_sign = (abs_a >= abs_b) ? (a & F32B_SIGN_MASK)
                                           : (b & F32B_SIGN_MASK);
        } else {
            result_mant = mant_b - mant_a;
            result_sign = (abs_b >= abs_a) ? (b & F32B_SIGN_MASK)
                                           : (a & F32B_SIGN_MASK);
        }
    }

    if (result_mant == 0) return F32B_ZERO;

    // Normalize
    while (result_mant >= (1u << 26)) {
        result_mant >>= 1;
        exp_a++;
    }
    while (result_mant < (1u << 25) && exp_a > 0) {
        result_mant <<= 1;
        exp_a--;
    }

    if (exp_a >= 255) return result_sign | F32B_INF;
    if (exp_a <= 0) return result_sign;  // underflow

    // Remove guard bits and implicit 1
    uint32_t frac = (result_mant >> 2) & 0x7FFFFFu;
    return result_sign | ((uint32_t)exp_a << 23) | frac;
}

// F32 subtraction: a - b
uint32_t ggml_gtx_f32b_sub(uint32_t a, uint32_t b) {
    return ggml_gtx_f32b_add(a, ggml_gtx_f32b_neg(b));
}

// F32 multiplication: a * b
uint32_t ggml_gtx_f32b_mul(uint32_t a, uint32_t b) {
    if (f32b_is_zero(a) || f32b_is_zero(b)) return F32B_ZERO;

    uint32_t result_sign = (a ^ b) & F32B_SIGN_MASK;
    int32_t exp_a = (a >> 23) & 0xFF;
    int32_t exp_b = (b >> 23) & 0xFF;
    int32_t exp_r = exp_a + exp_b - 127;

    uint32_t mant_a = (1u << 23) | (a & 0x7FFFFFu);
    uint32_t mant_b = (1u << 23) | (b & 0x7FFFFFu);

    // 24x24 → 48 bit product using 16-bit chunks to avoid 64-bit overflow
    uint32_t a_hi = mant_a >> 12;
    uint32_t a_lo = mant_a & 0xFFF;
    uint32_t b_hi = mant_b >> 12;
    uint32_t b_lo = mant_b & 0xFFF;

    uint32_t prod_hi = a_hi * b_hi;           // bits [47:24] approx
    uint32_t prod_mid = a_hi * b_lo + a_lo * b_hi;  // bits [35:12]
    // prod_lo = a_lo * b_lo;  // bits [23:0] — skip, only need top 24 bits

    // Combine: product ≈ (prod_hi << 24) + (prod_mid << 12) + prod_lo
    // We need bits [47:24] of full product
    uint32_t result_mant = prod_hi + (prod_mid >> 12);

    // Normalize: result_mant should have bit 23 as implicit 1
    // prod_hi is (12-bit * 12-bit) so up to 24 bits
    if (result_mant >= (1u << 24)) {
        result_mant >>= 1;
        exp_r++;
    }
    // Shouldn't need to shift left in normal cases

    if (exp_r >= 255) return result_sign | F32B_INF;
    if (exp_r <= 0) return result_sign;  // underflow

    uint32_t frac = result_mant & 0x7FFFFFu;
    return result_sign | ((uint32_t)exp_r << 23) | frac;
}

// F32 division: a / b
uint32_t ggml_gtx_f32b_div(uint32_t a, uint32_t b) {
    if (f32b_is_zero(a)) return F32B_ZERO;
    if (f32b_is_zero(b)) return (a & F32B_SIGN_MASK) | F32B_INF;

    uint32_t result_sign = (a ^ b) & F32B_SIGN_MASK;
    int32_t exp_a = (a >> 23) & 0xFF;
    int32_t exp_b = (b >> 23) & 0xFF;
    int32_t exp_r = exp_a - exp_b + 127;

    uint32_t mant_a = (1u << 23) | (a & 0x7FFFFFu);
    uint32_t mant_b = (1u << 23) | (b & 0x7FFFFFu);

    // Long division: shift mant_a left to get more precision
    // result = (mant_a << 23) / mant_b
    uint64_t num = (uint64_t)mant_a << 23;
    uint32_t result_mant = (uint32_t)(num / mant_b);

    // Normalize
    while (result_mant >= (1u << 24)) {
        result_mant >>= 1;
        exp_r++;
    }
    while (result_mant < (1u << 23) && exp_r > 0) {
        result_mant <<= 1;
        exp_r--;
    }

    if (exp_r >= 255) return result_sign | F32B_INF;
    if (exp_r <= 0) return result_sign;

    uint32_t frac = result_mant & 0x7FFFFFu;
    return result_sign | ((uint32_t)exp_r << 23) | frac;
}

// F32 exp(x) — range reduction + polynomial
// exp(x) = 2^(x * log2(e)) = 2^n * 2^f where n=int, f=frac
uint32_t ggml_gtx_f32b_exp(uint32_t x) {
    // exp(0) = 1
    if (f32b_is_zero(x)) return F32B_ONE;

    // Clamp to avoid overflow/underflow
    // exp(88.7) > FLT_MAX, exp(-87.3) < FLT_MIN
    // F32 bits for 88.0 ≈ 0x42B00000, -88.0 ≈ 0xC2B00000
    uint32_t abs_x = ggml_gtx_f32b_abs(x);
    if (ggml_gtx_f32b_gt(abs_x, 0x42B00000u)) {
        return f32b_is_neg(x) ? F32B_ZERO : F32B_INF;
    }

    // x * log2(e) → get integer and fractional parts
    uint32_t xlog2e = ggml_gtx_f32b_mul(x, F32B_LOG2E);

    // Floor to integer
    int32_t n = ggml_gtx_f32b_to_int(xlog2e);
    // Adjust floor for negative
    uint32_t n_f32 = ggml_gtx_f32b_from_int(n);
    if (ggml_gtx_f32b_gt(n_f32, xlog2e)) {
        n--;
        n_f32 = ggml_gtx_f32b_from_int(n);
    }

    // f = x*log2(e) - n  → in [0, 1)
    uint32_t f = ggml_gtx_f32b_sub(xlog2e, n_f32);

    // 2^f via polynomial (minimax on [0,1)):
    // 2^f ≈ 1 + 0.6931472*f + 0.2402265*f^2 + 0.0555041*f^3 + 0.00961813*f^4
    // Coefficients as F32 bits:
    // c1 = 0.6931472  → 0x3F317218
    // c2 = 0.2402265  → 0x3E75FDF0
    // c3 = 0.0555041  → 0x3D635847
    // c4 = 0.00961813 → 0x3C1D9530
    #define EXP_C1 0x3F317218u
    #define EXP_C2 0x3E75FDF0u
    #define EXP_C3 0x3D635847u
    #define EXP_C4 0x3C1D9530u

    // Horner: result = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    uint32_t p = ggml_gtx_f32b_mul(f, EXP_C4);
    p = ggml_gtx_f32b_add(p, EXP_C3);
    p = ggml_gtx_f32b_mul(p, f);
    p = ggml_gtx_f32b_add(p, EXP_C2);
    p = ggml_gtx_f32b_mul(p, f);
    p = ggml_gtx_f32b_add(p, EXP_C1);
    p = ggml_gtx_f32b_mul(p, f);
    p = ggml_gtx_f32b_add(p, F32B_ONE);

    // Multiply by 2^n: adjust exponent by n
    int32_t exp_p = (p >> 23) & 0xFF;
    exp_p += n;
    if (exp_p >= 255) return F32B_INF;
    if (exp_p <= 0) return F32B_ZERO;
    return (p & 0x807FFFFFu) | ((uint32_t)exp_p << 23);

    #undef EXP_C1
    #undef EXP_C2
    #undef EXP_C3
    #undef EXP_C4
}

// F32 log(x) — natural logarithm
// ln(x) = (exp_bits - 127) * ln(2) + ln(1 + f)
// where f = mantissa / 2^23
uint32_t ggml_gtx_f32b_log(uint32_t x) {
    // log(0) = -inf, log(neg) = NaN
    if (f32b_is_zero(x)) return F32B_SIGN_MASK | F32B_INF;
    if (f32b_is_neg(x)) return 0x7FC00000u;  // NaN
    if (x == F32B_ONE) return F32B_ZERO;

    int32_t exp_val = ((x >> 23) & 0xFF) - 127;
    // f = mantissa as fraction in [1.0, 2.0)
    // Reduce to [1, 2): set exponent to 127
    uint32_t f_bits = (x & 0x7FFFFFu) | 0x3F800000u;  // 1.0 + frac

    // f_bits is now F32 for value in [1.0, 2.0)
    // ln(f) ≈ polynomial on t = f - 1, t in [0, 1)
    uint32_t t = ggml_gtx_f32b_sub(f_bits, F32B_ONE);

    // ln(1+t) ≈ t - t^2/2 + t^3/3 - t^4/4 + t^5/5  (Taylor series)
    // For better accuracy: Padé or minimax
    // Using: ln(1+t) ≈ t*(1 + t*(-0.5 + t*(0.3333 + t*(-0.25 + t*0.2))))
    // Coefficients as F32 bits:
    // -0.5     → 0xBF000000
    // 0.33333  → 0x3EAAAAAB
    // -0.25    → 0xBE800000
    // 0.2      → 0x3E4CCCCDu
    #define LOG_C1 0xBF000000u
    #define LOG_C2 0x3EAAAAAB
    #define LOG_C3 0xBE800000u
    #define LOG_C4 0x3E4CCCCDu

    uint32_t p = ggml_gtx_f32b_mul(t, LOG_C4);
    p = ggml_gtx_f32b_add(p, LOG_C3);
    p = ggml_gtx_f32b_mul(p, t);
    p = ggml_gtx_f32b_add(p, LOG_C2);
    p = ggml_gtx_f32b_mul(p, t);
    p = ggml_gtx_f32b_add(p, LOG_C1);
    p = ggml_gtx_f32b_mul(p, t);
    p = ggml_gtx_f32b_add(p, F32B_ONE);
    uint32_t ln_frac = ggml_gtx_f32b_mul(p, t);

    // ln(x) = exp_val * ln(2) + ln_frac
    uint32_t exp_part = ggml_gtx_f32b_mul(
        ggml_gtx_f32b_from_int(exp_val), F32B_LN2);
    return ggml_gtx_f32b_add(exp_part, ln_frac);

    #undef LOG_C1
    #undef LOG_C2
    #undef LOG_C3
    #undef LOG_C4
}

// F32 softplus: log(1 + exp(x))
// With clamp: x > 20 → x, x < -20 → 0
uint32_t ggml_gtx_f32b_softplus(uint32_t x) {
    // 20.0f = 0x41A00000
    #define F32B_20 0x41A00000u
    if (ggml_gtx_f32b_gt(x, F32B_20)) return x;
    if (ggml_gtx_f32b_gt(ggml_gtx_f32b_neg(x), F32B_20)) return F32B_ZERO;
    uint32_t ex = ggml_gtx_f32b_exp(x);
    uint32_t one_plus = ggml_gtx_f32b_add(F32B_ONE, ex);
    return ggml_gtx_f32b_log(one_plus);
    #undef F32B_20
}

//==================================================================
// Tiling Plan Initialization
//==================================================================

void ggml_gtx_tile_plan_init(
    ggml_gtx_tile_plan_t * plan,
    uint16_t i_row,
    uint16_t i_col,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
    plan->i_row = i_row;
    plan->i_col = i_col;
    plan->unit_per_thread = i_col * GGML_GTX_DTYPE_SIZE;
    plan->max_cal_per_thread = GGML_GTX_L1_BANK_SIZE / plan->unit_per_thread;

    // Calculate effective nest count
    uint32_t cal_num_nest = i_row;
    plan->eff_nest_int = (cal_num_nest < nest_num) ? cal_num_nest : nest_num;
    plan->eff_nest_hex = GGML_GTX_EFF_MASK_64(plan->eff_nest_int);

    // Distribute work across nests
    uint32_t cal_quot_nest = cal_num_nest / plan->eff_nest_int;
    uint32_t cal_mod_nest = cal_num_nest % plan->eff_nest_int;

    uint64_t cumulative_offset = 0;

    for (uint8_t i = 0; i < plan->eff_nest_int; i++) {
        ggml_gtx_nest_tile_t * nest = &plan->nests[i];

        // Work per nest
        nest->cal_num = (i < cal_mod_nest) ? cal_quot_nest + 1 : cal_quot_nest;

        // Effective threads for this nest
        nest->eff_thread_int = (nest->cal_num < spu_num_per_nest) ?
                               nest->cal_num : spu_num_per_nest;
        nest->eff_thread_hex = GGML_GTX_EFF_MASK_16(nest->eff_thread_int);

        // Distribute work across threads
        uint32_t cal_quot_thread = nest->cal_num / nest->eff_thread_int;
        uint32_t cal_mod_thread = nest->cal_num % nest->eff_thread_int;

        for (uint8_t j = 0; j < nest->eff_thread_int; j++) {
            ggml_gtx_thread_tile_t * thread = &nest->threads[j];

            thread->cal_num = (j < cal_mod_thread) ? cal_quot_thread + 1 : cal_quot_thread;
            thread->iteration = thread->cal_num / plan->max_cal_per_thread;
            thread->remain = thread->cal_num % plan->max_cal_per_thread;
            thread->offset = cumulative_offset;

            cumulative_offset += thread->cal_num * plan->unit_per_thread;
        }
    }
}

void ggml_gtx_matmul_tile_init(
    ggml_gtx_matmul_tile_t * plan,
    uint16_t a_row,
    uint16_t h_col,
    uint16_t b_row,
    uint16_t h_col_origin,
    uint16_t res_col_origin,
    uint16_t tile_a_row,
    uint16_t tile_h_col,
    uint16_t tile_b_row,
    uint8_t  nest_num,
    uint8_t  spu_num_per_nest)
{
    plan->a_row = a_row;
    plan->h_col = h_col;
    plan->b_row = b_row;
    plan->h_col_origin = h_col_origin;
    plan->res_col_origin = res_col_origin;
    plan->tile_a_row = tile_a_row;
    plan->tile_h_col = tile_h_col;
    plan->tile_b_row = tile_b_row;

    // Calculate tile counts
    plan->num_tile_row_a = GGML_GTX_CEIL_DIV(a_row, tile_a_row);
    plan->num_tile_col_h = GGML_GTX_CEIL_DIV(h_col, tile_h_col);
    plan->num_tile_row_b = GGML_GTX_CEIL_DIV(b_row, tile_b_row);

    // Last tile sizes
    plan->last_row_a = GGML_GTX_LAST_TILE(a_row, tile_a_row);
    plan->last_col_h = GGML_GTX_LAST_TILE(h_col, tile_h_col);
    plan->last_row_b = GGML_GTX_LAST_TILE(b_row, tile_b_row);

    // Effective hardware (based on B tiling for nest distribution)
    plan->eff_nest_int = (plan->num_tile_row_b < nest_num) ?
                         plan->num_tile_row_b : nest_num;
    plan->eff_nest_hex = GGML_GTX_EFF_MASK_64(plan->eff_nest_int);

    // Distribute A rows across spreads (nests act as spreads for A tiling)
    for (uint8_t spread_idx = 0; spread_idx < plan->eff_nest_int; spread_idx++) {
        uint16_t row_a_quot = plan->num_tile_row_a / plan->eff_nest_int;
        uint16_t row_a_mod = plan->num_tile_row_a % plan->eff_nest_int;

        plan->num_row_a_per_spread[spread_idx] = (spread_idx < row_a_mod) ?
                                                  row_a_quot + 1 : row_a_quot;

        plan->eff_thread_int[spread_idx] = (plan->num_row_a_per_spread[spread_idx] < spu_num_per_nest) ?
                                            plan->num_row_a_per_spread[spread_idx] : spu_num_per_nest;
        plan->eff_thread_hex[spread_idx] = GGML_GTX_EFF_MASK_16(plan->eff_thread_int[spread_idx]);

        for (uint8_t thread_id = 0; thread_id < plan->eff_thread_int[spread_idx]; thread_id++) {
            uint16_t row_a_quot_thread = plan->num_row_a_per_spread[spread_idx] / plan->eff_thread_int[spread_idx];
            uint16_t row_a_mod_thread = plan->num_row_a_per_spread[spread_idx] % plan->eff_thread_int[spread_idx];

            plan->num_row_a_per_thread[spread_idx][thread_id] = (thread_id < row_a_mod_thread) ?
                                                                 row_a_quot_thread + 1 : row_a_quot_thread;
        }
    }

    // Distribute B rows across nests
    for (uint8_t nest_id = 0; nest_id < plan->eff_nest_int; nest_id++) {
        uint16_t row_b_quot = plan->num_tile_row_b / plan->eff_nest_int;
        uint16_t row_b_mod = plan->num_tile_row_b % plan->eff_nest_int;

        plan->num_row_b_per_nest[nest_id] = (nest_id < row_b_mod) ?
                                             row_b_quot + 1 : row_b_quot;
    }
}

//==================================================================
// Element-wise Operations (Tensor x Tensor)
//==================================================================
