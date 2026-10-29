/// Copyright by Syntacore LLC © 2016, 2017. See LICENSE for details
/// @file       <sc_print.c>
///
/// Printf implementation for baremetal RISC-V
/// Based on Marco Paland's printf implementation (MIT License)

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include "ggml-gtx/sc_print.h"

//==============================================================================
// Configuration
//==============================================================================

// Output port for simulator/hardware
#define SC_SIM_OUTPORT (0x000FFFE0)

// Buffer sizes
#define PRINTF_NTOA_BUFFER_SIZE  32U
#define PRINTF_FTOA_BUFFER_SIZE  32U

// Enable float support
#define PRINTF_SUPPORT_FLOAT

// Enable long long support
#define PRINTF_SUPPORT_LONG_LONG

//==============================================================================
// Internal flags
//==============================================================================

#define FLAGS_ZEROPAD   (1U << 0U)
#define FLAGS_LEFT      (1U << 1U)
#define FLAGS_PLUS      (1U << 2U)
#define FLAGS_SPACE     (1U << 3U)
#define FLAGS_HASH      (1U << 4U)
#define FLAGS_UPPERCASE (1U << 5U)
#define FLAGS_CHAR      (1U << 6U)
#define FLAGS_SHORT     (1U << 7U)
#define FLAGS_LONG      (1U << 8U)
#define FLAGS_LONG_LONG (1U << 9U)
#define FLAGS_PRECISION (1U << 10U)
#define FLAGS_WIDTH     (1U << 11U)

//==============================================================================
// Low-level output
//==============================================================================

//

static void sc_puts(const char *str, size_t len) {
    volatile char *out_ptr = (volatile char *)SC_SIM_OUTPORT;
    for (size_t i = 0; i < len; i++) {
        *out_ptr = str[i];
    }
}

#undef putchar
int putchar(int ch) {
    static char buf[64] __attribute__((aligned(64)));
    static int buflen = 0;

    buf[buflen++] = (char)ch;

    if (ch == '\n' || buflen == (int)sizeof(buf)) {
        sc_puts(buf, buflen);
        buflen = 0;
    }

    return 0;
}

// Output function type
typedef void (*out_fct_type)(char character, void *buffer, size_t idx, size_t maxlen);

// Internal character output
static void _out_char(char character, void *buffer, size_t idx, size_t maxlen) {
    (void)buffer; (void)idx; (void)maxlen;
    if (character) {
        putchar(character);
    }
}

// Internal buffer output
static void _out_buffer(char character, void *buffer, size_t idx, size_t maxlen) {
    if (idx < maxlen) {
        ((char *)buffer)[idx] = character;
    }
}

// Internal null output
static void _out_null(char character, void *buffer, size_t idx, size_t maxlen) {
    (void)character; (void)buffer; (void)idx; (void)maxlen;
}

//==============================================================================
// Helper functions
//==============================================================================

static inline unsigned int _strlen(const char *str) {
    const char *s;
    for (s = str; *s; ++s);
    return (unsigned int)(s - str);
}

static inline int _is_digit(char ch) {
    return (ch >= '0') && (ch <= '9');
}

static inline unsigned int _atoi(const char **str) {
    unsigned int i = 0U;
    while (_is_digit(**str)) {
        i = i * 10U + (unsigned int)(*((*str)++) - '0');
    }
    return i;
}

//==============================================================================
// Number formatting
//==============================================================================

static size_t _ntoa_format(out_fct_type out, char *buffer, size_t idx, size_t maxlen,
                           char *buf, size_t len, int negative, unsigned int base,
                           unsigned int prec, unsigned int width, unsigned int flags) {
    const size_t start_idx = idx;

    // Pad leading zeros
    while (!(flags & FLAGS_LEFT) && (len < prec) && (len < PRINTF_NTOA_BUFFER_SIZE)) {
        buf[len++] = '0';
    }
    while (!(flags & FLAGS_LEFT) && (flags & FLAGS_ZEROPAD) && (len < width) && (len < PRINTF_NTOA_BUFFER_SIZE)) {
        buf[len++] = '0';
    }

    // Handle hash
    if (flags & FLAGS_HASH) {
        if (((len == prec) || (len == width)) && (len > 0U)) {
            len--;
            if ((base == 16U) && (len > 0U)) {
                len--;
            }
        }
        if ((base == 16U) && !(flags & FLAGS_UPPERCASE) && (len < PRINTF_NTOA_BUFFER_SIZE)) {
            buf[len++] = 'x';
        }
        if ((base == 16U) && (flags & FLAGS_UPPERCASE) && (len < PRINTF_NTOA_BUFFER_SIZE)) {
            buf[len++] = 'X';
        }
        if (len < PRINTF_NTOA_BUFFER_SIZE) {
            buf[len++] = '0';
        }
    }

    // Handle sign
    if ((len == width) && (negative || (flags & FLAGS_PLUS) || (flags & FLAGS_SPACE))) {
        len--;
    }
    if (len < PRINTF_NTOA_BUFFER_SIZE) {
        if (negative) {
            buf[len++] = '-';
        } else if (flags & FLAGS_PLUS) {
            buf[len++] = '+';
        } else if (flags & FLAGS_SPACE) {
            buf[len++] = ' ';
        }
    }

    // Pad spaces up to given width
    if (!(flags & FLAGS_LEFT) && !(flags & FLAGS_ZEROPAD)) {
        for (size_t i = len; i < width; i++) {
            out(' ', buffer, idx++, maxlen);
        }
    }

    // Reverse string
    for (size_t i = 0U; i < len; i++) {
        out(buf[len - i - 1U], buffer, idx++, maxlen);
    }

    // Append pad spaces up to given width
    if (flags & FLAGS_LEFT) {
        while (idx - start_idx < width) {
            out(' ', buffer, idx++, maxlen);
        }
    }

    return idx;
}

static size_t _ntoa_long(out_fct_type out, char *buffer, size_t idx, size_t maxlen,
                         unsigned long value, int negative, unsigned long base,
                         unsigned int prec, unsigned int width, unsigned int flags) {
    char buf[PRINTF_NTOA_BUFFER_SIZE];
    size_t len = 0U;

    // Write if precision != 0 and value is != 0
    if (!(flags & FLAGS_PRECISION) || value) {
        do {
            const char digit = (char)(value % base);
            buf[len++] = digit < 10 ? '0' + digit : (flags & FLAGS_UPPERCASE ? 'A' : 'a') + digit - 10;
            value /= base;
        } while (value && (len < PRINTF_NTOA_BUFFER_SIZE));
    }

    return _ntoa_format(out, buffer, idx, maxlen, buf, len, negative, (unsigned int)base, prec, width, flags);
}

#if defined(PRINTF_SUPPORT_LONG_LONG)
static size_t _ntoa_long_long(out_fct_type out, char *buffer, size_t idx, size_t maxlen,
                              unsigned long long value, int negative, unsigned long long base,
                              unsigned int prec, unsigned int width, unsigned int flags) {
    char buf[PRINTF_NTOA_BUFFER_SIZE];
    size_t len = 0U;

    // Write if precision != 0 and value is != 0
    if (!(flags & FLAGS_PRECISION) || value) {
        do {
            const char digit = (char)(value % base);
            buf[len++] = digit < 10 ? '0' + digit : (flags & FLAGS_UPPERCASE ? 'A' : 'a') + digit - 10;
            value /= base;
        } while (value && (len < PRINTF_NTOA_BUFFER_SIZE));
    }

    return _ntoa_format(out, buffer, idx, maxlen, buf, len, negative, (unsigned int)base, prec, width, flags);
}
#endif

#if defined(PRINTF_SUPPORT_FLOAT)
static size_t _ftoa(out_fct_type out, char *buffer, size_t idx, size_t maxlen,
                    double value, unsigned int prec, unsigned int width, unsigned int flags) {
    char buf[PRINTF_FTOA_BUFFER_SIZE];
    size_t len = 0U;
    double diff = 0.0;

    // Powers of 10
    static const double pow10[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

    // Test for negative
    int negative = 0;
    if (value < 0) {
        negative = 1;
        value = 0 - value;
    }

    // Set default precision to 6, if not set explicitly
    if (!(flags & FLAGS_PRECISION)) {
        prec = 6U;
    }
    // Limit precision to 9
    while ((len < PRINTF_FTOA_BUFFER_SIZE) && (prec > 9U)) {
        buf[len++] = '0';
        prec--;
    }

    int whole = (int)value;
    double tmp = (value - whole) * pow10[prec];
    unsigned long frac = (unsigned long)tmp;
    diff = tmp - frac;

    if (diff > 0.5) {
        ++frac;
        if (frac >= (unsigned long)pow10[prec]) {
            frac = 0;
            ++whole;
        }
    } else if ((diff == 0.5) && ((frac == 0U) || (frac & 1U))) {
        ++frac;
    }

    // For very large numbers, just return 0
    if (value > (double)0x7FFFFFFF) {
        return 0U;
    }

    if (prec == 0U) {
        diff = value - (double)whole;
        if (diff > 0.5) {
            ++whole;
        } else if ((diff == 0.5) && (whole & 1)) {
            ++whole;
        }
    } else {
        unsigned int count = prec;
        while (len < PRINTF_FTOA_BUFFER_SIZE) {
            --count;
            buf[len++] = (char)(48U + (frac % 10U));
            if (!(frac /= 10U)) {
                break;
            }
        }
        while ((len < PRINTF_FTOA_BUFFER_SIZE) && (count-- > 0U)) {
            buf[len++] = '0';
        }
        if (len < PRINTF_FTOA_BUFFER_SIZE) {
            buf[len++] = '.';
        }
    }

    // Do whole part
    while (len < PRINTF_FTOA_BUFFER_SIZE) {
        buf[len++] = (char)(48 + (whole % 10));
        if (!(whole /= 10)) {
            break;
        }
    }

    // Pad leading zeros
    while (!(flags & FLAGS_LEFT) && (flags & FLAGS_ZEROPAD) && (len < width) && (len < PRINTF_FTOA_BUFFER_SIZE)) {
        buf[len++] = '0';
    }

    // Handle sign
    if ((len == width) && (negative || (flags & FLAGS_PLUS) || (flags & FLAGS_SPACE))) {
        len--;
    }
    if (len < PRINTF_FTOA_BUFFER_SIZE) {
        if (negative) {
            buf[len++] = '-';
        } else if (flags & FLAGS_PLUS) {
            buf[len++] = '+';
        } else if (flags & FLAGS_SPACE) {
            buf[len++] = ' ';
        }
    }

    // Pad spaces up to given width
    if (!(flags & FLAGS_LEFT) && !(flags & FLAGS_ZEROPAD)) {
        for (size_t i = len; i < width; i++) {
            out(' ', buffer, idx++, maxlen);
        }
    }

    // Reverse string
    for (size_t i = 0U; i < len; i++) {
        out(buf[len - i - 1U], buffer, idx++, maxlen);
    }

    // Append pad spaces up to given width
    if (flags & FLAGS_LEFT) {
        while (idx < width) {
            out(' ', buffer, idx++, maxlen);
        }
    }

    return idx;
}
#endif

//==============================================================================
// Internal vsnprintf
//==============================================================================

static int _vsnprintf(out_fct_type out, char *buffer, const size_t maxlen, const char *format, va_list va) {
    unsigned int flags, width, precision, n;
    size_t idx = 0U;

    if (!buffer) {
        out = _out_null;
    }

    while (*format) {
        // Format specifier? %[flags][width][.precision][length]
        if (*format != '%') {
            out(*format, buffer, idx++, maxlen);
            format++;
            continue;
        } else {
            format++;
        }

        // Evaluate flags
        flags = 0U;
        do {
            switch (*format) {
                case '0': flags |= FLAGS_ZEROPAD; format++; n = 1U; break;
                case '-': flags |= FLAGS_LEFT;   format++; n = 1U; break;
                case '+': flags |= FLAGS_PLUS;   format++; n = 1U; break;
                case ' ': flags |= FLAGS_SPACE;  format++; n = 1U; break;
                case '#': flags |= FLAGS_HASH;   format++; n = 1U; break;
                default:                                   n = 0U; break;
            }
        } while (n);

        // Evaluate width field
        width = 0U;
        if (_is_digit(*format)) {
            width = _atoi(&format);
        } else if (*format == '*') {
            const int w = va_arg(va, int);
            if (w < 0) {
                flags |= FLAGS_LEFT;
                width = (unsigned int)-w;
            } else {
                width = (unsigned int)w;
            }
            format++;
        }

        // Evaluate precision field
        precision = 0U;
        if (*format == '.') {
            flags |= FLAGS_PRECISION;
            format++;
            if (_is_digit(*format)) {
                precision = _atoi(&format);
            } else if (*format == '*') {
                const int prec = va_arg(va, int);
                precision = prec > 0 ? (unsigned int)prec : 0U;
                format++;
            }
        }

        // Evaluate length field
        switch (*format) {
            case 'l':
                flags |= FLAGS_LONG;
                format++;
                if (*format == 'l') {
                    flags |= FLAGS_LONG_LONG;
                    format++;
                }
                break;
            case 'h':
                flags |= FLAGS_SHORT;
                format++;
                if (*format == 'h') {
                    flags |= FLAGS_CHAR;
                    format++;
                }
                break;
            case 'z':
                flags |= (sizeof(size_t) == sizeof(long) ? FLAGS_LONG : FLAGS_LONG_LONG);
                format++;
                break;
            case 'j':
                flags |= (sizeof(intmax_t) == sizeof(long) ? FLAGS_LONG : FLAGS_LONG_LONG);
                format++;
                break;
            case 't':
                flags |= (sizeof(ptrdiff_t) == sizeof(long) ? FLAGS_LONG : FLAGS_LONG_LONG);
                format++;
                break;
            default:
                break;
        }

        // Evaluate specifier
        switch (*format) {
            case 'd':
            case 'i':
            case 'u':
            case 'x':
            case 'X':
            case 'o':
            case 'b': {
                // Set the base
                unsigned int base;
                if (*format == 'x' || *format == 'X') {
                    base = 16U;
                } else if (*format == 'o') {
                    base = 8U;
                } else if (*format == 'b') {
                    base = 2U;
                    flags &= ~FLAGS_HASH;
                } else {
                    base = 10U;
                    flags &= ~FLAGS_HASH;
                }
                // Uppercase
                if (*format == 'X') {
                    flags |= FLAGS_UPPERCASE;
                }

                // No plus or space flag for u, x, X, o, b
                if ((*format != 'i') && (*format != 'd')) {
                    flags &= ~(FLAGS_PLUS | FLAGS_SPACE);
                }

                // Convert the integer
                if ((*format == 'i') || (*format == 'd')) {
                    // Signed
                    if (flags & FLAGS_LONG_LONG) {
#if defined(PRINTF_SUPPORT_LONG_LONG)
                        const long long value = va_arg(va, long long);
                        idx = _ntoa_long_long(out, buffer, idx, maxlen,
                                              (unsigned long long)(value > 0 ? value : 0 - value),
                                              value < 0, base, precision, width, flags);
#endif
                    } else if (flags & FLAGS_LONG) {
                        const long value = va_arg(va, long);
                        idx = _ntoa_long(out, buffer, idx, maxlen,
                                         (unsigned long)(value > 0 ? value : 0 - value),
                                         value < 0, base, precision, width, flags);
                    } else {
                        const int value = (flags & FLAGS_CHAR) ? (char)va_arg(va, int) :
                                          (flags & FLAGS_SHORT) ? (short int)va_arg(va, int) :
                                          va_arg(va, int);
                        idx = _ntoa_long(out, buffer, idx, maxlen,
                                         (unsigned int)(value > 0 ? value : 0 - value),
                                         value < 0, base, precision, width, flags);
                    }
                } else {
                    // Unsigned
                    if (flags & FLAGS_LONG_LONG) {
#if defined(PRINTF_SUPPORT_LONG_LONG)
                        idx = _ntoa_long_long(out, buffer, idx, maxlen,
                                              va_arg(va, unsigned long long), 0, base,
                                              precision, width, flags);
#endif
                    } else if (flags & FLAGS_LONG) {
                        idx = _ntoa_long(out, buffer, idx, maxlen,
                                         va_arg(va, unsigned long), 0, base,
                                         precision, width, flags);
                    } else {
                        const unsigned int value = (flags & FLAGS_CHAR) ? (unsigned char)va_arg(va, unsigned int) :
                                                   (flags & FLAGS_SHORT) ? (unsigned short int)va_arg(va, unsigned int) :
                                                   va_arg(va, unsigned int);
                        idx = _ntoa_long(out, buffer, idx, maxlen, value, 0, base, precision, width, flags);
                    }
                }
                format++;
                break;
            }

#if defined(PRINTF_SUPPORT_FLOAT)
            case 'f':
            case 'F':
                idx = _ftoa(out, buffer, idx, maxlen, va_arg(va, double), precision, width, flags);
                format++;
                break;
#endif

            case 'c': {
                unsigned int l = 1U;
                // Pre padding
                if (!(flags & FLAGS_LEFT)) {
                    while (l++ < width) {
                        out(' ', buffer, idx++, maxlen);
                    }
                }
                // Char output
                out((char)va_arg(va, int), buffer, idx++, maxlen);
                // Post padding
                if (flags & FLAGS_LEFT) {
                    while (l++ < width) {
                        out(' ', buffer, idx++, maxlen);
                    }
                }
                format++;
                break;
            }

            case 's': {
                const char *p = va_arg(va, char *);
                if (p == NULL) {
                    p = "(null)";
                }
                unsigned int l = _strlen(p);
                // Pre padding
                if (flags & FLAGS_PRECISION) {
                    l = (l < precision ? l : precision);
                }
                if (!(flags & FLAGS_LEFT)) {
                    while (l++ < width) {
                        out(' ', buffer, idx++, maxlen);
                    }
                }
                // String output
                while ((*p != 0) && (!(flags & FLAGS_PRECISION) || precision--)) {
                    out(*(p++), buffer, idx++, maxlen);
                }
                // Post padding
                if (flags & FLAGS_LEFT) {
                    while (l++ < width) {
                        out(' ', buffer, idx++, maxlen);
                    }
                }
                format++;
                break;
            }

            case 'p': {
                width = sizeof(void *) * 2U;
                flags |= FLAGS_ZEROPAD | FLAGS_UPPERCASE;
#if defined(PRINTF_SUPPORT_LONG_LONG)
                const int is_ll = sizeof(uintptr_t) == sizeof(long long);
                if (is_ll) {
                    idx = _ntoa_long_long(out, buffer, idx, maxlen,
                                          (uintptr_t)va_arg(va, void *), 0, 16U,
                                          precision, width, flags);
                } else {
#endif
                    idx = _ntoa_long(out, buffer, idx, maxlen,
                                     (unsigned long)((uintptr_t)va_arg(va, void *)), 0, 16U,
                                     precision, width, flags);
#if defined(PRINTF_SUPPORT_LONG_LONG)
                }
#endif
                format++;
                break;
            }

            case '%':
                out('%', buffer, idx++, maxlen);
                format++;
                break;

            default:
                out(*format, buffer, idx++, maxlen);
                format++;
                break;
        }
    }

    // Termination
    out((char)0, buffer, idx < maxlen ? idx : maxlen - 1U, maxlen);

    return (int)idx;
}

//==============================================================================
// Public API
//==============================================================================

int sc_printf(const char *format, ...) {
    va_list va;
    va_start(va, format);
    char buffer[1];
    const int ret = _vsnprintf(_out_char, buffer, (size_t)-1, format, va);
    va_end(va);
    return ret;
}

int sc_sprintf(char *buffer, const char *format, ...) {
    va_list va;
    va_start(va, format);
    const int ret = _vsnprintf(_out_buffer, buffer, (size_t)-1, format, va);
    va_end(va);
    return ret;
}

int sc_snprintf(char *buffer, size_t count, const char *format, ...) {
    va_list va;
    va_start(va, format);
    const int ret = _vsnprintf(_out_buffer, buffer, count, format, va);
    va_end(va);
    return ret;
}

int sc_vsnprintf(char *buffer, size_t count, const char *format, va_list va) {
    return _vsnprintf(_out_buffer, buffer, count, format, va);
}
