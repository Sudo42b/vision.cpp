/**
 * @file sc_print.h
 * @brief Simulator Console Print declarations for GTX kernel builds.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GTX_SC_PRINT_H
#define GTX_SC_PRINT_H

#include <stdarg.h>

int putchar(int ch);
int printf(const char *format, ...);
int vprintf(const char *format, va_list va);

#endif /* GTX_SC_PRINT_H */
