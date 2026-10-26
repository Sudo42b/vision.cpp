#define GGML_GTX_UNARY_KERNEL_ID GGML_GTX_KERNEL_NEG_F16
#define GGML_GTX_UNARY_KERNEL_FN ggml_gtx_kernel_neg_f16
#define GGML_GTX_UNARY_INPUT_BANK GGML_GTX_UNARY_BANK_A
#define GGML_GTX_UNARY_OUTPUT_BANK GGML_GTX_UNARY_BANK_R
#define GGML_GTX_UNARY_APPLY(elems) __neg_v(elems)
#include "ggml-gtx-kernel-unary-ref.h"
