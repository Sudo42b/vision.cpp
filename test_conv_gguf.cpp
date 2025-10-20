// #include "ggml.h"
// #include "gguf.h"
// #include <cnpy.h>
// #include <vector>
#include <iostream>
// #include <cmath>
#include <cassert>
#include <cstring>

// float rmse(const std::vector<float>& a, const std::vector<float>& b) {
//     double s = 0;
//     for (size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]);
//     return std::sqrt(s / a.size());
// }

// float mae(const std::vector<float>& a, const std::vector<float>& b) {
//     double s = 0;
//     for (size_t i = 0; i < a.size(); i++) s += std::fabs(a[i] - b[i]);
//     return s / a.size();
// }

int main() {
    printf("Hello, World!\n");
    return 0;
}
//     const char* gguf_path = "/home/sw.lee/supergate/vision.cpp/models/yolov9t_converted.gguf";
//     const char* npz_path  = "/home/sw.lee/supergate/vision.cpp/tests/conv_test_ref.npz";

//     // === Load npz ===
//     auto npz = cnpy::npz_load(npz_path);
//     auto Xv = npz["input"].as_vec<float>();
//     auto Y_ref = npz["output_ref"].as_vec<float>();
//     auto Xshape = npz["input"].shape; // [1, C, H, W]
//     int B = Xshape[0], Cin = Xshape[1], H = Xshape[2], W = Xshape[3];

//     // === Load GGUF ===
//     struct gguf_context *gguf = gguf_init_from_file(gguf_path, (struct gguf_init_params){ false });
//     if (!gguf) {
//         fprintf(stderr, "❌ Failed to load GGUF file\n");
//         return 1;
//     }

//     int tidx_w = gguf_find_tensor(gguf, "model.0.conv.weight");
//     int tidx_b = gguf_find_tensor(gguf, "model.0.conv.bias");
//     if (tidx_w < 0) {
//         fprintf(stderr, "❌ Tensor not found: model.0.conv.weight\n");
//         gguf_free(gguf);
//         return 1;
//     }

//     const struct gguf_tensor_info *tinfo_w = gguf_get_tensor_info(gguf, tidx_w);
//     const struct gguf_tensor_info *tinfo_b = gguf_get_tensor_info(gguf, tidx_b);

//     int Kw = tinfo_w->ne[0];
//     int Kh = tinfo_w->ne[1];
//     int C_in = tinfo_w->ne[2];
//     int C_out = tinfo_w->ne[3];
//     printf("✅ Loaded weight: [%d,%d,%d,%d]\n", Kw, Kh, C_in, C_out);

//     size_t w_bytes = gguf_get_tensor_size(gguf, tidx_w);
//     size_t b_bytes = gguf_get_tensor_size(gguf, tidx_b);

//     std::vector<float> Wv(w_bytes / sizeof(float));
//     std::vector<float> Bv(b_bytes / sizeof(float));

//     memcpy(Wv.data(), gguf_get_tensor_data(gguf, tidx_w), w_bytes);
//     memcpy(Bv.data(), gguf_get_tensor_data(gguf, tidx_b), b_bytes);

//     gguf_free(gguf);

//     // === GGML Context ===
//     const size_t ctx_size = 512 * 1024 * 1024;
//     std::vector<uint8_t> buf(ctx_size);
//     ggml_init_params iparams = { ctx_size, buf.data(), false };
//     ggml_context* ctx = ggml_init(iparams);

//     ggml_tensor* X = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, B);
//     memcpy(X->data, Xv.data(), ggml_nbytes(X));

//     ggml_tensor* Wt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Kw, Kh, Cin, C_out);
//     memcpy(Wt->data, Wv.data(), ggml_nbytes(Wt));

//     ggml_tensor* Bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C_out);
//     memcpy(Bias->data, Bv.data(), ggml_nbytes(Bias));

//     ggml_tensor* Y = ggml_conv_2d(ctx, Wt, X, 1, 1, 1, 1, 1, 1);
//     ggml_tensor* Yb = ggml_add(ctx, Y, ggml_repeat(ctx, Bias, Y));

//     ggml_cgraph* gf = ggml_new_graph(ctx);
//     ggml_build_forward_expand(gf, Yb);
//     ggml_graph_compute(ctx, gf);

//     size_t N = ggml_nelements(Yb);
//     std::vector<float> out(N);
//     memcpy(out.data(), (float*)Yb->data, N * sizeof(float));

//     float err_rmse = rmse(out, Y_ref);
//     float err_mae  = mae(out, Y_ref);

//     printf("\n==== ✅ Conv Output Comparison ====\n");
//     printf("RMSE: %.8f\n", err_rmse);
//     printf("MAE : %.8f\n", err_mae);

//     ggml_free(ctx);
//     return 0;
// }
