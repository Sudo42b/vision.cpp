#include "yolov9t.h"
#include "visp/image.h"
#include "visp/ml.h"
#include "visp/nn.h"
#include "visp/util.h"
#include "util/math.h"
#include "visp/vision.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include "ggml.h"

#include <optional>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wpragmas"
using namespace visp;

enum PADDING_MODE { NONE = -INT16_MAX, YES = 1 };
namespace visp::yolov9t {


yolov9t_params yolov9t_detect_params(model_file const& /*file*/) {
    yolov9t_params params;
    
    
    params.num_classes = 80;
    params.input_size = 640;
    params.variant = "tiny";
    

    return params;
}
int autopad(int k, int p = -1, int d = 1) {
    if (d > 1) {
        k = d * (k - 1) + 1;
    }
    if (p == -1) { 
        p = k / 2;
    }
    return p;
}


inline tensor add_bias_if_exists(model_ref m, tensor x, const std::string& full_prefix) {
    auto old = m.prefix;
    m.prefix = tensor_name(full_prefix.c_str());
    tensor b = m.find("bias");
    m.prefix = old;
    if (!b) return x;
    
    GGML_ASSERT(b->ne[0] == x->ne[0]);
    b = ggml_reshape_4d(m, b, b->ne[0], 1, 1, 1);
    return ggml_add(m, x, b);
}


tensor Conv(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int k,
    int s,
    int p,
    bool act,
    bool bn,
    bool debug) {
    if (debug) {
        printf(
            "Conv In: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", (int)x->ne[0], (int)x->ne[1],
            (int)x->ne[2], (int)x->ne[3]);
    }
    
    int auto_pad = autopad(k, p, 1);

    
    tensor_name old_prefix = m.prefix;

    
    std::string weight_name = name;
    
    
    if (name.find(".conv") == std::string::npos) {
        weight_name += ".conv";
    }
    m.prefix = tensor_name(weight_name.c_str());
    tensor weight = m.weights("weight");
    
    if (debug) {
        printf("Weight shape: [%d,%d,%d,%d]\n", (int)weight->ne[0], (int)weight->ne[1], (int)weight->ne[2], (int)weight->ne[3]);
        printf("Input shape: [%d,%d,%d,%d]\n", (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }
    
    if (debug) {
        printf(
            "Conv: c1=%d, c2=%d, k=%d, s=%d, p=%d, g=1, d=1, act=%s\n", c1, c2, k, s, auto_pad,
            act ? "True" : "False");
    }

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(m, x);
    }
    
    
    x = conv_2d(m, x, s, auto_pad, 1);
    if (bn){
        printf("old_prefix: %s\n", name.c_str());
        std::string bn_name = name.c_str();
        if (auto pos = bn_name.rfind(".conv"); pos != std::string::npos) {
            bn_name.replace(pos, std::string(".conv").size(), ".bn");
        }
        if (bn_name.find(".bn") == std::string::npos) {
            bn_name += ".bn";
        }
        m.prefix = tensor_name(bn_name.c_str());
        x = batch_norm_2d(m, x);
    }
    
    if (act) {
        x = ggml_silu(m, x);
    }

    
    m.prefix = old_prefix;
    
    if (debug) {
        printf(
            "Conv Out: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", (int)x->ne[0], (int)x->ne[1],
            (int)x->ne[2], (int)x->ne[3]);
    }
    return x;
}



/*
class AConv(nn.Module):
    """Average pooling convolution for downsampling"""
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)  # 단순한 3x3 stride=2 컨볼루션
        self.avgpool = nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        pool = self.avgpool(x)
        x = self.cv1(pool)
        return x
*/
tensor conv_2d_batch_norm(model_ref m, tensor x, int stride = 1, int pad = 0) {
    return conv_2d(m, x, stride, pad); 
}
tensor mean_2d(model_ref m, tensor x) {
    auto [w, h, c, n] = nelements_whcn(m, x);
    x = contiguous_2d_to_whcn(m, x);
    x = ggml_reshape_3d(m, x, w * h, c, n);
    x = ggml_mean(m, x);
    x = is_cwhn(m) ? ggml_reshape_4d(m, x, c, 1, 1, n) : ggml_reshape_4d(m, x, 1, 1, c, n);
    return x;
}

tensor global_avg_pool(model_ref m, tensor x) {
    x = mean_2d(m, x);
    x = conv_2d_batch_norm(m[1], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}
tensor AConv(model_ref m, tensor x, std::string const& name, int c1, int c2, bool debug) {
    if (debug) {
        printf("AConv In  (C,W,H,N) = [%d,%d,%d,%d]\n",
               (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }

    
    
    printf("Before Pooling (C,W,H,N) = [%d,%d,%d,%d]\n",
           (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    x = contiguous_2d_to_whcn(m, x);
    
    printf("After Permute (W,H,C,N) = [%d,%d,%d,%d]\n",
           (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    tensor pooled = ggml_pool_2d(
        m, x,
        GGML_OP_POOL_AVG,
        /*kW*/ 2, /*kH*/ 2,
        /*sW*/ 1, /*sH*/ 1,
        /*pW*/ 0, /*pH*/ 0
    );
    
    printf("After Pooling (C,W,H,N) = [%d,%d,%d,%d]\n",
           (int)pooled->ne[0], (int)pooled->ne[1], (int)pooled->ne[2], (int)pooled->ne[3]);
           pooled = whcn_to_contiguous_2d(m, pooled);
    printf("After Permute (C,W,H,N) = [%d,%d,%d,%d]\n",
                  (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    if (debug) {
        printf("AConv Pool (C,W,H,N) = [%d,%d,%d,%d]\n",
               (int)pooled->ne[0], (int)pooled->ne[1], (int)pooled->ne[2], (int)pooled->ne[3]);
    }

    
    
    tensor out = Conv(m, pooled, name + ".cv1.conv", c1, c2, /*k=*/3, /*stride=*/2, /*pad=*/1, /*act=*/true);
    printf("%ld, %ld, %ld, %ld\n", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    if (debug) {
        printf("AConv Out  (C,W,H,N) = [%d,%d,%d,%d]\n",
               (int)out->ne[0], (int)out->ne[1], (int)out->ne[2], (int)out->ne[3]);
    }
    return out;
}

tensor ELAN1(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int c3,
    int c4,
    bool debug
) {
    
    if (debug) {
        printf("ELAN1 %s: x (C,W,H,N) = [%d,%d,%d,%d]\n",
               name.c_str(), (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }

    
    GGML_ASSERT(c3 % 2 == 0 && "ELAN1: c3 must be even for chunk(2)");
    const int c_half = c3 / 2;
    tensor cv1_out = Conv(m, x, name + ".cv1.conv", c1, c3, 1, 1, -1, true);

    
    
    tensor split1 = slice(m, cv1_out,
                          /*C*/ {0,       c_half},
                          /*W*/{}, /*H*/ {}, /*N*/ {});
    tensor split2 = slice(m, cv1_out,
                          /*C*/ {c_half, c3},
                          /*W*/ {}, /*H*/ {}, /*N*/ {});

    
    if (!ggml_is_contiguous(split1)) split1 = ggml_cont(m, split1);
    if (!ggml_is_contiguous(split2)) split2 = ggml_cont(m, split2);

    
    tensor cv2_out = Conv(m, split2, name + ".cv2.conv", c_half, c4, 3, 1, -1, true);

    
    tensor cv3_out = Conv(m, cv2_out, name + ".cv3.conv", c4, c4, 3, 1, -1, true);

    
    tensor cat12   = Concat(m, split1, split2, /*dim=*/0);
    tensor cat123  = Concat(m, cat12,  cv2_out, /*dim=*/0);
    tensor cat1234 = Concat(m, cat123, cv3_out, /*dim=*/0);

    if (debug) {
        printf("ELAN1 %s: cat (C,W,H,N) = [%d,%d,%d,%d]\n",
               name.c_str(), (int)cat1234->ne[0], (int)cat1234->ne[1], (int)cat1234->ne[2], (int)cat1234->ne[3]);
    }

    
    tensor out = Conv(m, cat1234, name + ".cv4.conv", c3 + (2 * c4), c2, 1, 1, -1, true);

    
    if (debug) {
        printf("ELAN1 %s: out (C,W,H,N) = [%d,%d,%d,%d]\n",
               name.c_str(), (int)out->ne[0], (int)out->ne[1], (int)out->ne[2], (int)out->ne[3]);
    }
    return out;
}

tensor Bottleneck(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    bool shortcut,
    int g,
    float e,
    bool debug) {
    if (debug) {
        printf(
            "Bottleneck %s: In shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3], (int)x->ne[2],
            (int)x->ne[1], (int)x->ne[0]);
    }
    int c_ = (int)(c2 * e); 

    
    tensor cv1_out = Conv(m, x, name + ".cv1.conv", c1, c_, 1, 1, -1, true);
    tensor cv2_out = Conv(m, cv1_out, name + ".cv2.conv", c_, c2, 3, 1, -1, true);

    
    if (debug) {
        printf(
            "Bottleneck %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)cv2_out->ne[3],
            (int)cv2_out->ne[2], (int)cv2_out->ne[1], (int)cv2_out->ne[0]);
    }
    return cv2_out;
}

tensor RepConv(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int k,
    int s,
    int p,
    int g,
    int d,
    bool act,
    bool bn,
    bool deploy,
    bool debug) {
    if (debug) {
        printf(
            "RepConv %s In: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", name.c_str(), (int)x->ne[0], (int)x->ne[1],
            (int)x->ne[2], (int)x->ne[3]);
    }
    
    
    if (k != 3 || p != 1) {
        throw std::invalid_argument("RepConv requires k=3 and p=1");
    }
    
    tensor output;
    
    if (deploy) {
        
        output = Conv(m, x, name + ".fused_conv", c1, c2, k, s, p, act, debug);
    } else {
        
        
        std::string cv11_name = name;
        
        tensor conv1_out = Conv(m, x, cv11_name + ".conv1.conv", c1, c2, k, s, p, false, debug);
        tensor conv2_out = Conv(m, x, cv11_name + ".conv2.conv", c1, c2, 1, s, (p-k)/2, false, debug);  

        if (bn){
            printf("batchnorm 처리해야함\n");
        }
        
        output = ggml_add(m, conv1_out, conv2_out);
        
        
        if (act) {
            output = ggml_silu(m, output); 
        }
    }
    
    if (debug) {
        printf(
            "RepConv %s Out: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", name.c_str(), 
            (int)output->ne[0], (int)output->ne[1], (int)output->ne[2], (int)output->ne[3]);
    }
    
    return output;
}



tensor RepBottleneck(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    bool shortcut,
    int g,
    int k,
    float e,
    bool debug) {
    if (debug) {
        printf(
            "RepBottleneck %s: In shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3],
            (int)x->ne[2], (int)x->ne[1], (int)x->ne[0]);
    }
    
    int c_ = (int)(c2 * e); 
    
    
    tensor cv1_out = RepConv(m, x, name + ".cv1", c1, c_, 3, 1, 1, 1, 1, true, false, false, debug);
    
    
    tensor cv2_out = Conv(m, cv1_out, name + ".cv2", c_, c2, 3, 1, -1, true, debug);
    
    
    if (shortcut && c1 == c2) {
        tensor result = ggml_add(m, x, cv2_out);
        if (debug) {
            printf(
                "RepBottleneck %s: Out shape with shortcut [%d,%d,%d,%d]\n", name.c_str(), 
                (int)result->ne[3], (int)result->ne[2], (int)result->ne[1], (int)result->ne[0]);
        }
        return result;
    }
    
    if (debug) {
        printf(
            "RepBottleneck %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)cv2_out->ne[3],
            (int)cv2_out->ne[2], (int)cv2_out->ne[1], (int)cv2_out->ne[0]);
    }
    
    return cv2_out;
}

tensor C3(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int n,
    bool shortcut,
    int g,
    float e,
    bool debug) {
    if (debug) {
        printf(
            "C3 %s: In shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3], (int)x->ne[2],
            (int)x->ne[1], (int)x->ne[0]);
    }
    int c_ = (int)(c2 * e); 
    
    
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c_, 1, 1, -1, true);

    
    tensor cv2_out = Conv(m, x, name + ".cv2", c1, c_, 1, 1, -1, true);

    
    tensor m_out = cv1_out;
    for (int i = 0; i < n; ++i) {
        std::string bottleneck_name = name + ".m." + std::to_string(i);
        m_out = Bottleneck(m, m_out, bottleneck_name, c_, c_, shortcut, g, 1.0);
    }

    
    tensor concat = ggml_concat(m, m_out, cv2_out, 0); 
    concat = ggml_cont(m, concat); 
    
    tensor output = Conv(m, concat, name + ".cv3", 2 * c_, c2, 1, 1, -1, true);
    
    if (debug) {
        printf(
            "C3 %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }
    return output;
}


tensor RepCSP(
    model_ref m,
    tensor x,
    std::string const& name,
    int c1,
    int c2,
    int n,
    bool shortcut,
    int g,
    float e,
    bool debug) {
    if (debug) {
        printf(
            "RepCSP %s: In shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3], (int)x->ne[2],
            (int)x->ne[1], (int)x->ne[0]);
    }
    
    int c_ = (int)(c2 * e); 
    
    
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c_, 1, 1, -1, true);
    tensor cv2_out = Conv(m, x, name + ".cv2", c1, c_, 1, 1, -1, true);

    
    tensor m_out = cv1_out;
    for (int i = 0; i < n; ++i) {
        std::string bottleneck_name = name + ".m." + std::to_string(i);
        m_out = RepBottleneck(m, m_out, bottleneck_name, c_, c_, shortcut, g, 3, 1.0);
    }

    
    tensor concat = ggml_concat(m, m_out, cv2_out, 0); 
    concat = ggml_cont(m, concat); 
    
    tensor output = Conv(m, concat, name + ".cv3", 2 * c_, c2, 1, 1, -1, true);
    
    if (debug) {
        printf(
            "RepCSP %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }
    return output;
}



tensor RepNCSPELAN4(
    model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int c4, int n, bool debug) {
    
    if (debug) {
        printf(
            "RepNCSPELAN4 %s In : x shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3],
            (int)x->ne[2], (int)x->ne[1], (int)x->ne[0]);
    }
    int c = c3 / 2;
    printf("Name: %s, c1=%d, c2=%d, c3=%d, c4=%d, n=%d\n", name.c_str(), c1, c2, c3, c4, n);
    
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c3, /*k=*/1, /*stride=*/1, /*pad=*/0, /*bias=*/true);

    
    tensor cv1_out_h0 = slice(m, cv1_out,
                      /*C*/ slice_t(0,       c),
                      /*W*/ slice_t(), /*H*/ slice_t(), /*N*/ slice_t());
    tensor cv1_out_h1 = slice(m, cv1_out,
                            /*C*/ slice_t(c, c3),
                            /*W*/ slice_t(), /*H*/ slice_t(), /*N*/ slice_t());

    if (!ggml_is_contiguous(cv1_out_h0)) 
        cv1_out_h0 = ggml_cont(m, cv1_out_h0);
    if (!ggml_is_contiguous(cv1_out_h1)) 
        cv1_out_h1 = ggml_cont(m, cv1_out_h1);

    auto rep_conv = [=](model_ref m, tensor x, std::string name, int c_in, int c_out, int n) {
        tensor y2 = RepCSP(m, x, name+ ".0", c_in, c_out, n);
        return Conv(m, y2, name + ".1", c_out, c_out, 3, 1, 1, true);
    };
    
    
    tensor cv2 = rep_conv(m, cv1_out_h1, name + ".cv2", c3 / 2, c4, n);
    tensor cv3 = rep_conv(m, cv2, name + ".cv3", c4, c4, n);
    
    tensor cat = concat(m, {cv1_out_h0, cv1_out_h1, cv2, cv3}, 0);
    if (!ggml_is_contiguous(cat))
    {
        printf("Making concatenated tensor contiguous for RepNCSPELAN4 %s\n", name.c_str());
        cat = ggml_cont(m, cat); 
    }
    
    
    tensor output = Conv(m, cat, name + ".cv4", c3 + (2 * c4), c2, 1, 1, -1, true);
    if (debug) {
        printf(
            "RepNCSPELAN4 %s: output shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }

    return output;
}



tensor SPPELAN(model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int k, bool debug) {
    if (debug) {
        printf(
            "SPPELAN %s: x shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3],
            (int)x->ne[2], (int)x->ne[1], (int)x->ne[0]);
    }
    
    tensor cv1 = Conv(m, x, name + ".cv1", c1, c3, 1, 1, -1, true);
    
    if (debug) {
        printf("cv1: [%ld, %ld, %ld, %ld]\n", 
               cv1->ne[0], cv1->ne[1], cv1->ne[2], cv1->ne[3]);
    }
    
    int pad = k / 2;
    
    
    
    
    auto max_pool = [=](model_ref m, tensor x) {
        x = contiguous_2d_to_whcn(m, x);
        x = ggml_pool_2d(m, x, GGML_OP_POOL_MAX, k, k, 1, 1, pad, pad);
        x = whcn_to_contiguous_2d(m, x);
        return x;
    };
    tensor cv2 = max_pool(m, cv1);
    tensor cv3 = max_pool(m, cv2);
    tensor cv4 = max_pool(m, cv3);
    

    
    
    tensor cv5 = concat(m, {cv1, cv2, cv3, cv4}, 0); 
    cv5 = ggml_cont(m, cv5); 

    tensor output = Conv(m, cv5, name + ".cv5", 4*c3, c2, 1, 1, -1, true);
    
    
    
    
    
    
    return output;
}


tensor Upsample(model_ref m, tensor x, int scale_factor, bool debug) {
    
    x = permute_cwhn_to_whcn(m, x);
    x = ggml_upscale(m, x, scale_factor, GGML_SCALE_MODE_NEAREST);
    x = permute_whcn_to_cwhn(m, x);
    x = ggml_cont(m, x);
    return  x; 
}

tensor Concat(model_ref m, tensor a, tensor b, int axis, bool debug) {
    
    
    int dim = (m.flags & model_build_flag::cwhn) ? 0 : 2;
    tensor output = ggml_concat(m, a, b, dim);
    
    output = ggml_cont(m, output); 
    return output;
}



std::map<int, tensor> yolov9t_backbone(model_ref m, tensor x) {
    std::map<int, tensor> features;
    
    tensor x0 = Conv(m, x, "model.0", 3, 16, 3, 2, -1, true, false);
    features[0] = x0;
    ggml_set_output(x0);

    
    tensor x1 = Conv(m, x0, "model.1", 16, 32, 3, 2, -1, true, false);
    features[1] = x1;
    ggml_set_output(x1);
    
    
    tensor x2 = ELAN1(m, x1, "model.2", 32, 32, 32, 16);
    ggml_set_output(x2);
    features[2] = x2;
    
    
    tensor x3 = AConv(m, x2, "model.3", 32, 64);
    ggml_set_output(x3);
    features[3] = x3;

    
    tensor x4 = RepNCSPELAN4(m, x3, "model.4", 64, 64, 64, 32, 3, false);
    features[4] = x4;
    ggml_set_output(x4);
    
    
    tensor x5 = AConv(m, x4, "model.5", 64, 96);
    features[5] = x5;
    ggml_set_output(x5);
    
    
    tensor x6 = RepNCSPELAN4(m, x5, "model.6", 96, 96, 96, 48, 3);
    features[6] = x6;
    ggml_set_output(x6);

    
    tensor x7 = AConv(m, x6, "model.7", 96, 128);
    features[7] = x7;
    ggml_set_output(x7);

    
    tensor x8 = RepNCSPELAN4(m, x7, "model.8", 128, 128, 128, 64, 3);
    features[8] = x8;
    ggml_set_output(x8);

    
    tensor x9 = SPPELAN(m, x8, "model.9", 128, 128, 64, 5, false);
    features[9] = x9;
    ggml_set_output(x9);

    
    tensor x10 = Upsample(m, x9, 2);
    features[10] = x10;
    ggml_set_output(x10);

    
    
    
    
    
    tensor x11 = Concat(m, x10, features[6], 2);
    features[11] = x11;
    ggml_set_output(x11);

    
    
    
    tensor x12 = RepNCSPELAN4(m, x11, "model.12", 224, 96, 96, 48, 3, false);
    features[12] = x12;
    ggml_set_output(x12);

    
    tensor x13 = Upsample(m, x12, 2);
    features[13] = x13;
    ggml_set_output(x13);

    
    tensor x14 = Concat(m, x13, features[4], 2);
    features[14] = x14;
    ggml_set_output(x14);

    
    tensor x15 = RepNCSPELAN4(m, x14, "model.15", 160, 64, 64, 32, 3);
    features[15] = x15;
    ggml_set_output(x15);

    
    tensor x16 = AConv(m, x15, "model.16", 64, 48);
    features[16] = x16;
    ggml_set_output(x16);

    
    tensor x17 = Concat(m, x16, features[12]);
    features[17] = x17;
    ggml_set_output(x17);

    
    tensor x18 = RepNCSPELAN4(m, x17, "model.18", 144, 96, 96, 48, 3);
    features[18] = x18;
    ggml_set_output(x18);

    
    tensor x19 = AConv(m, x18, "model.19", 96, 64);
    features[19] = x19;
    ggml_set_output(x19);

    
    tensor x20 = Concat(m, x19, features[9]);
    features[20] = x20;
    ggml_set_output(x20);

    
    tensor x21 = RepNCSPELAN4(m, x20, "model.21", 192, 128, 128, 64, 3);
    features[21] = x21;
    ggml_set_output(x21);
    
    
    
    return features;
}


tensor dist2bbox(model_ref m, tensor distance, tensor anchor_points, tensor stride_tensor, bool xywh) {
    
    printf("distance shape before permute: [%ld,%ld,%ld,%ld]\n",
        distance->ne[0], distance->ne[1], distance->ne[2], distance->ne[3]);
    
    distance = ggml_cont(m, distance);
    distance = ggml_permute(m, distance, 2, 0, 1, 3);
    printf("distance shape after permute: [%ld,%ld,%ld,%ld]\n",
        distance->ne[0], distance->ne[1], distance->ne[2], distance->ne[3]);
    distance = ggml_cont(m, distance);
    GGML_ASSERT(distance->ne[0] == 4);
    GGML_ASSERT(anchor_points->ne[0] == 2);

    tensor lt = slice(m, distance, slice_t{0, 2}, slice_t{}, slice_t{}, slice_t{});
    tensor rb = slice(m, distance, slice_t{2, 4}, slice_t{}, slice_t{}, slice_t{});

    
    tensor x1y1 = ggml_sub(m, anchor_points, lt);
    tensor x2y2 = ggml_add(m, anchor_points, rb);

    if (xywh) {
        tensor c_xy = ggml_scale(m, ggml_add(m, x1y1, x2y2), 0.5f);
        tensor wh   = ggml_sub(m, x2y2, x1y1);
        return ggml_concat(m, c_xy, wh, 0);
    } else {
        return ggml_concat(m, x1y1, x2y2, 0);
    }
    
}

tensor dfl_forward(model_ref m, tensor proj_tensor, tensor x, int reg_max, bool debug) {
    
    if (debug) {
        printf("dfl_forward: x shape: [%ld,%ld,%ld,%ld] reg_max=%d\n",
               x->ne[0], x->ne[1], x->ne[2], x->ne[3], reg_max);
    }

    
    if (!ggml_is_contiguous(x)) x = ggml_cont(m, x);

    
    int64_t e0 = x->ne[0];
    int64_t e1 = x->ne[1];
    int64_t e2 = x->ne[2];
    int64_t e3 = x->ne[3];
    int64_t total_elements = e0 * e1 * e2 * e3;
    int64_t batch = e3;
    int64_t expected_per_anchor = int64_t(reg_max) * 4 * batch;
    if (expected_per_anchor == 0) {
        throw std::runtime_error("Invalid reg_max or batch size in dfl_forward");
    }
    if (total_elements % expected_per_anchor != 0) {
        throw std::runtime_error("dfl_forward: input size not divisible by reg_max*4*batch");
    }
    int64_t anchors = total_elements / expected_per_anchor;

    tensor reshaped = ggml_reshape_4d(m, x, reg_max, 4, anchors, batch);

    tensor soft = ggml_soft_max(m, reshaped);

    if (!ggml_is_contiguous(proj_tensor)) proj_tensor = ggml_cont(m, proj_tensor);
    tensor proj = ggml_reshape_4d(m, proj_tensor, reg_max, 1, 1, 1);

    tensor weighted = ggml_mul(m, soft, proj);
    tensor summed = ggml_sum_rows(m, weighted);
    
    if (!ggml_is_contiguous(summed)) summed = ggml_cont(m, summed);
    return summed;
}

DetectOutput inference(model_ref m, 
                DetectOutput out, 
                std::vector<int> ch, 
                int reg_max, 
                int nc) {

    std::vector<tensor> reshaped_outputs;
    
    tensor x_cat = concat(m, out.features, 1); 
    printf("x_cat shape: [%ld,%ld,%ld, %ld]\n", x_cat->ne[0], x_cat->ne[1], x_cat->ne[2], x_cat->ne[3]);

    
    int64_t total_channels = x_cat->ne[0];
    int64_t total_anchors = x_cat->ne[1];

    std::vector<float> strides_vec;
    for (size_t i = 0; i < out.features.size(); ++i) {
        strides_vec.push_back(8.0f * std::pow(2.0f, (float)i));
    }
    std::vector<float> anchor_host;
    std::vector<float> stride_host;
    auto [anchor_points, stride_tensor] = make_anchors(m, out, anchor_host, stride_host, strides_vec, 0.5f);

    
    tensor box = ggml_view_4d(m, x_cat,
        reg_max * 4, total_anchors, 1, 1,
        x_cat->nb[1], x_cat->nb[2], x_cat->nb[3],
        0);
        
    box = ggml_cont(m, box);
    

    tensor cls = ggml_view_4d(m, x_cat,
        nc, total_anchors, 1, 1,
        x_cat->nb[1], x_cat->nb[2], x_cat->nb[3],
        reg_max * 4 * x_cat->nb[0]);
    cls = ggml_cont(m, cls);


    tensor cls_logits = ggml_dup(m, cls); 
    ggml_set_output(cls_logits);
    out.debug_cls_logits = cls_logits;


    
    tensor proj = ggml_new_tensor_1d(m.graph_context, GGML_TYPE_F32, reg_max);
    ggml_set_name(proj, "dfl_proj");
    out.dfl_proj = proj;
    out.dfl_proj_host_data.resize(reg_max);
    for (int i = 0; i < reg_max; ++i) out.dfl_proj_host_data[i] = float(i);
    out.reg_max = reg_max;
    printf("proj shape: [%ld,%ld,%ld,%ld]\n", proj->ne[0], proj->ne[1], proj->ne[2], proj->ne[3]);
    
    
    tensor dfl_output = dfl_forward(m, proj, box, reg_max, false);
    printf("dfl_output shape: [%ld,%ld,%ld,%ld]\n", dfl_output->ne[0], dfl_output->ne[1], dfl_output->ne[2], dfl_output->ne[3]);
    
    
    tensor dbox = dist2bbox(m, dfl_output, anchor_points, stride_tensor, false);

    printf("dbox shape: [%ld,%ld,%ld,%ld]\n", dbox->ne[0], dbox->ne[1], dbox->ne[2], dbox->ne[3]);
    
    tensor strides_bc = ggml_reshape_4d(m, stride_tensor, 1, total_anchors, 1, 1);

    dbox = ggml_mul(m, dbox, strides_bc);

    printf("strides_bc shape: [%ld,%ld,%ld,%ld]\n", strides_bc->ne[0], strides_bc->ne[1], strides_bc->ne[2], strides_bc->ne[3]);

    
    cls = ggml_sigmoid(m, cls);
    ggml_set_name(cls, "cls_prob");
    printf("cls shape: [%ld,%ld,%ld,%ld]\n", cls->ne[0], cls->ne[1], cls->ne[2], cls->ne[3]);
    

    out.predictions_bbox = std::move(dbox);
    out.predictions_cls = std::move(cls);
    out.anchor_points = std::move(anchor_points);
    out.strides_points = std::move(stride_tensor);
    out.anchor_host_data = std::move(anchor_host);
    out.stride_host_data = std::move(stride_host);
    return out;
}

DetectOutput detect_forward(model_ref m, 
                            std::vector<tensor> features, 
                            std::vector<int> ch, 
                            int nc,
                            bool training) {
    int reg_max = 16; 
    int c2 = std::max({16, ch[0] / 4, reg_max * 4});
    int c3 = std::max(ch[0], std::min(nc, 100));
    DetectOutput out;

    std::string reg_base = std::string("detect.cv2");
    std::string cls_base = std::string("detect.cv3");

    for (size_t i = 0; i < features.size(); ++i) {
        
        tensor_name old_prefix_i = m.prefix;
        tensor r0 = Conv(m, features[i], reg_base + "."+std::to_string(i)+".0.conv", ch[i], c2, 3, 1, -1, true, false);
        tensor r1 = Conv(m, r0          ,reg_base + "."+std::to_string(i)+".1.conv", c2, c2, 3, 1, -1, true, false);
        
        m.prefix = tensor_name((reg_base + "."+std::to_string(i)+".2").c_str());
        tensor r2 = conv_2d(m, r1, 1, 0, 1);

        if (!ggml_is_contiguous(r2)) r2 = ggml_cont(m, r2);
        
        tensor c0 = Conv(m, features[i], cls_base + "."+std::to_string(i)+".0.conv", ch[i], c3, 3, 1, -1, true, false);
        tensor c1 = Conv(m, c0,          cls_base + "."+std::to_string(i)+".1.conv", c3, c3, 3, 1, -1, true, false);
        
        m.prefix = tensor_name((cls_base + "." + std::to_string(i) + ".2").c_str());
        tensor c2 = conv_2d(m, c1, 1, 0, 1);

        if (!ggml_is_contiguous(c2)) c2 = ggml_cont(m, c2);
        
        tensor combined = Concat(m, r2, c2, 0);
        
        tensor reshaped_combined = ggml_reshape_2d(m, combined, combined->ne[0], combined->ne[1]*combined->ne[2]); 
        

        combined = ggml_cont(m, combined);
        reshaped_combined = ggml_cont(m, reshaped_combined);
        out.raw_outputs.push_back(std::move(combined));
        out.features.push_back(std::move(reshaped_combined));
        
        m.prefix = old_prefix_i;
    }

    if (training) {
        out.predictions_cls = nullptr;
        out.predictions_bbox = nullptr;
        return out;
    }
    
    DetectOutput detect = inference(m, out, ch, reg_max, nc);
    
    return detect;
}


DetectOutput yolov9t_forward(model_ref m, tensor x) {
    
    std::map<int, tensor> features = yolov9t_backbone(m, x);
    
    printf("features size: [%d]\n", (int)features.size());
    
    std::vector<int> channels = {64, 96, 128};
    
    std::vector<tensor> features_vector = {features[15], features[18], features[21]};
    DetectOutput d = detect_forward(m, features_vector, channels, 80, false);
    
    d.features_map = features; 
    printf("detect_forward complete\n");
    
    return d;
}


std::vector<std::string> const& get_coco_class_names() {
    static std::vector<std::string> const class_names = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush"};
    return class_names;
}

std::pair<tensor, tensor> make_anchors(
    model_ref m,
    DetectOutput const& out,
    std::vector<float>& anchor_host,
    std::vector<float>& stride_host,
    std::vector<float> const& strides,
    float grid_cell_offset) {
    
    GGML_ASSERT(out.raw_outputs.size() == strides.size());
    GGML_ASSERT(!out.raw_outputs.empty());
    
    
    int64_t total_anchors = 0;
    for (size_t i = 0; i < out.raw_outputs.size(); ++i) {
        auto ne = nelements(out.raw_outputs[i]);
        total_anchors += ne[1] * ne[2]; 
    }


    
    std::vector<float> anchor_data(2 * total_anchors);
    std::vector<float> stride_data(total_anchors);
    
    int64_t offset = 0;
    for (size_t i = 0; i < strides.size(); ++i) {
        int64_t w = out.raw_outputs[i]->ne[1];
        int64_t h = out.raw_outputs[i]->ne[2];
        
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                int64_t idx = offset + y * w + x;
                anchor_data[idx * 2 + 0] = x + grid_cell_offset;
                anchor_data[idx * 2 + 1] = y + grid_cell_offset;
                stride_data[idx] = strides[i];
            }
        }
        offset += w * h;
    }
    
    tensor anchor_points = compute_graph_input(
        m, GGML_TYPE_F32, {2, total_anchors, 1, 1}, "anchor_points");
    tensor stride_tensor = compute_graph_input(
        m, GGML_TYPE_F32, {1, total_anchors, 1, 1}, "stride_points");

    anchor_host = std::move(anchor_data);

    stride_host = std::move(stride_data);
    
    printf("anchor_points shape: [%ld,%ld]\n", anchor_points->ne[0], anchor_points->ne[1]);
    printf("stride_tensor shape: [%ld,%ld]\n", stride_tensor->ne[0], stride_tensor->ne[1]);
    
    return std::make_pair(anchor_points, stride_tensor);
}

int make_divisible(int x, int divisor) {
    
    if (divisor < 1) {
        return x;
    }
    return (x + divisor - 1) / divisor * divisor;
}
int check_img_size(int imgsz, int s, int floor) {
    int new_size = std::max(make_divisible(imgsz, int(s)), floor);
    if (new_size != imgsz) {
        printf("WARNING ⚠️ --img-size %d must be multiple of max stride %d, updating to %d\n", imgsz, s, new_size);
    }
    return new_size;
}

image_data image_add_border(image_data im, int top, int bottom,
                            int left, int right, u8x3 color) {
    
    int W = im.extent[0];
    int H = im.extent[1];
    int channels = 3;

    i32x2 new_extent = { W + left + right, H + top + bottom };
    image_data bordered = image_alloc(new_extent, im.format);

    
    for (int y = 0; y < new_extent[1]; ++y) {
        for (int x = 0; x < new_extent[0]; ++x) {
            int idx = (y * new_extent[0] + x) * channels;
            bordered.data[idx+0] = color[0];
            bordered.data[idx+1] = color[1];
            bordered.data[idx+2] = color[2];
        }
    }

    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int src = (y * W + x) * channels;
            int dst = ((y + top) * new_extent[0] + (x + left)) * channels;
            bordered.data[dst+0] = im.data[src+0];
            bordered.data[dst+1] = im.data[src+1];
            bordered.data[dst+2] = im.data[src+2];
        }
    }
    return bordered;
}


void resize_rgb_bilinear(
    uint8_t* src_data,
    i32x2 src_extent,
    uint8_t* dst_data,
    i32x2 dst_extent) {
    
    double inv_scale_x = static_cast<double>(src_extent[0]) / dst_extent[0];
    double inv_scale_y = static_cast<double>(src_extent[1]) / dst_extent[1];
    
    for (int y = 0; y < dst_extent[1]; ++y) {
        for (int x = 0; x < dst_extent[0]; ++x) {
            double src_xf = (x + 0.5) * inv_scale_x - 0.5;
            double src_yf = (y + 0.5) * inv_scale_y - 0.5;
            
            src_xf = std::max(0.0, src_xf);
            src_yf = std::max(0.0, src_yf);
            
            int x0 = static_cast<int>(src_xf);
            int y0 = static_cast<int>(src_yf);
            int x1 = std::min(x0 + 1, src_extent[0] - 1);
            int y1 = std::min(y0 + 1, src_extent[1] - 1);
            
            double wx = src_xf - x0;
            double wy = src_yf - y0;
            double w00 = (1 - wx) * (1 - wy);
            double w01 = wx * (1 - wy);
            double w10 = (1 - wx) * wy;
            double w11 = wx * wy;
            
            
            for (int c = 0; c < 3; ++c) {
                double value = 
                    src_data[(y0 * src_extent[0] + x0) * 3 + c] * w00 +
                    src_data[(y0 * src_extent[0] + x1) * 3 + c] * w01 +
                    src_data[(y1 * src_extent[0] + x0) * 3 + c] * w10 +
                    src_data[(y1 * src_extent[0] + x1) * 3 + c] * w11;
                    
                dst_data[(y * dst_extent[0] + x) * 3 + c] = 
                    static_cast<uint8_t>(std::min(255.0, std::max(0.0, value + 0.5)));
            }
        }
    }
}


image_data linear_image_resize(image_data im, i32x2 new_shape) {
    image_data resized = image_alloc(new_shape, im.format);
    resize_rgb_bilinear(im.data.get(), im.extent, resized.data.get(), new_shape);
    return resized;
}


LetterboxResult letterbox(image_data im, i32x2 new_shape, u8x3 color,
                          bool _auto, bool scaleFill, bool scaleup, int stride)
{
    i32x2 shape = im.extent;  
    int orig_w = shape[0];
    int orig_h = shape[1];

    if (new_shape[0] == 0)
        new_shape = {new_shape[1], new_shape[1]};

    printf("[LB] extent: width=%d, height=%d, target=(%d,%d)\n",
           orig_w, orig_h, new_shape[0], new_shape[1]);

    
    float r = std::min(
        (float)new_shape[1] / (float)orig_h,  
        (float)new_shape[0] / (float)orig_w   
    );

    if (!scaleup)
        r = std::min(r, 1.0f);

    
    int new_unpad_w = std::round(orig_w * r);
    int new_unpad_h = std::round(orig_h * r);

    float dw = (float)new_shape[0] - new_unpad_w;
    float dh = (float)new_shape[1] - new_unpad_h;

    if (_auto) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_unpad_w = new_shape[0];
        new_unpad_h = new_shape[1];
    }

    dw /= 2.0f;
    dh /= 2.0f;

    im = image_scale(im, {new_unpad_w, new_unpad_h});

    int left   = std::round(dw - 0.1f);
    int right  = std::round(dw + 0.1f);
    int top    = std::round(dh - 0.1f);
    int bottom = std::round(dh + 0.1f);
    im = image_add_border(std::move(im), top, bottom, left, right, color);

    printf("[LB] orig=[%d,%d] -> scaled=[%d,%d] -> new=[%d,%d], pad=[l=%d r=%d t=%d b=%d]\n",
           orig_w, orig_h, new_unpad_w, new_unpad_h,
           new_shape[0], new_shape[1], left, right, top, bottom);

    LetterboxResult out;
    out.img = std::move(im);
    out.gain = r;
    out.pad_w = dw;
    out.pad_h = dh;
    return out;
}


inline void xywh2xyxy(float* boxes, int n) {
    
    for (int i = 0; i < n; ++i) {
        float x = boxes[i * 4 + 0];
        float y = boxes[i * 4 + 1];
        float w = boxes[i * 4 + 2];
        float h = boxes[i * 4 + 3];
        boxes[i * 4 + 0] = x - w / 2.0f;
        boxes[i * 4 + 1] = y - h / 2.0f;
        boxes[i * 4 + 2] = x + w / 2.0f;
        boxes[i * 4 + 3] = y + h / 2.0f;
    }
}


inline float box_iou(const float* b1, const float* b2) {
    
    float ix1 = std::max(b1[0], b2[0]);
    float iy1 = std::max(b1[1], b2[1]);
    float ix2 = std::min(b1[2], b2[2]);
    float iy2 = std::min(b1[3], b2[3]);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    float union_ = area1 + area2 - inter;
    return union_ > 0.f ? (inter / union_) : 0.f;
}


std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes, const std::vector<float>& scores, const std::vector<int>& class_ids, float iou_thres, bool agnostic, int max_wh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    std::vector<bool> keep(boxes.size(), true);
    std::vector<int> kept;
    for (size_t i = 0; i < order.size(); ++i) {
        int idx_i = order[i];
        if (!keep[idx_i]) continue;
        kept.push_back(idx_i);
        const float* b1 = boxes[idx_i].data();
        int ci = agnostic ? 0 : class_ids[idx_i] * max_wh;
        for (size_t j = i+1; j < order.size(); ++j) {
            int idx_j = order[j];
            if (!keep[idx_j]) continue;
            int cj = agnostic ? 0 : class_ids[idx_j] * max_wh;
            float b2[4] = {boxes[idx_j][0]+cj, boxes[idx_j][1], boxes[idx_j][2]+cj, boxes[idx_j][3]};
            float bb1[4] = {b1[0]+ci, b1[1], b1[2]+ci, b1[3]};
            if (box_iou(bb1, b2) > iou_thres) {
                keep[idx_j] = false;
            }
        }
    }
    return kept;
}

std::vector<detected_obj> non_max_suppression(
    DetectOutput const& outputs,
    float conf_thres,
    float iou_thres,
    int max_det,
    int max_nms,
    int max_wh
) {
    if (!outputs.predictions_cls || !outputs.predictions_bbox) {
        return {};
    }
    
    if (!(0.0f <= conf_thres && conf_thres <= 1.0f)) throw std::runtime_error("Invalid Confidence threshold");
    if (!(0.0f <= iou_thres && iou_thres <= 1.0f)) throw std::runtime_error("Invalid IoU threshold");

    tensor_data td_cls = transfer_from_backend(outputs.predictions_cls);
    tensor_data td_box = transfer_from_backend(outputs.predictions_bbox);
    const float* cls_ptr = td_cls.as_f32().data();
    const float* box_ptr = td_box.as_f32().data();
    const int64_t nc = outputs.predictions_cls->ne[0]; 
    const int64_t na = outputs.predictions_cls->ne[1]; 
    float max_conf = -1.0f;
    float min_conf = 2.0f;
    int max_conf_anchor = -1;
    int max_conf_class = -1;
    
    for (int64_t j = 0; j < na; ++j) {
        for (int64_t c = 0; c < nc; ++c) {
            float conf = cls_ptr[c + j*nc];
            if (conf > max_conf) {
                max_conf = conf;
                max_conf_anchor = j;
                max_conf_class = c;
            }
            min_conf = std::min(min_conf, conf);
        }
    }
    
    if (max_conf_anchor >= 0) {
        float x1 = box_ptr[0 + max_conf_anchor*4];
        float y1 = box_ptr[1 + max_conf_anchor*4];
        float x2 = box_ptr[2 + max_conf_anchor*4];
        float y2 = box_ptr[3 + max_conf_anchor*4];
    }
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(na);
    scores.reserve(na);
    class_ids.reserve(na);
    
    for (int64_t j = 0; j < na; ++j) {
        float x1 = box_ptr[0 + j*4];
        float y1 = box_ptr[1 + j*4];
        float x2 = box_ptr[2 + j*4];
        float y2 = box_ptr[3 + j*4];
        for (int64_t c = 0; c < nc; ++c) {
            float conf = cls_ptr[c + j*nc];
            if (conf < conf_thres) continue;
            boxes.push_back({x1, y1, x2, y2});
            scores.push_back(conf);
            class_ids.push_back((int)c);
        }
    }
    if (boxes.empty()) {
        return {};
    }
    if ((int)boxes.size() > max_nms) {
        std::vector<int> order(boxes.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + max_nms, order.end(),
                          [&](int a, int b){ return scores[a] > scores[b]; });
        order.resize(max_nms);
        std::vector<std::array<float, 4>> b2;
        std::vector<float> s2;
        std::vector<int> c2;
        b2.reserve(order.size()); s2.reserve(order.size()); c2.reserve(order.size());
        for (int idx : order) {
            b2.push_back(boxes[idx]);
            s2.push_back(scores[idx]);
            c2.push_back(class_ids[idx]);
        }
        boxes.swap(b2);
        scores.swap(s2);
        class_ids.swap(c2);
    }
    if (!scores.empty()) {
        float max_score = *std::max_element(scores.begin(), scores.end());
    }
    if (boxes.empty()) {
        return {};
    }
    std::vector<int> keep = nms(boxes, scores, class_ids, iou_thres, /*agnostic=*/false, max_wh);
    
    if ((int)keep.size() > max_det) keep.resize(max_det);

    std::vector<detected_obj> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        detected_obj o;
        o.x1 = boxes[idx][0]; o.y1 = boxes[idx][1];
        o.x2 = boxes[idx][2]; o.y2 = boxes[idx][3];
        o.confidence = scores[idx];
        o.class_id = class_ids[idx];
        o.class_confidence = scores[idx];
        result.push_back(o);
    }
    return result;
}

void scale_boxes(
    std::vector<detected_obj>& detections,
    i32x2 model_shape,
    i32x2 img_shape,
    float gain,
    float pad_w,
    float pad_h)
{
    if (gain == 0.0f) {
        float ratio_w = (float)model_shape[0] / (float)img_shape[0];
        float ratio_h = (float)model_shape[1] / (float)img_shape[1];
        gain = std::min(ratio_w, ratio_h);
    }

    for (auto& det : detections) {
        det.x1 = (det.x1 - pad_w) / gain;
        det.x2 = (det.x2 - pad_w) / gain;
        det.y1 = (det.y1 - pad_h) / gain;
        det.y2 = (det.y2 - pad_h) / gain;

        det.x1 = std::clamp(det.x1, 0.0f, (float)img_shape[0]);
        det.x2 = std::clamp(det.x2, 0.0f, (float)img_shape[0]);
        det.y1 = std::clamp(det.y1, 0.0f, (float)img_shape[1]);
        det.y2 = std::clamp(det.y2, 0.0f, (float)img_shape[1]);
    }

    printf("[DBG] scale_boxes: gain=%.4f, pad=(%.1f,%.1f)\n", gain, pad_w, pad_h);
}

void draw_line(uint8_t* img, int width, int height, int channels,
    int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {

int dx = abs(x2 - x1);
int dy = abs(y2 - y1);
int sx = (x1 < x2) ? 1 : -1;
int sy = (y1 < y2) ? 1 : -1;
int err = dx - dy;

while (true) {

for (int ty = -thickness/2; ty <= thickness/2; ty++) {
 for (int tx = -thickness/2; tx <= thickness/2; tx++) {
     int px = x1 + tx;
     int py = y1 + ty;
     if (px >= 0 && px < width && py >= 0 && py < height) {
         int idx = (py * width + px) * channels;
         img[idx + 0] = r;
         img[idx + 1] = g;
         img[idx + 2] = b;
     }
 }
}

if (x1 == x2 && y1 == y2) break;
int e2 = 2 * err;
if (e2 > -dy) {
 err -= dy;
 x1 += sx;
}
if (e2 < dx) {
 err += dx;
 y1 += sy;
}
}
}


void draw_rectangle(uint8_t* img, int width, int height, int channels,
         int x1, int y1, int x2, int y2,
         uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {

x1 = std::max(0, std::min(x1, width - 1));
y1 = std::max(0, std::min(y1, height - 1));
x2 = std::max(0, std::min(x2, width - 1));
y2 = std::max(0, std::min(y2, height - 1));


draw_line(img, width, height, channels, x1, y1, x2, y1, r, g, b, thickness); 
draw_line(img, width, height, channels, x2, y1, x2, y2, r, g, b, thickness); 
draw_line(img, width, height, channels, x2, y2, x1, y2, r, g, b, thickness); 
draw_line(img, width, height, channels, x1, y1, x1, y2, r, g, b, thickness); 
}


void draw_filled_rectangle(uint8_t* img, int width, int height, int channels,
               int x1, int y1, int x2, int y2,
               uint8_t r, uint8_t g, uint8_t b) {
x1 = std::max(0, std::min(x1, width - 1));
y1 = std::max(0, std::min(y1, height - 1));
x2 = std::max(0, std::min(x2, width - 1));
y2 = std::max(0, std::min(y2, height - 1));

for (int y = y1; y <= y2; y++) {
for (int x = x1; x <= x2; x++) {
 int idx = (y * width + x) * channels;
 img[idx + 0] = r;
 img[idx + 1] = g;
 img[idx + 2] = b;
}
}
}


void draw_char(uint8_t* img, int width, int height, int channels,
    int x, int y, char c, uint8_t r, uint8_t g, uint8_t b) {

static const uint8_t font_5x7[][7] = {

{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},

{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},

{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},

{0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},

{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},

{0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},

{0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},

{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},

{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},

{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},

{0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}
};

int idx = -1;
if (c >= '0' && c <= '9') idx = c - '0';
else if (c == '%') idx = 10;
else return; 

for (int row = 0; row < 7; row++) {
uint8_t pattern = font_5x7[idx][row];
for (int col = 0; col < 5; col++) {
 if (pattern & (1 << (4 - col))) {
     int px = x + col;
     int py = y + row;
     if (px >= 0 && px < width && py >= 0 && py < height) {
         int pidx = (py * width + px) * channels;
         img[pidx + 0] = r;
         img[pidx + 1] = g;
         img[pidx + 2] = b;
     }
 }
}
}
}


void draw_text(uint8_t* img, int width, int height, int channels,
    int x, int y, const char* text,
    uint8_t r, uint8_t g, uint8_t b) {
int offset = 0;
while (*text) {
draw_char(img, width, height, channels, x + offset, y, *text, r, g, b);
offset += 6; 
text++;
}
}


void get_class_color(int class_id, uint8_t& r, uint8_t& g, uint8_t& b) {

int h = (class_id * 137) % 360; 
float s = 0.8f;
float v = 0.95f;


float c = v * s;
float x = c * (1 - std::fabs(std::fmod(h / 60.0f, 2) - 1));
float m = v - c;

float r1, g1, b1;
if (h < 60) { r1 = c; g1 = x; b1 = 0; }
else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
else { r1 = c; g1 = 0; b1 = x; }

r = (uint8_t)((r1 + m) * 255);
g = (uint8_t)((g1 + m) * 255);
b = (uint8_t)((b1 + m) * 255);
}

void draw_detections(
    image_data& img,
    std::vector<detected_obj> const& detections,
    std::vector<std::string> const& class_names) {
    
    
    int width = img.extent[0];
    int height = img.extent[1];
    int channels = 3;
    
    
    uint8_t* img_data = nullptr;
    
    
    if (img.format == image_format::rgb_u8 || img.format == image_format::rgba_u8) {
        img_data = img.data.get();
        channels = (img.format == image_format::rgba_u8) ? 4 : 3;
    } else if (img.format == image_format::rgb_f32 || img.format == image_format::rgba_f32) {
        
        
        channels = (img.format == image_format::rgba_f32) ? 4 : 3;
        
        
        image_data temp_u8 = image_f32_to_u8(img, 
            channels == 4 ? image_format::rgba_u8 : image_format::rgb_u8);
        
        
        img_data = temp_u8.data.get();
        
        
        for (auto const& det : detections) {
            int x1 = (int)det.x1;
            int y1 = (int)det.y1;
            int x2 = (int)det.x2;
            int y2 = (int)det.y2;

            uint8_t r, g, b;
            get_class_color(det.class_id, r, g, b);
            draw_rectangle(img_data, width, height, channels, x1, y1, x2, y2, r, g, b, 3);

            const char* class_name = (det.class_id >= 0 && (size_t)det.class_id < class_names.size()) 
                                    ? class_names[det.class_id].c_str() 
                                    : "unknown";
            
            char conf_str[16];
            snprintf(conf_str, sizeof(conf_str), "%d%%", (int)(det.class_confidence * 100));
            
            printf("Detection: %s %s at [%d, %d, %d, %d] (obj_conf: %.2f, cls_conf: %.2f)\n", 
                   class_name, conf_str, x1, y1, x2, y2, 
                   det.confidence, det.class_confidence);

            int text_width = strlen(conf_str) * 6;
            int text_height = 10;
            int label_y = std::max(0, y1 - text_height - 2);
            
            draw_filled_rectangle(img_data, width, height, channels,
                                x1, label_y, x1 + text_width, label_y + text_height,
                                r, g, b);
            draw_text(img_data, width, height, channels,
                     x1 + 2, label_y + 2, conf_str, 255, 255, 255);
        }
        
        
        img = std::move(temp_u8);
        return;
    } else {
        printf("Error: Unsupported image format for drawing\n");
        return;
    }

    
    for (auto const& det : detections) {
        int x1 = (int)det.x1;
        int y1 = (int)det.y1;
        int x2 = (int)det.x2;
        int y2 = (int)det.y2;

        uint8_t r, g, b;
        get_class_color(det.class_id, r, g, b);
        draw_rectangle(img_data, width, height, channels, x1, y1, x2, y2, r, g, b, 3);

        const char* class_name = (det.class_id >= 0 && (size_t)det.class_id < class_names.size()) 
                                ? class_names[det.class_id].c_str() 
                                : "unknown";
        
        char conf_str[16];
        snprintf(conf_str, sizeof(conf_str), "%d%%", (int)(det.class_confidence * 100));
        
        printf("Detection: %s %s at [%d, %d, %d, %d] (obj_conf: %.2f, cls_conf: %.2f)\n", 
               class_name, conf_str, x1, y1, x2, y2, 
               det.confidence, det.class_confidence);

        int text_width = strlen(conf_str) * 6;
        int text_height = 10;
        int label_y = std::max(0, y1 - text_height - 2);
        
        draw_filled_rectangle(img_data, width, height, channels,
                            x1, label_y, x1 + text_width, label_y + text_height,
                            r, g, b);
        draw_text(img_data, width, height, channels,
                 x1 + 2, label_y + 2, conf_str, 255, 255, 255);
    }
}
} 

namespace visp::yolov9t {

 float resize_longest_side(i32x2 extent, int target_longest_side) {
    int longest_side = std::max(extent[0], extent[1]);
    return float(target_longest_side) / float(longest_side);
}
int scale_coord(int coord, float scale) {
    return int(coord * scale + 0.5f);
}
i32x2 scale_extent(i32x2 extent, float scale) {
    return i32x2{scale_coord(extent[0], scale), scale_coord(extent[1], scale)};
}
image_data yolov9t_process_input2(image_view image, yolov9t_params const& p) {
    printf("extent[0]=%d, extent[1]=%d\n", image.extent[0], image.extent[1]);
    
    image_data resized;
    if (image.extent[0] != p.input_size || image.extent[1] != p.input_size) {
        resized = image_scale(image, {p.input_size, p.input_size});
        image = image_view(resized);
        

    }

    return image_u8_to_f32(image, image_format::rgb_f32, p.offset, 1.f / p.scale);
}

LetterboxResult yolov9t_process_input(image_data image, yolov9t_params const& p) {
    printf("[DBG] before letterbox: format=%d (0=rgb_u8, 1=bgr_u8?, 2=rgba_u8, 3=rgb_f32)\n", (int)image.format);

    
    LetterboxResult lb = letterbox(
        std::move(image),
        {p.input_size, p.input_size},
        u8x3{114,114,114},
        /*_auto=*/false, /*scaleFill=*/false, /*scaleup=*/true,
        yolov9t_params::stride);

    image_data result;

    if (lb.img.format == image_format::rgb_u8) {
        result = image_alloc({p.input_size, p.input_size}, image_format::rgb_f32);
        image_u8_to_f32(lb.img, result, f32x4(p.offset), f32x4(p.scale));
    } else {
        result = std::move(lb.img);
    }

    LetterboxResult out;
    out.img = std::move(result);
    out.gain = lb.gain;
    out.pad_w = lb.pad_w;
    out.pad_h = lb.pad_h;

    printf("[DBG] after yolov9t_process_input: format=%d, extent=(%d,%d)\n",
           (int)out.img.format, out.img.extent[0], out.img.extent[1]);

    return out;
}

static void write_tensor_txt(tensor t, char const* filename) {
    if (!t) {
        return;
    }
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Failed to open %s for writing\n", filename);
        return;
    }
    
    fprintf(
        f,
        "# %ld,%ld,%ld,%ld\n# %s\n",
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3], ggml_type_name(t->type));

    
    tensor_data td = transfer_from_backend(t);
    size_t n = ggml_nelements(t);

    if (t->type == GGML_TYPE_F32) {
        auto data = td.as_f32();
        for (size_t i = 0; i < data.size(); ++i) {
            fprintf(f, i + 1 == data.size() ? "%.4f\n" : "%.4f ", (double)data[i]);
        }
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* h = reinterpret_cast<const ggml_fp16_t*>(td.data.get());
        for (size_t i = 0; i < n; ++i) {
            float v = ggml_fp16_to_fp32(h[i]);
            fprintf(f, i + 1 == n ? "%.4f\n" : "%.4f ", (double)v);
        }
    } else if (t->type == GGML_TYPE_I32) {
        auto data = td.as_i32();
        for (size_t i = 0; i < data.size(); ++i) {
            fprintf(f, i + 1 == data.size() ? "%d\n" : "%d ", data[i]);
        }
    } else {
        
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(td.data.get());
        size_t nb = ggml_nbytes(t);
        for (size_t i = 0; i < nb; ++i) {
            fprintf(f, i + 1 == nb ? "%02X\n" : "%02X ", bytes[i]);
        }
    }
    fclose(f);
}

void save_features_to_txt(DetectOutput const& out, char const* base_path, std::vector<int> const& keys) {
    if (!base_path) {
        base_path = "features";
    }
    
    if (!keys.empty()) {
        for (int k : keys) {
            auto it = out.features_map.find(k);
            if (it == out.features_map.end()) continue;
            std::string fname = std::string(base_path) + "_layer_" + std::to_string(k) + ".txt";
            write_tensor_txt(it->second, fname.c_str());
            printf("-> feature layer %d saved to %s\n", k, fname.c_str());
        }
        return;
    }
    for (auto const& [k, t] : out.features_map) {
        std::string fname = std::string(base_path) + "_layer_" + std::to_string(k) + ".txt";
        write_tensor_txt(t, fname.c_str());
        printf("-> feature layer %d saved to %s\n", k, fname.c_str());
    }
}

} 

namespace visp::yolov9t {

void save_input_to_txt(tensor input, char const* filepath) {
    if (!input || !filepath) return;
    FILE* f = fopen(filepath, "w");
    if (!f) {
        printf("Failed to open %s for writing\n", filepath);
        return;
    }
    fprintf(f, "# input shape C,H,W,N = %ld,%ld,%ld,%ld\n# type = %s\n",
            (long)input->ne[0], (long)input->ne[1], (long)input->ne[2], (long)input->ne[3], ggml_type_name(input->type));
    
    tensor_data td = transfer_from_backend(input);
    if (input->type == GGML_TYPE_F32) {
        auto data = td.as_f32();
        for (size_t i = 0; i < data.size(); ++i) {
            fprintf(f, i + 1 == data.size() ? "%g\n" : "%g ", (double)data[i]);
        }
    } else if (input->type == GGML_TYPE_F16) {
        const ggml_fp16_t* h = reinterpret_cast<const ggml_fp16_t*>(td.data.get());
        size_t n = ggml_nelements(input);
        for (size_t i = 0; i < n; ++i) {
            float v = ggml_fp16_to_fp32(h[i]);
            fprintf(f, i + 1 == n ? "%g\n" : "%g ", (double)v);
        }
    } else if (input->type == GGML_TYPE_I32) {
        auto data = td.as_f32();
        for (size_t i = 0; i < data.size(); ++i) {
            fprintf(f, i + 1 == data.size() ? "%g\n" : "%g ", data[i]);
        }
    } else {
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(td.data.get());
        size_t nb = ggml_nbytes(input);
        for (size_t i = 0; i < nb; ++i) {
            fprintf(f, i + 1 == nb ? "%02X\n" : "%02X ", bytes[i]);
        }
    }
    fclose(f);
}

} 