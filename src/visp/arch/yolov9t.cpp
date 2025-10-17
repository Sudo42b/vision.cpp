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

// Parameter detection from model weights
yolov9t_params yolov9t_detect_params(model_file const& /*file*/) {
    yolov9t_params params;
    // Use default parameters since metadata functions aren't available
    // In actual implementation, these would be extracted from GGUF metadata
    params.num_classes = 80;
    params.input_size = 640;
    params.variant = "tiny";
    // Normalize inputs to [0, 1]

    return params;
}
int autopad(int k, int p = -1, int d = 1) {
    if (d > 1) {
        k = d * (k - 1) + 1;
    }
    if (p == -1) { // p is None in Python
        p = k / 2;
    }
    return p;
}

// Conv: Standard convolution with BN and SiLU (matching Python Conv class)
// Parameters: c1(input_ch), c2(output_ch), k(kernel), s(stride), p(padding), g(groups),
// d(dilation), act(activation)
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
    
    // Calculate padding using autopad if p == -1
    int auto_pad = autopad(k, p, 1);

    // Save current prefix
    tensor_name old_prefix = m.prefix;

    // Set prefix to the layer name so conv_2d can find "weight" and "bias"
    std::string weight_name = name;
    // printf("Weight name: %s\n", weight_name.c_str());
    // printf("Name: %s\n", name.c_str());
    if (name.find(".conv") == std::string::npos) {
        weight_name += ".conv";
    }
    m.prefix = tensor_name(weight_name.c_str());
    tensor weight = m.weights("weight");
    
    if (debug) {
        printf("Weight shape: [%d,%d,%d,%d]\n", (int)weight->ne[0], (int)weight->ne[1], (int)weight->ne[2], (int)weight->ne[3]);
        printf("Input shape: [%d,%d,%d,%d]\n", (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }

    // Get input channels from tensor
    // c1 = (int)x->ne[0]; // input channels
    // c2 = (int)weight->ne[3]; // output channels in GGML format
    
    if (debug) {
        printf(
            "Conv: c1=%d, c2=%d, k=%d, s=%d, p=%d, g=1, d=1, act=%s\n", c1, c2, k, s, auto_pad,
            act ? "True" : "False");
    }

    // Check if input is in CWHN format and needs conversion to WHCN
        
    // Ensure tensor is contiguous before convolution
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(m, x);
    }
    
    // Apply convolution: ggml_conv_2d expects (weight, input)
    x = conv_2d(m, x, s, auto_pad, 1);

    // Apply BatchNorm2d (if bias exists, it's typically the bn bias)
    // // if (m.find("bias")) {
    //     tensor bias = m.weights("bias");
    //     x = ggml_add(m, x, bias);
    // // }
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
    // Apply SiLU activation if act=true
    if (act) {
        x = ggml_silu(m, x);
    }

    // Restore prefix
    m.prefix = old_prefix;
    
    if (debug) {
        printf(
            "Conv Out: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", (int)x->ne[0], (int)x->ne[1],
            (int)x->ne[2], (int)x->ne[3]);
    }
    return x;
}



// AConv: Average pooling + Conv for downsampling
// Python signature: AConv(c1, c2) where c1=input_ch, c2=output_ch
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
    return conv_2d(m, x, stride, pad); // batch_norm is fused into conv_2d at model conversion
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

    // 1) AvgPool2d(k=2, s=1, p=0)  — 파이토치: nn.AvgPool2d(2, 1, 0, False, True)
    //    CWHN에서 풀링은 W,H 축(ne[1], ne[2])에 적용됨.
    printf("Before Pooling (C,W,H,N) = [%d,%d,%d,%d]\n",
           (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    x = contiguous_2d_to_whcn(m, x);
    // permute
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

    // 2) Conv 3x3, stride=2, pad=1  — PyTorch: Conv(c1, c2, 3, 2, 1)
    //    입력 채널 c1 → 출력 채널 c2
    tensor out = Conv(m, pooled, name + ".cv1.conv", c1, c2, /*k=*/3, /*stride=*/2, /*pad=*/1, /*bias=*/true);
    printf("%ld, %ld, %ld, %ld\n", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    if (debug) {
        printf("AConv Out  (C,W,H,N) = [%d,%d,%d,%d]\n",
               (int)out->ne[0], (int)out->ne[1], (int)out->ne[2], (int)out->ne[3]);
    }
    return out;
}

// ELAN1: Efficient Layer Aggregation Network block
// Python signature: ELAN1(c1, c2, c3, c4) where c1=input_ch, c2=output_ch, c3=cv2_ch, c4=cv3_ch
/*
class ELAN1(RepNCSPELAN4):
    """ELAN-1 block"""
    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ELAN1 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        x = self.cv4(torch.cat(y, 1))
        return x
*/
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
    // x: CWHN
    if (debug) {
        printf("ELAN1 %s: x (C,W,H,N) = [%d,%d,%d,%d]\n",
               name.c_str(), (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }

    // cv1: 1x1 conv -> c3 channels
    GGML_ASSERT(c3 % 2 == 0 && "ELAN1: c3 must be even for chunk(2)");
    const int c_half = c3 / 2;
    tensor cv1_out = Conv(m, x, name + ".cv1.conv", c1, c3, 1, 1, -1, true);

    // 채널을 반으로 분할 (PyTorch chunk(2, dim=1)와 동일; 여기선 dim0=C)
    // slice_t는 [begin, end) 규칙. step은 채널축에서 1만 허용(헬퍼가 그렇게 설계됨).
    tensor split1 = slice(m, cv1_out,
                          /*C*/ {0,       c_half},
                          /*W*/{}, /*H*/ {}, /*N*/ {});
    tensor split2 = slice(m, cv1_out,
                          /*C*/ {c_half, c3},
                          /*W*/ {}, /*H*/ {}, /*N*/ {});

    // 필요 시 연속화 (Conv가 비연속 view 입력을 못 받는 경우 대비)
    if (!ggml_is_contiguous(split1)) split1 = ggml_cont(m, split1);
    if (!ggml_is_contiguous(split2)) split2 = ggml_cont(m, split2);

    // cv2: 3x3 conv on split2 (c_half -> c4)
    tensor cv2_out = Conv(m, split2, name + ".cv2.conv", c_half, c4, 3, 1, -1, true);

    // cv3: 3x3 conv on cv2_out (c4 -> c4)
    tensor cv3_out = Conv(m, cv2_out, name + ".cv3.conv", c4, c4, 3, 1, -1, true);

    // concat: [split1, split2, cv2_out, cv3_out] along channel dim (dim=0 for CWHN)
    tensor cat12   = Concat(m, split1, split2, /*dim=*/0);
    tensor cat123  = Concat(m, cat12,  cv2_out, /*dim=*/0);
    tensor cat1234 = Concat(m, cat123, cv3_out, /*dim=*/0);

    if (debug) {
        printf("ELAN1 %s: cat (C,W,H,N) = [%d,%d,%d,%d]\n",
               name.c_str(), (int)cat1234->ne[0], (int)cat1234->ne[1], (int)cat1234->ne[2], (int)cat1234->ne[3]);
    }

    // cv4: 1x1 conv to final output channels: (c3 + 2*c4) -> c2
    tensor out = Conv(m, cat1234, name + ".cv4.conv", c3 + (2 * c4), c2, 1, 1, -1, true);

    // 레이아웃 변환 불필요 (네트워크 전반 CWHN 유지). 항등 permute/contiguous 강제 없음.
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
    int c_ = (int)(c2 * e); // hidden channels

    // cv1: 1x1 conv
    tensor cv1_out = Conv(m, x, name + ".cv1.conv", c1, c_, 1, 1, -1, true);

    // cv2: 3x3 conv
    tensor cv2_out = Conv(m, cv1_out, name + ".cv2.conv", c_, c2, 3, 1, -1, true);

    // Add shortcut if applicable
    // if (shortcut && x->ne[2] == cv2_out->ne[2]) {
    //     return ggml_add(m, x, cv2_out);
    // }
    if (debug) {
        printf(
            "Bottleneck %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)cv2_out->ne[3],
            (int)cv2_out->ne[2], (int)cv2_out->ne[1], (int)cv2_out->ne[0]);
    }
    return cv2_out;
}
// RepConv: Reparameterizable Convolution with training and deploy modes
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
    
    // Assert k == 3 and p == 1 as in Python implementation
    if (k != 3 || p != 1) {
        throw std::invalid_argument("RepConv requires k=3 and p=1");
    }
    
    tensor output;
    
    if (deploy) {
        // In deploy mode, use fused convolution (would be pre-computed)
        output = Conv(m, x, name + ".fused_conv", c1, c2, k, s, p, act, debug);
    } else {
        // Training mode: PyTorch structure with conv1 (3x3) + conv2 (1x1)
        
        std::string cv11_name = name;
        // suffix가 .cv1이면 .cv1.conv1로 교체
        tensor conv1_out = Conv(m, x, cv11_name + ".conv1.conv", c1, c2, k, s, p, false, debug);
        
        
        tensor conv2_out = Conv(m, x, cv11_name + ".conv2.conv", c1, c2, 1, s, (p-k)/2, false, debug);  // stride, pad=1, dilate=1
        
        // id_out
        if (bn){
            printf("batchnorm 처리해야함\n");
        }
        // Add conv1 + conv2
        output = ggml_add(m, conv1_out, conv2_out);
        
        // Apply activation
        if (act) {
            output = ggml_silu(m, output); // SiLU is default activation
        }
    }
    
    if (debug) {
        printf(
            "RepConv %s Out: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n", name.c_str(), 
            (int)output->ne[0], (int)output->ne[1], (int)output->ne[2], (int)output->ne[3]);
    }
    
    return output;
}

// RepBottleneck: Bottleneck with RepConv (inherits from Bottleneck behavior)
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
    
    int c_ = (int)(c2 * e); // hidden channels
    
    // cv1: RepConv(c1, c_, 3, 1, 1) - RepConv with k=3, s=1, p=1
    tensor cv1_out = RepConv(m, x, name + ".cv1", c1, c_, 3, 1, 1, 1, 1, true, false, false, debug);
    
    // cv2: Conv(c_, c2, 3, 1) - Regular Conv  
    tensor cv2_out = Conv(m, cv1_out, name + ".cv2", c_, c2, 3, 1, -1, true, debug);
    
    // Add shortcut if applicable (same as original Bottleneck)
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
// C3: CSP Bottleneck with 3 convolutions
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
    int c_ = (int)(c2 * e); // hidden channels
    
    // cv1: 1x1 conv (c1 -> c_)
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c_, 1, 1, -1, true);

    // cv2: 1x1 conv (c1 -> c_)
    tensor cv2_out = Conv(m, x, name + ".cv2", c1, c_, 1, 1, -1, true);

    // Process cv1_out through n Bottleneck blocks
    tensor m_out = cv1_out;
    for (int i = 0; i < n; ++i) {
        std::string bottleneck_name = name + ".m." + std::to_string(i);
        m_out = Bottleneck(m, m_out, bottleneck_name, c_, c_, shortcut, g, 1.0);
    }

    // Concatenate m_out and cv2_out (2 * c_ channels)
    tensor concat = ggml_concat(m, m_out, cv2_out, 0); // Channel dimension concat
    concat = ggml_cont(m, concat); // Ensure contiguous memory
    // cv3: 1x1 conv (2*c_ -> c2)
    tensor output = Conv(m, concat, name + ".cv3", 2 * c_, c2, 1, 1, -1, true);
    
    if (debug) {
        printf(
            "C3 %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }
    return output;
}

// RepCSP: Repeatable Cross Stage Partial Network (inherits from C3 behavior)
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
    
    int c_ = (int)(c2 * e); // hidden channels
    
    // cv1: 1x1 conv (c1 -> c_)
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c_, 1, 1, -1, true);

    // cv2: 1x1 conv (c1 -> c_)
    tensor cv2_out = Conv(m, x, name + ".cv2", c1, c_, 1, 1, -1, true);

    // Process cv1_out through n RepBottleneck blocks (key difference from C3)
    tensor m_out = cv1_out;
    for (int i = 0; i < n; ++i) {
        std::string bottleneck_name = name + ".m." + std::to_string(i);
        m_out = RepBottleneck(m, m_out, bottleneck_name, c_, c_, shortcut, g, 3, 1.0);
    }

    // Concatenate m_out and cv2_out (2 * c_ channels)
    tensor concat = ggml_concat(m, m_out, cv2_out, 0); // Channel dimension concat
    concat = ggml_cont(m, concat); // Ensure contiguous memory
    // cv3: 1x1 conv (2*c_ -> c2)
    tensor output = Conv(m, concat, name + ".cv3", 2 * c_, c2, 1, 1, -1, true);
    
    if (debug) {
        printf(
            "RepCSP %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }
    return output;
}
// RepNCSPELAN4: RepNCSP + ELAN structure
// Python signature: RepNCSPELAN4(c1, c2, c3, c4, n) where c1=input_ch, c2=output_ch,
// c3=hidden_ch, c4=branch_ch, n=num_blocks
tensor RepNCSPELAN4(
    model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int c4, int n, bool debug) {
    
    if (debug) {
        printf(
            "RepNCSPELAN4 %s In : x shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3],
            (int)x->ne[2], (int)x->ne[1], (int)x->ne[0]);
    }
    int c = c3 / 2;
    printf("Name: %s, c1=%d, c2=%d, c3=%d, c4=%d, n=%d\n", name.c_str(), c1, c2, c3, c4, n);
    // cv1: 1x1 conv (c1 -> c3), stride=1, pad=0
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c3, /*k=*/1, /*stride=*/1, /*pad=*/0, /*bias=*/true);

    // 채널을 반으로 분할: y1 = [0:c_half), y2_input = [c_half:c3)
    tensor cv1_out_h0 = slice(m, cv1_out,
                      /*C*/ slice_t(0,       c),
                      /*W*/ slice_t(), /*H*/ slice_t(), /*N*/ slice_t());
    tensor cv1_out_h1 = slice(m, cv1_out,
                            /*C*/ slice_t(c, c3),
                            /*W*/ slice_t(), /*H*/ slice_t(), /*N*/ slice_t());
    

    // Conv가 비연속 view를 받지 못할 수 있으니 안전하게 연속화
    if (!ggml_is_contiguous(cv1_out_h0)) cv1_out_h0 = ggml_cont(m, cv1_out_h0);
    if (!ggml_is_contiguous(cv1_out_h1)) cv1_out_h1 = ggml_cont(m, cv1_out_h1);

    auto rep_conv = [=](model_ref m, tensor x, std::string name, int c_in, int c_out, int n) {
        tensor y2 = RepCSP(m, x, name+ ".0", c_in, c_out, n);
        return Conv(m, y2, name + ".1", c_out, c_out, 3, 1, 1, true);
    };
    
    // y2 = RepCSP(m, y2, name + ".cv3.0", c4, c4, n);
    // tensor y3 = Conv(m, y2, name + ".cv3.1", c4, c4, 3, 1, 1, true);
    // cv2: RepCSP(c3//2 -> c4) + Conv(c4 -> c4)
    tensor cv2 = rep_conv(m, cv1_out_h1, name + ".cv2", c3 / 2, c4, n);
    // cv3: RepCSP(c4 -> c4) + Conv(c4 -> c4)
    tensor cv3 = rep_conv(m, cv2, name + ".cv3", c4, c4, n);
    
    // Use dimension 0 for channel concat in CWHN
    // tensor cat = ggml_concat(m, y1, y2_input, 0);
    // cat = ggml_concat(m, cat, y2, 0);
    // cat = ggml_concat(m, cat, y3, 0);
    tensor cat = concat(m, {y1, y2_input, cv2, cv3}, 0);
    if (!ggml_is_contiguous(cat))
    {
        printf("Making concatenated tensor contiguous for RepNCSPELAN4 %s\n", name.c_str());
        cat = ggml_cont(m, cat); // Ensure contiguous memory
    }
    
    // cv4: 1x1 conv (c + 2*c4 -> c2)
    tensor output = Conv(m, cat, name + ".cv4", c3 + (2 * c4), c2, 1, 1, -1, true);
    if (debug) {
        printf(
            "RepNCSPELAN4 %s: output shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }

    return output;
}

// SPPELAN: Spatial Pyramid Pooling - ELAN
// Python signature: SPPELAN(c1, c2, c3) where c1=input_ch, c2=output_ch, c3=hidden_ch
tensor SPPELAN(model_ref m, tensor x, std::string const& name, int c1, int c2, int c3, int k, bool debug) {
    if (debug) {
        printf(
            "SPPELAN %s: x shape [%d,%d,%d,%d]\n", name.c_str(), (int)x->ne[3],
            (int)x->ne[2], (int)x->ne[1], (int)x->ne[0]);
    }
    // cv1: 1x1 conv
    tensor cv1_out = Conv(m, x, name + ".cv1", c1, c3, 1, 1, -1, true);
    // 디버깅: cv1_out shape 확인
    if (debug) {
        printf("cv1_out: [%ld, %ld, %ld, %ld]\n", 
               cv1_out->ne[0], cv1_out->ne[1], cv1_out->ne[2], cv1_out->ne[3]);
    }

    // Three MaxPool layers with same kernel size (k=5 for YOLOv9t)
    int pad = k / 2;
    tensor cv2 = ggml_pool_2d(m, cv1_out, GGML_OP_POOL_MAX, k, k, 1, 1, pad, pad);
    cv2 = ggml_cont(m, cv2); // Ensure contiguous memory
    if (debug) {
        printf("cv2 (after pool1): [%ld, %ld, %ld, %ld]\n", 
               cv2->ne[0], cv2->ne[1], cv2->ne[2], cv2->ne[3]);
    }
    tensor cv3 = ggml_pool_2d(m, cv2, GGML_OP_POOL_MAX, k, k, 1, 1, pad, pad);
    cv3 = ggml_cont(m, cv3); // Ensure contiguous memory
    if (debug) {
        printf("cv3 (after pool2): [%ld, %ld, %ld, %ld]\n", 
               cv3->ne[0], cv3->ne[1], cv3->ne[2], cv3->ne[3]);
    }
    tensor cv4 = ggml_pool_2d(m, cv3, GGML_OP_POOL_MAX, k, k, 1, 1, pad, pad);
    cv4 = ggml_cont(m, cv4); // Ensure contiguous memory
    if (debug) {
        printf("cv4 (after pool3): [%ld, %ld, %ld, %ld]\n", 
               cv4->ne[0], cv4->ne[1], cv4->ne[2], cv4->ne[3]);
    }

    // Ensure all tensors are contiguous before concatenation
    // if (!ggml_is_contiguous(cv1_out)) {
    //     cv1_out = ggml_cont(m, cv1_out);
    // }
    // if (!ggml_is_contiguous(cv2)) {
    //     cv2 = ggml_cont(m, cv2);
    // }
    // if (!ggml_is_contiguous(cv3)) {
    //     cv3 = ggml_cont(m, cv3);
    // }
    // if (!ggml_is_contiguous(cv4)) {
    //     cv4 = ggml_cont(m, cv4);
    // }
    // tensor tensors[4] = { cv1_out, cv2, cv3, cv4 };
    tensor y = ggml_concat(m, cv1_out, cv2, 0); // Channel dimension concat in CWHN
    y = ggml_concat(m, y, cv3, 0);
    y = ggml_concat(m, y, cv4, 0);
    // printf("After concat: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n",
        //    (int)y->ne[0], (int)y->ne[1], (int)y->ne[2], (int)y->ne[3]);
    // y = contiguous_2d_to_cwhn(m, y);
    // printf("After conti: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n",
    //        (int)y->ne[0], (int)y->ne[1], (int)y->ne[2], (int)y->ne[3]);
    // if (!ggml_is_contiguous(y)){
    //     y = ggml_cont(m, y); // Ensure contiguous memory
    // }
    tensor output = Conv(m, y, name + ".cv5", 4*c3, c2, 1, 1, -1, true);
    if (debug){
        printf(
            "SPPELAN %s: Out shape [%d,%d,%d,%d]\n", name.c_str(), (int)output->ne[3],
            (int)output->ne[2], (int)output->ne[1], (int)output->ne[0]);
    }
    
    return output;
}

// Upsample: Nearest neighbor upsampling
tensor Upsample(model_ref m, tensor x, int scale_factor, bool debug) {
    // NHWC to WHNC
    x = permute_cwhn_to_whcn(m, x);
    x = ggml_upscale(m, x, scale_factor, GGML_SCALE_MODE_NEAREST);
    x = permute_whcn_to_cwhn(m, x);
    x = ggml_cont(m, x);
    return  x; // Ensure contiguous memory
}
// Concatenate tensors along channel dimension
tensor Concat(model_ref m, tensor a, tensor b, int axis, bool debug) {
    // In WHCN: channels are at ne[2]
    // In CWHN: channels are at ne[0]
    int dim = (m.flags & model_build_flag::cwhn) ? 0 : 2;
    tensor output = ggml_concat(m, a, b, dim);
    // output = ggml_permute(m, output, 2, 1, 0, 3); // NHWC -> WHNC
    output = ggml_cont(m, output); // Ensure contiguous memory
    return output;
}

// YOLOv9t backbone implementation (matching Python structure)
// std::vector<tensor> yolov9t_backbone(model_ref m, tensor x) {
std::map<int, tensor> yolov9t_backbone(model_ref m, tensor x) {
    std::map<int, tensor> features;
    // Layer 0: Conv(3, 16, 3, 2) - P1/2
    tensor x0 = Conv(m, x, "model.0", 3, 16, 3, 2, -1, true, false, true);
    features[0] = x0;
    ggml_set_output(x0);

    // Layer 1: Conv(16, 32, 3, 2) - P2/4
    tensor x1 = Conv(m, x0, "model.1", 16, 32, 3, 2, -1, true, false, true);
    features[1] = x1;
    ggml_set_output(x1);
    
    // Layer 2: ELAN1(32, 32, 32, 16)
    tensor x2 = ELAN1(m, x1, "model.2", 32, 32, 32, 16, false);
    ggml_set_output(x2);
    features[2] = x2;
    
    // Layer 3: AConv(32, 64) - P3/8
    tensor x3 = AConv(m, x2, "model.3", 32, 64, false);
    ggml_set_output(x3);
    features[3] = x3;

    // Layer 4: RepNCSPELAN4(64, 64, 64, 32, 3)
    tensor x4 = RepNCSPELAN4(m, x3, "model.4", 64, 64, 64, 32, 3, false);
    features[4] = x4;
    ggml_set_output(x4);
    
    // Layer 5: AConv(64, 96) - P4/16
    tensor x5 = AConv(m, x4, "model.5", 64, 96);
    features[5] = x5;
    ggml_set_output(x5);
    
    // Layer 6: RepNCSPELAN4(96, 96, 96, 48, 3)
    tensor x6 = RepNCSPELAN4(m, x5, "model.6", 96, 96, 96, 48, 3);
    features[6] = x6;
    ggml_set_output(x6);

    // Layer 7: AConv(96, 128) - P5/32
    tensor x7 = AConv(m, x6, "model.7", 96, 128);
    features[7] = x7;
    ggml_set_output(x7);

    // Layer 8: RepNCSPELAN4(128, 128, 128, 64, 3)
    tensor x8 = RepNCSPELAN4(m, x7, "model.8", 128, 128, 128, 64, 3);
    features[8] = x8;
    ggml_set_output(x8);

    // Layer 9: SPPELAN(128, 128, 64)
    tensor x9 = SPPELAN(m, x8, "model.9", 128, 128, 64, 5, false);
    features[9] = x9;
    ggml_set_output(x9);

    // Layer 10: Upsample(None, 2, 'nearest')
    tensor x10 = Upsample(m, x9, 2);
    features[10] = x10;
    ggml_set_output(x10);

    // printf("After Upsample layer 10: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n",
    //        (int)x10->ne[0], (int)x10->ne[1], (int)x10->ne[2], (int)x10->ne[3]);
    // printf("Feature map from layer 6 (P4): ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n",
    //        (int)features[6]->ne[0], (int)features[6]->ne[1], (int)features[6]->ne[2], (int)features[6]->ne[3]);
    // Layer 11: Concat(1) - with P4 (layer 6)
    tensor x11 = Concat(m, x10, features[6], 2);
    features[11] = x11;
    ggml_set_output(x11);

    // printf("After Concat layer 11: ne[0]=%d, ne[1]=%d, ne[2]=%d, ne[3]=%d\n",
    //        (int)x11->ne[0], (int)x11->ne[1], (int)x11->ne[2], (int)x11->ne[3]);
    // Layer 12: RepNCSPELAN4(224, 96, 96, 48, 3)
    tensor x12 = RepNCSPELAN4(m, x11, "model.12", 224, 96, 96, 48, 3, false);
    features[12] = x12;
    ggml_set_output(x12);

    // Layer 13: Upsample(None, 2, 'nearest')
    tensor x13 = Upsample(m, x12, 2);
    features[13] = x13;
    ggml_set_output(x13);

    // Layer 14: Concat(1) - with P3 (layer 4)
    tensor x14 = Concat(m, x13, features[4], 2);
    features[14] = x14;
    ggml_set_output(x14);

    // Layer 15: RepNCSPELAN4(160, 64, 64, 32, 3) - N3 output
    tensor x15 = RepNCSPELAN4(m, x14, "model.15", 160, 64, 64, 32, 3);
    features[15] = x15;
    ggml_set_output(x15);

    // Layer 16: AConv(64, 48)
    tensor x16 = AConv(m, x15, "model.16", 64, 48);
    features[16] = x16;
    ggml_set_output(x16);

    // Layer 17: Concat(1) - with P4 (layer 12)
    tensor x17 = Concat(m, x16, features[12]);
    features[17] = x17;
    ggml_set_output(x17);

    // Layer 18: RepNCSPELAN4(144, 96, 96, 48, 3) - N4 output
    tensor x18 = RepNCSPELAN4(m, x17, "model.18", 144, 96, 96, 48, 3);
    features[18] = x18;
    ggml_set_output(x18);

    // Layer 19: AConv(96, 64)
    tensor x19 = AConv(m, x18, "model.19", 96, 64);
    features[19] = x19;
    ggml_set_output(x19);

    // Layer 20: Concat(1) - with P5 (layer 9)
    tensor x20 = Concat(m, x19, features[9]);
    features[20] = x20;
    ggml_set_output(x20);

    // Layer 21: RepNCSPELAN4(192, 128, 128, 64, 3) - N5 output
    tensor x21 = RepNCSPELAN4(m, x20, "model.21", 192, 128, 128, 64, 3);
    features[21] = x21;
    ggml_set_output(x21);
    
    // Return detection outputs: N3(x15), N4(x18), N5(x21)
    // return {x15, x18, x21};
    return features;
}
/*

*/


// PyTorch 코드를 참고해서 GGML로 변환한 dist2bbox 구현
// PyTorch 코드:
// def dist2bbox(distance:Tensor, anchor_points:Tensor, xywh:bool=True, dim:int=-1) -> Tensor:
//     """Transform distance(ltrb) to box(xywh or xyxy)."""
//     lt, rb = torch.split(distance, 2, dim)
//     x1y1 = anchor_points - lt
//     x2y2 = anchor_points + rb
//     if xywh:
//         c_xy = (x1y1 + x2y2) / 2
//         wh = x2y2 - x1y1
//         return torch.concatenate((c_xy, wh), dim)  # xywh bbox
//     return torch.concatenate((x1y1, x2y2), dim)  # xyxy bbox

tensor dist2bbox(model_ref m, tensor distance, tensor anchor_points, bool xywh/*=true*/) {
    // distance: [4, N, ...], anchor_points: [2, N, ...]
    printf("distance shape: [%ld,%ld,%ld,%ld]\n", distance->ne[0], distance->ne[1], distance->ne[2], distance->ne[3]);
    
    distance = ggml_cont(m, distance);
    distance = ggml_permute(m, distance, 2, 0, 1, 3);
    
    distance = ggml_cont(m, distance);
    GGML_ASSERT(distance->ne[0] == 4);
    GGML_ASSERT(anchor_points->ne[0] == 2);

    // lt, rb = torch.split(distance, 2, dim=0)
    // lt = distance[0:2, ...], rb = distance[2:4, ...]
    tensor lt = slice(m, distance, slice_t{0, 2}, slice_t{}, slice_t{}, slice_t{});
    tensor rb = slice(m, distance, slice_t{2, 4}, slice_t{}, slice_t{}, slice_t{});
    // x1y1 = anchor_points - lt
    tensor x1y1 = ggml_sub(m, anchor_points, lt);

    // x2y2 = anchor_points + rb
    tensor x2y2 = ggml_add(m, anchor_points, rb);

    if (xywh) {
        // c_xy = (x1y1 + x2y2) / 2
        tensor c_xy = ggml_scale(m, ggml_add(m, x1y1, x2y2), 0.5f);
        // wh = x2y2 - x1y1
        tensor wh = ggml_sub(m, x2y2, x1y1);
        // return torch.concatenate((c_xy, wh), dim=0)
        return ggml_concat(m, c_xy, wh, 0);
    } else {
        // return torch.concatenate((x1y1, x2y2), dim=0)
        return ggml_concat(m, x1y1, x2y2, 0);
    }
}

// DFL (Distribution Focal Loss) layer implementation
tensor dfl_forward(model_ref m, tensor x, int reg_max, tensor& proj, bool debug) {
    // Input x: [4*reg_max, A, 1, 1] (CWHN). Compute expected value per coordinate.
    if (debug) {
        printf("x shape: [%d,%d,%d,%d]\n", (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
    }
    // x(box): [reg_max*4, num_anchors, 1, 1]
    // Split into 4 coordinates, apply softmax on reg_max bins, then weighted sum
    // Output: [4, num_anchors, 1, 1]
    
    int64_t num_anchors = x->ne[1];
    
    // Reshape: [reg_max, 4, num_anchors, 1]
    tensor reshaped = ggml_reshape_4d(m, x, reg_max, 4, num_anchors, 1);
    
    // Softmax along reg_max dimension (dim 0)
    tensor softmaxed = ggml_soft_max_ext(m, reshaped, nullptr, 1.0f, 0.0f);
    
    // Reshape provided projection weights for broadcasting: [reg_max, 1, 1, 1]
    proj = ggml_reshape_4d(m, proj, reg_max, 1, 1, 1);
    
    // Weighted sum: element-wise multiply and sum along dim 0
    tensor weighted = ggml_mul(m, softmaxed, proj);
    
    // Sum along reg_max dimension
    tensor result = ggml_sum_rows(m, weighted);
    
    // Result: [4, num_anchors, 1, 1]
    return result;
}
//(m, out.raw_outputs, ch, reg_max, nc);
DetectOutput inference(model_ref m, 
                DetectOutput out, // List of feature maps from different detection layers.
                std::vector<int> ch, 
                int reg_max, 
                int nc) {

    std::vector<tensor> reshaped_outputs;
    // Concat along anchor dimension (dim 1)
    tensor x_cat = concat(m, out.features, 1); // [144, 8400, 1]
    printf("x_cat shape: [%ld,%ld,%ld, %ld]\n", x_cat->ne[0], x_cat->ne[1], x_cat->ne[2], x_cat->ne[3]);

    // anchors, strides shape is torch.Size([2, 8400]) torch.Size([1, 8400])
    int64_t total_channels = x_cat->ne[0];
    int64_t total_anchors = x_cat->ne[1];
    
    // printf("After concat: channels=%ld, anchors=%ld\n", total_channels, total_anchors);
    // Done!
    // === Generate anchors inline ===
    std::vector<float> strides_vec;
    for (size_t i = 0; i < out.features.size(); ++i) {
        strides_vec.push_back(8.0f * std::pow(2.0f, (float)i));
    }
    std::vector<float> anchor_host;
    std::vector<float> stride_host;
    auto [anchor_points, stride_tensor] = make_anchors(m, out, anchor_host, stride_host, strides_vec, 0.5f);

    // Split into box and cls
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
    
    // Prepare DFL projection tensor and host data
    tensor proj = ggml_new_tensor_1d(m.graph_context, GGML_TYPE_F32, reg_max);
    ggml_set_name(proj, "dfl_proj");
    out.dfl_proj = proj;
    out.dfl_proj_host_data.resize(reg_max);
    for (int i = 0; i < reg_max; ++i) out.dfl_proj_host_data[i] = float(i);
    out.reg_max = reg_max;
    printf("proj shape: [%ld,%ld,%ld,%ld]\n", proj->ne[0], proj->ne[1], proj->ne[2], proj->ne[3]);
    
    // Apply DFL
    tensor dfl_output = dfl_forward(m, box, reg_max, proj, false);
    printf("dfl_output shape: [%ld,%ld,%ld,%ld]\n", dfl_output->ne[0], dfl_output->ne[1], dfl_output->ne[2], dfl_output->ne[3]);
    // Decode bounding boxes
    tensor dbox = dist2bbox(m, dfl_output, anchor_points, false);
    printf("dbox shape: [%ld,%ld,%ld,%ld]\n", dbox->ne[0], dbox->ne[1], dbox->ne[2], dbox->ne[3]);
    // Multiply by strides
    tensor strides_bc = ggml_reshape_4d(m, stride_tensor, 1, total_anchors, 1, 1);

    dbox = ggml_mul(m, dbox, strides_bc);
    printf("strides_bc shape: [%ld,%ld,%ld,%ld]\n", strides_bc->ne[0], strides_bc->ne[1], strides_bc->ne[2], strides_bc->ne[3]);

    // Apply sigmoid to classes
    cls = ggml_sigmoid(m, cls);
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
    int reg_max = 16; // DFL bins
    int c2 = std::max({16, ch[0] / 4, reg_max * 4});
    int c3 = std::max(ch[0], std::min(nc, 100));
    DetectOutput out;

    std::string reg_base = std::string("detect.cv2");
    std::string cls_base = std::string("detect.cv3");

    for (size_t i = 0; i < features.size(); ++i) {
        // printf("features[%d] shape: [%d,%d,%d,%d]\n", (int)i, (int)features[i]->ne[0], (int)features[i]->ne[1], (int)features[i]->ne[2], (int)features[i]->ne[3]);
        
        // std::string idx_str = std::to_string(i);
        tensor_name old_prefix_i = m.prefix;
        // Regression head: two convs, second outputs 4*reg_max channels
        tensor r0 = Conv(m, features[i], reg_base + "."+std::to_string(i)+".0.conv", ch[i], c2, 3, 1, -1, true, false);
        
        tensor r1 = Conv(m, r0          ,reg_base + "."+std::to_string(i)+".1.conv", c2, c2, 3, 1, -1, true, false);
        
        // Final regression conv: outputs 4*reg_max channels
        // Final regression conv: nn.Conv2d (NO .conv suffix)
        m.prefix = tensor_name((reg_base + "."+std::to_string(i)+".2").c_str());
        tensor r2 = conv_2d(m, r1, 1, 0, 1);
        // printf("r2 shape: [%d,%d,%d,%d]\n", (int)r2->ne[0], (int)r2->ne[1], (int)r2->ne[2], (int)r2->ne[3]);
        if (!ggml_is_contiguous(r2)) r2 = ggml_cont(m, r2);
        
        // Classification head: two convs, second outputs nc channels
        // m.prefix = old_prefix_i;
        tensor c0 = Conv(m, features[i], cls_base + "."+std::to_string(i)+".0.conv", ch[i], c3, 3, 1, -1, true, false);
        tensor c1 = Conv(m, c0,          cls_base + "."+std::to_string(i)+".1.conv", c3, c3, 3, 1, -1, true, false);
        // Final classification conv: nn.Conv2d (NO .conv suffix)
        
        m.prefix = tensor_name((cls_base + "." + std::to_string(i) + ".2").c_str());
        tensor c2 = conv_2d(m, c1, 1, 0, 1);
        // printf("c2 shape: [%d,%d,%d,%d]\n", (int)c2->ne[0], (int)c2->ne[1], (int)c2->ne[2], (int)c2->ne[3]);
        if (!ggml_is_contiguous(c2)) c2 = ggml_cont(m, c2);
        
        
        // Combine along channel dim and flatten spatial dims to anchors
        tensor combined = Concat(m, r2, c2, 0);
        // xi.view(shape[0], self.no, -1)
        tensor reshaped_combined = ggml_reshape_2d(m, combined, combined->ne[0], combined->ne[1]*combined->ne[2]); 
        //reshaped to [144, H, W, N] -> [144, H*W, N] (144, 8400, 1)

        combined = ggml_cont(m, combined);
        reshaped_combined = ggml_cont(m, reshaped_combined);
        out.raw_outputs.push_back(std::move(combined));
        out.features.push_back(std::move(reshaped_combined));
        // torch.Size([1, 144, 80, 80])
        // torch.Size([1, 144, 40, 40])
        // torch.Size([1, 144, 20, 20])
        
        m.prefix = old_prefix_i;
    }

    if (training) {
        out.predictions_cls = nullptr;
        out.predictions_bbox = nullptr;
        return out;
    }
    // Inference function
    DetectOutput detect = inference(m, out, ch, reg_max, nc);
    
    return detect;
}

// Main YOLOv9t forward pass with complete Detect head
DetectOutput yolov9t_forward(model_ref m, tensor x) {
    
    // Run backbone + neck
    std::map<int, tensor> features = yolov9t_backbone(m, x);
    
    // features = [N3(64 channels), N4(96 channels), N5(128 channels)]   
    printf("features size: [%d]\n", (int)features.size());
    
    // channels for N3, N4, N5
    std::vector<int> channels = {64, 96, 128};
    // std::string base_name = std::to_string(i);
    // {x15, x18, x21};
    std::vector<tensor> features_vector = {features[15], features[18], features[21]};
    DetectOutput d = detect_forward(m, features_vector, channels, 80, false);
    
    d.features_map = features; // expose backbone/neck features for dumping
    printf("detect_forward complete\n");
    
    return d;
}

// COCO class names
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
/*

def make_anchors(feats:Tensor, strides:Tensor, grid_cell_offset:float=0.5) -> Tuple[Tensor, Tensor]:
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), fill_value=stride, dtype=dtype, device=device))
    return torch.concatenate(anchor_points), torch.concatenate(stride_tensor)

*/

// struct anchor_result {
//     tensor anchor_points;  // [total_anchors, 2]
//     tensor stride_tensor;  // [total_anchors, 1]
// };
std::pair<tensor, tensor> make_anchors(
    model_ref m,
    DetectOutput const& out,
    std::vector<float>& anchor_host,
    std::vector<float>& stride_host,
    std::vector<float> const& strides,
    float grid_cell_offset) {
    
    GGML_ASSERT(out.raw_outputs.size() == strides.size());
    GGML_ASSERT(!out.raw_outputs.empty());
    
    // Calculate total number of anchors
    int64_t total_anchors = 0;
    for (size_t i = 0; i < out.raw_outputs.size(); ++i) {
        auto ne = nelements(out.raw_outputs[i]);
        total_anchors += ne[1] * ne[2]; // w * h
    }


    // CPU에서 먼저 데이터 준비
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
    
    // graph_context에 tensor 생성 (no_alloc이므로 data는 nullptr)
    tensor anchor_points = ggml_new_tensor_2d(m.graph_context, GGML_TYPE_F32, 2, total_anchors);
    tensor stride_tensor = ggml_new_tensor_2d(m.graph_context, GGML_TYPE_F32, 1, total_anchors);
    
    ggml_set_name(anchor_points, "anchor_points");
    ggml_set_name(stride_tensor, "stride_tensor");
    
    // Save host data to upload later after graph allocation
    anchor_host = std::move(anchor_data);
    stride_host = std::move(stride_data);
    
    printf("anchor_points shape: [%ld,%ld]\n", anchor_points->ne[0], anchor_points->ne[1]);
    printf("stride_tensor shape: [%ld,%ld]\n", stride_tensor->ne[0], stride_tensor->ne[1]);
    
    return std::make_pair(anchor_points, stride_tensor);
}
/*
def check_img_size(imgsz, s=32, floor=0):
    def make_divisible(x, divisor):
        # Returns nearest x divisible by divisor
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size
*/

int make_divisible(int x, int divisor) {
    // Returns nearest x divisible by divisor
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
    i32x2 new_extent = {im.extent[0] + top + bottom, 
                       im.extent[1] + left + right};
    image_data bordered = image_alloc(new_extent, im.format);

    // 전체 영역을 먼저 테두리 색상으로 채우기
    for (int y = 0; y < new_extent[0]; ++y) {
        for (int x = 0; x < new_extent[1]; ++x) {
            for (int c = 0; c < 3; ++c) {
                bordered.data[(y * new_extent[1] + x) * 3 + c] = color[c];
            }
        }
    }

    // 원본 이미지를 중앙에 복사
    for (int y = 0; y < im.extent[0]; ++y) {
        for (int x = 0; x < im.extent[1]; ++x) {
            for (int c = 0; c < 3; ++c) {
                int src_idx = (y * im.extent[1] + x) * 3 + c;
                int dst_idx = ((y + top) * new_extent[1] + (x + left)) * 3 + c;
                bordered.data[dst_idx] = im.data[src_idx];
            }
        }
    }

    return bordered;
}

// RGB 이미지를 직접 처리하는 특화된 버전
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
            
            // 3채널 동시 처리
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

// 사용 예시
image_data linear_image_resize(image_data im, i32x2 new_shape) {
    image_data resized = image_alloc(new_shape, im.format);
    resize_rgb_bilinear(im.data.get(), im.extent, resized.data.get(), new_shape);
    return resized;
}
image_data letterbox(image_data im, i32x2 new_shape, u8x3 color, 
              bool _auto, bool scaleFill, bool scaleup, int stride) {
    i32x2 shape = im.extent;  // 현재 이미지의 크기 [height, width]
    if (new_shape[0] == 0) {
        new_shape = {new_shape[1], new_shape[1]};
    }

    // 스케일 비율 (new / old)
    float r = std::min(static_cast<float>(new_shape[0]) / shape[0], static_cast<float>(new_shape[1]) / shape[1]);

    if (!scaleup) {  // 스케일 업을 하지 않음
        r = std::min(r, 1.0f);
    }

    // 패딩 계산
    i32x2 new_unpad = {static_cast<int>(round(shape[1] * r)), static_cast<int>(round(shape[0] * r))};
    float dw = new_shape[1] - new_unpad[0];
    float dh = new_shape[0] - new_unpad[1];

    if (_auto) {  // 최소 직사각형
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {  // 스트레치
        dw = 0.0f;
        dh = 0.0f;
        new_unpad = {new_shape[1], new_shape[0]};
    }

    dw /= 2;  // 패딩을 양쪽으로 나눔
    dh /= 2;

    
    im = image_scale(im, new_unpad);  // 이미지 크기 조정
    // Bilinear interpolation 수행
    // im = linear_image_resize(std::move(im), new_unpad);
    

    int top = static_cast<int>(round(dh - 0.1f));
    int bottom = static_cast<int>(round(dh + 0.1f));
    int left = static_cast<int>(round(dw - 0.1f));
    int right = static_cast<int>(round(dw + 0.1f));
    im = image_add_border(std::move(im), top, bottom, left, right, color);  // 테두리 추가

    return im;
}

// 1. xywh -> xyxy box conversion, inplace
inline void xywh2xyxy(float* boxes, int n) {
    // inplace conversion. boxes: n x 4 [x,y,w,h]
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

// IoU(box1, box2) for NMS
inline float box_iou(const float* b1, const float* b2) {
    // box: x1, y1, x2, y2
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

// NMS - similar to torchvision.ops.nms (offset by class for agnostic==false!)
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

// PyTorch-like non_max_suppression for YOLOv8/YOLOv9 outputs, closely following PyTorch logic.
/*
def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680
):
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,8400)
    nc = prediction.shape[1] - 4  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  # to track idxs
    
    # Settings
    time_limit = 2.0 + 0.05 * bs  # seconds to quit after
    
    prediction = prediction.transpose(-1, -2)  # shape(1,84,8400) to shape(1,8400,84)
    
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy


    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x, xk = x[filt], xk[filt]
        
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)
        
        conf, j = cls.max(1, keepdim=True)
        filt = conf.view(-1) > conf_thres
        x = torch.cat((box, conf, j.float(), mask), 1)[filt]
        xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output
*/
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
    if (!(0.0f <= conf_thres && conf_thres <= 1.0f)) {
        throw std::runtime_error("Invalid Confidence threshold");
    }
    if (!(0.0f <= iou_thres && iou_thres <= 1.0f)) {
        throw std::runtime_error("Invalid IoU threshold");
    }

    const int64_t num_classes = outputs.predictions_cls->ne[0];
    const int64_t num_anchors = outputs.predictions_cls->ne[1];

    // Fetch data from backend to host
    tensor_data td_cls = transfer_from_backend(outputs.predictions_cls);
    tensor_data td_box = transfer_from_backend(outputs.predictions_bbox);
    const float* cls_ptr = td_cls.as_f32().data();
    const float* box_ptr = td_box.as_f32().data();

    // Collect candidates
    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(num_anchors);
    scores.reserve(num_anchors);
    class_ids.reserve(num_anchors);

    for (int64_t j = 0; j < num_anchors; ++j) {
        float best_conf = -1.0f;
        int best_cls = -1;
        for (int64_t c = 0; c < num_classes; ++c) {
            float v = cls_ptr[c * num_anchors + j]; // [nc, N, 1, 1]
            if (v > best_conf) {
                best_conf = v;
                best_cls = (int)c;
            }
        }
        if (best_conf < conf_thres) continue;

        // predictions_bbox is already in xyxy
        float x1 = box_ptr[0 * num_anchors + j];
        float y1 = box_ptr[1 * num_anchors + j];
        float x2 = box_ptr[2 * num_anchors + j];
        float y2 = box_ptr[3 * num_anchors + j];

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(best_conf);
        class_ids.push_back(best_cls);
    }

    if (boxes.empty()) return {};

    // Limit candidates for NMS
    if ((int)boxes.size() > max_nms) {
        std::vector<int> order(boxes.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + max_nms, order.end(), [&](int a, int b) {
            return scores[a] > scores[b];
        });
        order.resize(max_nms);

        std::vector<std::array<float, 4>> b2;
        std::vector<float> s2;
        std::vector<int> c2;
        b2.reserve(order.size());
        s2.reserve(order.size());
        c2.reserve(order.size());
        for (int idx : order) {
            b2.push_back(boxes[idx]);
            s2.push_back(scores[idx]);
            c2.push_back(class_ids[idx]);
        }
        boxes.swap(b2);
        scores.swap(s2);
        class_ids.swap(c2);
    }

    std::vector<int> keep = nms(boxes, scores, class_ids, iou_thres, false, max_wh);
    if ((int)keep.size() > max_det) keep.resize(max_det);

    std::vector<detected_obj> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        detected_obj o;
        o.x1 = boxes[idx][0];
        o.y1 = boxes[idx][1];
        o.x2 = boxes[idx][2];
        o.y2 = boxes[idx][3];
        o.confidence = scores[idx];
        o.class_id = class_ids[idx];
        o.class_confidence = scores[idx];
        result.push_back(o);
    }
    return result;
}




// Scale boxes 함수도 detected_obj에 맞게 수정
void scale_boxes(
    std::vector<detected_obj>& detections,
    i32x2 model_shape,  // [height, width] of model input
    i32x2 img_shape) {   // [height, width] of original image
    
    // Calculate gain and padding (same as letterbox)
    float gain = std::min(
        (float)model_shape[0] / img_shape[0],
        (float)model_shape[1] / img_shape[1]
    );
    
    float pad_w = (model_shape[1] - img_shape[1] * gain) / 2.0f;
    float pad_h = (model_shape[0] - img_shape[0] * gain) / 2.0f;
    
    for (auto& det : detections) {
        // Remove padding
        det.x1 = (det.x1 - pad_w) / gain;
        det.y1 = (det.y1 - pad_h) / gain;
        det.x2 = (det.x2 - pad_w) / gain;
        det.y2 = (det.y2 - pad_h) / gain;
        
        // Clip to image boundaries
        det.x1 = std::max(0.0f, std::min(det.x1, (float)img_shape[1]));
        det.y1 = std::max(0.0f, std::min(det.y1, (float)img_shape[0]));
        det.x2 = std::max(0.0f, std::min(det.x2, (float)img_shape[1]));
        det.y2 = std::max(0.0f, std::min(det.y2, (float)img_shape[0]));
    }
}


// Helper: Draw a line on image
void draw_line(uint8_t* img, int width, int height, int channels,
    int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
// Bresenham's line algorithm
int dx = abs(x2 - x1);
int dy = abs(y2 - y1);
int sx = (x1 < x2) ? 1 : -1;
int sy = (y1 < y2) ? 1 : -1;
int err = dx - dy;

while (true) {
// Draw thick line by drawing multiple pixels
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

// Helper: Draw rectangle
void draw_rectangle(uint8_t* img, int width, int height, int channels,
         int x1, int y1, int x2, int y2,
         uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
// Clamp coordinates
x1 = std::max(0, std::min(x1, width - 1));
y1 = std::max(0, std::min(y1, height - 1));
x2 = std::max(0, std::min(x2, width - 1));
y2 = std::max(0, std::min(y2, height - 1));

// Draw four lines
draw_line(img, width, height, channels, x1, y1, x2, y1, r, g, b, thickness); // Top
draw_line(img, width, height, channels, x2, y1, x2, y2, r, g, b, thickness); // Right
draw_line(img, width, height, channels, x2, y2, x1, y2, r, g, b, thickness); // Bottom
draw_line(img, width, height, channels, x1, y1, x1, y2, r, g, b, thickness); // Left
}

// Helper: Draw filled rectangle (for text background)
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

// Simple 5x7 bitmap font for digits and basic chars
void draw_char(uint8_t* img, int width, int height, int channels,
    int x, int y, char c, uint8_t r, uint8_t g, uint8_t b) {
// Simple 5x7 font patterns (only digits and %)
static const uint8_t font_5x7[][7] = {
// '0'
{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
// '1'
{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
// '2'
{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
// '3'
{0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},
// '4'
{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
// '5'
{0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
// '6'
{0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
// '7'
{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
// '8'
{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
// '9'
{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
// '%'
{0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}
};

int idx = -1;
if (c >= '0' && c <= '9') idx = c - '0';
else if (c == '%') idx = 10;
else return; // Unknown character

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

// Draw text string
void draw_text(uint8_t* img, int width, int height, int channels,
    int x, int y, const char* text,
    uint8_t r, uint8_t g, uint8_t b) {
int offset = 0;
while (*text) {
draw_char(img, width, height, channels, x + offset, y, *text, r, g, b);
offset += 6; // 5 pixels + 1 space
text++;
}
}

// Get color for class (simple color wheel)
void get_class_color(int class_id, uint8_t& r, uint8_t& g, uint8_t& b) {
// Generate distinct colors for different classes
int h = (class_id * 137) % 360; // Golden angle for good distribution
float s = 0.8f;
float v = 0.95f;

// HSV to RGB
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
// Main drawing function - 이미지를 직접 수정하고 반환
void draw_detections(
    image_data& img,
    std::vector<detected_obj> const& detections,
    std::vector<std::string> const& class_names) {
    
    // Get image properties
    int width = img.extent[1];
    int height = img.extent[0];
    int channels = 3;
    
    // Get raw pointer to image data
    uint8_t* img_data = nullptr;
    // bool is_f32 = false;
    
    if (img.format == image_format::rgb_u8 || img.format == image_format::rgba_u8) {
        img_data = img.data.get();
        channels = (img.format == image_format::rgba_u8) ? 4 : 3;
    } else if (img.format == image_format::rgb_f32 || img.format == image_format::rgba_f32) {
        // F32 format - we'll need to convert after drawing
        // is_f32 = true;
        channels = (img.format == image_format::rgba_f32) ? 4 : 3;
        
        // Convert to U8 for drawing
        image_data temp_u8 = image_f32_to_u8(img, 
            channels == 4 ? image_format::rgba_u8 : image_format::rgb_u8);
        
        // Draw on the U8 image
        img_data = temp_u8.data.get();
        
        // Draw each detection
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
        
        // Replace original image with drawn U8 image
        img = std::move(temp_u8);
        return;
    } else {
        printf("Error: Unsupported image format for drawing\n");
        return;
    }

    // Draw each detection (for U8 format)
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
} // namespace visp::yolov9t

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
    
    image_data resized;
    if (image.extent[0] != p.input_size || image.extent[1] != p.input_size) {
        resized = image_scale(image, {p.input_size, p.input_size});
        image = image_view(resized);
    }

    return image_u8_to_f32(image, image_format::rgb_f32, p.offset, 1.f / p.scale);
}

image_data yolov9t_process_input(image_data image, yolov9t_params const& p) {
    // 1) letterbox to square (p.input_size)
    image_data lb = letterbox(std::move(image), {p.input_size, p.input_size}, u8x3{114,114,114}, true, false, true, yolov9t_params::stride);
    // 2) convert to f32 and normalize to [0,1]
    image_data result = image_alloc({p.input_size, p.input_size}, image_format::rgb_f32);
    image_u8_to_f32(lb, result, f32x4(p.offset), f32x4(p.scale));
    return result;
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
    // Header with shape and type
    fprintf(
        f,
        "# %ld,%ld,%ld,%ld\n# %s\n",
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3], ggml_type_name(t->type));

    // Fetch data from backend
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
        // Raw bytes as hex fallback
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
    // Decide which keys to dump
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

} // namespace visp::yolov9t

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
    // Ensure data is fetched from backend
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

} // namespace visp::yolov9t