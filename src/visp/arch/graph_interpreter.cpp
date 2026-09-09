// 범용 그래프 인터프리터 구현. (graph_interpreter.h 참조)
#include "visp/arch/graph_interpreter.h"
#include "visp/nn.h"

#include <ggml.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace visp {

static std::vector<std::string> split(std::string_view s, char d) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t p = s.find(d, start);
        size_t len = (p == std::string_view::npos) ? std::string_view::npos : p - start;
        out.emplace_back(std::string(s.substr(start, len)));
        if (p == std::string_view::npos) break;
        start = p + 1;
    }
    return out;
}

static std::vector<int> parse_ids(std::string_view s) {
    std::vector<int> v;
    for (auto& x : split(s, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

graph_spec parse_graph(model_file const& file) {
    graph_spec g;
    for (auto& line : split(file.get_string("graph.nodes"), '\n')) {
        if (line.empty()) continue;
        auto f = split(line, '|');  // id|op|inputs|weight|attrs|out_shape
        if (f.size() < 2) continue;
        graph_node n;
        n.id = std::atoi(f[0].c_str());
        n.op = f[1];
        if (f.size() > 2) n.inputs = parse_ids(f[2]);
        if (f.size() > 3) n.weight = f[3];
        if (f.size() > 4 && !f[4].empty())
            for (auto& kv : split(f[4], ';')) {
                auto e = kv.find('=');
                if (e != std::string::npos)
                    n.attrs[kv.substr(0, e)] = (float)std::atof(kv.c_str() + e + 1);
            }
        if (f.size() > 5 && !f[5].empty())
            for (auto& x : split(f[5], ',')) n.out_shape.push_back(std::atoll(x.c_str()));
        g.nodes.push_back(std::move(n));
    }
    g.inputs = parse_ids(file.get_string("graph.inputs"));
    g.outputs = parse_ids(file.get_string("graph.outputs"));
    return g;
}

static int iattr(graph_node const& n, char const* k, int def) {
    auto it = n.attrs.find(k);
    return it == n.attrs.end() ? def : (int)it->second;
}

tensor graph_interpret(model_ref m, tensor input, graph_spec const& g) {
    std::map<int, tensor> t;
    tensor x0 = cwhn_to_contiguous_2d(m, input);
    for (int id : g.inputs) t[id] = x0;

    for (auto& n : g.nodes) {
        if (n.op == "input") continue;
        auto in = [&](int i) -> tensor { return t.at(n.inputs[(size_t)i]); };
        model_ref mw = n.weight.empty() ? m : m[n.weight.c_str()];
        tensor y = nullptr;

        if (n.op == "conv2d") {
            int s = iattr(n, "stride", 1), p = iattr(n, "padding", 0),
                d = iattr(n, "dilation", 1), grp = iattr(n, "groups", 1);
            y = grp > 1 ? conv_2d_grouped(mw, in(0), s, p, d, grp)
                        : conv_2d(mw, in(0), s, p, d);
        } else if (n.op == "batch_norm") {
            y = batch_norm_2d(mw, in(0));
        } else if (n.op == "relu") {
            y = ggml_relu(m, in(0));
        } else if (n.op == "elemwise_add") {
            y = ggml_add(m, in(0), in(1));
        } else if (n.op == "maxpool") {
            int k = iattr(n, "kernel_size", 2), s = iattr(n, "stride", k), p = iattr(n, "padding", 0);
            y = ggml_pool_2d(m, in(0), GGML_OP_POOL_MAX, k, k, s, s, p, p);
        } else if (n.op == "adaptive_avg_pool2d") {
            tensor a = in(0);
            y = ggml_pool_2d(m, a, GGML_OP_POOL_AVG, a->ne[0], a->ne[1], a->ne[0], a->ne[1], 0, 0);
        } else if (n.op == "flatten") {
            tensor a = ggml_cont(m, in(0));
            y = ggml_reshape_2d(m, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]);
        } else if (n.op == "dense" || n.op == "linear") {
            y = linear(mw, in(0));
        } else if (n.op == "resize") {
            // FPN 업샘플 등. 정적 out_shape(torch NCHW) 역순 = ggml ne [W,H,C,N] 로 interpolate.
            int64_t ne[4] = {1, 1, 1, 1};
            int k = 0;
            for (int i = (int)n.out_shape.size() - 1; i >= 0 && k < 4; --i) ne[k++] = n.out_shape[(size_t)i];
            y = ggml_interpolate(m, in(0), ne[0], ne[1], ne[2], ne[3], GGML_SCALE_MODE_NEAREST);
        } else if (n.op == "shape") {
            // 메타 op(차원 질의) — resize 는 정적 out_shape 를 쓰므로 tensor 로는 미사용. passthrough.
            y = n.inputs.empty() ? nullptr : in(0);
            if (!y) continue;
        } else {
            fprintf(stderr, "[graph] 미지원 op '%s' (n%d) — passthrough\n", n.op.c_str(), n.id);
            y = in(0);
        }
        ggml_format_name(y, "n%d_%s", n.id, n.op.c_str());
        t[n.id] = y;
    }

    tensor last = nullptr;
    int oi = 0;
    for (int oid : g.outputs) {
        tensor out = contiguous_2d_to_cwhn(m, t.at(oid));
        char nm[32];
        snprintf(nm, sizeof(nm), "out_%d", oi++);
        last = compute_graph_output(m, out, tensor_name(nm));
    }
    return last;
}

} // namespace visp
