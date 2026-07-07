// 컴포넌트 조립형 vision.cpp — Registry + 조립 엔진 구현. (component.h 참조)
#include "visp/arch/component.h"
#include "visp/nn.h"
#include "util/string.h"

#include <string>

namespace visp {

std::map<std::string, component_fn>& component_registry() {
    static std::map<std::string, component_fn> reg;
    return reg;
}

component_register::component_register(char const* id, component_fn fn) {
    component_registry()[id] = fn;
}

// "role|impl|prefix" 한 줄씩. 공백 트림, 빈 줄/필드부족 줄은 스킵.
compose_spec parse_compose_pipeline(std::string_view text) {
    auto trim = [](std::string_view s) -> std::string_view {
        size_t b = s.find_first_not_of(" \t\r");
        if (b == std::string_view::npos) return {};
        size_t e = s.find_last_not_of(" \t\r");
        return s.substr(b, e - b + 1);
    };
    compose_spec spec;
    size_t start = 0;
    while (start <= text.size()) {
        size_t nl = text.find('\n', start);
        std::string_view line =
            text.substr(start, nl == std::string_view::npos ? std::string_view::npos : nl - start);
        line = trim(line);
        if (!line.empty()) {
            std::string_view fields[3];
            int nf = 0;
            size_t fs = 0;
            for (int i = 0; i < 3 && fs <= line.size(); ++i) {
                size_t bar = line.find('|', fs);
                std::string_view f =
                    line.substr(fs, bar == std::string_view::npos ? std::string_view::npos : bar - fs);
                fields[nf++] = trim(f);
                if (bar == std::string_view::npos) break;
                fs = bar + 1;
            }
            if (nf == 3 && !fields[1].empty()) {
                spec.stages.push_back(
                    {std::string(fields[0]), std::string(fields[1]), std::string(fields[2])});
            }
        }
        if (nl == std::string_view::npos) break;
        start = nl + 1;
    }
    return spec;
}

tensor compose_forward(model_ref m, tensor x, compose_spec const& spec) {
    // 입구: 이미지 레이아웃 → contiguous_2d (부품 내부는 경계 변환 안 함).
    std::vector<tensor> t = {cwhn_to_contiguous_2d(m, x)};

    for (compose_stage const& s : spec.stages) {
        auto it = component_registry().find(s.impl);
        if (it == component_registry().end()) {
            throw except("compose: 등록 안 된 부품 '{}' (role={})", s.impl, s.role);
        }
        compose_params p;
        // 가중치는 단일 gguf 를 prefix 로 스코프: m["backbone"] 안에서 부품이 m["conv1"] →
        // "backbone.conv1" 로 해소된다(chain_prefix, dot 결합).
        model_ref sub = s.prefix.empty() ? m : m[s.prefix.c_str()];
        t = it->second(sub, t, p);
    }

    // 출구: 각 출력 텐서를 2d→cwhn 후 out_N 으로 등록 (host 가 out_0.. 순회 수집).
    for (size_t i = 0; i < t.size(); ++i) {
        std::string nm = "out_" + std::to_string(i);
        compute_graph_output(m, contiguous_2d_to_cwhn(m, t[i]), nm.c_str());
    }
    return t.empty() ? x : t.back();
}

} // namespace visp
