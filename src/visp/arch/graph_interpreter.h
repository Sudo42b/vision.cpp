// 범용 그래프 인터프리터 — gguf 의 그래프 '데이터'(graph.nodes)를 읽어 런타임에 ggml
// 그래프를 조립한다. g2c 가 .cpp 부품을 생성하는 대신, 모델을 노드 리스트로 직렬화하고
// (tools/graph_export/serialize.py) vision.cpp 가 여기서 해석한다 (ONNX 방식).
#pragma once

#include "visp/ml.h"

#include <map>
#include <string>
#include <vector>

namespace visp {

struct graph_node {
    int id = 0;
    std::string op;               // "conv2d", "batch_norm", "relu", ...
    std::vector<int> inputs;      // 활성 입력 = producer 노드 id
    std::string weight;           // 가중치 prefix (m[weight]); 무가중치면 빈 문자열
    std::map<std::string, float> attrs;  // 스칼라 attr (stride/padding/...)
    std::vector<int64_t> out_shape;
};

struct graph_spec {
    std::vector<graph_node> nodes;   // id 오름차순(topo)
    std::vector<int> inputs;
    std::vector<int> outputs;
};

// gguf KV(graph.nodes/inputs/outputs) 파싱
graph_spec parse_graph(model_file const& file);

// 그래프 데이터 → ggml 그래프 조립. out_0..N 등록, 마지막 출력 반환.
tensor graph_interpret(model_ref m, tensor input, graph_spec const& g);

} // namespace visp
