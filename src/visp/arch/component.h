// 컴포넌트 조립형 vision.cpp — 부품(backbone/neck/head) Registry + 조립 엔진.
//
// mmdet 모델을 통짜 arch 로 편입하는 대신, g2c 가 부품 단위로 생성한 컴포넌트 함수들을
// self-register 로 등록해 두고, gguf 메타의 조립 스펙(compose.pipeline)만으로
// backbone→neck→head 를 이어붙여 실행한다. 가중치는 단일 gguf(prefix 네임스페이스)를
// model_ref prefix 커서로 스코프해서 각 부품에 넘긴다.
#pragma once
#include "visp/ml.h"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace visp {

// 부품에 넘길 하이퍼파라미터(선택). 현재는 비어 있고, 필요 시 gguf 메타를 읽어 채운다.
struct compose_params {
    model_file const* file = nullptr;  // "<impl>.<key>" 스칼라 하이퍼파라미터 조회용(선택)
};

// 모든 부품(backbone/neck/head) 공통 시그니처.
// 입력/출력은 contiguous_2d 레이아웃의 텐서 리스트(경계 변환은 compose_forward 가 담당).
//   backbone: {image}   -> {C2,C3,C4,C5}
//   neck:     {C2..C5}  -> {P3..P7}
//   head:     {P3..P7}  -> {cls_l0, bbox_l0, ...}
using component_fn = std::vector<tensor> (*)(model_ref, std::vector<tensor> const&, compose_params const&);

// 부품 identity(문자열) → 함수. Meyers singleton (정적 초기화 순서 안전).
VISP_API std::map<std::string, component_fn>& component_registry();

// 각 생성 부품 .cpp 말미의 file-scope static 이 이 ctor 로 자기 자신을 등록한다.
//   static visp::component_register _reg_resnet50("resnet50", &resnet50_component);
struct VISP_API component_register {
    component_register(char const* id, component_fn fn);
};

// 조립 스펙: 한 스테이지 = (역할, 부품 impl, 가중치 prefix).
struct compose_stage {
    std::string role;    // "backbone"/"neck"/"head" (설명용)
    std::string impl;    // registry 키, 예: "resnet50"
    std::string prefix;  // 가중치 prefix, 예: "backbone"
};

struct compose_spec {
    std::vector<compose_stage> stages;
};

// gguf 메타 "compose.pipeline" 파싱 — 한 줄당 "role|impl|prefix", 줄 구분 '\n'.
VISP_API compose_spec parse_compose_pipeline(std::string_view text);

// 조립 실행: 입구 cwhn→2d 1회 → 스테이지 순서대로 부품 체이닝(prefix 스코프) →
// 출구 2d→cwhn 후 각 텐서를 out_N 으로 등록. 마지막 텐서 반환(단일출력 러너 호환).
VISP_API tensor compose_forward(model_ref m, tensor x, compose_spec const& spec);

} // namespace visp
