// mmdet_run.h — mmdet 검출기 한 벌을 백본부터 디코드까지 끝낸다.
//
// 계열마다 조립기(mmdet_head.cpp)와 디코더(postproc.cpp)의 조합이 다르고, DDQ 는 층 사이에
// 호스트 NMS 가 끼어 그래프가 여러 벌로 갈린다. 그 분기를 한자리에 두어 vision-cli 와 검증
// 하네스가 같은 코드를 쓴다.
#pragma once

#include "visp/mmdet_head.h"
#include "visp/ml.h"
#include "visp/postproc.h"

#include <span>
#include <vector>

namespace visp {

// 백본 그래프를 세우는 함수. arch_registry 의 `arch_forward_fn` 과 같은 시그니처이며,
// 생성물의 `<Arch>_forward(m, x, <Arch>_detect_params(file))` 를 감싼 것이다.
using mmdet_backbone_fn = tensor (*)(model_ref, tensor, model_file const&);

// 전처리까지 끝난 CWHN 입력을 받아 박스를 낸다. 좌표계는 입력 해상도(`input_w`×`input_h`)이며,
// 원본 이미지로 되돌리는 일은 호출한 쪽이 한다.
//
// `cfg.det.input_w` / `input_h` 는 여기서 인자 값으로 덮어쓴다.
std::vector<detection> mmdet_run(backend_device& backend, model_weights& weights,
                                 model_file const& file, mmdet_backbone_fn backbone,
                                 mmdet_cfg cfg, std::span<const float> input_cwhn,
                                 int input_w, int input_h);

}  // namespace visp
