// draw.h — 검출 결과를 이미지에 그린다. **순수 CPU** (ggml 무관).
//
// vision.cpp 에 그리기가 없었다 — 기존 CLI 5개(sam·birefnet·depthany·migan·esrgan)가
// 전부 이미지→이미지(마스크·깊이맵·확대본)라 사각형을 그릴 일이 없었기 때문이다.
#pragma once

#include "visp/image.h"
#include "visp/postproc.h"

#include <string>
#include <vector>

namespace visp {

struct draw_style {
    int thickness = 2;
    bool labels = true;      // 클래스 이름 + 점수를 박스 위에 찍는다
    int text_scale = 2;      // 5x7 비트맵 폰트의 배율
};

// 좌표는 **모델 입력 크기 기준**이다(예: 640×640). 원본 이미지에 그리려면 스케일이 필요하다 —
// 호출부가 `scale`(원본/입력)을 준다. letterbox 를 쓰면 오프셋도 여기서 빼야 한다.
void draw_detections(image_span const& img, std::vector<detection> const& dets,
                     std::vector<std::string> const& class_names,
                     float scale_x = 1.0f, float scale_y = 1.0f,
                     draw_style const& style = {});

} // namespace visp
