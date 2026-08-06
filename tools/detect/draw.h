// 검출 결과를 이미지 위에 그린다. 러너의 기본 출력이 이미지이기 위한 최소한의 부품.
//
// 글자는 그리지 않는다 — 폰트를 들여오면 러너가 무거워지고, 클래스·점수는 러너가 표로
// 찍는다. 이미지는 "어디"를, 터미널 표는 "무엇"을 담당한다.
#pragma once

#include "visp/image.h"
#include "visp/postproc.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace visp {

// 클래스끼리 구분되게. 범례 없이도 같은 클래스가 같은 색으로 보이면 충분하다.
inline std::array<uint8_t, 3> detection_colour(int label) {
    static constexpr uint8_t table[][3] = {
        {230, 60, 60},   {60, 160, 230}, {70, 190, 110}, {240, 160, 40}, {170, 100, 220},
        {40, 200, 200},  {230, 110, 170}, {150, 160, 60}, {110, 130, 240}, {200, 90, 60},
    };
    int n = int(sizeof(table) / sizeof(table[0]));
    int i = ((label % n) + n) % n;
    return {table[i][0], table[i][1], table[i][2]};
}

namespace detail {

inline void put_pixel(image_span const& img, int x, int y, std::array<uint8_t, 3> c) {
    if (x < 0 || y < 0 || x >= img.extent[0] || y >= img.extent[1]) {
        return;
    }
    int nc = n_channels(img.format);
    auto* p = static_cast<uint8_t*>(img.data) + size_t(y) * img.stride + size_t(x) * nc;
    p[0] = c[0];
    if (nc > 1) p[1] = c[1];
    if (nc > 2) p[2] = c[2];
}

}  // namespace detail

// 박스 외곽선 하나. thickness 픽셀 두께로 안쪽에 그린다.
inline void draw_box(image_span const& img, float x1, float y1, float x2, float y2,
                     std::array<uint8_t, 3> colour, int thickness = 2) {
    int ix1 = int(std::min(x1, x2)), ix2 = int(std::max(x1, x2));
    int iy1 = int(std::min(y1, y2)), iy2 = int(std::max(y1, y2));
    for (int t = 0; t < thickness; ++t) {
        for (int x = ix1; x <= ix2; ++x) {
            detail::put_pixel(img, x, iy1 + t, colour);
            detail::put_pixel(img, x, iy2 - t, colour);
        }
        for (int y = iy1; y <= iy2; ++y) {
            detail::put_pixel(img, ix1 + t, y, colour);
            detail::put_pixel(img, ix2 - t, y, colour);
        }
    }
}

// 검출 목록을 그린다.
// 좌표는 검출기가 돈 정사각 입력 기준이므로 원본 크기로 되돌린다(scale_x/scale_y).
// 반환: 실제로 그린 개수.
inline int draw_detections(image_span const& img, std::vector<detection> const& dets,
                           float scale_x, float scale_y, float threshold = 0.3f) {
    int thickness = std::max(2, img.extent[1] / 300);
    int drawn = 0;
    for (detection const& d : dets) {
        if (d.score < threshold) {
            continue;
        }
        draw_box(img, d.x1 * scale_x, d.y1 * scale_y, d.x2 * scale_x, d.y2 * scale_y,
                 detection_colour(d.label), thickness);
        ++drawn;
    }
    return drawn;
}

}  // namespace visp
