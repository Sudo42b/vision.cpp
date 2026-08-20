#include "visp/draw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace visp {

namespace {

// 5x7 비트맵 폰트. 라벨에 필요한 글자만 담는다(대문자·숫자·소수점·공백·하이픈).
// 폰트 라이브러리를 끌어오지 않으려고 최소한만 둔다 — 의존성 하나가 빌드를 인질로 잡는다.
struct glyph { char c; uint8_t rows[7]; };

constexpr glyph FONT[] = {
    {' ', {0,0,0,0,0,0,0}},          {'-', {0,0,0,0x1F,0,0,0}},
    {'.', {0,0,0,0,0,0x0C,0x0C}},    {':', {0,0x0C,0x0C,0,0x0C,0x0C,0}},
    {'%', {0x19,0x1A,0x02,0x04,0x08,0x0B,0x13}},
    {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
    {'2', {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}}, {'3', {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}},
    {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
    {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
    {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'B', {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
    {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}}, {'D', {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
    {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}}, {'H', {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'I', {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}}, {'J', {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
    {'K', {0x11,0x12,0x14,0x18,0x14,0x12,0x11}}, {'L', {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
    {'M', {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}}, {'N', {0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
    {'O', {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'P', {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q', {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}}, {'R', {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
    {'S', {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T', {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
    {'U', {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'V', {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    {'W', {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}}, {'X', {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
    {'Y', {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}}, {'Z', {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
};

glyph const* find_glyph(char c) {
    if (c >= 'a' && c <= 'z') {
        c = char(c - 'a' + 'A');       // 소문자는 대문자로 접는다
    }
    for (glyph const& g : FONT) {
        if (g.c == c) {
            return &g;
        }
    }
    return nullptr;
}

// 클래스마다 다른 색. HSV 를 돌리는 대신 소수 곱으로 흩어 놓는다 — 인접 클래스가 붙어 있을 때
// 색이 비슷하면 박스를 구분 못 한다.
void class_color(int label, uint8_t rgb[3]) {
    const float h = std::fmod(float(label) * 0.6180339887f, 1.0f) * 6.0f;
    const int i = int(h);
    const float f = h - float(i);
    const uint8_t v = 255, p = 40;
    const uint8_t q = uint8_t(255.0f * (1.0f - f * 0.84f));
    const uint8_t t = uint8_t(255.0f * (0.16f + f * 0.84f));
    switch (i % 6) {
        case 0: rgb[0] = v; rgb[1] = t; rgb[2] = p; break;
        case 1: rgb[0] = q; rgb[1] = v; rgb[2] = p; break;
        case 2: rgb[0] = p; rgb[1] = v; rgb[2] = t; break;
        case 3: rgb[0] = p; rgb[1] = q; rgb[2] = v; break;
        case 4: rgb[0] = t; rgb[1] = p; rgb[2] = v; break;
        default: rgb[0] = v; rgb[1] = p; rgb[2] = q; break;
    }
}

struct canvas {
    uint8_t* data;
    int w, h, ch;

    void px(int x, int y, uint8_t const rgb[3], float a = 1.0f) {
        if (x < 0 || y < 0 || x >= w || y >= h) {
            return;                     // 잘라낸다 — 박스가 이미지 밖으로 나가는 건 정상이다
        }
        uint8_t* q = data + ((size_t)y * w + x) * ch;
        for (int c = 0; c < 3 && c < ch; ++c) {
            q[c] = uint8_t(float(q[c]) * (1.0f - a) + float(rgb[c]) * a);
        }
        if (ch == 4) {
            q[3] = 255;
        }
    }

    void fill(int x0, int y0, int x1, int y1, uint8_t const rgb[3], float a) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                px(x, y, rgb, a);
            }
        }
    }

    void rect(int x0, int y0, int x1, int y1, uint8_t const rgb[3], int t) {
        for (int k = 0; k < t; ++k) {
            for (int x = x0 - k; x <= x1 + k; ++x) {
                px(x, y0 - k, rgb); px(x, y1 + k, rgb);
            }
            for (int y = y0 - k; y <= y1 + k; ++y) {
                px(x0 - k, y, rgb); px(x1 + k, y, rgb);
            }
        }
    }

    void text(int x, int y, std::string const& s, uint8_t const rgb[3], int scale) {
        int cx = x;
        for (char c : s) {
            glyph const* g = find_glyph(c);
            if (g) {
                for (int row = 0; row < 7; ++row) {
                    for (int col = 0; col < 5; ++col) {
                        if (g->rows[row] & (1 << (4 - col))) {
                            fill(cx + col * scale, y + row * scale,
                                 cx + (col + 1) * scale, y + (row + 1) * scale, rgb, 1.0f);
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
};

} // namespace

void draw_detections(image_span const& img, std::vector<detection> const& dets,
                     std::vector<std::string> const& class_names,
                     float scale_x, float scale_y, draw_style const& style) {
    const int ch = n_channels(img.format);
    if (is_float(img.format) || ch < 3) {
        // f32 캔버스나 회색조에 그리는 경로는 아직 없다. **조용히 건너뛰지 않고** 말한다 —
        // 아무 일도 안 일어나면 "박스가 하나도 안 나왔다" 로 오해한다.
        fprintf(stderr, "draw_detections: u8 RGB/RGBA 만 지원한다 (format=%d, ch=%d)\n",
                (int)img.format, ch);
        return;
    }
    canvas cv{static_cast<uint8_t*>(img.data), img.extent[0], img.extent[1], ch};

    for (detection const& d : dets) {
        uint8_t rgb[3];
        class_color(d.label, rgb);
        const int x0 = int(d.x1 * scale_x), y0 = int(d.y1 * scale_y);
        const int x1 = int(d.x2 * scale_x), y1 = int(d.y2 * scale_y);
        cv.rect(x0, y0, x1, y1, rgb, style.thickness);

        if (!style.labels) {
            continue;
        }
        std::string name = (d.label >= 0 && d.label < (int)class_names.size())
                               ? class_names[d.label]
                               : ("CLASS " + std::to_string(d.label));
        char pct[8];
        snprintf(pct, sizeof(pct), " %d%%", int(d.score * 100.0f + 0.5f));
        std::string label = name + pct;

        const int tw = int(label.size()) * 6 * style.text_scale;
        const int th = 7 * style.text_scale;
        // 박스 위에 붙이되, 이미지 위쪽으로 넘치면 박스 **안쪽**으로 내린다.
        const int ly = (y0 - th - 2 >= 0) ? (y0 - th - 2) : (y0 + 2);
        cv.fill(x0, ly, x0 + tw + 2, ly + th + 2, rgb, 0.75f);
        const uint8_t black[3] = {0, 0, 0};
        cv.text(x0 + 1, ly + 1, label, black, style.text_scale);
    }
}

} // namespace visp
