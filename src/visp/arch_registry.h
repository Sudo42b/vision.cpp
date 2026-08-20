// arch_registry.h — g2c 가 생성한 arch 를 **이름으로** 찾아 실행한다.
//
// 왜 있나: 생성물을 CLI 에 붙일 때마다 enum·도움말·인자파싱·전방선언·switch·본체를
// 손으로 고쳐야 했다. `inference_yolov9t.cpp` 가 그렇게 7곳을 고쳤는데 CMake 한 줄을
// 빠뜨려 **빌드에서 통째로 빠진 채 죽은 코드**가 됐다(아무도 모르고 있었다).
// 생성물은 시그니처가 균일하므로 등록으로 대신할 수 있다:
//
//     tensor <Arch>_forward(model_ref, tensor, <Arch>_params const&);
//     <Arch>_params <Arch>_detect_params(model_file const&);
//
// 손코딩 arch(birefnet·esrgan·migan·sam·depth-anything)는 전처리·타일링·후처리가
// 제각각이라 **여기 넣지 않는다.** 이 레지스트리는 g2c 생성물 전용이다.
//
// ⚠️ **정적 링크에서는 등록이 통째로 사라진다.** 등록 객체를 아무도 참조하지 않으므로
//    링커가 그 TU 를 버린다. `libvisioncpp` 가 공유 라이브러리라 지금은 안전하다.
//    static 으로 바꾸려면 `--whole-archive` 가 필요하다 → `DECISIONS.md`.
#pragma once

#include "visp/ml.h"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace visp {

// 타입 지움: `<Arch>_params` 는 arch 마다 **다른 타입**이라 시그니처에 못 싣는다.
// 등록 TU 가 `<Arch>_detect_params(f)` 를 안에서 부르고 forward 까지 호출한다.
using arch_forward_fn = tensor (*)(model_ref, tensor, model_file const&);

// 그래프 뒤에 무엇을 해야 하는지. **gguf 에 이 정보가 없다** — g2c 가 안 쓴다.
// g2c 를 고치지 않기로 했으므로 `tools/install_arch.py` 가 모델을 보고 등록 TU 에 박는다.
enum class arch_kind { raw, detect_yolo };

struct arch_task {
    arch_kind kind = arch_kind::raw;

    // detect_yolo 전용. 그래프가 내는 것은 **격자 단위 ltrb + 클래스 로짓**이고,
    // 격자점 곱·시그모이드·top-k 는 호스트가 한다(g2c 가 그 앞에서 끊는다).
    int num_classes = 80;
    std::vector<float> strides{8, 16, 32};
    bool nms_free = true;          // YOLO26/v10: one2one 이라 NMS 없이 top-k
    float score_thr = 0.25f;
    float nms_thr = 0.7f;          // nms_free 면 안 쓴다
    int max_det = 300;
    int input_size = 640;

    // 전처리 정규화 `(v - mean) / std`. 기본값은 0~1 (YOLO 규약).
    // ⚠️ **모델마다 다르다.** torchvision 분류기는 ImageNet 통계를 쓰는데, 0~1 로 돌리면
    //    크래시 없이 값만 틀린다(ResNet-18 실측 상대 L1 5.1e-02, argmax 는 우연히 일치).
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> stdv{255.0f, 255.0f, 255.0f};
    // 그래프 출력 중 몇 번이 박스/점수인지. 생성 모델마다 다르다(one2many 분기가 앞에 올 수 있다).
    int box_out = -1;              // -1 = 자동(뒤에서 두 번째 4채널 출력)
    int score_out = -1;
    std::vector<std::string> class_names;   // 비면 "class N"
};

struct arch_entry {
    std::string_view name;         // gguf `general.architecture` 와 대조
    arch_forward_fn forward;
    arch_task task;
};

// 이름으로 찾는다. 없으면 nullptr.
arch_entry const* arch_find(std::string_view name);

// 등록된 전부(도움말·오류 메시지용). **등록 순서는 링크 순서라 보장되지 않는다** — 정렬해 쓸 것.
std::span<arch_entry const> arch_all();

// 전역 객체로 만들면 등록된다. 생성 arch 옆의 `<name>_register.cpp` 가 만든다.
struct arch_registrar {
    arch_registrar(std::string_view name, arch_forward_fn forward, arch_task task = {});
};

} // namespace visp
