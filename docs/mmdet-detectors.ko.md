# MMDetection 검출기 실행 가이드

[English](mmdet-detectors.md)

[MMDetection](https://github.com/open-mmlab/mmdetection) 의 검출기를 _vision_.cpp 에서
실행하는 방법을 설명한다. 라이브러리가 처음이면 [개요](overview.ko.md) 를 먼저 읽는다.

MMDetection 은 검출기를 백본·넥·head 의 조합으로 정의한다. 백본과 넥은 단순 순전파 망이라
ggml 그래프로 그대로 옮겨진다. head 는 그렇지 않다 — NMS, 동적 offset, 가변 proposal 개수처럼
**데이터에 따라 흐름이 갈리는 부분**이 있어 trace 로 잡히지 않는다. 그래서 _vision_.cpp 는
검출기를 **하이브리드**로 돌린다. 특징 추출기는 컴파일된 ggml 그래프로, head·디코드·후처리는
라이브러리 프리미티브로 조립한 C++ 로 실행한다.

```
 이미지 ──▶ 백본 + 넥 (컴파일된 ggml 그래프) ──▶ FPN features
                                                     │
                                              head   │  tools/detect/head.cpp
                                                     ▼
                                              raw cls / bbox
                                                     │
                                     디코드 + NMS     │  visp/postproc.h
                                                     ▼
                                                  검출 결과
```

그래프 이후는 전부 평범한 CPU 코드다. 프레임워크 config 객체가 아니라 단순 struct 로
파라미터를 받기 때문에, 디코더는 MMDetection 밖에서도 그대로 쓸 수 있다.

## 목차

- [설계](#설계)
- [준비물](#준비물)
- [파이프라인](#파이프라인)
- [검출 head](#검출-head)
- [후처리 API](#후처리-api)
- [Two-stage 검출기](#two-stage-검출기)
- [인스턴스 분할](#인스턴스-분할)
- [다중 객체 추적](#다중-객체-추적)
- [설정 파일 레퍼런스](#설정-파일-레퍼런스)

## 설계

프레임워크 지식은 **한 디렉터리에만** 둔다. 나머지는 프레임워크 중립이라, 다른 검출
프레임워크를 붙이려면 frontend 하나만 새로 쓰면 된다.

```
tools/
  frontend/
    mmdet/            MMDetection 전용. mmdet 을 import 하는 유일한 곳.
      mmdet_wrap.py     traceable 모듈 + config 추출
      mmdet_to_pt.py    CLI: config → backbone.pt + postproc.json
      frcnn_wrap.py     two-stage / mask 서브그래프
      frcnn_to_pt.py    위의 CLI
  detect/             프레임워크 중립 head 부품. 러너와 함께 컴파일된다.
    head.h  head.cpp
  verify/             태스크별 E2E 러너
    backbone/   run_mmdet.cpp  run_dump.cpp
    dense_head/ run_vfnet_head.cpp
    roi/        run_frcnn.cpp  run_roi_verify.cpp  run_rpn_verify.cpp
    seg/        run_maskrcnn.cpp
    tracking/   run_bytetrack_verify.cpp
  build/              러너 빌드 스크립트
```

이 격리를 유지하는 규칙이 둘 있다.

- **`detect/head.cpp` 는 `libvisioncpp` 에 들어가지 않는다.** 러너와 함께 컴파일되므로
  프레임워크에 종속된 구조가 코어 라이브러리로 새지 않는다.
- **디코드는 frontend 가 아니라 라이브러리에 있다.** `detect_anchor`·`roi_align`·
  `rpn_proposals` 는 config 객체가 아니라 숫자를 받는다.

## 준비물

- 소스에서 빌드한 _vision_.cpp. 러너는 `libvisioncpp` 와 ggml 라이브러리에 링크한다.
- MMDetection 이 설치된 Python 환경 — **export 단계에만** 필요하다.
  런타임 경로에는 Python 의존이 없다.

## 파이프라인

네 단계다. 1·2 는 모델당 한 번, 3·4 가 실제 배포 경로다.

### 1단계 — 검출기 export

`mmdet_to_pt.py` 가 MMDetection config 를 읽어, 백본과 넥이 traceable `nn.Module` 이 되도록
감싼 뒤 파일 두 개를 쓴다.

```sh
python tools/frontend/mmdet/mmdet_to_pt.py \
    --config /path/to/retinanet_r18_fpn_1x_coco.py \
    --checkpoint retinanet_r18.pth \
    --out backbone.pt \
    --size 512
```

| 옵션 | 설명 |
| :--- | :--- |
| `--config` | MMDetection config 파일(`.py`). 필수. |
| `--checkpoint` | 가중치(`.pth`). 생략하면 config 초기화값을 쓴다 — shape 확인용이지 정확도 검증용이 아니다. |
| `--out` | traceable 모듈(`.pt`) 출력 경로. 필수. |
| `--size` | trace 에 쓸 정사각 입력 해상도. 기본 `512`. |

`backbone.pt`
:   백본과 넥을 담은 traceable 모듈. head 는 가중치를 `state_dict` 에 남기기 위해 attribute 로
    붙어 있지만 forward 에는 참여하지 않는다.

`backbone.postproc.json`
:   C++ 쪽이 head 를 다시 조립하고 출력을 디코드하는 데 필요한 값 전부 — anchor 생성기 설정,
    bbox coder 통계, head conv 구조, config 의 `data_preprocessor` 에서 뽑은 정규화 값.
    [설정 파일 레퍼런스](#설정-파일-레퍼런스) 참조.

`.pt` 는 모듈명 `mmdet_wrap` 으로 피클되므로, 다시 로드할 때 그 디렉터리가 `PYTHONPATH` 에
있어야 한다.

### 2단계 — 백본 컴파일

`backbone.pt` 는 평범한 PyTorch 모듈이고, PyTorch→ggml 모델 컴파일러가 이를 _vision_.cpp arch
모듈로 컴파일한다. 그 컴파일러 자체는 이 문서의 범위 밖이다. 여기서 중요한 건 **생성된 코드가
지켜야 할 인터페이스**다.

**생성 파일** — 아키텍처 이름이 `MMDetBackbone` 일 때:

| 파일 | 내용 |
| :--- | :--- |
| `MMDetBackbone.h` | 아래 선언들. |
| `MMDetBackbone.cpp` | 백본·넥의 ggml 그래프 구성. |
| `MMDetBackbone.gguf` | **백본과 head 양쪽** 가중치. 원래 `state_dict` 이름 그대로. |

**헤더 계약** — 러너는 매크로로 전개되는 이름 세 개로 그래프에 닿는다.

```c++
namespace visp {

struct MMDetBackbone_params { /* ... */ };

tensor MMDetBackbone_forward(model_ref m, tensor x, MMDetBackbone_params const& p);
MMDetBackbone_params MMDetBackbone_detect_params(model_file const& f);

}  // namespace visp
```

**그래프 계약**

- 입력 텐서 이름은 `x`, 타입 `f32`, shape 는 ggml `ne` 순서로 `{3, size, size, 1}`
  (CWHN — 채널이 가장 빨리 변한다).
- FPN 레벨마다 그래프 텐서에 `out_0`, `out_1`, … `out_{L-1}` 이름을 붙인다. 고해상도 레벨이
  먼저다. 러너가 `ggml_graph_get_tensor` 로 찾으므로 이름이 정확히 일치해야 한다.
- head 가중치가 config 가 쓰는 이름 그대로 GGUF 에 있어야 한다. 예를 들어
  `bbox_head.cls_convs.0.conv.weight`, `bbox_head.retina_cls.weight`. head 부품이 prefix 로
  찾아 쓴다.

레벨을 못 찾으면 러너가 없는 `out_n` 을 알리고 멈춘다 — 생성된 그래프가 출력 이름을 규약대로
붙이지 않았다는 뜻이다.

### 3단계 — 러너 빌드

```sh
bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone
```

스크립트가 아래 셋을 함께 컴파일해 `libvisioncpp` 에 링크한다.

- `tools/verify/backbone/run_mmdet.cpp` — 러너
- `tools/detect/head.cpp` — head 부품
- `output/MMDetBackbone/MMDetBackbone.cpp` — 생성된 그래프

`build_mmdet_cpp.sh <gen_dir> [arch_name]`
:   `gen_dir` 은 생성된 `.cpp`·`.h`·`.gguf` 가 있는 디렉터리다. `arch_name` 을 생략하면 거기 있는
    `.cpp` 의 파일명을 쓴다.

라이브러리는 `build/` 에서 찾는다 — [Building](../README.md#building) 이 거기에 만든다.
다른 곳에 빌드했다면 그 디렉터리를 `VISP_BUILD` 로 알려준다.

```sh
VISP_BUILD=/path/to/that/directory bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone
```

결과물은 `<gen_dir>/run_mmdet` 이다.

### 4단계 — 실행

```sh
output/MMDetBackbone/run_mmdet \
    output/MMDetBackbone/MMDetBackbone.gguf \
    image.jpg \
    backbone.postproc.json \
    boxes.bin \
    512
```

```
run_mmdet <gguf> <input> <postproc.json> <out.bin> [size=512]
```

`<gguf>`
:   2단계에서 나온 가중치.

`<input>`
:   이미지(`.jpg`·`.jpeg`·`.png`·`.bmp`) 또는 전처리가 끝난 텐서(`.bin`).
    이미지면 JSON 에 기록된 mean·std·채널 순서로 `preprocess()` 가 리사이즈·정규화까지 처리한다.
    `.bin` 이면 그대로 쓰므로 `3 × size × size` 개의 `float32` 가 CWHN 순서로 들어 있어야 한다.

`<postproc.json>`
:   1단계에서 나온 사이드카.

`<out.bin>`
:   출력 경로. 검출 하나가 `float32` 여섯 개다 — `x1, y1, x2, y2, score, label`.
    좌표는 입력 이미지 픽셀 단위.

`[size]`
:   입력 해상도. 1단계의 `--size`, 그리고 그래프를 컴파일한 shape 와 같아야 한다.

## 검출 head

head 부품은 FPN features 를 디코더가 기대하는 레벨별 raw 텐서로 바꾼다.
선언은 `tools/detect/head.h` 에 있다.

### `anchor_head_forward`

공유 conv 타워 뒤에 분류·회귀 conv 가 붙는 구조 — RetinaNet·ATSS·GFL 등 anchor 기반 dense head
가 쓰는 배치다. 모든 레벨이 가중치 한 벌을 공유한다.

```c++
void anchor_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c,
                         std::vector<tensor>& cls_out, std::vector<tensor>& box_out);
```

| `anchor_head_cfg` | 기본값 | 설명 |
| :--- | :--- | :--- |
| `stacked_convs` | `4` | 공유 cls/reg 타워의 깊이. |
| `feat_channels` | `256` | 타워 내부 채널 수. |
| `num_base` | `9` | location 당 anchor 수. |
| `num_classes` | `80` | 분류 출력 채널 수. |
| `cls_convs_prefix` | `bbox_head.cls_convs` | cls 타워 가중치 이름 prefix. |
| `reg_convs_prefix` | `bbox_head.reg_convs` | reg 타워 가중치 이름 prefix. |
| `cls_head` | `bbox_head.retina_cls` | 최종 분류 conv. |
| `reg_head` | `bbox_head.retina_reg` | 최종 회귀 conv. |
| `head_has_norm` | `false` | 타워에 정규화 레이어가 있는지. |

레벨 `l` 의 출력 shape (ggml `ne` 순서):

- `cls_out[l]` — `{num_base * num_classes, feat_w, feat_h, 1}`
- `box_out[l]` — `{num_base * 4, feat_w, feat_h, 1}`

이걸 [`detect_anchor`](#detect_anchor) 에 넣는다.

### `vfnet_head_forward`

VFNet 은 anchor delta 대신 거리를 예측하고, 첫 bbox 예측에서 계산한 offset 으로 star 형태
deformable conv 를 돌려 결과를 보정한다. **그 offset 계산이 정확히 trace 가 안 되는 부분**이라,
여기서 그래프로 직접 조립한다.

```c++
void vfnet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        vfnet_head_cfg const& c, tensor dcn_base,
                        std::vector<tensor>& cls_out, std::vector<tensor>& box_out);
```

`dcn_base` 는 고정 3×3 샘플링 그리드로 shape 는 `{18, 1, 1, 1}`, 호출부가 채워서 넘긴다.
부품은 `offset = star_dcn_offset(bbox_pred) − dcn_base` 를 계산해 라이브러리 프리미티브
`conv_2d_deform` 에 넣는다. 레벨별로 `cls_out[l]` 은 `{num_classes, w, h, 1}`,
`box_out[l]` 은 `{4, w, h, 1}` 이다.

`vfnet_head_cfg` 에는 `gn_groups`(타워의 GroupNorm 그룹 수), `strides`(레벨별, offset 을 feature
스케일로 투영), `reg_denoms`(레벨별, `bbox_pred = exp(reg) · reg_denom`) 가 더 있다.

### head 추가하기

1. `tools/detect/head.cpp` 에 FPN features → 레벨별 raw 텐서를 만드는
   `<name>_head_forward` 를 추가한다. 라이브러리 프리미티브(`conv_2d`·`group_norm`·
   `conv_2d_deform`)를 쓰고, **프레임워크에 종속된 코드를 `src/visp` 에 넣지 않는다.**
2. `mmdet_wrap.postproc_cfg` 에서 head 의 구조 파라미터를 뽑아 사이드카 JSON 에 싣는다.
3. raw 출력을 `visp/postproc.h` 의 대응 디코더에 연결한다. 디코드 방식이 새로우면 디코더를
   추가한다.

## 후처리 API

`src/visp/postproc.h` 에 선언돼 있고, ggml 의존 없는 순수 CPU 코드다. 레벨별 입력은 전부 CWHN
순서의 평탄한 `float` 버퍼(`index = (y · W + x) · C + c`)이며, 레벨별 `(feat_h, feat_w)` 를 함께
넘긴다.

```c++
struct detection {
    float x1, y1, x2, y2;  // 픽셀 좌표
    float score;
    int label;
};
```

### 전처리

`std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c, int out_size, float const mean[3], float const std[3], bool to_rgb, int* out_w = nullptr, int* out_h = nullptr)`
:   `out_size × out_size` 로 리사이즈하고 `(v − mean) / std` 로 정규화한다. 채널 순서 교환은
    선택. 그래프 입력에 바로 넣을 수 있는 CWHN `float32` 텐서를 돌려준다.

### Dense head

<a id="detect_anchor"></a>
`std::vector<detection> detect_anchor(cls_scores, bbox_preds, feat_hw, det_params const& p)`
:   anchor 기반 디코드 — anchor 생성, delta 디코드, 레벨별 top-k, score 임계, NMS.
    RetinaNet·ATSS·GFL 및 RPN 계열 head 가 쓴다.

`std::vector<detection> detect_fcos(cls_scores, bbox_preds, centerness, feat_hw, fcos_params const& p)`
:   anchor-free 거리 디코드 + centerness 가중.

`std::vector<detection> detect_yolox(cls, box, obj, feat_hw, yolox_params const& p)`
:   objectness 분기가 있는 grid 디코드. score 는 `sigmoid(cls) · sigmoid(obj)`.

`std::vector<detection> detect_detr(float const* cls, float const* bbox, detr_params const& p)`
:   set prediction. query logit 과 정규화된 `cxcywh` 박스를 받아 top-k 만 하고 **NMS 는 하지
    않는다.** Deformable-DETR 계열이면 `use_sigmoid` 를 켠다.

`det_params` 에는 anchor 생성기(`strides`·`octave_base_scale`·`octave_scales`·`ratios`·
`center_offset`), bbox coder(`means`·`stds`), 테스트 시 임계값(`score_thr`·`nms_thr`·
`nms_pre`·`max_per_img`) 이 들어간다. `input_w`/`input_h` 는 박스를 이미지 안으로 자른다.

### Two-stage 부품

`std::vector<float> rpn_proposals(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`
:   RPN 출력에서 region proposal 생성 — anchor 디코드, 레벨별 top-k, 레벨 통합 NMS.
    이미지 좌표의 `M × 4` 박스를 돌려준다(`M ≤ max_per_img`).

`std::vector<float> roi_align(feats, feat_hw, float const* rois, int m, roi_align_params const& p)`
:   MMCV 호환 RoIAlign(`aligned = true`, 적응적 `sampling_ratio`). 레벨 배정은
    `clamp(floor(log2(sqrt(w·h) / finest_scale + 1e-6)), 0, L−1)` 을 따른다.
    `M × C × out × out` 을 NCHW 순서로 돌려준다.

`std::vector<detection> detect_roi(float const* scores, float const* bbox_deltas, float const* proposals, int n, roi_params const& p)`
:   RoI head 최종 디코드 — 클래스별 delta 디코드 + 클래스별 NMS. `scores` 는 softmax 를 마친
    값이고 배경이 마지막이다. `bbox_pred` 가 `num_classes · 4` 가 아니라 4열이면
    `class_agnostic` 을 켠다.

### 마스크·키포인트

`std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw, detection const& box, float thr = 0.5f, int* out_h = nullptr, int* out_w = nullptr)`
:   sigmoid → 박스 크기로 리사이즈 → 임계. 박스 영역을 덮는 이진 마스크를 돌려준다.

`std::vector<float> decode_keypoints(float const* heatmap, int k, int hm_h, int hm_w, float stride)`
:   heatmap 에서 키포인트별 argmax. `k × 3` 을 `(x, y, score)` 로 돌려준다.

### 구성 요소

`gen_anchors`·`gen_points`·`delta2bbox`·`distance2bbox`·`nms` 는 개별로도 노출돼 있어
직접 디코더를 만들 때 쓸 수 있다.

## Two-stage 검출기

RPN proposal 과 RoIAlign 은 데이터 의존적이다 — proposal 개수가 망을 돌려 봐야 정해진다.
그래서 two-stage 검출기는 단일 그래프가 될 수 없다. 컴파일된 서브그래프 둘로 나누고 그 사이를
host 코드가 잇는다.

```
 이미지
   │ SubA (백본 + 넥 + RPN)          출력 14개: P2-P5, rpn_cls×5, rpn_bbox×5
   ▼
   │ rpn_proposals (host)           디코드 + 레벨별 NMS → proposal 1000개
   │ roi_align (host)               proposal + P2-P5 → roi_feat (N,256,7,7)
   ▼
   │ SubB (bbox head, Shared2FC)    → cls_score (N,81), bbox_pred (N,320)
   ▼
   │ detect_roi (host)              softmax + delta 디코드 + 클래스별 NMS
   ▼ 최종 박스
```

`frcnn_to_pt.py` 로 서브그래프 둘을 export 하고, 각각 컴파일한 뒤 빌드·실행한다.

```sh
python tools/frontend/mmdet/frcnn_to_pt.py \
    --config faster-rcnn_r50_fpn_1x_coco.py --checkpoint frcnn.pth --out /tmp/frcnn
# /tmp/frcnn/FRCNN_SubA.pt 는 1,3,800,800 로, FRCNN_SubB.pt 는 4,256,7,7 로 컴파일

bash tools/build/build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB

output/FRCNN_SubA/run_frcnn \
    output/FRCNN_SubA/FRCNN_SubA.gguf output/FRCNN_SubB/FRCNN_SubB.gguf \
    /tmp/frcnn/frcnn.json input.bin 800
```

`run_roi_verify` 와 `run_rpn_verify` 는 두 host 단계를 격리해서 레퍼런스 구현의 덤프와 대조한다.

## 인스턴스 분할

Mask R-CNN 은 위 흐름에 최종 박스에 대한 두 번째 RoIAlign(출력 크기 14), mask 서브그래프,
host 쪽 마스크 붙이기를 더한다.

```
 최종 박스
   │ roi_align (out=14, host)   → mask_feat (M,256,14,14)
   │ SubC (mask head FCN)       → mask_logits (M,80,28,28)
   │ paste_mask (host)          → 인스턴스별 이진 마스크
   ▼
```

```
run_maskrcnn <SubA.gguf> <SubB.gguf> <SubC.gguf> <mask.json> <input.bin> <out_prefix> [size=800]
```

> **주의**
> ggml 의 `conv_transpose_2d_p0` 는 배치를 지원하지 않는다. 그래서 `run_maskrcnn` 은 mask
> 서브그래프를 RoI 하나씩 돌린다. 배치로 돌리면 첫 RoI 만 맞는다.

## 다중 객체 추적

추적은 신경망이 아니라 상태 관리라 컴파일할 대상이 없다. `ByteTracker`
(`src/visp/tracker.h`) 가 프레임 간 track 상태를 유지하며, 검출기와 무관하게 위 디코더 중
무엇이 낸 `std::vector<detection>` 이든 받는다.

```c++
ByteTracker tracker;                       // 임계값은 byte_params 로 조정
for (int frame = 0; frame < n; ++frame) {
    std::vector<detection> dets = /* 검출기 실행 */;
    std::vector<track_result> tracks = tracker.track(dets, frame);
    // tracks[i].id 는 프레임이 바뀌어도 유지된다
}
```

호출할 때마다 8-state `cxcyah` 모델로 Kalman 예측을 하고, IoU 2단계 매칭(고득점 검출 먼저, 그
다음 저득점)을 거쳐, track 수명을 관리한다 — `num_tentatives` 회 연속 매칭되면 confirmed,
`num_frames_retain` 프레임 동안 매칭이 없으면 삭제. `frame_id == 0` 을 넘기면 추적기가
초기화된다.

## 설정 파일 레퍼런스

`<name>.postproc.json` 은 export 단계가 쓰고 러너가 읽는다. 필드를 쓰는 쪽 기준으로 묶었다.

**전처리** — 러너에 `.bin` 이 아니라 이미지를 줄 때만 쓴다.

| 필드 | 설명 |
| :--- | :--- |
| `img_mean`, `img_std` | 채널별 정규화 값. config 의 `data_preprocessor` 에서 뽑는다. |
| `to_rgb` | 정규화 전에 채널 순서를 바꿀지. |

**head 재구성** — `anchor_head_cfg` 에 대응한다.

| 필드 | 설명 |
| :--- | :--- |
| `head_type` | 지원하는 dense head 면 `anchor`, features 만 export 했으면 `raw`. |
| `stacked_convs`, `feat_channels` | 공유 타워의 형태. |
| `cls_convs_prefix`, `reg_convs_prefix` | 두 타워의 가중치 이름 prefix. |
| `cls_head`, `reg_head` | 최종 conv 의 이름. |
| `head_has_norm` | 타워에 정규화 레이어가 있는지. |

**디코드** — `det_params` 에 대응한다.

| 필드 | 설명 |
| :--- | :--- |
| `strides` | FPN 레벨별 stride. 길이가 곧 레벨 수 `L` 이다. |
| `octave_base_scale`, `octave_scales`, `ratios`, `center_offset` | anchor 생성기. |
| `num_base` | location 당 anchor 수, `len(octave_scales) · len(ratios)`. |
| `means`, `stds` | delta coder 통계. |
| `num_classes`, `use_sigmoid` | 분류 출력 형태와 활성 함수. |

`head_type` 이 `raw` 면 config 의 head 를 인식하지 못했다는 뜻이다. 백본은 그대로 export 되지만
디코드는 호출부가 직접 해야 한다.

단계별 격리 harness 가 `tools/verify/` 에 있다 — dense head 는 `run_vfnet_head`,
two-stage host 부품은 `run_rpn_verify`·`run_roi_verify`, 추적은 `run_bytetrack_verify`.
각각 한 단계만 떼어 레퍼런스 구현의 덤프와 대조하므로, 값이 어긋날 때 원인 위치를 가장 빨리
좁힐 수 있다.
