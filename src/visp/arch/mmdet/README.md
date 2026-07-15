# vision.cpp / src/visp/arch/mmdet

mmdet 검출기를 vision.cpp 에서 실행하기 위한 **부품 모음**(arch 모듈). 구조는
**backbone/neck = g2c 생성(그대로 사용) + head = 여기 C++ 부품** 하이브리드다.
검출 head 는 trace 가 안 되므로(NMS 등) ggml 로 억지 변환하지 않고 손코딩 C++ 부품으로 조립한다.
**g2c 코어는 건드리지 않는다.**

배치(역할=레포 경계):
- **C++ head 부품**(`head.h/cpp`) = 이 폴더(vision.cpp, libvisioncpp 에 1회 컴파일).
- **러너**(`tools/run_mmdet.cpp` + `build_mmdet_cpp.sh`) = **g2c 레포**(run_yolo_cpp 옆). output 직접 컴파일.
- **Python 전처리**(mmdet→.pt) = **g2c 레포** `frontends/mmdet/`(g2c 코어 무수정, mmdet 의존은 여기만).

## 흐름 (main 의 run_yolo_cpp 러너 패턴)

```
mmdet config
   │  ① [g2c] frontends/mmdet/mmdet_to_pt.py     (전처리 — mmdet 지식은 여기만)
   ▼
backbone.pt  +  <name>.postproc.json
   │  ② [g2c] g2c --model backbone.pt --name <Model>   (g2c 정식 CLI, 무수정)
   ▼
output/<Model>/{<Model>.cpp, .h, .gguf}   (백본 forward + 가중치[backbone+head])
   │  ③ [g2c] tools/build_mmdet_cpp.sh output/<Model>   (러너 + output/.cpp 컴파일, arch 복사·REG 없음)
   ▼
output/<Model>/run_mmdet
   │  ④ run_mmdet <gguf> <input> <postproc.json> <out.bin>
   ├─ <Model>_forward   : g2c 백본(output/.cpp) → FPN features out_0..L-1
   ├─ C++ head 부품     : head.cpp anchor_head_forward → raw cls/box
   └─ detect_anchor     : decode + NMS → 박스
```

> 대안(컴파일 0): `serialize.py`(tools/graph_export) 로 backbone 을 graph.nodes gguf 로 만들어
> `vision-cli run` 인터프리터로 실행하는 경로도 있다(head 부품·detect_anchor 동일 재사용).

## 파일

**이 폴더 (`src/visp/arch/mmdet/`, vision.cpp) — C++ arch 부품:**

| 파일 | 역할 |
|---|---|
| `head.h` / `head.cpp` | C++ head 부품. `anchor_head_forward`: features → 공유 cls/reg conv 타워 + 최종 cls/reg conv → raw cls_score/bbox_pred(cwhn). libvisioncpp 에 컴파일됨. |

**`tools/` (g2c 레포 GTX_Compiler) — 러너 (g2c output 직접 컴파일, arch 복사·REG 없음):**

| 파일 | 역할 |
|---|---|
| `run_mmdet.cpp` | 러너 1개(제네릭, `-DARCH`). `<Model>_forward`(백본) → `anchor_head_forward`(head) → `detect_anchor`. |
| `build_mmdet_cpp.sh` | `output/<Model>.cpp` + `run_mmdet.cpp` 를 libvisioncpp 와 컴파일 → `run_mmdet`. |

**`frontends/mmdet/` (g2c 레포 GTX_Compiler) — Python 전처리:**

| 파일 | 역할 |
|---|---|
| `mmdet_wrap.py` | `MMDetBackbone`(backbone+neck features nn.Module, head 는 가중치 유지 attribute) + `postproc_cfg`(anchor/decode + head-conv 구조 추출). **유일한 mmdet 의존 지점.** |
| `mmdet_to_pt.py` | CLI: mmdet config → `backbone.pt` + `<name>.postproc.json` |

decode+NMS 는 vision.cpp `src/visp/postproc.{h,cpp}` 의 `detect_anchor` 를 그대로 쓴다(이미 구현됨).

## 사용 예 (RetinaNet r18)

```bash
PY=<g2c venv python>;  G2C=<GTX_Compiler>;  V=$G2C/vision.cpp
CFG=<mmdetection>/configs/retinanet/retinanet_r18_fpn_1x_coco.py

# ① [g2c] mmdet → backbone.pt + postproc.json  (mmdet 환경, mmdet 지식은 여기만)
PYTHONPATH=$G2C $PY $G2C/frontends/mmdet/mmdet_to_pt.py --config $CFG --out /tmp/rn.pt --size 512

# ② [g2c] g2c 정식 CLI → output/MMDetBackbone/{cpp,h,gguf}  (백본, g2c 무수정)
PYTHONPATH=$G2C $PY -m shared.compile.pipeline --model /tmp/rn.pt --name MMDetBackbone \
    --input-shape 1,3,512,512 --output output/MMDetBackbone

# ③ [g2c] 러너 컴파일 (output/.cpp + run_mmdet + libvisioncpp)
VISP_BUILD=<libvisioncpp build> bash $G2C/tools/build_mmdet_cpp.sh output/MMDetBackbone

# ④ 실행 (백본 + C++ head + detect_anchor → 박스)
#   입력이 이미지(.jpg/.png)면 preprocess()(mmdet mean/std, postproc.json)로 자동 전처리,
#   .bin 이면 이미 전처리된 CWHN f32 텐서로 간주.
output/MMDetBackbone/run_mmdet output/MMDetBackbone/MMDetBackbone.gguf \
    image.jpg /tmp/rn.postproc.json boxes.bin 512
```

pre(전처리)도 손코딩이 아니라 **범용 부품 + config 추출**: `postproc.cpp` 의 `preprocess()`
(resize+normalize+to_rgb, CPU 스칼라) + mmdet_wrap 이 `data_preprocessor` 에서 `img_mean/img_std/
to_rgb` 자동 추출 → postproc.json. yolo 처럼 pre 를 모델별로 손코딩하지 않는다.

## 검증 (RetinaNet r18, 512, CPU)

- C++ head raw cls/box(5레벨×2) : PyTorch head 대비 **cos 0.999999 ~ 1.0**
- 최종 박스(러너) : PyTorch `predict_by_feat` 대비 **IoU>0.99 매칭 100/100**, top 서브픽셀 일치
  (인터프리터 경로도 99~100/100)

## 확장 (다른 head)

`postproc_cfg` 가 `head_type=="anchor"`(DeltaXYWHBBoxCoder)만 지원. ATSS 등 같은 anchor head 는
그대로 동작(cls/reg conv 이름 자동 탐지). FCOS(distance)/DETR/two-stage 는 `head.cpp` 에 부품
추가 + `postproc.{h,cpp}` 의 `detect_fcos`/`detect_detr`/`detect_roi` 연결로 확장.
