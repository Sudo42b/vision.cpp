# vision.cpp / tools — 검출 frontend & 검증

검출기를 vision.cpp 에서 실행하기 위한 부품을 **기능(function)별로 분리**한다. 프레임워크(mmdet 등)
지식은 `frontend/<framework>/` 에만 격리하고, 검출 head/decode 부품(`detect/`)과 검증 러너(`verify/`)는
**프레임워크 중립**으로 둔다 — mmdet 외 프레임워크도 여기 붙일 수 있게. **g2c 코어·libvisioncpp
코어는 건드리지 않는다.** 구조는 **backbone/neck = g2c 생성(그대로) + head = `detect/` C++ 부품** 하이브리드.
검출 head 는 trace 가 안 되므로(NMS·동적 offset 등) ggml 로 억지 변환하지 않고 손코딩 C++ 로 조립한다.

## 폴더 구조 (기능별)

```
tools/
  detect/                # 검출 공통 C++ 부품 (프레임워크 무관) — head/decode
      head.h  head.cpp   #   anchor_head_forward · vfnet_head_forward …
  frontend/
      mmdet/             # mmdet 전용: 검출기 → traceable .pt + postproc.json (torch-side)
          mmdet_wrap.py  mmdet_to_pt.py  frcnn_wrap.py  frcnn_to_pt.py
      #                  # (다른 프레임워크는 frontend/<name>/ 로 추가)
  verify/                # E2E 검증 러너 (task별)
      dense_head/  run_vfnet_head.cpp
      roi/         run_frcnn.cpp  run_roi_verify.cpp  run_rpn_verify.cpp
      seg/         run_maskrcnn.cpp
      tracking/    run_bytetrack_verify.cpp
      backbone/    run_dump.cpp  run_mmdet.cpp
  build/                 # 빌드 스크립트  build_{mmdet,frcnn,maskrcnn}_cpp.sh
```

- **`detect/head.cpp` 는 라이브러리가 아니라 러너와 함께 컴파일**된다(`build/build_mmdet_cpp.sh`).
  libvisioncpp 에 넣지 않는다 = 프레임워크 지식이 코어로 새지 않음.
- head 가 쓰는 `conv_2d`/`group_norm`/`conv_2d_deform` 등은 vision.cpp 라이브러리 프리미티브(무수정).

## 흐름 (main 의 run_yolo_cpp 러너 패턴)

```
mmdet config
   │  ① python mmdet_to_pt.py       (전처리 — mmdet 지식은 여기만 → backbone.pt + postproc.json)
   ▼
backbone.pt  +  <name>.postproc.json
   │  ② g2c --model backbone.pt --name <Model>    (g2c 정식 CLI, 코어 무수정·mmdet 코드 없음)
   ▼
output/<Model>/{<Model>.cpp, .h, .gguf}   (백본 forward + 가중치[backbone+head])
   │  ③ build_mmdet_cpp.sh output/<Model>   (run_mmdet + head.cpp + output/.cpp 컴파일, libvisioncpp 링크)
   ▼
output/<Model>/run_mmdet
   │  ④ run_mmdet <gguf> <input> <postproc.json> <out.bin>
   ├─ <Model>_forward   : g2c 백본(output/.cpp) → FPN features out_0..L-1
   ├─ C++ head 부품     : head.cpp (anchor / vfnet …) → raw cls/box
   └─ detect_anchor …   : decode + NMS → 박스
```

## 파일 (기능별 위치)

| 파일 | 역할 |
|---|---|
| `mmdet_wrap.py` | `MMDetBackbone`(backbone+neck features nn.Module, head 는 가중치 유지 attribute) + `postproc_cfg`(anchor/decode + head-conv 구조 + 전처리 메타 추출). **유일한 mmdet 의존 지점.** |
| `mmdet_to_pt.py` | CLI: mmdet config → `backbone.pt` + `<name>.postproc.json`. 피클 모듈명 = `mmdet_wrap` (self-contained import). |
| `head.h` / `head.cpp` | C++ head 부품. `anchor_head_forward`(RetinaNet/ATSS: 공유 cls/reg conv 타워) · `vfnet_head_forward`(VFNet: star deformable offset 계산 + `conv_2d_deform`). 러너와 함께 컴파일. |
| `run_mmdet.cpp` | 러너(제네릭, `-DARCH`). `<Model>_forward`(백본) → head 부품 → `detect_anchor`. |
| `run_vfnet_head.cpp` | VFNet head 격리 검증 harness (torch FPN features → head → cls/box 덤프). |
| `build_mmdet_cpp.sh` | `run_mmdet.cpp` + `head.cpp` + `output/<Model>.cpp` 를 libvisioncpp 와 컴파일. |

decode+NMS·전처리는 vision.cpp 라이브러리(`src/visp/postproc.{h,cpp}` 의 `detect_anchor`/`preprocess`)를
그대로 쓴다. `conv_2d_deform`(DCN)도 라이브러리 프리미티브(ggml `conv_2d_deform` 커널 래퍼).

## 사용 예 (RetinaNet r18)

```bash
PY=<g2c venv python>;  G2C=<GTX_Compiler>;  V=$G2C/vision.cpp
FE=$V/tools/frontend/mmdet   # mmdet frontend (torch-side, mmdet 지식 유일 지점)
BUILD=$V/tools/build         # 빌드 스크립트
CFG=<mmdetection>/configs/retinanet/retinanet_r18_fpn_1x_coco.py

# ① mmdet → backbone.pt + postproc.json  (frontend/mmdet = mmdet 지식 유일 지점)
PYTHONPATH=$FE $PY $FE/mmdet_to_pt.py --config $CFG --out /tmp/rn.pt --size 512

# ② g2c 정식 CLI → output/MMDetBackbone/{cpp,h,gguf}  (g2c 코어 무수정, .pt 는 generic torch 모듈)
PYTHONPATH=$G2C:$FE $PY -m shared.compile.pipeline --model /tmp/rn.pt --name MMDetBackbone \
    --input-shape 1,3,512,512 --output output/MMDetBackbone

# ③ 러너 컴파일 (output/.cpp + verify/backbone/run_mmdet + detect/head.cpp + libvisioncpp)
VISP_BUILD=$V/build bash $BUILD/build_mmdet_cpp.sh output/MMDetBackbone

# ④ 실행 (백본 + C++ head + detect_anchor → 박스). 입력이 이미지면 preprocess() 자동 전처리.
output/MMDetBackbone/run_mmdet output/MMDetBackbone/MMDetBackbone.gguf \
    image.jpg /tmp/rn.postproc.json boxes.bin 512
```

pre(전처리)도 손코딩이 아니라 **범용 부품 + config 추출**: `postproc.cpp` 의 `preprocess()`
(resize+normalize+to_rgb, CPU 스칼라) + mmdet_wrap 이 `data_preprocessor` 에서 `img_mean/img_std/
to_rgb` 자동 추출 → postproc.json.

## 검증

- **RetinaNet r18** (anchor head): C++ head raw cls/box cos 0.999999~1.0, 최종 박스 `predict_by_feat`
  대비 IoU>0.99 매칭 100/100.
- **VFNet r50** (distance + star DCN head): `vfnet_head_forward` (star deformable offset 계산 +
  `conv_2d_deform`) 를 torch head 와 비교 → 5레벨 cls/box **cos 0.999998~1.0** (격리 검증
  `run_vfnet_head`). DCN offset 계산이 자동변환 안 되는 부분을 손코딩으로 해결한 사례.

## Two-stage (Faster R-CNN)

RPN proposal·RoIAlign 은 데이터 의존(proposal 개수 가변)이라 단일 그래프에 안 들어감 → g2c 로
**두 subgraph**(SubA/SubB)만 뽑고, 그 사이는 host C++ op 으로 오케스트레이션.

```
이미지
  │ g2c SubA (backbone+neck+RPN)                    [frcnn_wrap.FRCNN_SubA → 14 출력]
  ▼ P2-P5 + rpn_cls×5 + rpn_bbox×5
  │ rpn_proposals (host)   RPN decode + level NMS → 1000 proposals
  │ roi_align (host)       proposal + P2-P5 → roi_feat (N,256,7,7)
  ▼
  │ g2c SubB (bbox_head Shared2FC)                  [frcnn_wrap.FRCNN_SubB → cls,bbox]
  ▼ cls_score(N,81) + bbox_pred(N,320)
  │ detect_roi (host)      softmax + delta decode + per-class NMS → 박스
  ▼ 최종 박스
```

파일: `frcnn_wrap.py`(SubA/SubB + config), `frcnn_to_pt.py`(→ .pt×2 + frcnn.json),
`run_frcnn.cpp`(오케스트레이션 러너), `build_frcnn_cpp.sh`. host op 은 `postproc.cpp` 의
`rpn_proposals`/`roi_align`/`detect_roi` (라이브러리). 검증 harness: `run_roi_verify.cpp`,
`run_rpn_verify.cpp`.

```bash
python tools/frontend/mmdet/frcnn_to_pt.py --config faster-rcnn_r50_fpn_1x_coco.py --checkpoint frcnn.pth --out /tmp/frcnn
g2c --model /tmp/frcnn/FRCNN_SubA.pt --name FRCNN_SubA --input-shape 1,3,800,800 --output output/FRCNN_SubA
g2c --model /tmp/frcnn/FRCNN_SubB.pt --name FRCNN_SubB --input-shape 4,256,7,7 --output output/FRCNN_SubB
bash tools/build/build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB
output/FRCNN_SubA/run_frcnn output/FRCNN_SubA/FRCNN_SubA.gguf output/FRCNN_SubB/FRCNN_SubB.gguf \
    /tmp/frcnn/frcnn.json input.bin 800
```

**검증 (Faster R-CNN r50, 800, trained, demo.jpg):**
- RoIAlign : torch `bbox_roi_extractor` 대비 **cos 1.0, max|Δ|=7e-07** (1000 proposals)
- RPN proposals : torch `RPNHead.predict_by_feat` 대비 **1000/1000 IoU>0.99**
- E2E 박스 : torch 풀 two-stage 대비 **score>0.3 20/20 매칭(IoU>0.95), score>0.05 48/49**

## Instance segmentation (Mask R-CNN)

Faster R-CNN + mask 분기. 최종 박스에 **2nd RoIAlign(out=14)** → g2c SubC(mask_head FCN:
4×conv + deconv + conv_logits) → `paste_mask`(host).

```
Faster R-CNN → 최종 박스
  │ roi_align(out=14, host)   박스 → mask_feat (M,256,14,14)
  │ g2c SubC (mask_head)      → mask_logits (M,80,28,28)     [frcnn_wrap.MaskRCNN_SubC]
  │ paste_mask (host)         label mask → sigmoid+resize+threshold → 인스턴스 마스크
  ▼
```

파일: `frcnn_wrap.MaskRCNN_SubC`, `run_maskrcnn.cpp`, `build_maskrcnn_cpp.sh`. host op 은
`postproc.cpp` 의 `roi_align`/`paste_mask`.

> ⚠️ **ggml `conv_transpose_2d_p0` 는 batch(N>1) 미지원** → run_maskrcnn 은 SubC 를 **roi 별
> (batch=1)** 로 실행한다. (배치로 돌리면 첫 roi 만 정확: batch0 cos 1.0, batch1+ 깨짐.)

**검증 (Mask R-CNN r50, 800, trained, demo.jpg):** torch 풀 Mask R-CNN 대비
- 박스 20/20 매칭, **mask IoU 평균 0.984, IoU>0.9 20/20** (score>0.3);
- score>0.05: mask IoU 평균 0.975, IoU>0.9 42/43.

## Tracking (MOT — ByteTrack)

검출(그래프)은 그대로 쓰고, 프레임 간 **association**만 host 부품으로 한다. tracking 은 신경망이
아니라 상태추적 로직 → g2c 변환 대상 없음. 상태 유지 class `ByteTracker`(라이브러리 `tracker.{h,cpp}`).

```
프레임별 검출 (위 검출기 그대로)
  → ByteTracker.track(dets, frame_id)
       ① Kalman 예측 (SORT 8-state cxcyah)
       ② IoU 2단계 매칭 (high-score 먼저 → low-score) + tentative/confirmed
       ③ track 생성/유지/삭제 (num_frames_retain)
  → track ID (프레임 간 유지)
```

`ByteTracker` 는 검출기 무관(generic). 검증 harness: `run_bytetrack_verify.cpp`.

**검증:** 합성 검출 시퀀스(10 프레임, 5 물체: 등속 이동 + 등장/소멸)를 mmdet `ByteTracker` 와
동일 입력으로 비교 → **track ID 44/44 완전 일치, 충돌 0** (Kalman·2단계 매칭 복제).

## 확장 (다른 head)

- **anchor**(RetinaNet/ATSS, DeltaXYWHBBoxCoder): `anchor_head_forward` 그대로(cls/reg conv 이름 자동 탐지).
- **VFNet**(distance + DCN): `vfnet_head_forward`. offset = star_dcn_offset(bbox_pred) - dcn_base →
  `conv_2d_deform`. distance decode 는 `postproc.cpp` 의 `detect_fcos` 연결로 박스화(진행 중).
- **FCOS/DETR/two-stage**: `head.cpp` 에 부품 추가 + `postproc` 의 `detect_fcos`/`detect_detr`/`detect_roi` 연결.
