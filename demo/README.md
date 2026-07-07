# demo — mmdet 검출 모델을 vision.cpp 로 실행

mmdet(PyTorch) 검출 모델을 g2c 로 컴파일한 gguf(compose 부품)를 vision.cpp 로 실행하는 두 예제.

```
mmdet config → g2c → vision.cpp 부품 + gguf → 실행 → 박스
```
모델 계산(backbone/neck/head)은 vision.cpp(NPU/CPU), 전·후처리는 아래 두 경로 중 선택.

---

## 데모 1 — Python 프론트  `python/detect_mmdet_npu.py`
전·후처리는 mmdet, 모델 계산은 vision.cpp(NPU).
```python
model = init_detector(cfg)
model.extract_feat = npu_forward(vmodel)   # 계산만 vision.cpp
result = inference_detector(model, image)  # 전·후처리는 mmdet
```
실행:
```sh
python demo/python/detect_mmdet_npu.py  <config.py>  <model.gguf>  <image.jpg>
```
요건: Python + mmdet + visioncpp 바인딩.

## 데모 2 — C++ 자족  `cpp/detect_standalone.cpp`
전처리·모델·후처리 전부 C++ (Python 불필요).
```cpp
auto in   = preprocess(img, ...);            // 전처리
compose_forward(m, x, spec);                 // 모델 (backbone→neck→head)
auto dets = detect_anchor(cls, box, hw, p);  // 후처리
```
빌드·실행:
```sh
cmake -B build -DVISP_DEMO=ON
cmake --build build --target detect_standalone
./build/bin/detect_standalone  <model.gguf>  <image.jpg>  [cpu|gpu]
```
요건: gguf 만.

---

## 후처리 함수 (`src/visp/postproc.h`)
| 계열 | 함수 |
|---|---|
| anchor (RetinaNet/ATSS/GFL/RPN) | `detect_anchor` |
| anchor-free (FCOS) | `detect_fcos` |
| two-stage (Faster/Mask/Cascade RCNN) | `detect_roi` |
| YOLO (YOLOX) | `detect_yolox` |
| DETR | `detect_detr` |
| mask / keypoint | `paste_mask` / `decode_keypoints` |
| 전처리 | `preprocess` (resize + normalize) |

파라미터(anchor scale·strides·score_thr·means/stds)는 `det_params`/`fcos_params` 등으로 지정.
데모 2 는 anchor-based 기준 — 다른 계열은 `detect_fcos`/`detect_yolox`/`detect_detr` 로 교체.

## 구성
- `src/visp/arch/component.{h,cpp}` — compose (config 로 backbone/neck/head 조립)
- `src/visp/arch/{backbone,neck,head}_<hash>` — g2c 생성 부품
- `src/visp/nn.{cpp,h}` — 커널 (group_norm·conv_2d_grouped·conv_2d dilation·conv_2d_deform grouped·conv_2d_wt·pixel_shuffle)
- `src/visp/postproc.{h,cpp}` — 전·후처리
- `demo/` — 위 두 예제. `VISP_DEMO=ON` 으로 빌드.
