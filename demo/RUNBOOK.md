# RUNBOOK — 빌드부터 실행까지 (단계별)

mmdet 검출 모델을 vision.cpp 로 실행하는 전 과정. 위→아래 순서대로.

## 0. 사전 준비
```sh
git submodule update --init --recursive     # depend/llama(ggml), fmt, stb
cmake --version                             # 3.20+
```

## 1. 라이브러리 빌드
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)              # → build/lib/libvisioncpp, libggml*
```
확인: `[100%] Built target visioncpp`, `Built target vision-cli`.

## 2. gguf 준비 (compose 모델)
gguf 는 g2c(GTX_Compiler)로 mmdet 모델을 컴파일해 생성.
```sh
# GTX_Compiler 쪽에서 (참고 — 이 저장소 아님)
uv run g2c --model "mmdet:retinanet_r50_fpn_1x_coco" --output out/retinanet
# → retinanet.gguf (weights + compose.pipeline) + arch/*.cpp 부품
```
생성된 부품(`backbone_*.cpp` 등)을 `src/visp/arch/` 에 두면 glob 이 자동 빌드.
gguf 확인:
```sh
strings retinanet.gguf | grep "compose.pipeline"   # backbone|...|neck|...|head|...
```

## 3-A. 실행 — Python 프론트 (Flow B)
```sh
pip install -e bindings/python                     # visioncpp 바인딩
python demo/python/detect_mmdet_npu.py \
    configs/retinanet_r50_fpn_1x_coco.py \
    retinanet.gguf \
    dog.jpg
# 출력: box=[x1,y1,x2,y2] label=16 score=0.94 ...
```

## 3-B. 실행 — C++ 자족 (Flow C)
```sh
cmake -B build -DVISP_DEMO=ON
cmake --build build --target detect_standalone
./build/bin/detect_standalone  retinanet.gguf  dog.jpg  gpu
# 출력: 검출 N 개: box=(x1,y1,x2,y2) label=16 score=0.94 ...
#   gpu = NPU(GTX 백엔드 있을 때) / cpu = CPU
```

## 4. 후처리 검증 (선택 — mmdet 대비)
```sh
# 후처리 함수(detect_anchor 등)는 mmdet 레퍼런스 대비 검증됨
# anchors/decode/nms 정확 일치, detect 100~96% recall
# 검증 스크립트는 개발 히스토리 참조(pp_refgen.py / pp_test.cpp)
```

## 5. 다른 검출 계열
데모 2(C++)는 anchor-based(RetinaNet/ATSS/GFL) 기준. 계열별 교체:
| 모델 | 후처리 함수 | params |
|---|---|---|
| RetinaNet/ATSS/GFL/RPN | `detect_anchor` | `det_params` |
| FCOS | `detect_fcos` | `fcos_params` |
| Faster/Mask/Cascade RCNN | `detect_roi` | `roi_params` |
| YOLOX | `detect_yolox` | `yolox_params` |
| DETR | `detect_detr` | `detr_params` |

## 트러블슈팅
| 증상 | 조치 |
|---|---|
| `undefined reference to ggml_*` | submodule 미init → `git submodule update --init --recursive` |
| `compose: 등록 안 된 부품 '...'` | gguf 가 참조하는 부품 .cpp 가 `src/visp/arch/` 에 없음 → g2c 재생성 후 재빌드 |
| `ggml_nelements` / `can_repeat` assert | 부품이 옛 API 로 생성 → g2c 재생성(현 nn.h 기준) |
| gpu 인데 CPU 로 돎 | GTX 백엔드 미빌드(포크 main 은 CPU only). NPU 는 GTX 백엔드 통합 필요 |
| 박스가 안 나옴/이상 | `det_params`(anchor scale·strides·num_classes·means/stds) 모델과 일치하는지 확인 |

## 요약 (한눈)
```
submodule init → cmake build → gguf 준비(g2c) →
  Flow B: python demo/python/detect_mmdet_npu.py <cfg> <gguf> <img>
  Flow C: cmake -DVISP_DEMO=ON → detect_standalone <gguf> <img> [cpu|gpu]
```
