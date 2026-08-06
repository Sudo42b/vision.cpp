# 라이브러리로 쓰기

[English](using-the-library.md)

`vision-cli` 가 하는 일은 전부 `libvisioncpp` 호출 몇 개다. 이 문서는 그 호출의 생김새를 보여준다 —
모델을 내 프로그램 안에 넣기 위해서다.

## 기본형

내장 모델은 전부 같은 세 단계다. 장치를 고르고, 가중치를 읽고, 계산한다.

```c++
#include <visp/vision.h>
using namespace visp;

int main() {
    backend_device dev = backend_init();                       // 1. 장치
    birefnet_model model = birefnet_load_model("BiRefNet-lite-F16.gguf", dev);   // 2. 가중치

    image_data input = image_load("photo.jpg");
    image_data mask = birefnet_compute(model, input);          // 3. 계산

    image_save(mask, "mask.png");
}
```

망의 구조는 이미 라이브러리 안에 있다 — 그래서 로딩이 가중치 파일 하나로 끝난다. 시작할 때
텐서를 읽는 것 말고는 파싱하거나 계획하는 단계가 없다.

## 장치

```c++
backend_device dev = backend_init();                 // 가능한 최선
backend_device cpu = backend_init(backend_type::cpu);
backend_device gpu = backend_init(backend_type::gpu);
```

인자 없는 `backend_init()` 은 Vulkan 으로 빌드됐고 장치가 있으면 GPU 를, 없으면 CPU 를 고른다.
모델 하나에 장치 하나를 쓴다. 로드 함수에 넘기면 가중치와 계산이 어디에 놓일지 그게 정한다.

## 모델들

모델마다 `_load_model` 과 `_compute` 가 있다. 차이는 무엇이 들어가고 무엇이 나오는지뿐이다.

| 모델 | 로드 | 계산 |
| :--- | :--- | :--- |
| BiRefNet | `birefnet_load_model(path, dev)` | `birefnet_compute(m, image)` → 알파 마스크 |
| Depth-Anything | `depthany_load_model(path, dev)` | `depthany_compute(m, image)` → 깊이, [0,1] f32 |
| MI-GAN | `migan_load_model(path, dev)` | `migan_compute(m, image, mask)` → 채워진 이미지 |
| ESRGAN | `esrgan_load_model(path, dev)` | `esrgan_compute(m, image)` → 확대된 이미지 |
| MobileSAM | `sam_load_model(path, dev)` | 아래 참조 — 호출 두 번 |

SAM 만 갈라져 있다. 무거운 부분이 프롬프트와 무관하기 때문이다. 이미지를 한 번 인코딩해 두고
물체를 원하는 만큼 물어보면 된다.

```c++
sam_model sam = sam_load_model("MobileSAM-F16.gguf", dev);

sam_encode(sam, image);                                  // 이미지당 한 번

image_data a = sam_compute(sam, i32x2{300, 200});                    // 점으로
image_data b = sam_compute(sam, box_2d{{420, 120}, {650, 430}});     // 박스로
```

프롬프트 좌표는 픽셀 단위이고 원점은 좌상단이다.

## 이미지

`image_data` 는 픽셀을 소유하고, `image_view` 는 남이 소유한 픽셀을 가리킨다. 함수들이 view 를
받으므로 이미 갖고 있는 데이터를 복사 없이 그대로 넘길 수 있다.

```c++
image_data img = image_load("photo.jpg");     // 디스크에서
image_save(img, "out.png");                   // 디스크로

image_view v{extent, image_format::rgba_u8, my_buffer};   // 내 메모리를 감싸기
```

마지막 형태가 카메라·디코더·애플리케이션의 다른 부분에서 프레임이 올 때 쓰는 것이다. 파일을
거칠 필요가 없다.

## 더 내려가기

위의 한 방 호출들은 조합이다. 모델마다 단계가 따로 노출돼 있다 — 파라미터 검출, 전처리, 그래프
구성, 후처리.

```c++
birefnet_params p = birefnet_detect_params(file);   // GGUF 에서 형태·변종을 읽는다
image_data in = birefnet_process_input(image, p);   // 리사이즈·정규화
tensor out = birefnet_predict(m, input_tensor, p);  // 그래프 구성
image_data mask = birefnet_process_output(data, target_extent, p);
```

배치로 묶어야 하거나, 단계 사이에 텐서를 장치에 남겨둬야 하거나, 전처리를 다른 데서 하거나,
계산 그래프를 여러 호출이 공유해야 할 때 이쪽을 쓴다. 그 아래 부품은 `visp/ml.h` 에 있다 —
`model_load`·`model_transfer`·`compute_graph_init`·`compute`.

## 검출 후처리

내장 모델을 쓰는 게 아니라 검출기를 만드는 중이라면, `visp/postproc.h` 에 신경망이 아닌 부분이
들어 있다 — anchor 생성, 박스 디코드, NMS, RoIAlign, 마스크 붙이기. `visp/tracker.h` 에는 프레임
간 신원을 유지하는 ByteTrack 이 있다. 둘 다 순수 CPU 코드이고 프레임워크 config 가 아니라 struct 를
받는다.

[MMDetection 가이드](mmdet-detectors.ko.md) 가 이것들을 조립해 검출기를 만드는 예를 보여준다.

## Python

바인딩은 스크립팅과 대조 작업을 위해 같은 모델들을 덮는다.

```python
from visioncpp import Device, Model, Backend

device = Device.init(Backend.auto)
model = Model.load("BiRefNet-lite-F16.gguf", device)
mask = model.compute(image)
```

`bindings/python` 에 있다. 정본은 C++ API 이고 바인딩이 그걸 따라간다.

## 다음

- [커맨드라인으로 쓰기](using-the-cli.ko.md) — 코드 없이 같은 모델 쓰기.
- [모델 구현 가이드](model-implementation-guide.md) — 라이브러리에 없는 모델 추가하기.
