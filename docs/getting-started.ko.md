# 시작하기

[English](getting-started.md)

이 튜토리얼에서는 _vision_.cpp 로 사진에서 물체를 오려낸다. 5분쯤 걸리고, 릴리스 패키지와 모델
파일 하나, 이미지 하나면 된다 — 빌드도, Python 도, 변환도 필요 없다.

끝나면 이 두 개가 생긴다.

| | |
| :--- | :--- |
| `mask.png` | 물체 영역을 흑백으로 나타낸 마스크 |
| `object.png` | 원본 사진에서 배경만 어둡게 걷어낸 것 |

## 1단계 — 실행 파일 받기

[릴리스 패키지](https://github.com/Acly/vision.cpp/releases) 를 받아 푼다. `bin` 폴더에
`vision-cli` 가 있다.

실행되는지 확인한다.

```sh
vision-cli --help
```

명령 목록이 보이면 된다 — `sam`, `birefnet`, `depthany`, `migan`, `esrgan`.

> 소스에서 빌드하고 싶으면 [Building](../README.md#building) 을 먼저 따르고 돌아온다.
> `vision-cli` 는 `build/bin` 에 생긴다.

## 2단계 — 모델과 이미지 준비

실행 파일에는 신경망의 **구조**가 들어 있지만 **가중치**는 없다. 받는다.

```sh
curl -L -O https://huggingface.co/Acly/BiRefNet-GGUF/resolve/main/BiRefNet-lite-F16.gguf
```

BiRefNet — 사진에서 주제와 배경을 갈라내는 모델이다. 이 파일은
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 로, 가중치 말고는 아무것도
들어 있지 않다.

입력은 주제가 뚜렷한 사진이면 아무거나 된다. 저장소를 클론했다면 `docs/media/input.jpg` 가 있다.
모델 파일 옆에 `input.jpg` 라는 이름으로 둔다.

## 3단계 — 실행

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i input.jpg -o mask.png --composite object.png
```

진행 상황이 그대로 찍힌다.

```
Initializing backend... done (1.1 ms)
- device: CPU - Intel(R) Core(TM) i3-14100
Loading model weights from 'BiRefNet-lite-F16.gguf'... done (151.3 ms)
- float type: f16
- tensor layout: cwhn
- model image size: 1024
- inference image size: 1024x1024
- flash attention: off
Running inference... complete (5372.6 ms)
-> mask saved to mask.png
-> image composited and saved to object.png
```

데스크톱 CPU 에서 추론은 몇 초 걸린다. 가중치 로딩이 1초도 안 걸리는 것이 이 프로젝트가 노리는
지점이고, 어느 장비에서든 비슷하다.

## 4단계 — 결과 보기

`object.png` 를 연다. 주제는 그대로고 배경만 사라져 있다.

`mask.png` 가 모델이 실제로 만들어낸 것이다 — 주제가 있는 곳은 희고 나머지는 검다.
`object.png` 는 전부 이 마스크에서 계산된 결과다.

**이게 전부다.** 신경망을 이미 알고 있는 실행 파일, 가중치를 담은 `.gguf`, 이미지 하나를 넣고
결과 하나를 받는 것.

## 하나만 더

같은 실행 파일이 다른 내장 모델도 돌린다. 바뀌는 건 명령과 가중치뿐이다.

```sh
curl -L -O https://huggingface.co/Acly/Real-ESRGAN-GGUF/resolve/main/RealESRGAN-x4plus_anime-6B-F16.gguf

vision-cli esrgan -m RealESRGAN-x4plus_anime-6B-F16.gguf -i input.jpg -o upscaled.png
```

이건 이미지를 4배로 키운다. 타일 단위로 돌아서 눈에 띄게 오래 걸리고, 타일 세는 게 보인다.

## 다음

- [개요](overview.ko.md) — 이 라이브러리가 무엇이고, 왜 구조와 가중치를 갈라놨는지.
- [README](../README.md#features) — 나머지 내장 모델과 각각이 하는 일.
- [모델 구현 가이드](model-implementation-guide.md) — 원하는 모델이 목록에 없어서 직접 넣어야 할 때.
- [MMDetection 검출기](mmdet-detectors.ko.md) — 구조를 손으로 쓰지 않고 생성해서 쓰는 검출기.
