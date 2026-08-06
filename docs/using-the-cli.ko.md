# 커맨드라인으로 쓰기

[English](using-the-cli.md)

`vision-cli` 는 내장 모델 전부를 코드 한 줄 없이 돌린다. 실행 파일과 `.gguf` 가중치만 있으면 된다.

아직 아무것도 안 돌려봤다면 [시작하기](getting-started.ko.md) 부터 본다.

## 명령의 생김새

```sh
vision-cli <명령> -m <가중치.gguf> -i <입력> -o <출력>
```

명령이 모델을 고르고, `-m` 이 가중치를 지정하고, `-i`·`-o` 는 파일이다.

| 명령 | 하는 일 | 입력 | 출력 |
| :--- | :--- | :--- | :--- |
| `birefnet` | 배경 제거 | 이미지 | 마스크 |
| `sam` | 가리킨 물체 하나를 분리 | 이미지 + 프롬프트 | 마스크 |
| `depthany` | 깊이 추정 | 이미지 | 깊이 맵 |
| `migan` | 인페인팅 — 지정 영역 채우기 | 이미지 + 마스크 | 이미지 |
| `esrgan` | 업스케일 | 이미지 | 커진 이미지 |

## 옵션

`-m, --model <파일>`
:   `.gguf` 가중치. 필수.

`-i, --input <이미지> [<이미지> ...]`
:   입력 이미지. `migan` 은 두 개를 받는다 — 이미지와 마스크.

`-o, --output <파일>`
:   출력 파일. 기본 `output.png`.

`-p, --prompt <x> [<y> ...]`
:   프롬프트를 받는 모델용. `sam` 은 점(`x y`) 또는 박스(`x1 y1 x2 y2`) 를 픽셀 단위로 받는다.
    원점은 좌상단.

`-b, --backend <cpu|gpu>`
:   실행할 장치. 기본은 자동 — Vulkan 으로 빌드됐고 장치가 있으면 GPU, 아니면 CPU.

`--composite <파일>`
:   마스크만 저장하지 않고, 입력 이미지에 마스크를 합친 결과도 같이 쓴다.

`--tile <크기>`
:   큰 입력을 이 크기의 타일로 쪼갠다. `esrgan` 이 메모리를 묶어두기 위해 쓴다.

`-h, --help`
:   명령 목록을 찍고 끝낸다.

## 가중치 받기

모델마다 GGUF 저장소가 따로 있다. 파일을 받아 `-m` 으로 준다.

| 모델 | 가중치 |
| :--- | :--- |
| MobileSAM | [Acly/MobileSAM-GGUF](https://huggingface.co/Acly/MobileSAM-GGUF) |
| BiRefNet | [Acly/BiRefNet-GGUF](https://huggingface.co/Acly/BiRefNet-GGUF) |
| Depth-Anything V2 | [Acly/Depth-Anything-V2-GGUF](https://huggingface.co/Acly/Depth-Anything-V2-GGUF) |
| MI-GAN | [Acly/MIGAN-GGUF](https://huggingface.co/Acly/MIGAN-GGUF) |
| Real-ESRGAN | [Acly/Real-ESRGAN-GGUF](https://huggingface.co/Acly/Real-ESRGAN-GGUF) |

모델마다 변종이 여러 개 있다 — 크기나 해상도가 다르다. 어느 것을 받았는지는 실행 파일이 파일
메타데이터에서 읽으므로 따로 알려줄 필요가 없다.

## 배경 지우기

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i photo.jpg -o mask.png --composite cutout.png
```

`mask.png` 는 주제가 있는 곳이 희다. `cutout.png` 는 배경이 걷힌 사진이다.

## 물체 하나만 분리

배경 제거와 달리 **어느 물체인지** 알려줘야 한다. 안쪽의 점을 주거나,

```sh
vision-cli sam -m MobileSAM-F16.gguf -i photo.jpg -p 300 200 -o mask.png
```

둘러싸는 박스를 준다.

```sh
vision-cli sam -m MobileSAM-F16.gguf -i photo.jpg -p 420 120 650 430 -o mask.png
```

물체가 서로 붙어 있을 때는 박스가 대체로 안정적이다.

## 깊이 추정

```sh
vision-cli depthany -m Depth-Anything-V2-Small-F16.gguf -i photo.jpg -o depth.png
```

출력은 단일 채널 이미지다 — 밝으면 가깝고 어두우면 멀다. 값은 그 이미지 안에서의 상대값이지
실제 거리가 아니다.

## 영역 채우기

인페인팅은 입력을 둘 받는다. 이미지와, 무엇을 바꿀지 표시한 마스크다.

```sh
vision-cli migan -m MIGAN-512-places2-F16.gguf -i photo.jpg mask.png -o filled.png
```

마스크에서 흰 부분이 채울 영역이다. 그 마스크는 `birefnet` 이나 `sam` 으로 만들 수 있어서,
물체 지우기가 두 단계 작업이 된다.

## 업스케일

```sh
vision-cli esrgan -m RealESRGAN-x4plus_anime-6B-F16.gguf -i photo.jpg -o large.png
```

배율은 가중치에서 나온다 — 위 모델은 4배다. 큰 입력은 타일로 나눠 처리하고, 타일 세는 게 보이며,
다른 모델보다 훨씬 오래 걸린다.

## 장치 고르기

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i photo.jpg -o mask.png -b gpu
```

GPU 는 Vulkan 을 켜고 빌드해야 쓸 수 있다 — [Building](../README.md#building) 참조. 없으면
`-b gpu` 로 고를 대상이 없어 CPU 로 돈다. 실제로 어느 장치를 썼는지는 출력 첫 두 줄에 항상 찍힌다.

## 내 가중치 쓰기

라이브러리가 이미 구현한 아키텍처의 체크포인트를 갖고 있다면, 만들어진 파일을 찾지 말고 GGUF 로
변환한다.

```sh
uv run scripts/convert.py <arch> MyModel.pth
```

`<arch>` 는 `sam`·`sam3`·`birefnet`·`depth-anything`·`migan`·`esrgan` 중 하나다.
결과는 `models/` 에 생긴다.

| 옵션 | 설명 |
| :--- | :--- |
| `-o, --output` | 출력 디렉터리 또는 파일. 기본 `models`. |
| `-q, --quantize f16` | 실수 가중치를 f16 으로 저장 — 파일 크기가 대략 절반. |
| `-l, --layout whcn\|cwhn` | 2D 연산의 텐서 레이아웃. 필요한 이유가 분명하지 않으면 지정하지 않는다. |
| `--model-name` | 파일 메타데이터에 기록할 이름. |
| `-v, --verbose` | 변환되는 텐서를 하나씩 찍는다. |

변환은 텐서를 재배치하고 일부를 미리 계산하기도 한다 — 순수한 포맷 변환이 아니다. 체크포인트를
그대로 못 읽는 이유가 이것이다.

이 경로는 **라이브러리에 이미 있는 아키텍처**만 해당한다. 그 밖은
[모델 구현 가이드](model-implementation-guide.md) 로 간다.

## 다음

- [라이브러리로 쓰기](using-the-library.ko.md) — 같은 모델을 내 C++·Python 코드에서.
- [개요](overview.ko.md) — 가중치와 구조를 왜 갈라놨는지.
