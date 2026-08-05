#!/usr/bin/env bash
#
# build_mmdet_cpp.sh — g2c 가 생성한 mmdet 백본(output/<ARCH>.cpp)을 러너(verify/backbone/run_mmdet)·
# head 부품(detect/head.cpp)과 함께 컴파일해 libvisioncpp 에 링크한다.
# head.cpp 는 라이브러리가 아니라 여기서 러너와 함께 컴파일된다 → g2c output/.cpp 를 직접
# 컴파일(arch/ 복사·cli REG 없음).
#
# 사용:  build_mmdet_cpp.sh <gen_dir> [arch_name]
#   gen_dir  = g2c --output 디렉토리 (예: output/MMDetBackbone) — <ARCH>.cpp/.h/.gguf 있음
#   arch_name= 클래스명(생략 시 gen_dir 의 *.cpp 에서 자동)
# env:  VISP_BUILD = libvisioncpp 빌드 디렉토리 (기본: <vision.cpp>/build)
#
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # vision.cpp/tools/build (빌드 스크립트)
V="$(cd "$SELF/../.." && pwd)"            # vision.cpp (libvisioncpp 소스)
DETECT="$V/tools/detect"                  # 공용 head/decode 부품 (head.cpp/head.h)
RUN="$V/tools/verify"                     # E2E 검증 러너
GEN="${1:?usage: build_mmdet_cpp.sh <gen_dir> [arch_name]}"
GEN="$(cd "$GEN" && pwd)"
ARCH="${2:-}"
if [ -z "$ARCH" ]; then
  ARCH="$(basename "$(ls "$GEN"/*.cpp | grep -v run_ | head -1)" .cpp)"
fi
BUILD="${VISP_BUILD:-$V/build}"
LIB="$BUILD/lib"
[ -f "$LIB/libvisioncpp.so" ] || { echo "libvisioncpp.so 없음: $LIB (VISP_BUILD 설정?)"; exit 1; }

# VISP_FMT_LIB: fmt 라이브러리 사용 플래그. libvisioncpp 가 fmt 로 빌드됐으면 -DVISP_FMT_LIB +
# fmt include 필요. 아니면(내장 fallback) 정의하지 않는다. FMT_INC 있으면 자동으로 켠다.
FMT_INC="$BUILD/_deps/fmt-src/include"
FMT_FLAGS=""
[ -f "$FMT_INC/fmt/format.h" ] && FMT_FLAGS="-DVISP_FMT_LIB -I$FMT_INC"

echo "arch=$ARCH gen=$GEN build=$BUILD fmt=${FMT_FLAGS:-fallback}"
INC="$GEN/inc"
mkdir -p "$INC/visp/arch"
cp "$GEN/$ARCH.h" "$INC/visp/arch/$ARCH.h"

# run_mmdet.cpp + head.cpp(러너와 함께 컴파일, 라이브러리 아님) + g2c 백본 output/.cpp
g++ -std=c++20 -O2 $FMT_FLAGS \
  -DARCH="$ARCH" -DVISP_ARCH_HEADER="\"visp/arch/$ARCH.h\"" \
  -I"$DETECT" -I"$INC" -I"$V/include" -I"$V/src" \
  -I"$V/depend/llama/ggml/include" -I"$V/depend/llama/vendor" \
  "$RUN/backbone/run_mmdet.cpp" "$DETECT/head.cpp" "$GEN/$ARCH.cpp" \
  -L"$LIB" -lvisioncpp -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$LIB" \
  -o "$GEN/run_mmdet"
echo "built: $GEN/run_mmdet (arch=$ARCH)"
