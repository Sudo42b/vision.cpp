#!/usr/bin/env bash
#
# build_mmdet_cpp.sh — g2c 가 생성한 mmdet 백본(output/<ARCH>.cpp)을 러너(verify/backbone/run_mmdet)·
# head 부품(detect/head.cpp)과 함께 컴파일해 libvisioncpp 에 링크한다.
# head.cpp 는 라이브러리가 아니라 여기서 러너와 함께 컴파일된다 → g2c output/.cpp 를 직접
# 컴파일(arch/ 복사·cli REG 없음).
#
# usage: build_mmdet_cpp.sh <gen_dir> <params.h> [arch_name]
#   gen_dir  = g2c --output 디렉토리 (예: output/MMDetBackbone) — <ARCH>.cpp/.h/.gguf 있음
#   params.h = <name>.postproc.h written by mmdet_to_pt.py next to its --out. When omitted,
#              a *.postproc.h inside gen_dir is used.
#   arch_name= 클래스명(생략 시 gen_dir 의 *.cpp 에서 자동)
#
# options:
#   --build <dir>  libvisioncpp 빌드 디렉토리 (기본: <vision.cpp>/build)
#                  ⚠️ **어떤 라이브러리로 빌드했는지는 명령줄에 남아야 한다.** 예전엔 env
#                     (`VISP_BUILD`)로만 받았는데, 낡은 build/ 를 참조해 링크가 깨졌을 때
#                     로그만 봐서는 원인을 알 수 없었다. env 도 계속 읽지만 인자가 우선이다.
#
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # vision.cpp/tools/build (빌드 스크립트)
V="$(cd "$SELF/../.." && pwd)"            # vision.cpp (libvisioncpp 소스)
DETECT="$V/tools/detect"                  # 공용 head/decode 부품 (head.cpp/head.h)
RUN="$V/tools/verify"                     # E2E 검증 러너
# `--build <dir>` 를 어디에 두든 받는다 — 위치 인자와 섞이지 않게 먼저 걷어낸다.
ARG_BUILD=""
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --build) ARG_BUILD="${2:?--build needs a directory}"; shift 2 ;;
    --build=*) ARG_BUILD="${1#--build=}"; shift ;;
    *) POS+=("$1"); shift ;;
  esac
done
set -- "${POS[@]}"

GEN="${1:?usage: build_mmdet_cpp.sh [--build <dir>] <gen_dir> [params.h] [arch_name]}"
GEN="$(cd "$GEN" && pwd)"
# The second argument is either the parameters header (.h) or an architecture name.
PARAMS=""
ARCH=""
case "${2:-}" in
  *.h) PARAMS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; ARCH="${3:-}" ;;
  "")  ;;
  *)   ARCH="$2" ;;
esac
if [ -z "$PARAMS" ]; then
  PARAMS="$(ls "$GEN"/*.postproc.h 2>/dev/null | head -1)"
fi
if [ ! -f "$PARAMS" ]; then
  echo "no parameters header. mmdet_to_pt.py writes <name>.postproc.h next to its --out."
  echo "  usage: $(basename "$0") $GEN <name>.postproc.h"
  exit 1
fi
if [ -z "$ARCH" ]; then
  ARCH="$(basename "$(ls "$GEN"/*.cpp | grep -v run_ | head -1)" .cpp)"
fi
BUILD="${ARG_BUILD:-${VISP_BUILD:-$V/build}}"
LIB="$BUILD/lib"
[ -f "$LIB/libvisioncpp.so" ] || { echo "libvisioncpp.so 없음: $LIB (--build <dir> 로 지정)"; exit 1; }

# VISP_FMT_LIB: fmt 라이브러리 사용 플래그. libvisioncpp 가 fmt 로 빌드됐으면 -DVISP_FMT_LIB +
# fmt include 필요. 아니면(내장 fallback) 정의하지 않는다. FMT_INC 있으면 자동으로 켠다.
FMT_INC="$BUILD/_deps/fmt-src/include"
FMT_FLAGS=""
[ -f "$FMT_INC/fmt/format.h" ] && FMT_FLAGS="-DVISP_FMT_LIB -I$FMT_INC"

echo "arch=$ARCH gen=$GEN params=$PARAMS build=$BUILD fmt=${FMT_FLAGS:-fallback}"
INC="$GEN/inc"
mkdir -p "$INC/visp/arch"
cp "$GEN/$ARCH.h" "$INC/visp/arch/$ARCH.h"

# run_mmdet.cpp + head.cpp(러너와 함께 컴파일, 라이브러리 아님) + g2c 백본 output/.cpp
g++ -std=c++20 -O2 $FMT_FLAGS \
  -DARCH="$ARCH" -DVISP_ARCH_HEADER="\"visp/arch/$ARCH.h\"" \
  -DMMDET_PARAMS_HEADER="\"$PARAMS\"" \
  -I"$DETECT" -I"$INC" -I"$V/include" -I"$V/src" \
  -I"$V/depend/llama/ggml/include" -I"$V/depend/llama/vendor" \
  "$RUN/backbone/run_mmdet.cpp" "$DETECT/head.cpp" "$GEN/$ARCH.cpp" \
  -L"$LIB" -lvisioncpp -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$LIB" \
  -o "$GEN/run_mmdet"
echo "built: $GEN/run_mmdet (arch=$ARCH)"
