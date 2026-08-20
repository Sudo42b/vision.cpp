#!/usr/bin/env bash
#
# build_frcnn_cpp.sh — Faster R-CNN two-stage 러너(run_frcnn) 컴파일.
# g2c 가 생성한 SubA(backbone+neck+RPN)·SubB(bbox_head) 산출을 run_frcnn.cpp 와 함께
# 컴파일해 libvisioncpp 와 링크. (host op: rpn_proposals/roi_align/detect_roi 는 라이브러리)
#
# ⚠️ 빌드 라인은 verify_heads.py(3_build2 단계)와 같아야 한다 — 하네스가 정본이다.
#    생성 .cpp 는 `visp/arch/<Arch>.h` 를 include 하므로 헤더를 그 형태로 스테이징한다.
#
# 사용:  build_frcnn_cpp.sh [--build <dir>] <SubA_gen_dir> <SubB_gen_dir> [SubB_arch]
#   예:  build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB
# env:  VISP_BUILD = libvisioncpp 빌드 디렉토리 (기본: <vision.cpp>/build)
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"
V="$(cd "$SELF/../.." && pwd)"
RUN="$V/tools/verify"

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

GA="$(cd "${1:?usage: build_frcnn_cpp.sh [--build <dir>] <SubA_dir> <SubB_dir> [SubB_arch]}" && pwd)"
GB="$(cd "${2:?SubB_dir 필요}" && pwd)"
ARCH_B="${3:-}"
if [ -z "$ARCH_B" ]; then
  ARCH_B="$(basename "$(ls "$GB"/*.cpp | grep -v run_ | head -1)" .cpp)"
fi
BUILD="${ARG_BUILD:-${VISP_BUILD:-$V/build}}"; LIB="$BUILD/lib"
[ -f "$LIB/libvisioncpp.so" ] || { echo "libvisioncpp.so 없음: $LIB (--build <dir> 로 지정)"; exit 1; }
FMT_INC="$BUILD/_deps/fmt-src/include"; FMT_FLAGS=""
[ -f "$FMT_INC/fmt/format.h" ] && FMT_FLAGS="-DVISP_FMT_LIB -I$FMT_INC"

echo "SubA=$GA SubB=$GB arch_b=$ARCH_B build=$BUILD"
INC_A="$GA/inc"; INC_B="$GB/inc"
mkdir -p "$INC_A/visp/arch" "$INC_B/visp/arch"
cp "$GA/FRCNN_SubA.h" "$INC_A/visp/arch/"
cp "$GB/$ARCH_B.h"    "$INC_B/visp/arch/"

g++ -std=c++20 -O2 $FMT_FLAGS \
  -DARCH_A=FRCNN_SubA -DARCH_B="$ARCH_B" \
  -DVISP_ARCH_HEADER_A='"visp/arch/FRCNN_SubA.h"' \
  -DVISP_ARCH_HEADER_B="\"visp/arch/$ARCH_B.h\"" \
  -I"$INC_A" -I"$INC_B" -I"$V/include" -I"$V/src" \
  -I"$V/depend/llama/ggml/include" -I"$V/depend/llama/vendor" \
  "$RUN/backbone/run_frcnn.cpp" "$GA/FRCNN_SubA.cpp" "$GB/$ARCH_B.cpp" \
  -L"$LIB" -lvisioncpp -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$LIB" \
  -o "$GA/run_frcnn"
echo "built: $GA/run_frcnn"
