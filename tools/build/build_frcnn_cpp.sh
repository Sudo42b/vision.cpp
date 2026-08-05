#!/usr/bin/env bash
#
# build_frcnn_cpp.sh — Faster R-CNN two-stage 러너(run_frcnn) 컴파일.
# g2c 가 생성한 SubA(backbone+neck+RPN)·SubB(bbox_head) output/.cpp 를 run_frcnn.cpp 와 함께
# 컴파일해 libvisioncpp 와 링크. (host op: rpn_proposals/roi_align/detect_roi 는 라이브러리)
#
# 사용:  build_frcnn_cpp.sh <SubA_gen_dir> <SubB_gen_dir>
#   예:  build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB
# env:  VISP_BUILD = libvisioncpp 빌드 디렉토리 (기본: <vision.cpp>/build)
#
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"
V="$(cd "$SELF/../.." && pwd)"
DETECT="$V/tools/detect"                  # head.h (run_frcnn 이 include)
RUN="$V/tools/verify"                     # E2E 검증 러너
GA="$(cd "${1:?usage: build_frcnn_cpp.sh <SubA_dir> <SubB_dir>}" && pwd)"
GB="$(cd "${2:?SubB_dir 필요}" && pwd)"
BUILD="${VISP_BUILD:-$V/build}"; LIB="$BUILD/lib"
[ -f "$LIB/libvisioncpp.so" ] || { echo "libvisioncpp.so 없음: $LIB"; exit 1; }
FMT_INC="$BUILD/_deps/fmt-src/include"; FMT_FLAGS=""
[ -f "$FMT_INC/fmt/format.h" ] && FMT_FLAGS="-DVISP_FMT_LIB -I$FMT_INC"

echo "SubA=$GA SubB=$GB build=$BUILD"
g++ -std=c++20 -O2 $FMT_FLAGS \
  -I"$GA" -I"$GB" -I"$DETECT" -I"$V/include" -I"$V/src" \
  -I"$V/depend/llama/ggml/include" -I"$V/depend/llama/vendor" \
  "$RUN/roi/run_frcnn.cpp" "$GA/FRCNN_SubA.cpp" "$GB/FRCNN_SubB.cpp" \
  -L"$LIB" -lvisioncpp -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$LIB" \
  -o "$GA/run_frcnn"
echo "built: $GA/run_frcnn"
