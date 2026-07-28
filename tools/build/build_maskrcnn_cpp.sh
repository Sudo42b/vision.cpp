#!/usr/bin/env bash
#
# build_maskrcnn_cpp.sh — Mask R-CNN E2E 러너(run_maskrcnn) 컴파일.
# g2c 3 subgraph(SubA backbone+neck+RPN, SubB bbox_head, SubC mask_head)를 run_maskrcnn.cpp 와
# 함께 컴파일. host op(rpn_proposals/roi_align/detect_roi/paste_mask)은 libvisioncpp.
#
# 사용:  build_maskrcnn_cpp.sh <SubA_dir> <SubB_dir> <SubC_dir>
# env:  VISP_BUILD = libvisioncpp 빌드 디렉토리 (기본: <vision.cpp>/build)
#
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"
V="$(cd "$SELF/../.." && pwd)"
DETECT="$V/tools/detect"                  # head.h (run_maskrcnn 이 include)
RUN="$V/tools/verify"                     # E2E 검증 러너
GA="$(cd "${1:?usage: build_maskrcnn_cpp.sh <SubA> <SubB> <SubC>}" && pwd)"
GB="$(cd "${2:?SubB 필요}" && pwd)"
GC="$(cd "${3:?SubC 필요}" && pwd)"
BUILD="${VISP_BUILD:-$V/build}"; LIB="$BUILD/lib"
[ -f "$LIB/libvisioncpp.so" ] || { echo "libvisioncpp.so 없음: $LIB"; exit 1; }
FMT_INC="$BUILD/_deps/fmt-src/include"; FMT_FLAGS=""
[ -f "$FMT_INC/fmt/format.h" ] && FMT_FLAGS="-DVISP_FMT_LIB -I$FMT_INC"

# g2c .cpp 는 "visp/arch/<ARCH>.h" 를 include → 각 gen dir 에 inc/visp/arch 구성.
for d in "$GA" "$GB" "$GC"; do
  arch="$(basename "$(ls "$d"/*.cpp | grep -v run_ | head -1)" .cpp)"
  mkdir -p "$d/inc/visp/arch"; cp "$d/$arch.h" "$d/inc/visp/arch/$arch.h"
done

echo "SubA=$GA SubB=$GB SubC=$GC"
g++ -std=c++20 -O2 $FMT_FLAGS \
  -I"$GA" -I"$GB" -I"$GC" -I"$GA/inc" -I"$GB/inc" -I"$GC/inc" -I"$DETECT" \
  -I"$V/include" -I"$V/src" -I"$V/depend/llama/ggml/include" -I"$V/depend/llama/vendor" \
  "$RUN/seg/run_maskrcnn.cpp" "$GA/FRCNN_SubA.cpp" "$GB/FRCNN_SubB.cpp" "$GC/MaskRCNN_SubC.cpp" \
  -L"$LIB" -lvisioncpp -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$LIB" \
  -o "$GA/run_maskrcnn"
echo "built: $GA/run_maskrcnn"
