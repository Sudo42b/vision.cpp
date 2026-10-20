#!/usr/bin/env bash
# build_routing_probe.sh — gtx_smoke 컴파일 (libggml + libggml-gtx 링크).
# 전제: vision.cpp 가 -DGGML_GTX=ON 으로 빌드돼 libggml-gtx.so 가 있을 것.
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"; V="$(cd "$SELF/.." && pwd)"
BUILD="${VISP_BUILD:-$V/build}"; LIB="$BUILD/lib"
[ -f "$LIB/libggml-gtx.so" ] || { echo "libggml-gtx.so 없음: $LIB (cmake -DGGML_GTX=ON 필요)"; exit 1; }
g++ -std=c++20 -O2 -I"$V/depend/llama/ggml/include" \
  "$SELF/gtx_smoke.cpp" \
  -L"$LIB" -lggml -lggml-base -lggml-cpu -lggml-gtx -Wl,-rpath,"$LIB" \
  -o "${1:-$BUILD/bin}/gtx_smoke"
echo "built: ${1:-$BUILD/bin}/gtx_smoke"
