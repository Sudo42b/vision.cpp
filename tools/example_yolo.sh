#!/usr/bin/env bash
# example_yolo.sh — g2c 생성 arch 를 붙여 **끝까지** 돌리는 예제.
#
#   ./tools/example_yolo.sh                 # yolo26m, 기본 테스트 이미지
#   ./tools/example_yolo.sh yolo26s         # 다른 크기
#   ./tools/example_yolo.sh yolo26m my.jpg
#
# 생성물을 저장소에 커밋하지 않는 이유: `.pt` 는 ultralytics 가 받아주고 `.cpp`/`.gguf` 는
# g2c 가 만든다 — 커밋해봐야 g2c 가 바뀌면 조용히 낡는다. 대신 **이 스크립트로 재현**한다.
set -euo pipefail

MODEL="${1:-yolo26m}"
IMAGE="${2:-}"
SIZE="${SIZE:-640}"

VCPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G2C="${G2C_ROOT:-$(cd "$VCPP/.." && pwd)}"
BUILD="${VISP_BUILD:-$VCPP/build}"
WORK="${WORK:-/tmp/visp-example-$MODEL}"

# 클래스명 첫 글자를 대문자로: yolo26m → Yolo26m (g2c --name 규약)
CLS="$(printf '%s' "${MODEL:0:1}" | tr '[:lower:]' '[:upper:]')${MODEL:1}"

[ -z "$IMAGE" ] && IMAGE="$VCPP/tests/input/cat-and-hat.jpg"

if [ ! -f "$G2C/shared/compile/pipeline.py" ]; then
    echo "g2c 를 못 찾았다: $G2C" >&2
    echo "G2C_ROOT=<GTX_Compiler 또는 GTX_port 경로> 로 지정할 것." >&2
    exit 1
fi

echo "== 1/4  g2c 컴파일 ($MODEL → $WORK) =="
# ⚠️ trace 는 thread 폭주로 hang 할 수 있다 — 한 개로 묶는다.
OMP_NUM_THREADS=1 PYTHONPATH="$G2C" \
    uv run --project "$G2C" python -m shared.compile.pipeline \
    --model "ultralytics.YOLO('$MODEL.pt')" --name "$CLS" \
    --output "$WORK" --input-shape "1,3,$SIZE,$SIZE"

# 성공 판정은 종료코드가 아니라 **파일 유무**로 한다 — g2c 는 실패해도 exit 0 + "완료!" 를 낸다.
[ -f "$WORK/$CLS.gguf" ] || { echo "gguf 가 안 나왔다 — 위 로그를 볼 것" >&2; exit 2; }

echo "== 2/4  arch 등록 =="
python3 "$VCPP/tools/install_arch.py" "$WORK" --name "$CLS" --detect-yolo --size "$SIZE"

echo "== 3/4  빌드 =="
cmake --build "$BUILD" -j"$(nproc)" > /dev/null

echo "== 4/4  실행 =="
OUT="$WORK/detected.jpg"
"$BUILD/bin/vision-cli" "$MODEL" -m "$WORK/$CLS.gguf" -i "$IMAGE" -o "$OUT"

echo
echo "결과: $OUT"
