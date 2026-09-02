#!/usr/bin/env bash
# add_model.sh — 모델 하나를 **컴파일 → 등록 → 재빌드** 까지 한 번에 한다.
#
#   tools/add_model.sh --model mymodel.pt --name MyNet -- --mean 123.675,116.28,103.53 \
#                                                          --std 58.395,57.12,57.375
#   tools/add_model.sh --model resnet18 --name R18 -- --mean 123.675,116.28,103.53 --std 58.395,57.12,57.375
#   tools/add_model.sh --model yolo11n.pt --name Y11 --input-shape 1,3,640,640 -- --detect-yolo --nms --classes 3
#
# `--` 뒤의 플래그는 **그대로 `install_arch.py` 로 넘어간다**(전처리·검출 파라미터).
# 그 값들은 `.gguf` 에 없다 — 여기서 안 넣으면 모델은 도는데 값이 조용히 틀린다.
#
# 손으로 밟을 때 빠뜨리는 검사 둘을 여기서 **막는다**:
#   ① `→ gguf:` 가 안 나와도 g2c 는 종료 코드 0 을 낸다 → 파일 유무로 판정한다
#   ② `unhandled op` 이 있으면 그 연산이 그래프에서 통째로 빠진 채 돈다 → 세서 멈춘다
set -euo pipefail

VCPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-$VCPP/build}"
OUT=""; MODEL=""; NAME=""; SHAPE=""; PTH=""
JOBS="${JOBS:-4}"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --name)        NAME="$2";  shift 2 ;;
        --out)         OUT="$2";   shift 2 ;;
        --input-shape) SHAPE="$2"; shift 2 ;;
        --pth)         PTH="$2";   shift 2 ;;
        --)            shift; break ;;
        -h|--help)     sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "모르는 인자: $1 (install_arch.py 플래그는 -- 뒤에 둬라)" >&2; exit 2 ;;
    esac
done

[ -n "$MODEL" ] || { echo "--model 이 필요하다" >&2; exit 2; }
[ -n "$NAME" ]  || { echo "--name 이 필요하다" >&2; exit 2; }
OUT="${OUT:-$PWD/out-$(echo "$NAME" | tr '[:upper:]' '[:lower:]')}"

# g2c 를 찾는다 — wheel 로 깐 경우(배포판)와 소스 체크아웃 둘 다 받는다.
if [ -n "${G2C_BIN:-}" ]; then      :
elif [ -x "$PWD/.venv/bin/g2c" ];      then G2C_BIN="$PWD/.venv/bin/g2c"
elif [ -x "$VCPP/../.venv/bin/g2c" ];  then G2C_BIN="$VCPP/../.venv/bin/g2c"
elif command -v g2c >/dev/null 2>&1;   then G2C_BIN="$(command -v g2c)"
else
    echo "g2c 를 못 찾았다. wheel 을 깔았으면 그 venv 를 쓰거나 G2C_BIN=<경로> 로 지정해라." >&2
    exit 3
fi
PY_BIN="$(dirname "$G2C_BIN")/python"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python3)"

# gguf-py 는 이 트리 안에 있다. 안 잡히면 가중치가 안 나온다.
export PYTHONPATH="$VCPP/depend/llama/gguf-py${PYTHONPATH:+:$PYTHONPATH}"

echo "== 1/3  컴파일  ($G2C_BIN)"
"$G2C_BIN" --model "$MODEL" --name "$NAME" --output "$OUT" \
    ${SHAPE:+--input-shape "$SHAPE"} ${PTH:+--pth "$PTH"}

# ① 종료 코드로 판정하지 않는다
[ -f "$OUT/$NAME.gguf" ] || {
    echo "실패: $OUT/$NAME.gguf 가 안 나왔다. 위 로그에서 '→ gguf:' 줄을 찾아라." >&2
    echo "      대개 gguf-py 를 못 찾은 것이다 ($VCPP/depend/llama/gguf-py)." >&2
    exit 4
}
# ② 빠진 연산이 있으면 뒤 숫자는 전부 무의미하다
UNH=$(grep -c "unhandled op" "$OUT/$NAME.cpp" || true)
[ "$UNH" = "0" ] || {
    echo "실패: unhandled op $UNH 건 — 그 연산이 그래프에서 빠진 채 돈다." >&2
    grep -n "unhandled op" "$OUT/$NAME.cpp" >&2
    exit 5
}

echo "== 2/3  등록"
"$PY_BIN" "$VCPP/tools/install_arch.py" "$OUT" --name "$NAME" "$@"

echo "== 3/3  재빌드"
[ -d "$BUILD" ] || cmake -S "$VCPP" -B "$BUILD" -D VISP_TESTS=OFF
cmake --build "$BUILD" -j"$JOBS"

ARCH="$(echo "$NAME" | tr '[:upper:]' '[:lower:]')"
"$BUILD/bin/vision-cli" --help | grep -i "generated archs" || true
echo
echo "돌리려면:"
echo "  $BUILD/bin/vision-cli $ARCH -m $OUT/$NAME.gguf -i $VCPP/tests/input/cat-and-hat.jpg -o result"
