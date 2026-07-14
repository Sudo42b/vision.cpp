#!/usr/bin/env bash
#
# 예제: mmdet RetinaNet(r18) → 그래프 gguf(decode 포함) → vision-cli graph (CPU / NPU)
#
# g2c 는 안 건드리고, mmdet 검출기를 raw+decode nn.Module 로 감싸(serialize.py 의 _MMDetStatic)
# anchor delta2bbox 까지 그래프 op 으로 전개 → 인터프리터가 박스 좌표+점수를 바로 출력.
# (host postproc.cpp 불필요. NMS 만 그래프 밖.)  출력 = out_0..9 (박스5레벨 + 점수5레벨)
#
# 사전:  - mmdet 설치 + config .py 경로 : $CFG
#        - GTX_Compiler 경로(g2c 파서)  : $G2C
#        - PYTHON: g2c 의존성 환경 (예: PYTHON="uv run --no-sync python")
#        - 빌드된 vision-cli            : $VCLI
#
set -euo pipefail

G2C="${G2C:-$HOME/GTX_Compiler}"
GGUF_PY="$G2C/vision.cpp/depend/llama/gguf-py"
VCLI="${VCLI:-$(cd "$(dirname "$0")/../../.." && pwd)/build/bin/vision-cli}"
CFG="${CFG:-$HOME/mmdetection/configs/retinanet/retinanet_r18_fpn_1x_coco.py}"
SIZE="${SIZE:-512}"
OUT="${OUT:-/tmp/demo_retinanet}"

# ── 1) mmdet 검출기 → 그래프 gguf (decode-in-graph) ────────────────────────
OMP_NUM_THREADS=1 PYTHONPATH="$G2C:$GGUF_PY" \
    ${PYTHON:-python} "$(dirname "$0")/../serialize.py" \
        --model "mmdet:$CFG" --size "$SIZE" --out "$OUT.gguf" --golden "$OUT"

# ── 2) CPU 인터프리터 실행 ─────────────────────────────────────────────────
echo "== CPU =="
"$VCLI" graph -m "$OUT.gguf" -i "$OUT.in.bin" -o "$OUT.cpu.bin" -b cpu

# ── 3) NPU(GTX) 실행 ──────────────────────────────────────────────────────
#   torchvision_resnet18.sh 의 NPU 주석과 동일 (0.9.5 GTX 백엔드 + Spike 서버).
if [ "${RUN_NPU:-0}" = "1" ]; then
    echo "== NPU (-b gpu) =="
    "$VCLI" graph -m "$OUT.gguf" -i "$OUT.in.bin" -o "$OUT.npu.bin" -b gpu
fi

# 검증: out_0..9 를 $OUT.golden.0..9.bin 과 cos 비교 → 박스5+점수5 전부 1.0
echo "완료: $OUT.gguf, $OUT.cpu.bin.0..9.bin (RUN_NPU=1 이면 $OUT.npu.bin.0..9.bin)"
