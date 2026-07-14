#!/usr/bin/env bash
#
# 예제: torchvision resnet18 → 그래프 gguf → vision-cli graph 인터프리터 (CPU / NPU)
#
# 흐름:  serialize.py (모델→그래프 데이터 gguf)  →  vision-cli graph (조립·실행)
#        모델별 .cpp 생성/재빌드 없음. gguf 하나에 구조(graph.nodes)+가중치.
#
# 사전:  - GTX_Compiler 경로(g2c 파서)      : $G2C
#        - PYTHON: g2c 의존성 환경 (예: PYTHON="uv run --no-sync python")
#        - 빌드된 vision-cli               : $VCLI   (NPU 는 -DGGML_BACKEND_DL=ON 빌드)
#
set -euo pipefail

G2C="${G2C:-$HOME/GTX_Compiler}"
GGUF_PY="$G2C/vision.cpp/depend/llama/gguf-py"
VCLI="${VCLI:-$(cd "$(dirname "$0")/../../.." && pwd)/build/bin/vision-cli}"
OUT="${OUT:-/tmp/demo_resnet18}"

# ── 1) 모델 → 그래프 gguf (+ 검증용 golden) ────────────────────────────────
OMP_NUM_THREADS=1 PYTHONPATH="$G2C:$GGUF_PY" \
    ${PYTHON:-python} "$(dirname "$0")/../serialize.py" \
        --model resnet18 --out "$OUT.gguf" --golden "$OUT"

# ── 2) CPU 인터프리터 실행 ─────────────────────────────────────────────────
echo "== CPU =="
"$VCLI" graph -m "$OUT.gguf" -i "$OUT.in.bin" -o "$OUT.cpu.bin" -b cpu

# ── 3) NPU(GTX) 실행 ──────────────────────────────────────────────────────
#   필요: 0.9.5 GTX 백엔드(libggml-gtx.so) + 상주 Spike RPC 서버(uart-vp).
#   아래 env 는 GTX 개발환경(gtx_ggml_zephyr_host) 기준 — 경로만 맞추면 됨.
#     GTX_ROOT=<gtx_ggml_zephyr_host>
#     source $GTX_ROOT/tools/gtx_spike_model_env.sh   # GGML_BACKEND_PATH/transport
#     export LD_LIBRARY_PATH="$(dirname "$VCLI")/../lib:$LD_LIBRARY_PATH"  # vision-cli ggml 앞세움
#     bash $GTX_ROOT/tools/run_gtx_spike_server.sh &   # 서버 미기동 시
if [ "${RUN_NPU:-0}" = "1" ]; then
    echo "== NPU (-b gpu) =="
    "$VCLI" graph -m "$OUT.gguf" -i "$OUT.in.bin" -o "$OUT.npu.bin" -b gpu
    # sched routing 로그: GTX(NPU)=N, CPU=M — conv/matmul 이 NPU 로 감.
fi

# 검증: $OUT.cpu.bin.0.bin / $OUT.npu.bin.0.bin 을 $OUT.golden.0.bin 과 cos 비교 → 1.0
echo "완료: $OUT.gguf, $OUT.cpu.bin.0.bin (RUN_NPU=1 이면 $OUT.npu.bin.0.bin)"
