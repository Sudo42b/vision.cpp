#!/usr/bin/env bash
# example_torchvision.sh — torchvision 분류 모델을 **실제 ImageNet 가중치**로 끝까지 돌린다.
#
#   ./tools/example_torchvision.sh                # resnet18
#   ./tools/example_torchvision.sh resnet50
#
# yolo 예제와 다른 점: 분류 모델은 박스가 없다. `vision-cli` 대신 생성된 `.py` 로
# ggml 커널에서 돌리고 **torch 와 값을 대조**한다 — 그게 이 갈래의 "끝까지" 다.
#
# ⚠️ `weights='DEFAULT'` 를 쓴다. 랜덤 초기화로 재지 마라 — 항등 초기값(γ=1·β=0)이
#    빠진 연산을 덮어 검증을 통과시킨다. 실제로 VFNet 의 scale 하드코딩이 그렇게 숨었다.
set -euo pipefail

MODEL="${1:-resnet18}"
SIZE="${SIZE:-224}"

VCPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G2C="${G2C_ROOT:-$(cd "$VCPP/.." && pwd)}"
WORK="${WORK:-/tmp/visp-example-$MODEL}"

# resnet18 → ResNet18 (g2c --name 규약: 클래스명이 파일명이 된다)
CLS="$(printf '%s' "${MODEL:0:1}" | tr '[:lower:]' '[:upper:]')${MODEL:1}"

if [ ! -f "$G2C/shared/compile/pipeline.py" ]; then
    echo "g2c 를 못 찾았다: $G2C" >&2
    echo "G2C_ROOT=<GTX_Compiler 경로> 로 지정할 것." >&2
    exit 1
fi

echo "== 1/3  g2c 컴파일 ($MODEL, 실제 ImageNet 가중치 → $WORK) =="
# torchvision 이름을 그대로 준다. g2c 가 weights='DEFAULT' 로 받아 온다.
OMP_NUM_THREADS=1 PYTHONPATH="$G2C" \
    uv run --project "$G2C" python -m shared.compile.pipeline \
    --model "torchvision.models.$MODEL(weights='DEFAULT')" --name "$CLS" \
    --output "$WORK" --input-shape "1,3,$SIZE,$SIZE"

# 성공 판정은 종료코드가 아니라 **파일 유무**다 — g2c 는 실패해도 exit 0 + "완료!" 를 낸다.
[ -f "$WORK/$CLS.gguf" ] || { echo "gguf 가 안 나왔다 — 위 로그를 볼 것" >&2; exit 2; }

echo "== 2/3  ggml 커널로 실행 =="
# 생성된 .py 는 GGUF 를 libggml.so 로 eager 실행하는 진입점이다. 파이썬 바인딩이 필요하다.
OMP_NUM_THREADS=1 uv run --project "$G2C" --extra ggml python "$WORK/$CLS.py"

echo "== 3/3  torch 와 대조 =="
OMP_NUM_THREADS=1 uv run --project "$G2C" --extra ggml python - "$MODEL" "$WORK" "$CLS" <<'PY'
import importlib.util, os, sys, numpy as np, torch, torchvision

name, work, cls = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed(0); torch.set_num_threads(1)
x = torch.randn(1, 3, 224, 224)

ref = getattr(torchvision.models, name)(weights="DEFAULT").eval()
with torch.no_grad():
    want = ref(x).numpy().ravel()

# 생성 .py 는 클래스만 정의한다. 실행은 `nn.run_gguf(Model(), gguf, 파일경로, x)` 다
# (그 파일의 `__main__` 블록이 쓰는 것과 같은 진입점).
import nn
nn.set_backend("ggml")
gen_path = os.path.join(work, cls + ".py")
spec = importlib.util.spec_from_file_location("gen", gen_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out = nn.run_gguf(getattr(mod, cls)(), os.path.join(work, cls + ".gguf"), gen_path, x.numpy())
got = np.asarray(out[0] if isinstance(out, (list, tuple)) else out).ravel()

# ⚠️ cosine 으로 재지 마라 — 스케일 불변이라 크기가 통째로 틀려도 1.0 이 나온다.
#    rtmdet 이 cos 0.999450 으로 통과했는데 값은 97% 틀렸다. 거리로 잰다.
rel = np.abs(got - want).sum() / np.abs(want).sum()
print(f"상대 L1 {rel:.2e}  ·  argmax torch={want.argmax()} ggml={got.argmax()}")
print("PASS" if rel < 5e-2 and want.argmax() == got.argmax() else "FAIL")
PY

echo
echo "생성물: $WORK"
