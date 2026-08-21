#!/usr/bin/env bash
# example_custom.sh — **직접 정의한 nn.Module** 을 끝까지 돌린다.
#
#   ./tools/example_custom.sh
#
# 앞의 세 예제는 라이브러리가 모델을 준다(ultralytics·torchvision·mmdet). 이건 아니다 —
# 클래스를 내가 쓰고, `.pt` 로 저장하고, g2c 에 그 `.pt` 를 준다. 다른 부서가 자기 모델을
# 얹을 때 밟는 경로가 이것이다.
#
# ⚠️ **모델 클래스는 별도 `.py` 에 둔다.** `torch.save(model)` 는 클래스를 **모듈 이름으로**
#    피클하므로, `__main__` 에서 정의하면 로드하는 쪽이 그 이름을 못 찾아
#    `ModuleNotFoundError` 로 죽는다. 그래서 아래도 `mymodel.py` 를 따로 쓴다.
#
# ⚠️ **가중치는 실제 학습된 것을 쓴다.** 사전학습 ResNet 의 앞단을 그대로 가져와 조립한다.
#    랜덤 초기화로 재면 항등 초기값이 빠진 연산을 덮어 검증을 통과시킨다.
set -euo pipefail

SIZE="${SIZE:-224}"
VCPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G2C="${G2C_ROOT:-$(cd "$VCPP/.." && pwd)}"
WORK="${WORK:-/tmp/visp-example-custom}"

if [ ! -f "$G2C/shared/compile/pipeline.py" ]; then
    echo "g2c 를 못 찾았다: $G2C" >&2
    echo "G2C_ROOT=<GTX_Compiler 경로> 로 지정할 것." >&2
    exit 1
fi

mkdir -p "$WORK"

echo "== 1/4  모델 클래스 (별도 .py — 피클이 이름으로 찾는다) =="
cat > "$WORK/mymodel.py" <<'PY'
"""사전학습 ResNet18 의 앞단 + 직접 쓴 head. 가중치는 전부 실제 학습된 것이다."""
import torch.nn as nn
import torchvision


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights="DEFAULT")
        # 사전학습 레이어를 그대로 가져온다 — 여기까지가 "실제 가중치" 다.
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        # 직접 쓴 부분. 사전학습 conv 를 1x1 로 줄여 쓰므로 여기도 학습된 값에서 나온다.
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # head 도 학습된 값에서 채운다 — 난수를 한 값도 안 쓴다.
        # ⚠️ 채널 수가 다르다: layer2 는 128, 우리 head 는 64다. **앞 64개만 자른다.**
        #    `load_state_dict` 로 통째로 넣으면 size mismatch 로 죽는다(strict=False 도 못 막는다).
        src_conv = r.layer2[-1].conv2.weight.detach()               # [128,128,3,3]
        self.head[0].weight.data.copy_(
            src_conv.mean(dim=(2, 3))[:64].unsqueeze(-1).unsqueeze(-1))  # → [64,128,1,1]
        bn = r.layer2[-1].bn2
        self.head[1].weight.data.copy_(bn.weight.detach()[:64])
        self.head[1].bias.data.copy_(bn.bias.detach()[:64])
        self.head[1].running_mean.data.copy_(bn.running_mean.detach()[:64])
        self.head[1].running_var.data.copy_(bn.running_var.detach()[:64])

    def forward(self, x):
        return self.head(self.stem(x))
PY

echo "== 2/4  실제 가중치로 저장 =="
OMP_NUM_THREADS=1 uv run --project "$G2C" python - "$WORK" <<'PY'
import sys, torch
sys.path.insert(0, sys.argv[1])
from mymodel import MyNet

torch.set_num_threads(1)
m = MyNet().eval()
torch.save(m, f"{sys.argv[1]}/mynet.pt")
print(f"  saved: {sys.argv[1]}/mynet.pt  ({sum(p.numel() for p in m.parameters()):,} params)")
PY

echo "== 3/4  g2c 컴파일 =="
# PYTHONPATH 에 .pt 가 있는 폴더를 넣어야 피클이 `mymodel` 을 찾는다.
OMP_NUM_THREADS=1 PYTHONPATH="$WORK:$G2C" \
    uv run --project "$G2C" python -m shared.compile.pipeline \
    --model "$WORK/mynet.pt" --name MyNet \
    --output "$WORK" --input-shape "1,3,$SIZE,$SIZE"

# 성공 판정은 종료코드가 아니라 **파일 유무**다 — g2c 는 실패해도 exit 0 + "완료!" 를 낸다.
[ -f "$WORK/MyNet.gguf" ] || { echo "gguf 가 안 나왔다 — 위 로그를 볼 것" >&2; exit 2; }

echo "== 4/4  torch 와 대조 =="
OMP_NUM_THREADS=1 PYTHONPATH="$WORK" \
    uv run --project "$G2C" --extra ggml python - "$WORK" "$SIZE" <<'PY'
import importlib.util, sys, numpy as np, torch
# ⚠️ 이름이 겹친다 — 내가 쓴 클래스도 `MyNet`, g2c 가 생성한 클래스도 `MyNet` 이다.
#    (후자는 `nn.QuantModel` 을 상속한 별개 타입이다.) 별칭으로 갈라 둔다.
from mymodel import MyNet as TorchNet

work, size = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0); torch.set_num_threads(1)
x = torch.randn(1, 3, size, size)

with torch.no_grad():
    want = TorchNet().eval()(x).numpy().ravel()

# 생성 .py 는 클래스만 정의한다. 실행 진입점은 `nn.run_gguf(Model(), gguf, 파일경로, x)` 다.
import nn
nn.set_backend("ggml")
gen_path = f"{work}/MyNet.py"
spec = importlib.util.spec_from_file_location("gen", gen_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out = nn.run_gguf(mod.MyNet(), f"{work}/MyNet.gguf", gen_path, x.numpy())
got = np.asarray(out[0] if isinstance(out, (list, tuple)) else out).ravel()

# ⚠️ cosine 은 스케일 불변이라 크기가 통째로 틀려도 1.0 이 나온다. 거리로 잰다.
rel = np.abs(got - want).sum() / np.abs(want).sum()
print(f"상대 L1 {rel:.2e}  ({want.size} 값)")
print("PASS" if rel < 5e-2 else "FAIL")
PY

echo
echo "생성물: $WORK"
