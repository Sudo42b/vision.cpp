#!/usr/bin/env bash
# example_mmdet.sh — MMDetection 검출기를 **학습된 체크포인트**로 박스까지 돌린다.
#
#   ./tools/example_mmdet.sh                  # retinanet
#   ./tools/example_mmdet.sh fcos             # 다른 계열
#   MMDET=~/mmbuild/mmdetection ./tools/example_mmdet.sh
#
# yolo·torchvision 예제와 갈리는 지점: **mmdet head 는 트레이스를 통과하지 못한다.**
# 그래서 `.pt` 로 나가는 것은 backbone+neck 뿐이고, head 는 C++ 부품이 조립한다.
# `bbox_head` 는 속성으로 남아 state_dict(→GGUF)에 실린다 — 연산만 빼고 가중치는 남긴다.
# 자세한 배경은 `docs/mmdet-detectors.md`.
#
# ⚠️ **학습된 체크포인트로 잰다.** config 만으로 지은 랜덤 초기화는 항등 초기값
#    (γ=1·β=0, scale=1)이 빠진 연산을 덮어 검증을 통과시킨다.
set -euo pipefail

FAM="${1:-retinanet}"
SIZE="${SIZE:-512}"

VCPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMDET="${MMDET:-$HOME/mmbuild/mmdetection}"
WORK="${WORK:-/tmp/visp-example-$FAM}"
DH="$VCPP/tools/verify/mmdet/dense_head"

# ⚠️ 이 예제만 **mmdet 이 설치된 인터프리터**가 필요하다. 앞의 세 예제와 갈리는 지점이다 —
#    거긴 `uv run --project` 로 g2c 환경을 쓰면 됐지만, 여기선 하네스가 mmdet 을 import 한다.
#    `python3` 를 그냥 쓰면 시스템 파이썬으로 가서 `No module named 'yaml'` 같은
#    엉뚱한 곳에서 죽는다 — 진짜 원인(mmdet 환경이 아님)이 안 보인다.
PYTHON="${PYTHON:-python3}"

if [ ! -d "$MMDET/configs" ]; then
    echo "MMDetection 체크아웃을 못 찾았다: $MMDET" >&2
    echo "MMDET=<mmdetection 경로> 로 지정할 것. configs/ 와 checkpoints/ 가 있어야 한다." >&2
    exit 1
fi

if ! "$PYTHON" -c "import mmdet, yaml" 2>/dev/null; then
    echo "'$PYTHON' 에 mmdet(또는 pyyaml)이 없다." >&2
    echo "PYTHON=<mmdet 설치된 파이썬> ./tools/example_mmdet.sh $FAM" >&2
    echo "확인: \$PYTHON -c 'import mmdet; print(mmdet.__version__)'" >&2
    exit 1
fi

echo "== 1/2  config·체크포인트 짝 고르기 =="
# ⚠️ 짝을 손으로 고르지 마라 — 계열마다 변종이 여럿이라 **남의 가중치를 재게** 된다.
#    metafile.yml 이 계열마다 Config → Weights 를 갖고 있으므로 그걸 읽는다.
read -r CFG CKPT <<EOF
$(cd "$DH" && "$PYTHON" - "$FAM" <<'PY'
import os, sys
sys.path.insert(0, ".")
import mmdet_families as MF
mm = os.path.expanduser(os.environ.get("MMDET", "~/mmbuild/mmdetection"))
cfg, ckpt = MF.resolve_pair(os.path.join(mm, "configs"), sys.argv[1])
print(cfg, os.path.join(mm, "checkpoints", ckpt))
PY
)
EOF
echo "  config     $CFG"
echo "  checkpoint $CKPT"
[ -f "$CKPT" ] || { echo "체크포인트가 없다. mmdet metafile 의 Weights URL 에서 받을 것." >&2; exit 2; }

echo "== 2/2  박스 대조 (mmdet 자신의 predict_by_feat 와) =="
# 하네스가 export → g2c → head 조립 → 실행 → 대조를 한 번에 한다.
# ⚠️ vision.cpp 를 먼저 빌드해 둬야 한다 — 러너가 libvisioncpp 에 링크한다.
cd "$DH"
OMP_NUM_THREADS=1 "$PYTHON" verify_heads.py "$FAM" \
    --set "paths.workdir=$WORK" --set run.workers=1 --set "run.size=$SIZE"

GEN="$WORK/$FAM/out"
[ -x "$GEN/run_mmdet" ] || { echo "러너가 안 나왔다 — 위 로그를 볼 것" >&2; exit 3; }

OMP_NUM_THREADS=1 "$PYTHON" verify_postproc.py \
    "$GEN" "$CFG" "$CKPT" "$VCPP/tests/input/cat-and-hat.jpg" "$SIZE"

echo
echo "생성물: $WORK/$FAM"
