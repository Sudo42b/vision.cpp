"""mmseg_wrap.py — mmseg 세그멘터를 '이미지 하나 → seg_logits' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지). `torch.save` 는 클래스를
`__module__` 로 피클하므로, CLI(`mmseg_to_pt.py`)와 로더가 **같은 이름으로 import** 해야
동일 클래스로 복원된다(mmdet 쪽 `mmdet_wrap` 과 같은 규약).

**mmdet 보다 훨씬 단순하다** — 앵커도 NMS 도 박스 디코드도 없다. head 를 C++ 로 조립할
이유가 없어 `decode_head` 까지 통째로 g2c 로 컴파일한다(GLIP 융합헤드에서 통한 경로).
"""
import os
import sys

import torch
import torch.nn as nn

# mmdet 프론트엔드의 trace 호환 패치를 그대로 쓴다 — mmcv custom op(CARAFE·DCN 등)은
# 두 라이브러리가 같은 것을 쓴다. 없으면 그 op 들이 trace 에서 삼켜진다.
_FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_FE, "mmdet"))
import mmdet_compat                                   # noqa: E402
import mmdet_wrap                                     # noqa: E402

trace_friendly_ops = mmdet_compat.patch_ops


class MMSegWrap(nn.Module):
    """`backbone → (neck) → decode_head` 까지. 출력은 **resize 전** seg_logits.

    mmseg 는 `encode_decode` 에서 head 출력을 입력 크기로 bilinear resize 하는데, 그건
    **후처리**라 그래프 밖에 둔다 — 러너가 원하는 크기로 올린다. 그래프에 넣으면 입력 크기가
    바뀔 때마다 다시 구워야 한다.

    `auxiliary_head` 는 학습 전용이라 뺀다(`predict` 경로가 안 부른다).
    """

    def __init__(self, seg):
        super().__init__()
        self.backbone = seg.backbone
        self.neck = seg.neck if getattr(seg, "with_neck", False) else None
        self.decode_head = seg.decode_head

    def forward(self, x):
        f = self.backbone(x)
        if self.neck is not None:
            f = self.neck(f)
        out = self.decode_head(f)
        # 대부분 단일 텐서다. cascade head 처럼 여럿을 내는 것도 있어 튜플이면 그대로
        # 흘린다 — 러너가 `out_i` 로 순서대로 덤프한다.
        return out if isinstance(out, torch.Tensor) else tuple(out)


def build(config, checkpoint=None, size=512):
    """mmseg config(.py) → (MMSegWrap(eval), 출력 shape 목록)."""
    from mmseg.apis import init_model                  # mmseg 만 import (g2c 무관)
    trace_friendly_ops()
    # PyTorch 2.6 부터 `torch.load` 의 `weights_only` 기본값이 True 라 mmengine 이 넣은
    # 학습 메타에 걸린다. mmdet 쪽과 같은 사정이므로 그 구현을 재사용한다.
    mmdet_wrap.allow_mmengine_checkpoint_globals()
    seg = init_model(config, checkpoint, device="cpu")
    seg.eval()
    m = MMSegWrap(seg)
    m.eval()
    with torch.no_grad():
        outs = m(torch.randn(1, 3, size, size))
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    return m, [tuple(o.shape) for o in outs]
