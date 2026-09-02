"""mmpose_wrap.py — mmpose 추정기를 '이미지 하나 → head 출력' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지) — `mmseg_wrap`·`mmdet_wrap` 과 같은 규약.

mmseg 와 같은 이유로 단순하다: 앵커도 NMS 도 없다. topdown 계열은 이미 잘린 사람 이미지를
받아 **히트맵**(또는 SimCC 의 1D 벡터)을 내고, 디코드는 argmax + 소수점 보정뿐이라 그래프
밖이다. `backbone → (neck) → head` 를 한 그래프로 컴파일하고 출력 텐서를 대조한다.
"""
import os
import sys

import torch
import torch.nn as nn

_FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_FE, "mmdet"))
import mmdet_compat                                   # noqa: E402
import mmdet_wrap                                     # noqa: E402

trace_friendly_ops = mmdet_compat.patch_ops


def _stub_xtcocotools():
    """`xtcocotools` 를 `pycocotools` 로 잇는다 — **없으면 모델 로드조차 못 한다.**

    `mmpose.apis.init_model` 은 레지스트리를 채우려고 `mmpose.datasets` 를 통째로 import
    하는데, 그 안의 COCO 계열 데이터셋이 `from xtcocotools.coco import COCO` 를 한다.
    xtcocotools 는 C 확장이라 이 환경에서 빌드가 실패한다(gcc exit 1).

    ⚠️ **우리는 데이터셋을 안 쓴다.** 재는 것은 `backbone → head` 의 텐서뿐이고, 이 심은
    import 를 통과시키는 용도다. 평가지표(AP 등)를 계산하려 들면 **이 심으로는 안 된다** —
    xtcocotools 는 crowdpose/wholebody 용으로 확장된 것이라 API 가 더 넓다.
    그때는 진짜 xtcocotools 를 빌드해야 한다.
    """
    if "xtcocotools.coco" in sys.modules or _has_module("xtcocotools"):
        return
    import types
    import pycocotools.coco
    import pycocotools.cocoeval
    import pycocotools.mask
    pkg = types.ModuleType("xtcocotools")
    pkg.__path__ = []
    sys.modules["xtcocotools"] = pkg
    for sub in ("coco", "cocoeval", "mask"):
        sys.modules[f"xtcocotools.{sub}"] = getattr(pycocotools, sub)
        setattr(pkg, sub, getattr(pycocotools, sub))


def _mmpose_pkg_dir():
    import mmpose
    return os.path.dirname(os.path.abspath(mmpose.__file__))


class _chdir:
    """`with` 로 cwd 를 잠깐 바꾼다 (py3.11 의 `contextlib.chdir` 과 같다)."""

    def __init__(self, path):
        self.path, self.prev = path, None

    def __enter__(self):
        self.prev = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.prev)
        return False


def _has_module(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ⚠️ **import 시점에 건다.** `torch.load` 로 `MMPoseWrap` 를 되살리는 쪽(g2c 서브프로세스)은
#    피클이 이 모듈을 먼저 import 하는데, 거기서 mmpose 가 딸려 들어온다. 함수 안에서만
#    걸면 그 경로는 심을 못 보고 `ModuleNotFoundError: xtcocotools` 로 컴파일이 죽는다.
_stub_xtcocotools()


class MMPoseWrap(nn.Module):
    """`backbone → (neck) → head.forward` 까지. 디코드(argmax·보정)는 그래프 밖.

    `head.forward` 는 mmpose 의 **순수 forward** 다(`base.py:_forward` 가 부르는 것과 같다).
    `predict` 는 flip TTA·디코드가 붙어 있어 쓰지 않는다 — 그건 후처리다.

    출력은 head 마다 다르다:
      · `HeatmapHead`  → (1, K, H/4, W/4) 텐서 하나
      · `SimCCHead`    → (x, y) 1D 로짓 **두 개** (튜플로 그대로 흘린다)
    """

    def __init__(self, pose):
        super().__init__()
        self.backbone = pose.backbone
        self.neck = pose.neck if getattr(pose, "with_neck", False) else None
        self.head = pose.head

    def forward(self, x):
        f = self.backbone(x)
        if self.neck is not None:
            f = self.neck(f)
        out = self.head.forward(f)
        # ⚠️ **항상 튜플이다.** 예전엔 출력이 하나면 벌거벗은 Tensor 를 돌려줬는데,
        #    `build()` 는 shape 을 늘 리스트로 알려줘서 **호출자가 두 겹 벗기는 사고**가
        #    났다 — `outs[0][0]` 이 19장짜리 클래스 맵에서 1장만 남겼다(2026-09-01 실측).
        #    형태를 하나로 고정해 그 사고 자체를 없앤다. 러너는 `out_i` 로 순서대로 덤프한다.
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def build(config, checkpoint=None, size=(256, 192)):
    """mmpose config(.py) → (MMPoseWrap(eval), 출력 shape 목록).

    ⚠️ **입력이 정방이 아니다.** topdown 은 사람 박스를 256x192(H x W)로 잘라 넣는다 —
    정방으로 넣으면 히트맵 크기가 config 와 안 맞아 head 가 죽거나 조용히 다른 것을 잰다.
    """
    _stub_xtcocotools()
    from mmpose.apis import init_model                 # mmpose 만 import (g2c 무관)
    trace_friendly_ops()
    mmdet_wrap.allow_mmengine_checkpoint_globals()
    # ⚠️ **mmpose 의 dataset metainfo 경로는 cwd 기준이다.** config 가
    #    `from_file='configs/_base_/datasets/aic.py'` 같은 **상대경로**를 들고 있고,
    #    `parse_pose_metainfo` 는 그걸 cwd 에서 찾는다. 폴백은 `mmpose/.mim/configs/` 인데
    #    editable 설치에는 그 폴더가 없다 → `FileNotFoundError` 로 모델 로드가 죽는다.
    #    레포 루트에서 부르고 되돌린다(산출물은 원래 cwd 에 쓰인다).
    with _chdir(os.path.dirname(_mmpose_pkg_dir())):
        pose = init_model(config, checkpoint, device="cpu")
    pose.eval()
    m = MMPoseWrap(pose)
    m.eval()
    h, w = size
    with torch.no_grad():
        outs = m(torch.randn(1, 3, h, w))
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    return m, [tuple(o.shape) for o in outs]


def input_size(config):
    """config 의 `codec.input_size` → (H, W). 없으면 topdown 관례값 (256, 192).

    codec 은 `(W, H)` 순서다 — **torch 와 반대다.** 뒤집어 쓰면 히트맵이 전치돼 나오는데
    정사각이 아니면 shape 에서 죽고, 정사각이면 조용히 틀린다.
    """
    _stub_xtcocotools()
    from mmengine.config import Config
    cfg = Config.fromfile(config)
    codec = cfg.get("codec")
    if isinstance(codec, (list, tuple)):
        codec = codec[0]
    sz = (codec or {}).get("input_size") if codec else None
    if sz and len(sz) == 2:
        return int(sz[1]), int(sz[0])
    return 256, 192


def save(model, path):
    """`.pt` 를 쓰고 **여는 데 필요한 모듈을 그 옆에 전부 복사한다.**

    `torch.save` 는 클래스를 `__module__` 이름으로 절이므로 여는 쪽이 그 이름을 import
    할 수 있어야 한다. g2c 는 `.pt` 가 있는 디렉터리를 `sys.path` 에 넣으므로, 모듈이
    거기 **있기만 하면** `PYTHONPATH` 없이 열린다.

    ⚠️ **자기 파일 하나로는 부족하다.** 이 래퍼는 `mmdet_wrap`·`mmdet_compat` 을 import
    하므로 그것들도 같이 날라야 한다 — 안 그러면 컴파일이
    `ModuleNotFoundError: No module named 'mmdet_compat'` 에서 멈춘다(2026-09-01 실측).
    mmdet 쪽 `install_loader_modules` 가 그 일을 이미 한다.
    """
    import torch

    torch.save(model, path)
    return mmdet_compat.install_loader_modules(path, "mmpose_wrap", "mmdet_wrap")
