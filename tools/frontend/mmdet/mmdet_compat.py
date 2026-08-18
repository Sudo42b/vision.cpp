"""mmdet_compat.py — mmdet/mmpretrain 을 **import 가능하게** 만드는 호환 패치.

import 하는 것만으로 걸린다. `mmdet_wrap` · `frcnn_wrap` 둘 다 여기를 거친다 —
우회를 진입점마다 따로 두면 한쪽에서 조용히 실패한다(실제로 겪었다: 검증 하네스에만
`_stub.py` 를 두었더니 `mmdet_to_pt.py` 직접 실행에서 mmdet import 가 죽었고,
그 탓에 CARAFE 패치가 안 걸려 한참 뒤 "CPU 커널 없음" 으로 나타났다).

**패키지를 설치하지 않는다.** mm 계열은 레지스트리 자동 임포트라 설치 자체가 부작용이고,
전 계열을 깨뜨린 전례가 두 번 있다 → 위키 `openmim-설치가-setuptools를-되돌려-전부-깨뜨린다`.
"""
import sys
import types

import torch.nn as nn


def _relax_torch_load():
    """torch 2.6 부터 `torch.load` 기본값이 `weights_only=True` 다.

    mmengine 이 체크포인트를 읽을 때 ConfigDict·numpy 스칼라 같은 비-텐서 객체에서
    `_pickle.UnpicklingError` 로 죽는다(crowddet·maskformer 실측). 호출부가 mmengine
    안이라 인자를 넘길 수 없으므로 기본값을 되돌린다.

    ⚠️ 신뢰하는 파일 전용이다 — 이 하네스는 mmdet 공식 metafile URL 로 받은 것만 읽는다.
    """
    import torch

    if getattr(torch.load, "_visp_relaxed", False):
        return
    _orig = torch.load

    def load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)

    load._visp_relaxed = True
    torch.load = load


def apply():
    """`mmpretrain` 의 BLIP 모듈을 빈 껍데기로 막는다 — **mmdet import 를 살리려고**.

    `mmdet.models` → `reid_data_preprocessor` → `mmpretrain.models` 로 딸려 들어가는데,
    그 안 BLIP 이 설치된 transformers 버전과 안 맞아 import 시점에 죽는다:

        TypeError: NoneType takes no arguments   (BertPreTrainedModel 정의 중)

    우회가 **호출자마다 따로** 있으면(검증 하네스만 `_stub.py` 를 두는 식) 다른 진입점에서
    조용히 실패한다 — `trace_friendly_ops` 가 mmdet import 에 실패해 CARAFE 패치를 못 걸었고,
    한참 뒤 "CPU 커널 없음" 이라는 엉뚱한 에러로 나타났다. mmdet 으로 들어가는 문은
    여기 하나이므로 여기서 막는다. 이미 import 된 상태면 건드리지 않는다.
    """
    _relax_torch_load()
    # ⚠️ **패키지를 통째로 막지 마라.** 처음엔 `...blip` 자체를 빈 모듈로 바꿨는데,
    #    그러면 형제 모듈(`blip_retrieval` 등)을 못 찾아 오히려 더 넓게 깨진다.
    #    죽는 건 `language_model.py` 하나뿐이다(거기서 `PreTrainedModel` 이 None 이 된다).
    #    그 파일만 **아무 이름이든 내주는** 더미로 바꾼다 — `from ... import A, B, C` 가 통과한다.
    # ① 먼저 **설치 없이** 고칠 수 있는지 본다. transformers 5.x 는 몇 심볼을
    #    `modeling_utils` → `pytorch_utils` 로 옮겼는데 mmpretrain 은 옛 위치를 import 한다.
    #    별칭만 이어주면 되므로 **패키지를 건드리지 않는다**(설치는 레지스트리 자동 임포트
    #    때문에 전 계열을 깨뜨린 전례가 두 번 있다 → 위키).
    # 옮겨간 심볼을 **하나씩 쫓지 않는다** — 없는 이름을 물어오면 형제 모듈에서 찾아 준다.
    # (5.x 에서 apply_chunking_to_forward → pytorch_utils, GenerationMixin → generation …)
    try:
        import importlib as _il
        _mu = _il.import_module("transformers.modeling_utils")
        _srcs = ["transformers.pytorch_utils", "transformers.generation",
                 "transformers.modeling_layers", "transformers"]

        class _Fallback(type(_mu)):
            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                for s in _srcs:
                    try:
                        v = getattr(_il.import_module(s), name)
                    except Exception:
                        continue
                    setattr(_mu, name, v)      # 한 번 찾으면 캐시
                    return v
                # 5.x 에서 **삭제된** 이름은 어디에도 없다. import 만 통과시키고
                # 실제로 부르면 터뜨린다 — 조용히 잘못된 값을 내는 것보다 낫다.
                # (head pruning 유틸이라 추론 경로에서는 안 불린다)
                if name in ("find_pruneable_heads_and_indices",):
                    def _removed(*a, _n=name, **k):
                        raise NotImplementedError(
                            f"transformers 5.x 에서 삭제된 함수다: {_n}")
                    setattr(_mu, name, _removed)
                    return _removed
                raise AttributeError(name)

        _mu.__class__ = _Fallback
    except Exception:
        pass                                   # transformers 가 없으면 아래 더미로 간다

    # ② 데이터셋 전용 의존성은 더미로 막는다. 우리는 **모델 구조만** 쓰므로 데이터 로더가
    #    없어도 된다. 이것들은 레지스트리 프레임워크가 아니라 잎 패키지라 부작용이 없다.
    for dep in ("mat4py",):
        if dep not in sys.modules:
            try:
                __import__(dep)
            except Exception:
                sys.modules[dep] = types.ModuleType(dep)

    # ③ 토크나이저의 삭제된 메서드. `batch_encode_plus` 는 5.x 에서 빠졌고 `__call__` 이 같은 일을 한다.
    #    (GroundingDINO·GLIP 이 프롬프트를 토큰화할 때 부른다)
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _T
        if not hasattr(_T, "batch_encode_plus"):
            _T.batch_encode_plus = lambda self, *a, **k: self(*a, **k)
    except Exception:
        pass

    n = "mmpretrain.models.multimodal.blip.language_model"
    if n in sys.modules:
        return

    class _AnyNames(types.ModuleType):
        def __getattr__(self, name):
            # ⚠️ 던더는 가로채면 안 된다 — import 기계가 `__file__`·`__path__` 를 물어보는데
            #    빈 클래스를 돌려주면 엉뚱한 곳에서 터진다
            #    (`type object '__file__' has no attribute 'endswith'`).
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return type(name, (), {})          # 요청한 이름의 빈 클래스를 만들어 준다

    sys.modules[n] = _AnyNames(n)




def patch_ops():
    """trace 가 통째로 삼키는 mmdet 커스텀 op 을 **등가 수식**으로 바꾼다.

    `torch.autograd.Function` 은 forward/backward 를 직접 정의한 불투명 단위라
    `torch.jit.trace` 가 내부를 안 편다 — 그래프에 원자 노드 하나로 남고, 컴파일러는
    `unhandled op '<클래스명>'` 을 낸다. 렌더러를 새로 쓸 일이 아니라 **여기서 풀 일**이다.
    이 클래스들은 학습(역전파 수치안정) 때문에 존재하고 순전파는 등가이기 때문이다.

    호출은 멱등. mmdet 이 없거나 구조가 바뀌었으면 조용히 넘어간다.
    """
    try:
        import mmdet.models.dense_heads.tood_head as _tood
    except Exception as e:
        # ⚠️ **조용히 넘어가지 않는다.** 여기서 return 하면 아래 패치가 전부 안 걸리고,
        #    한참 뒤 "CPU 커널 없음" 같은 엉뚱한 에러로 나타난다(실제로 겪었다).
        print(f"  ⚠️ trace_friendly_ops: mmdet import 실패 — 패치를 못 걸었다: "
              f"{type(e).__name__}: {e}")
        return
    # TOOD. mmdet docstring 이 직접 밝힌다 — "substitutes the autograd function of
    # (x.sigmoid() * y.sigmoid()).sqrt()". 학습용 해석적 gradient 라 추론 값은 같다.
    if getattr(_tood.sigmoid_geometric_mean, "__module__", "") != __name__:
        def sigmoid_geometric_mean(x, y):
            return (x.sigmoid() * y.sigmoid()).sqrt()
        _tood.sigmoid_geometric_mean = sigmoid_geometric_mean

    _patch_carafe()
    _patch_swin_mask()
    _patch_sac_dilation()



def _patch_sac_dilation():
    """SAC(DetectoRS)의 dilation-3 deform conv 를 **오프셋 상수 이동**으로 등가 변환한다.

    `SAConv2d.forward` 의 out_l 갈래는 `deform_conv2d(x, offset, w, stride, 3·pad, 3·dil)`
    인데, ggml `conv_2d_deform` 과 mmcv-Function 렌더러는 **dilation 인자가 없다** —
    pad=(k-1)//2 · dil=1 로 추론해 렌더하므로 out_l 의 base 샘플링 격자가 통째로 어긋난다.
    크래시는 없고 out_l 만 조용히 틀린다(detectors 실측: out_s 0.001 vs **out_l 0.753**,
    layer3 까지 0.94 로 증폭).

    deform conv 의 탭 (i,j) 샘플 위치는 `h0·s − p + i·d + Δh` 다. (p→3p, d→3d) 는
    오프셋에 상수 `2·d·i − 2·p` 를 더한 (p, d) 와 **정확히 같은 위치**를 읽는다
    (출력 크기도 같다: (H+6−6−1)/s+1 = (H+2−2−1)/s+1). 그래서 탭별 상수를 버퍼로 구워
    offset 에 더하고 dilation 1 로 부른다 — 렌더러의 가정이 참이 되고 torch 값은 불변이다.
    버퍼는 첫 eager forward 에서 등록된다(→ export 는 저장 전 dummy forward 필수,
    frcnn_to_pt.py 가 이미 그렇게 한다).
    """
    try:
        from mmcv.ops.saconv import SAConv2d
    except Exception:
        return                                  # SAC 를 안 쓰는 환경 — 조용히 넘어간다
    if getattr(SAConv2d, "_visp_sac_patched", False):
        return
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mmcv.ops.deform_conv import deform_conv2d

    def forward(self, x):
        # pre-context (원식 그대로)
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch (원식 그대로)
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        zero_bias = torch.zeros(
            self.out_channels, device=weight.device, dtype=weight.dtype)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            out_s = deform_conv2d(x, offset, weight, self.stride, self.padding,
                                  self.dilation, self.groups, 1)
        else:
            out_s = nn.Conv2d._conv_forward(self, x, weight, zero_bias)
        weight = weight + self.weight_diff
        if self.use_deform:
            # dilation·padding 3배 대신: 탭 (i,j) 마다 offset 에 (2·d·i−2·p, 2·d·j−2·p)
            # 를 더한다. 값은 원식과 정확히 같고, 그래프에는 dilation 1 conv 만 남는다.
            if getattr(self, "_visp_dil_shift", None) is None:
                kh, kw = self.kernel_size
                d0, p0 = self.dilation, self.padding
                sh = torch.zeros(2 * kh * kw, 1, 1)
                for i in range(kh):
                    for j in range(kw):
                        sh[2 * (i * kw + j) + 0] = 2.0 * d0[0] * i - 2.0 * p0[0]
                        sh[2 * (i * kw + j) + 1] = 2.0 * d0[1] * j - 2.0 * p0[1]
                self.register_buffer("_visp_dil_shift", sh, persistent=True)
            offset = self.offset_l(avg_x) + self._visp_dil_shift
            out_l = deform_conv2d(x, offset, weight, self.stride, self.padding,
                                  self.dilation, self.groups, 1)
        else:
            # 비-deform 은 평범한 conv 라 dilation 을 렌더러가 받는다 — 원식 그대로.
            ori_p, ori_d = self.padding, self.dilation
            self.padding = tuple(3 * p for p in self.padding)
            self.dilation = tuple(3 * d for d in self.dilation)
            out_l = nn.Conv2d._conv_forward(self, x, weight, zero_bias)
            self.padding, self.dilation = ori_p, ori_d
        out = switch * out_s + (1 - switch) * out_l
        # post-context (원식 그대로)
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out

    SAConv2d.forward = forward
    SAConv2d._visp_sac_patched = True

def _patch_swin_mask():
    """Swin 의 shifted-window 어텐션 마스크를 **버퍼로 미리 굽는다.**

    `ShiftWindowMSA.forward` 는 `torch.zeros` 로 img_mask 를 만들고 **슬라이스 대입**
    (`img_mask[:, h, w, :] = cnt`, swin.py:201-212)으로 9개 영역 번호를 채우는데, trace 에서
    이 in-place 대입(`aten::index_put_`)이 그래프에 안 실려 **마스크가 전부 0** 이 된다.
    0 마스크는 크래시가 없다 — shifted 블록마다 경계 윈도가 roll 로 이어붙은 반대편 픽셀에
    어텐션하고 값만 조용히 틀린다(swin-t 실측: shifted 블록당 rel L1 +0.004~0.06,
    stage 가 깊을수록 경계 윈도 비율이 10%→56% 로 커져 최종 rpn 출력 0.82).

    입력 크기가 고정이면 이 마스크는 **상수**다. eager (export dry-run) forward 에서
    mmdet 원식 그대로 계산해 버퍼로 등록해 두고, trace 는 버퍼를 읽게 한다 —
    `relative_position_index` 와 같은 경로로 GGUF 에 실린다. 값은 동일하므로 torch
    기준값 쪽도 이 패치를 지나도 결과가 같다.
    """
    try:
        from mmdet.models.backbones.swin import ShiftWindowMSA
    except Exception:
        return                                  # swin 이 없는 환경 — 조용히 넘어간다
    if getattr(ShiftWindowMSA, "_visp_mask_patched", False):
        return
    import torch
    import torch.nn.functional as F

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        ws, ss = self.window_size, self.shift_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if ss > 0:
            shifted_query = torch.roll(query, shifts=(-ss, -ss), dims=(1, 2))
            # 원식 그대로 — 단 한 번(eager)만 계산하고 결과를 버퍼에 박는다.
            if getattr(self, "_visp_mask_hw", None) != (H_pad, W_pad):
                with torch.no_grad():
                    img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
                    sl = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
                    cnt = 0
                    for h in sl:
                        for w in sl:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                    mask_windows = self.window_partition(img_mask)
                    mask_windows = mask_windows.view(-1, ws * ws)
                    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)
                                                      ).masked_fill(attn_mask == 0, float(0.0))
                if hasattr(self, "_visp_attn_mask"):
                    del self._visp_attn_mask
                self.register_buffer("_visp_attn_mask", attn_mask, persistent=True)
                self._visp_mask_hw = (H_pad, W_pad)
            attn_mask = self._visp_attn_mask
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, ws**2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        if ss > 0:
            x = torch.roll(shifted_x, shifts=(ss, ss), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    ShiftWindowMSA.forward = forward
    ShiftWindowMSA._visp_mask_patched = True


def _patch_carafe():
    """CARAFE 를 torch 로 대체한다 — mmcv 커널이 **CUDA 전용**이라 CPU 에서 죽는다.

        RuntimeError: carafe_forward_impl: implementation for device cpu not found.

    "CPU 커널이 없다" 는 검증 불가가 아니라 재구현 과제다. 커널 소스
    (`carafe_naive_cuda_kernel.cuh`)의 인덱싱을 그대로 옮긴다:

        out[n,c,ph,pw] = Σ_{my,mx} feat[n,c,iy,ix] · mask[n,(g·k+my)·k+mx, ph, pw]
        g = c // (C/group),  (iy,ix) 는 (ph//s, pw//s) 주변 k×k, 경계 밖은 0

    벡터화하면 unfold + nearest 확대 + 가중합이다.

    ⚠️ **`import mmcv.ops.carafe as m` 은 모듈이 아닐 수 있다** — `mmcv/ops/__init__.py`
       가 같은 이름의 함수를 re-export 해 서브모듈을 가린다. `importlib` 로 꺼낸다.
    ⚠️ **패치 실패를 조용히 삼키지 마라.** 안 걸린 패치는 원래 버그보다 찾기 어렵다.
    """
    import importlib
    import torch.nn.functional as F
    try:
        mod = importlib.import_module("mmcv.ops.carafe")
    except Exception:
        return                                  # CARAFE 를 안 쓰는 환경 — 조용히 넘어간다
    if getattr(mod, "_visp_patched", False):
        return

    def carafe(feats, masks, kernel_size, group_size, scale_factor):
        n, c, h, w = feats.shape
        ho, wo = h * scale_factor, w * scale_factor
        k, g = kernel_size, group_size
        u = F.unfold(feats, k, padding=(k - 1) // 2).view(n, c * k * k, h, w)
        u = F.interpolate(u, size=(ho, wo), mode="nearest")
        u = u.view(n, g, c // g, k * k, ho, wo)
        m = masks.view(n, g, k * k, ho, wo).unsqueeze(2)
        return (u * m).sum(3).view(n, c, ho, wo)

    class _CARAFE(nn.Module):
        def __init__(self, kernel_size, group_size, scale_factor):
            super().__init__()
            self.kernel_size, self.group_size = kernel_size, group_size
            self.scale_factor = scale_factor

        def forward(self, feats, masks):
            return carafe(feats, masks, self.kernel_size, self.group_size, self.scale_factor)

    mod.carafe = carafe
    mod.CARAFE = _CARAFE
    mod._visp_patched = True
    # `mmcv.ops` 가 re-export 한 이름도 같이 갈아야 한다(그쪽을 import 한 코드가 있다).
    ops = importlib.import_module("mmcv.ops")
    ops.carafe = carafe
    ops.CARAFE = _CARAFE
    try:
        import mmcv.ops.carafe as _chk
        assert getattr(_chk, "_visp_patched", False) or _chk is carafe
    except Exception as e:                       # 실패는 **반드시 말한다**
        print(f"  ⚠️ CARAFE 패치 확인 실패: {type(e).__name__}: {e}")



def install_loader_modules(out_path, *names):
    """`out_path`(.pt 또는 그 디렉토리) 옆에 래퍼 모듈을 복사한다.

    `torch.save` 는 클래스를 `__module__` 이름으로 절이므로, 로드하는 쪽이 그 이름을
    import 할 수 있어야 한다. 컴파일러는 `.pt` 의 디렉토리를 `sys.path` 에 넣으므로
    거기에 모듈이 **있기만** 하면 환경변수 없이 열린다.

    자기 자신(`mmdet_compat`)도 같이 나른다 — 래퍼가 import 한다.
    """
    import os
    import shutil

    here = os.path.dirname(os.path.abspath(__file__))
    dst_dir = out_path if os.path.isdir(out_path) else os.path.dirname(os.path.abspath(out_path))
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in (*names, "mmdet_compat"):
        src = os.path.join(here, name + ".py")
        dst = os.path.join(dst_dir, name + ".py")
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        shutil.copyfile(src, dst)
        copied.append(name + ".py")
    return copied


apply()
patch_ops()
