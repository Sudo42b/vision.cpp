"""mmseg_wrap.py — mmseg 세그멘터를 '이미지 하나 → seg_logits' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지). `torch.save` 는 클래스를
`__module__` 로 피클하므로, CLI(`mmseg_to_pt.py`)와 로더가 **같은 이름으로 import** 해야
동일 클래스로 복원된다(mmdet 쪽 `mmdet_wrap` 과 같은 규약).

**mmdet 보다 훨씬 단순하다** — 앵커도 NMS 도 박스 디코드도 없다. head 를 C++ 로 조립할
이유가 없어 `decode_head` 까지 통째로 g2c 로 컴파일한다(GLIP 융합헤드에서 통한 경로).
"""
import inspect
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
        # ⚠️ **`backbone` 이 없는 세그멘터가 있다.** `MultimodalEncoderDecoder`(san)는
        #    `image_encoder` + `text_encoder` 로 되어 있어 `seg.backbone` 을 무조건 집으면
        #    `AttributeError: ... has no attribute 'backbone'` 로 죽는다. 「범위 밖」이
        #    아니라 **하네스가 못 세운 것**이었다(2026-08-31).
        self.multimodal = (not hasattr(seg, "backbone")
                           and hasattr(seg, "image_encoder")
                           and hasattr(seg, "text_encoder"))
        if self.multimodal:
            self.backbone = None
            self.neck = None
            self.image_encoder = seg.image_encoder
            self.text_encoder = seg.text_encoder
            # `encode_decode` 가 이미지 인코더에만 줄인 입력을 넣는다(san 은 0.5배).
            self.asymetric_input = getattr(seg, "asymetric_input", False)
            self.encoder_resolution = getattr(seg, "encoder_resolution", 1.0)
        else:
            self.backbone = seg.backbone
            self.neck = seg.neck if getattr(seg, "with_neck", False) else None
        self.decode_head = seg.decode_head
        self.cascade = isinstance(seg.decode_head, nn.ModuleList)
        # ⚠️ **cascade 라고 다 `forward(x, prev)` 로 끝나지 않는다.** `PointHead`(point_rend)의
        #    `forward` 는 `(fine_grained_point_feats, coarse_point_feats)` 라 특징 튜플을
        #    그대로 넘기면 `torch.cat` 이 `expected Tensor ... but got tuple` 로 죽는다.
        #    mmseg 자신도 `cascade_encoder_decoder.py:135` 에 "TODO support PointRend
        #    tensor mode" 라고 적어 두었다 — 추론 경로는 마지막 단계만 `predict` 다
        #    (`encode_decode`: 0..n-2 는 forward, n-1 은 predict).
        self.cascade_predict = (
            self.cascade
            and "prev_output" in inspect.signature(
                seg.decode_head[-1].predict).parameters
            and "fine_grained_point_feats" in inspect.signature(
                seg.decode_head[-1].forward).parameters
        )
        self.test_cfg = getattr(seg, "test_cfg", None)
        # ⚠️ **MaskFormer 계열의 head 는 `forward(x)` 가 아니다** —
        #    `forward(x, batch_data_samples)` 라 그냥 부르면
        #    `forward() missing 1 required positional argument` 로 죽는다.
        #    두 번째 인자는 이름과 달리 **입력 크기를 넘기는 통로**일 뿐이고
        #    (mmseg `maskformer_head.py:151` 주석이 그렇게 말한다), 실제 추론 경로는
        #    `predict(x, batch_img_metas, test_cfg)` 다. 그쪽으로 부른다.
        self.needs_metas = (
            not self.cascade
            and not self.multimodal
            and "batch_data_samples" in inspect.signature(
                seg.decode_head.forward).parameters
        )

    def forward(self, x):
        # **계열 무관 프로브.** 값이 틀릴 때 백본/넥/헤드 중 어디서 벌어지는지 먼저 가른다 —
        # 계열마다 프로브를 새로 짜지 않으려고 여기 둔다. 튜플을 그대로 돌려주면
        # 하네스가 `out_i` 로 각각 덤프하고 `ref.shapes.txt` 도 여러 줄이 된다.
        _probe = os.environ.get("VISP_SEG_PROBE")
        if self.multimodal:
            return self._forward_multimodal(x, _probe)
        f = self.backbone(x)
        if _probe == "bb":
            return f if isinstance(f, torch.Tensor) else tuple(f)
        if self.neck is not None:
            f = self.neck(f)
        if _probe == "neck":
            return f if isinstance(f, torch.Tensor) else tuple(f)
        if self.cascade:
            # ⚠️ **`CascadeEncoderDecoder` 의 head 는 `nn.ModuleList` 다.** 그냥 부르면
            #    `Module [ModuleList] is missing the required forward` 로 죽는다 —
            #    「범위 밖」처럼 보이지만 **하네스가 못 부른 것**이다(ocrnet·point_rend).
            #    단계 규약은 mmseg `cascade_encoder_decoder.py:78` 과 같다: 0단계는
            #    `forward(x)`, 이후는 `forward(x, prev)`. 마지막 단계의 `predict` 는
            #    `forward(x, prev)` 뒤 **resize** 만 하는데 그건 그래프 밖이라 뺀다.
            n = len(self.decode_head)
            out = self.decode_head[0](f)
            # `cascade_predict` 면 마지막 단계는 forward 가 아니라 predict 로 부른다.
            # 그 외(ocrnet 등)는 예전 그대로 끝까지 forward 다.
            for i in range(1, (n - 1) if self.cascade_predict else n):
                out = self.decode_head[i](f, out)
            if self.cascade_predict:
                h, w = int(x.shape[-2]), int(x.shape[-1])
                metas = [{"img_shape": (h, w), "batch_input_shape": (h, w)}]
                out = self.decode_head[n - 1].predict(f, out, metas, self.test_cfg)
        elif self.needs_metas:
            # `predict` 는 마스크를 **입력 크기로** 올린 뒤 클래스 점수와 곱해
            # seg_logits 를 만든다(`einsum('bqc,bqhw->bchw')`). 다른 계열의
            # 「resize 는 그래프 밖」 규약과 달리 이건 head 안의 계산이라 그래프에 둔다 —
            # 여기서 빼면 남는 게 마스크 로짓이라 비교 대상이 달라진다.
            h, w = int(x.shape[-2]), int(x.shape[-1])
            metas = [{"img_shape": (h, w), "batch_input_shape": (h, w)}]
            probe = os.environ.get("VISP_SEG_PROBE")
            if probe == "cs":
                # cumsum 그 자체만. 여기서 틀리면 렌더러, 맞으면 PE 의 뒷단
                # (stride-2 슬라이스·stack·view)이 범인이다.
                feat = f[-1]
                mask = feat.new_zeros(
                    (int(feat.shape[0]), int(feat.shape[2]), int(feat.shape[3])))
                not_mask = 1 - mask
                # ⚠️ 배치축(dim=0)에 붙이면 안 된다 — 하네스가 `t[0]` 로 배치를
                #    벗기므로 절반만 비교된다. 채널축으로 붙인다.
                out = (torch.stack([not_mask.cumsum(1), not_mask.cumsum(2)], dim=1)
                       + feat[:, 0:1, :, :] * 0)
            elif probe in ("pe", "pe_n", "pe_d"):
                # 위치인코딩만. `cumsum` 이 여기 있고 MHA(baddbmm)는 없다 —
                # 둘 중 누구인지 가르는 자리.
                # `pe_n`·`pe_d` 는 그 안을 다시 셋으로 가른다. `pe` 는 전체다.
                #   pe_n → cumsum + normalize(⚠️ **음수 인덱스 슬라이스** `[:, -1:, :]`)
                #   pe_d → 거기에 dim_t 브로드캐스트 나눗셈까지
                #   pe   → 거기에 stride-2 슬라이스 + stack(dim=4) + view 까지
                feat = f[-1]
                mask = feat.new_zeros(
                    (int(feat.shape[0]), int(feat.shape[2]), int(feat.shape[3])))
                zero = feat[:, :1] * 0
                if probe == "pe":
                    out = self.decode_head.decoder_pe(mask) + zero
                else:
                    out = _pe_stage(self.decode_head.decoder_pe, mask, probe) + zero
            elif probe == "mask":
                # 값이 어디서 벌어지는지 가르는 자리 — 디코더까지만 내고
                # `predict` 의 interpolate·softmax·einsum 은 뺀다.
                from mmseg.structures import SegDataSample
                _, all_mask_preds = self.decode_head(
                    f, [SegDataSample(metainfo=metas[0])])
                out = all_mask_preds[-1]
            else:
                out = self.decode_head.predict(f, metas, None)
        else:
            out = self.decode_head(f)
        # 대부분 단일 텐서다. 여럿을 내는 것도 있어 튜플이면 그대로 흘린다 —
        # 러너가 `out_i` 로 순서대로 덤프한다.
        return out if isinstance(out, torch.Tensor) else tuple(out)

    def _forward_multimodal(self, x, probe=None):
        """`MultimodalEncoderDecoder`(san) 전용 경로.

        mmseg `multimodal_encoder_decoder.py:121` 의 `encode_decode` 와 **같은 순서**다:
        (필요하면 입력을 줄여) `image_encoder` → `text_encoder()` → 
        `decode_head.predict([원본입력, 시각특징, 클래스임베딩], metas, test_cfg)`.
        다르게 쓰면 재는 대상이 달라지므로 순서를 바꾸지 않는다.

        `text_encoder()` 는 입력을 안 받는다 — 클래스 이름 임베딩이라 이미지와 무관하다.
        `predict` 안에서 마스크를 입력 크기로 올려 클래스 점수와 곱하므로
        (`einsum('bqc,bqhw->bchw')`) 그 resize 는 head 안의 계산이고 그래프에 남긴다
        — maskformer 계열(`needs_metas`)과 같은 사정이다.
        """
        import torch.nn.functional as F
        clip_x = x
        if self.asymetric_input:
            clip_x = F.interpolate(x, scale_factor=self.encoder_resolution,
                                   mode="bilinear")
        feats = self.image_encoder(clip_x)
        if probe == "bb":
            return feats if isinstance(feats, torch.Tensor) else tuple(feats)
        cls_embeds = self.text_encoder()
        h, w = int(x.shape[-2]), int(x.shape[-1])
        metas = [{"img_shape": (h, w), "batch_input_shape": (h, w)}]
        out = self.decode_head.predict([x, feats, cls_embeds], metas, self.test_cfg)
        return out if isinstance(out, torch.Tensor) else tuple(out)


def _pe_stage(pe, mask, stage):
    """`SinePositionalEncoding.forward` 를 단계별로 끊어 낸다(프로브 전용).

    본체(mmdet `positional_encoding.py`)와 **같은 식**을 그대로 옮긴 것이다 —
    다르게 쓰면 재는 대상이 달라진다. 출력은 항상 `[B, C, H, W]` 로 맞춘다:
    하네스가 `t[0]` 로 배치를 벗기므로 채널축에 실어야 전부 비교된다.
    """
    B, H, W = mask.size()
    mask = mask.to(torch.int)
    not_mask = 1 - mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if pe.normalize:
        # ⚠️ 음수 인덱스 슬라이스다. 마지막 행/열을 집는 이 두 줄이 g2c 에서
        #    어떻게 나가는지가 미확인이었다.
        y_embed = (y_embed + pe.offset) / (y_embed[:, -1:, :] + pe.eps) * pe.scale
        x_embed = (x_embed + pe.offset) / (x_embed[:, :, -1:] + pe.eps) * pe.scale
    if stage == "pe_n":
        return torch.stack([y_embed, x_embed], dim=1)
    dim_t = torch.arange(pe.num_feats, dtype=torch.float32)
    dim_t = pe.temperature ** (2 * (dim_t // 2) / pe.num_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    if stage == "pe_d":
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    raise ValueError(stage)

def crop_size(config):
    """config 의 `data_preprocessor.size` → (H, W). 없으면 (512, 512).

    ⚠️ **계열마다 다르다** — cityscapes 는 512x1024, ade20k 는 512x512, beit 는 640x640,
    cgnet 은 680x680, bisenetv2 는 1024x1024. 하나로 고정하면 **잰 것이 그 모델이 아니다.**
    ViT 계열은 조용히 넘어가지도 않는다: `pos_embed` 토큰 수가 입력에 묶여 있어
    `The size of tensor a (1025) must match ...` 로 죽는다(512/16 → 1024+1 vs 640/16 → 1600+1).
    `data_preprocessor.size` 는 학습·평가 crop 이라 그 모델이 실제로 보는 크기다.
    """
    from mmengine.config import Config
    cfg = Config.fromfile(config)
    sz = (cfg.get("model") or {}).get("data_preprocessor", {}).get("size")
    if sz and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return 512, 512


def build(config, checkpoint=None, size=512):
    """mmseg config(.py) → (MMSegWrap(eval), 출력 shape 목록).

    `size` 는 int(정방) 또는 (H, W) 다.
    """
    from mmseg.apis import init_model                  # mmseg 만 import (g2c 무관)
    trace_friendly_ops()
    # PyTorch 2.6 부터 `torch.load` 의 `weights_only` 기본값이 True 라 mmengine 이 넣은
    # 학습 메타에 걸린다. mmdet 쪽과 같은 사정이므로 그 구현을 재사용한다.
    mmdet_wrap.allow_mmengine_checkpoint_globals()
    seg = init_model(config, checkpoint, device="cpu")
    seg.eval()
    m = MMSegWrap(seg)
    m.eval()
    h, w = (size, size) if isinstance(size, int) else size
    with torch.no_grad():
        outs = m(torch.randn(1, 3, h, w))
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    return m, [tuple(o.shape) for o in outs]
