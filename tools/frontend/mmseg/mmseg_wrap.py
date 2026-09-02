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
            elif probe in ("pix", "pix_ms"):
                # ⚠️ **head 안을 픽셀 디코더와 트랜스포머 디코더로 가른다.**
                # `pixel_decoder`(MSDeformAttn) 는 head 의 앞단이다. 여기서 틀리면
                # deform-attn 이 범인이고, 여기가 맞으면 뒤의 디코더 층이다.
                mask_features, multi_scale_memorys = self.decode_head.pixel_decoder(f)
                out = (mask_features if probe == "pix"
                       else _flatten_tensors(multi_scale_memorys))
            elif probe and ((probe.startswith("am") and probe[2:].isdigit()) or (
                    probe.startswith("ami") and probe[3:].isdigit())):
                # `_forward_head` 가 만드는 **attn_mask** 자체를 그래프 출력으로 뺀다.
                # 계단 함수(`sigmoid()<0.5`)라 여기가 한 비트만 틀려도 뒤가 크게 벌어진다 —
                # 「자를 의심하기 전에 자가 맞는지 재라」에 해당하는 자리.
                head = self.decode_head
                want = int(probe[3:] if probe.startswith("ami") else probe[2:])
                grabbed = {}
                orig = head._forward_head

                import torch.nn.functional as _F
                logits = probe.startswith("ami")

                def _spy(decoder_out, mask_feature, attn_mask_target_size):
                    cls_pred, mask_pred, attn_mask = orig(
                        decoder_out, mask_feature, attn_mask_target_size)
                    # `ami` 는 **문턱 전 로짓**을 잡는다. 여기가 통과선 안이면
                    # 마스크 비트 차이는 계단 함수의 민감도지 별개의 버그가 아니다.
                    grabbed.setdefault(
                        len(grabbed),
                        _F.interpolate(mask_pred, attn_mask_target_size,
                                       mode="bilinear", align_corners=False)
                        if logits else attn_mask)
                    return cls_pred, mask_pred, attn_mask

                head._forward_head = _spy
                try:
                    from mmseg.structures import SegDataSample
                    head(f, [SegDataSample(metainfo=metas[0])])
                finally:
                    head._forward_head = orig
                am = grabbed[want]
                # bool → float. 하네스는 f32 만 비교한다.
                out = am.to(torch.float32).unsqueeze(0)
            elif probe and probe.startswith("mask") and probe[4:].isdigit():
                # 디코더 **층별** 마스크 예측. `all_mask_preds[0]` 은 층을 하나도 안 거친
                # 예측이다 — 거기서 맞으면 범인은 디코더 층(마스크드 어텐션)이다.
                from mmseg.structures import SegDataSample
                _, all_mask_preds = self.decode_head(
                    f, [SegDataSample(metainfo=metas[0])])
                out = all_mask_preds[int(probe[4:])]
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
        # ⚠️ **항상 튜플이다.** 예전엔 출력이 하나면 벌거벗은 Tensor 를 돌려줬는데,
        #    `build()` 는 shape 을 늘 리스트로 알려줘서 **호출자가 두 겹 벗기는 사고**가
        #    났다 — `outs[0][0]` 이 19장짜리 클래스 맵에서 1장만 남겼다(2026-09-01 실측).
        #    형태를 하나로 고정해 그 사고 자체를 없앤다. 러너는 `out_i` 로 순서대로 덤프한다.
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)

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
            # ⚠️ **san 의 image_encoder 는 리스트 안에 리스트를 낸다.** 그냥 `tuple(feats)`
            #    로 내면 하네스가 `.shape` 을 찾다 죽는다(REF_FAIL 로 보여 「프로브가
            #    안 되는 계열」처럼 읽힌다 — 실제로는 평탄화만 하면 된다).
            return _flatten_tensors(feats)
        cls_embeds = self.text_encoder()
        h, w = int(x.shape[-2]), int(x.shape[-1])
        metas = [{"img_shape": (h, w), "batch_input_shape": (h, w)}]
        if probe in ("san_pe", "san_fuse"):
            # `encode_feature` 의 **앞머리만** 재현한다(mmseg san_head.py:114-137 과 같은 식).
            #   san_pe   : patch_embed + pos_embed(필요하면 bicubic resize) + query 결합까지
            #   san_fuse : 거기에 첫 CLIP 융합(fuse_clip)까지
            # 블록을 돌기 전에 이미 틀렸는지부터 가른다.
            from mmseg.models.utils import resize as _resize
            san = self.decode_head.side_adapter_network
            xx, hwshape = san.patch_embed(x)
            ori_h, ori_w = san.patch_embed.init_out_size
            pos_embed = san.pos_embed
            if san.pos_embed.shape[1] != xx.shape[1]:
                pos_embed = _resize(
                    san.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                    size=hwshape, mode="bicubic", align_corners=False,
                ).flatten(2).permute(0, 2, 1)
            pos_embed = torch.cat(
                [san.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1)
            xx = torch.cat([san.query_embed.expand(xx.shape[0], -1, -1), xx], dim=1)
            xx = xx + pos_embed
            if probe == "san_pe":
                return xx
            L = hwshape[0] * hwshape[1]
            if san.fusion_index[0] == 0:
                xx = san.fuse_clip(0, xx, feats[0][0], hwshape, L)
            return xx
        if probe == "san_enc":
            # SAN 안을 다시 가른다 — `encode_feature`(경량 ViT + CLIP 융합)까지만 잰다.
            # 여기서 맞으면 범인은 `decode_feature`(MLPMaskDecoder)다.
            san = self.decode_head.side_adapter_network
            return _flatten_tensors(san.encode_feature(x, feats, []))
        if probe in ("san_mask", "san_cls", "san_head"):
            # `predict` 는 `forward` → `predict_by_feat`(업샘플·softmax·einsum) 이다.
            # 그 둘을 갈라야 어느 쪽이 틀렸는지 짚인다 — `predict` 전체만 재면
            # 마스크 오류와 클래스 오류가 한 숫자로 섞인다.
            mask_props, mask_logits = self.decode_head.forward([x, feats, cls_embeds], [])
            if probe == "san_mask":
                return _flatten_tensors(mask_props[-1])
            if probe == "san_cls":
                return _flatten_tensors(mask_logits[-1])
            return _flatten_tensors((mask_props[-1], mask_logits[-1]))
        out = self.decode_head.predict([x, feats, cls_embeds], metas, self.test_cfg)
        return out if isinstance(out, torch.Tensor) else tuple(out)



def _flatten_tensors(obj):
    """중첩 list/tuple 을 텐서 튜플로 편다(프로브 전용). 텐서 하나면 그대로 돌려준다."""
    if isinstance(obj, torch.Tensor):
        return obj
    out = []

    def _walk(o):
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for e in o:
                _walk(e)
        elif isinstance(o, dict):
            # san 의 `encode_feature` 는 `{'query':…, 'x':…}` 리스트를 낸다.
            # 안 걸으면 빈 리스트가 돼 `out[0]` 이 IndexError 로 죽는다.
            for k in sorted(o):
                _walk(o[k])

    _walk(obj)
    return out[0] if len(out) == 1 else tuple(out)


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



def _pin_nmf_random_bases():
    """SegNeXt(`LightHamHead`)의 NMF2D 를 **결정적**으로 만든다.

    `NMF2D._build_bases` 는 추론마다 `torch.rand` 로 기저를 새로 뽑는다(`rand_init=True` 가
    config 의 실제 값이다). 그래서 **torch 두 번이 서로 어긋난다** — 실측 rel L1 1.09e-02
    (2026-08-28) · 1.275e-02 (2026-08-31). 기준값 자체가 흔들리면 컴파일러를 못 잰다.

    게다가 `aten::rand` 는 ggml 로 낼 수 없다 — 렌더러가 없어 passthrough 로 떨어지고
    「낼 수 없는 op 1개」로 컴파일이 멈춘다.

    고정 시드로 한 번 뽑아 **모듈 버퍼로 등록**한다. 그러면
      · 기준값(ref.py)과 컴파일(trace)이 **같은 값**을 쓴다 — 같은 프로세스에서 만든
        `seg.pt` 한 벌을 양쪽이 읽기 때문이다
      · 그래프에서 난수가 사라지고 GGUF 가중치가 된다
    `build()` 의 워밍업 forward 가 `torch.save` 보다 앞서므로 저장 시점에 버퍼가 잡혀 있다.

    ⚠️ **이건 원 모델과 다른 구성이다.** 원 모델은 매 추론 기저를 다시 뽑는다. 여기서
       재는 것은 「고정 기저에서 컴파일러가 맞게 번역하는가」이지 「난수 초기화까지 같은가」가
       아니다. 수치를 적을 때 이 조건을 같이 적어라.
    """
    try:
        from mmseg.models.decode_heads.ham_head import NMF2D
        import torch.nn.functional as _F
    except Exception:
        return
    if getattr(NMF2D, "_visp_pinned", False):
        return

    def _build_bases(self, B, S, D, R, device=None):
        # ⚠️ **버퍼 이름을 shape 으로 만들지 마라.** trace 중에는 `D = C // S` 가 정수가
        #    아니라 그래프 텐서로 와서 이름이 매번 달라지고, 버퍼가 새로 등록돼
        #    `torch.jit.trace` 가 「state_dict changed after running the tracer」로 죽는다.
        #    한 모델 안에서 shape 은 고정이므로 이름 하나면 된다.
        buf = getattr(self, "visp_bases", None)
        if buf is None:
            n = int(B) * int(S)
            g = torch.Generator().manual_seed(0)
            bases = _F.normalize(torch.rand((n, int(D), int(R)), generator=g), dim=1)
            # persistent=True — state_dict 에 남아야 GGUF 로 나간다.
            self.register_buffer("visp_bases", bases, persistent=True)
            buf = getattr(self, "visp_bases")
        return buf

    NMF2D._build_bases = _build_bases
    NMF2D._visp_pinned = True


# ⚠️ **모듈 import 시점에 건다.** `build()` 안에서만 걸면 기준값 프로세스에만 먹고
#    **컴파일 프로세스에는 안 먹는다** — 거기서는 `seg.pt` 를 언피클할 뿐 `build()` 를
#    부르지 않기 때문이다(언피클이 이 모듈을 import 하므로 여기 두면 양쪽 다 걸린다).
#    실측: build() 안에만 뒀을 때 생성물에 `aten::rand` 가 그대로 남았다.
_pin_nmf_random_bases()


def build(config, checkpoint=None, size=512):
    """mmseg config(.py) → (MMSegWrap(eval), 출력 shape 목록).

    `size` 는 int(정방) 또는 (H, W) 다.
    """
    from mmseg.apis import init_model                  # mmseg 만 import (g2c 무관)
    trace_friendly_ops()
    _pin_nmf_random_bases()
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
    return mmdet_compat.install_loader_modules(path, "mmseg_wrap", "mmdet_wrap")
