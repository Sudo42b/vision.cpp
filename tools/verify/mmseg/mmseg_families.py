"""mmseg_families.py — 계열 → (대표 config, 체크포인트 URL) 의 **예외 목록**.

기본은 `configs/<계열>/metafile.yaml` 의 첫 모델이다(`verify_seg.families()`).
손목록을 정본으로 쓰면 거기 없는 계열이 존재하지 않는 것처럼 보이므로 **열거는 그대로
metafile 이 하고, 여기서는 틀린 줄만 덮어쓴다.** mmdet 쪽
`tools/verify/mmdet/dense_head/mmdet_families.py` 의 `_OVERRIDE_LIST` 와 같은 구조다.

⚠️ **저장소(`~/mmbuild/mmsegmentation`)의 metafile 을 직접 고치지 마라.** vendor 트리라
   재클론하면 날아가고 왜 고쳤는지도 안 남는다. 예외는 전부 이 파일에 적는다.
"""

# (계열, config 상대경로, 체크포인트 URL 또는 "" = metafile 것을 그대로 쓴다)
_OVERRIDE_LIST = [
    # ⚠️ metafile 오타. `configs/emanet/metafile.yaml:22` 가
    #    `eemanet_r50-d8_4xb2-80k_cityscapes-512x1024.py`(맨 앞 `e` 가 둘)를 가리킨다.
    #    실제 파일은 `emanet_r50-...` 로 **존재한다** — 「범위 밖」이 아니라 오타 하나로
    #    `CONFIG_NONE` 이 되던 것이다(2026-08-31 확인, mmsegmentation main).
    #    체크포인트 URL 은 metafile 것이 맞으므로 덮어쓰지 않는다.
    ("emanet", "emanet/emanet_r50-d8_4xb2-80k_cityscapes-512x1024.py", ""),
]

OVERRIDE = {f: (c, w) for f, c, w in _OVERRIDE_LIST}


def apply(fam, cfg, weights, configs_root):
    """metafile 이 준 (cfg, weights) 에 예외를 얹어 돌려준다.

    `configs_root` 는 `<mmseg>/configs`. 예외가 없으면 받은 것을 그대로 낸다.
    """
    import os
    if fam not in OVERRIDE:
        return cfg, weights
    c, w = OVERRIDE[fam]
    if c:
        cfg = os.path.join(configs_root, c)
    if w:
        weights = w
    return cfg, weights
