"""mmdet_families.py — 계열 → (대표 config, 체크포인트) 매핑. **검증 도구들이 공유한다.**

왜 공유해야 하나
---------------
config 와 체크포인트는 **짝이 맞아야** 한다. 도구마다 대표 config 를 따로 고르면
`fetch_checkpoints.py` 가 받은 가중치와 `verify_heads.py` 가 컴파일하는 config 가
어긋나고, 그러면 로드는 되는데(strict=False) 일부가 랜덤으로 남아 **조용히 틀린다.**

그리고 매핑을 **손으로 적으면** 목록에 없는 계열이 존재하지 않는 것처럼 보인다 —
`pisa`·`rpn` 이 실제로 그랬다. mmdet 은 계열마다 `metafile.yml` 에
`Config → Weights` 를 적어 둔다. 그걸 읽는다.
"""
import glob
import os

# 모델이 아닌 폴더(데이터셋 변형·학습 레시피 변형·뼈대 조각). 대조 스크립트로 확인했다 —
# 이 안에 다른 폴더에 없는 아키텍처는 하나도 없다. `sweep_integ.py` 와 같은 목록.
SKIP_DIRS = {"_base_", "common", "misc", "legacy_1.x", "strong_baselines", "selfsup_pretrain",
             "scratch", "dsdl", "objects365", "lvis", "openimages", "cityscapes", "wider_face",
             "pascal_voc", "deepfashion", "v3det"}

# 계열별 시험 이미지. **여기 없으면 하네스의 기본 이미지를 쓴다.**
#
# ⚠️ 왜 필요한가 — MOT 트래커는 `num_classes=1`(보행자)로 학습됐다. 고양이 사진을 넣으면
#    mmdet 쪽도 우리 쪽도 **0건**이 나온다. 그러면 하네스가 `EMPTY`("한쪽이 비었다")를
#    내므로 **조용히 통과하지는 않지만**, 그 계열을 아예 못 재게 된다. 못 잰 것을
#    "대상이 못 한다" 로 적지 않으려면 맞는 사진을 줘야 한다.
#
# `bench-image.jpg` 는 이미 저장소에 있고(새 자산·라이선스 불필요) 다섯 계열 전부에서
# 2~5건이 나오는 것을 실측했다(2026-08-19, 800px·score>0.30):
#   deepsort 2 · sort 2 · qdtrack 3 · masktrack_rcnn 5 · bytetrack 2
_IMAGE = {f: "bench-image.jpg" for f in
          ("bytetrack", "deepsort", "ocsort", "qdtrack", "sort", "strongsort",
           "masktrack_rcnn")}
# 이미지는 **저장소 안**에 있다. 사용자가 준 `--image` 의 디렉토리에서 찾으면
# 엉뚱한 곳(`~/pics/bench-image.jpg`)을 가리키고, 상대경로면 하위 프로세스의 cwd 로 풀린다.
_IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "..", "tests", "input"))


def test_image(fam, default):
    """이 계열로 잴 때 쓸 이미지 경로. 지정이 없거나 파일이 없으면 `default` 그대로."""
    name = _IMAGE.get(fam)
    if not name:
        return default
    p = os.path.join(_IMG_DIR, name)
    return p if os.path.exists(p) else default


def pick_cfgs(fam_dir):
    """대표 config 후보(점수순). `_` 로 시작하거나 `_base` 로 끝나는 뼈대는 뺀다.

    후보를 여럿 돌려주는 이유: 1순위가 없는 형제 config 를 상속해 죽는 계열이 있다
    (`sort` 의 mot20 → FileNotFoundError). 차순위까지 시도해야 한다.
    """
    cands = [f for f in glob.glob(os.path.join(fam_dir, "*.py"))
             if not os.path.basename(f).startswith("_")
             and not os.path.splitext(os.path.basename(f))[0].endswith("_base")]

    def score(f):
        n, s = os.path.basename(f), 0
        for kw, w in (("r50", 4), ("fpn", 3), ("1x", 3), ("coco", 4), ("r18", 2)):
            if kw in n:
                s += w
        return -s
    return sorted(cands, key=score)


def weights_map(configs_root, fam):
    """{config 파일명: 체크포인트 URL} — 그 계열 `metafile.yml` 에서."""
    import yaml
    mf = os.path.join(configs_root, fam, "metafile.yml")
    if not os.path.exists(mf):
        return {}
    try:
        d = yaml.safe_load(open(mf)) or {}
    except Exception:
        return {}
    out = {}
    for m in (d.get("Models") or []):
        cfg, w = m.get("Config"), m.get("Weights")
        # ⚠️ `Weights` 가 **리스트인 계열이 있다**(여러 체크포인트 나열). 첫 개를 쓴다.
        if isinstance(w, (list, tuple)):
            w = w[0] if w else None
        if cfg and isinstance(w, str) and w.startswith("http"):
            out[os.path.basename(cfg)] = w
    return out


def resolve(configs_root, fam):
    """(config 상대경로, 체크포인트 파일명, URL). 못 찾으면 해당 항목이 None.

    **metafile 에 가중치가 있는 첫 후보**를 고른다 — 그래야 config 와 체크포인트가 짝이 맞는다.
    가중치가 하나도 없으면 1순위 config 와 `None` 을 돌려준다(그 계열은 CKPT_NONE 이 된다).
    """
    cands = pick_cfgs(os.path.join(configs_root, fam))
    if not cands:
        return None, None, None
    wm = weights_map(configs_root, fam)
    for cfg in cands:
        url = wm.get(os.path.basename(cfg))
        if url:
            return os.path.relpath(cfg, configs_root), os.path.basename(url), url
    return os.path.relpath(cands[0], configs_root), None, None


def families(configs_root):
    """[(계열, config 상대경로, 체크포인트 파일명 or None)] — 아키텍처 계열 전부."""
    out = []
    for name in sorted(os.listdir(configs_root)):
        d = os.path.join(configs_root, name)
        if not os.path.isdir(d) or name in SKIP_DIRS:
            continue
        cfg, ckpt, _ = resolve(configs_root, name)
        if cfg:
            out.append((name, cfg, ckpt))
    return out
