#!/usr/bin/env python3
"""append_head_weights.py — g2c 가 구운 GGUF 에 **그래프 밖 가중치**를 덧붙인다.

왜 g2c 가 아니라 여기인가
------------------------
g2c 는 `torch.jit.trace` 가 만든 그래프의 노드에서만 가중치를 모은다. 그건 옳다 —
"그래프에 쓰인 것 = 실행에 필요한 것" 이 보통은 성립하기 때문이다.

그 등식이 **이 경로에서만** 깨진다. head 를 C++ 부품으로 조립하기로 했으므로
`MMDetBackbone.forward` 는 backbone+neck 까지만 돈다. 그래서 `bbox_head`(그리고 DETR
계열이면 `encoder`/`decoder`/`query_embedding`)가 그래프에 안 들어가고, GGUF 에도 안 실린다.
C++ 부품이 이름으로 찾다가 `tensor not found: bbox_head...` 로 죽는다.

이걸 g2c 안에서 처리하면 컴파일러에 "그래프 밖 가중치" 라는 개념이 하나 늘어난다.
**mmdet 을 아는 쪽이 자기 것을 덧붙이는 편이 맞다** — g2c 는 원본 그대로 두고, 산출물은
여전히 파일 하나다(러너도 안 바뀐다).

무엇을 싣나
----------
`state_dict` 를 통째로 붓지 않는다. fold 로 흡수된 BN weight/bias 까지 되살아나는데,
`.cpp` 가 안 쓰는 데다 **fold 이전 값**이라 conv 와 어긋난 채 파일에 남는다.
`MMDetBackbone.head_weight_prefixes` 가 선언한 prefix 아래만, 그리고 GGUF 에 아직 없는
것만 싣는다.

사용:
    python append_head_weights.py bb.pt out/Fam.gguf
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    # 검증 하네스가 두는 import 우회(mmpretrain 의 transformers 충돌). 없으면 그냥 넘어간다.
    import _stub  # noqa: F401
except Exception:
    pass

# BN 의 running 통계는 학습 파라미터가 아니다. conv 로 접혔거나 추론에 안 쓴다.
_SKIP_SUFFIX = ("num_batches_tracked", "running_mean", "running_var")


def _gguf():
    """vendor 된 gguf-py 를 import. vision.cpp 안에서만 경로를 찾는다."""
    here = os.path.dirname(os.path.abspath(__file__))
    v = os.path.abspath(os.path.join(here, "..", "..", ".."))   # …/vision.cpp
    sys.path.insert(0, os.path.join(v, "depend", "llama", "gguf-py"))
    import gguf
    return gguf


# g2c 와 **같은 축약 규칙**을 쓴다 — 따로 구현하면 언젠가 갈린다.
# 이 파일은 `<g2c>/vision.cpp/tools/frontend/mmdet/` 에 있으므로 네 단계 위가 g2c 루트다.
_G2C_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
if _G2C_ROOT not in sys.path:
    sys.path.insert(0, _G2C_ROOT)
from shared.compile.tensor_names import short_name  # noqa: E402


def append(pt_path: str, gguf_path: str, prefixes=None) -> int:
    """`gguf_path` 를 제자리에서 다시 써서 선언된 가중치를 덧붙인다. 추가된 개수를 돌려준다."""
    import torch

    gguf = _gguf()
    model = torch.load(pt_path, weights_only=False)
    if prefixes is None:
        prefixes = tuple(getattr(model, "head_weight_prefixes", ()) or ())
    if not prefixes:
        return 0

    reader = gguf.GGUFReader(gguf_path)
    arch = "fam"
    fld = reader.fields.get("general.architecture")
    if fld is not None:
        arch = bytes(fld.parts[-1]).decode()
    # ⚠️ reader 의 `.data` 는 **torch 순서**(shape 가 ne 의 역순)로 나온다 —
    #    writer 가 다시 뒤집으므로 그대로 넘기면 왕복이 맞는다.
    existing = [(t.name, np.asarray(t.data)) for t in reader.tensors]
    have = {n for n, _ in existing}

    extra = []
    for key, tensor in model.state_dict().items():
        if key in have or not key.startswith(prefixes) or key.endswith(_SKIP_SUFFIX):
            continue
        arr = tensor.detach().cpu().numpy()
        # 정수 버퍼를 fp16 으로 구우면 값이 뭉개진다 — 부동소수만 내린다.
        # ⚠️ **본체가 어떤 정밀도로 구워졌는지 따라가야 한다.** g2c 의 `generate_gguf` 는
        #    `GTX_GGUF_F32=1` 이면 fp32 로 굽는데 여기만 fp16 을 하드코딩하고 있었다.
        #    한 gguf 안에 두 정밀도가 섞이면 러너가 **세그폴트로 죽는다**(실측: fp32 로
        #    구운 yolact·rpn 이 `run_mmdet` 진입 직후 SIGSEGV).
        #    「정밀도냐 버그냐」를 가르는 fp32 대조가 one-stage 계열에서 **통째로 막혀
        #    있었다** — two-stage 는 head 를 덧붙이지 않아 안 걸렸다.
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32
                             if os.environ.get("GTX_GGUF_F32") == "1" else np.float16)
        # ⚠️ **여기도 이름을 줄여야 한다.** g2c 가 굽는 쪽(`generate_gguf`)만 줄이면
        #    나중에 덧붙는 head 가중치가 64자를 넘어 로드가 거부된다
        #    (`mask_head.mask_feature_head.convs_all_levels.…` 실측 64자).
        extra.append((short_name(key), arr))
    if not extra:
        return 0

    tmp = gguf_path + ".tmp"
    # ⚠️ `GGUFWriter(path, arch=...)` 가 이미 `general.architecture` 를 쓴다. 여기서
    #    `add_architecture()` 를 또 부르면 최신 gguf-py 가 stderr 로
    #    `Duplicated key name 'general.architecture'` 를 뱉는다. 값이 같아 산출물은 멀쩡한데
    #    그 줄이 상위 하네스의 "실패 이유" 로 잡혀 원인을 오도한다.
    writer = gguf.GGUFWriter(tmp, arch=arch)
    for name, arr in existing + extra:
        writer.add_tensor(name, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.replace(tmp, gguf_path)
    return len(extra)


def main(argv=None):
    a = (argv if argv is not None else sys.argv[1:])
    if len(a) < 2:
        print(__doc__)
        return 2
    n = append(a[0], a[1])
    print(f"  → appended {n} weights that the graph does not use: {a[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
