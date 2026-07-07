"""Flow B — Python 프론트: mmdet 전/후처리 + vision.cpp(NPU) 모델 계산.

철학: 무거운 모델 계산(backbone+neck+head)만 vision.cpp(NPU)로 넘기고, 전처리·후처리는
mmdet 이 그대로 담당(검증됨·generic). 모델 전체를 '한 그래프'로 NPU 실행(모듈별로 쪼개지
않음) → 경계 1회(NPU→CPU) → 효율적.

  mmdet 모델 ─(extract_feat 를 NPU 로 patch)→ inference_detector(img) → 박스
             전처리·후처리는 mmdet, 무거운 계산만 NPU 한 덩어리

요건: mmdet + visioncpp(파이썬 바인딩). gguf 는 g2c 로 컴파일된 compose 모델.
"""
import numpy as np
import torch

import visioncpp  # vision.cpp 파이썬 바인딩 (bindings/python)


def load_npu_model(gguf_path: str, use_npu: bool = True):
    """compose gguf 를 vision.cpp 로 로드 (한 그래프). NPU(gpu) 또는 CPU."""
    dev = visioncpp.Device(visioncpp.Backend.gpu if use_npu else visioncpp.Backend.cpu)
    return visioncpp.Model.load(gguf_path, dev)


def make_npu_extract_feat(vmodel, num_levels: int):
    """mmdet detector.extract_feat 를 대체: 입력 이미지텐서 → NPU 계산 → head 원시출력(torch).

    vision.cpp compose_forward 가 backbone→neck→head 를 '한 그래프'로 실행하고 out_0..N 을
    반환한다(CWHN). 이를 mmdet 이 기대하는 [N,C,H,W] torch 리스트로 되돌린다.
    """
    def extract(img_tensor: torch.Tensor):
        # img_tensor [1,3,H,W] → vision.cpp 입력(CWHN)
        cwhn = img_tensor[0].permute(1, 2, 0).contiguous().numpy().astype("float32")
        outs = vmodel.compute_tensors(cwhn)          # NPU 한 덩어리 → [np(HWC), ...]
        # HWC → NCHW torch 로 복원 (mmdet head 입력 형식)
        feats = []
        for o in outs:
            # o: (H, W, C) → (1, C, H, W)
            t = torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0)
            feats.append(t)
        return tuple(feats)
    return extract


def run(cfg_path: str, gguf_path: str, image_path: str, use_npu: bool = True):
    from mmdet.apis import init_detector, inference_detector

    # 1) mmdet 모델 (전/후처리·config 담당)
    model = init_detector(cfg_path, None, device="cpu").eval()

    # 2) vision.cpp 로 모델 계산부 로드 + extract_feat 를 NPU 로 patch
    vmodel = load_npu_model(gguf_path, use_npu)
    num_levels = len(model.bbox_head.prior_generator.strides)
    model.extract_feat = make_npu_extract_feat(vmodel, num_levels)  # 계산만 NPU

    # 3) mmdet inference (전처리·후처리 mmdet 이 generic 하게)
    result = inference_detector(model, image_path)
    return result  # DetDataSample: bboxes/scores/labels


if __name__ == "__main__":
    import sys
    cfg, gguf, img = sys.argv[1], sys.argv[2], sys.argv[3]
    res = run(cfg, gguf, img)
    inst = res.pred_instances
    for b, s, l in zip(inst.bboxes, inst.scores, inst.labels):
        if s > 0.3:
            print(f"box={b.tolist()} label={int(l)} score={float(s):.2f}")
