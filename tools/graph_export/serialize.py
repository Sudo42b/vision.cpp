#!/usr/bin/env python3
"""graph_export/serialize.py — 모델 → 그래프 '데이터'(gguf) 직렬화기.

g2c 를 **read-only import** 만 한다(수정 없음). 모델을 통짜 trace 해서
Graph IR 를 노드 리스트(gguf KV `graph.nodes`)로 직렬화하고, BN-fold fp16 가중치와
함께 단일 gguf 로 쓴다. vision.cpp 의 범용 인터프리터(`arch=="graph"`)가 런타임에
이 데이터를 읽어 ggml 그래프를 조립한다. (.cpp 부품 생성 없음)

사용:
  G2C=~/.../GTX_Compiler  GGUF_PY=$G2C/vision.cpp/depend/llama/gguf-py
  PYTHONPATH=$G2C:$GGUF_PY python serialize.py --model resnet18 --out resnet18.gguf
"""
import argparse
import os
import sys

import numpy as np
import torch

# gguf-py 를 최우선 경로로 (uv env 의 다른 gguf 패키지 shadow 방지). 스크립트 기준 상대경로.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "..", "depend", "llama", "gguf-py"),
              os.environ.get("GGUF_PY", "")):
    if _cand and os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break

# g2c (read-only import) — attr/weight/out_shape/op 추출 헬퍼 재사용
from shared.compile.render_api import op_value, weight_key, attr, out_shape, scalarize
from parse import TorchParser
from parse.rich_in_out_helper import StandardInputData

torch.set_num_threads(1)

# 노드가 가중치를 갖는 op (weight prefix 를 emit). 나머지(relu/add/pool/flatten)는 무가중치.
_WEIGHTED = {"conv2d", "batch_norm", "dense", "linear", "group_norm", "layer_norm",
            "depthwise_conv2d", "instance_norm"}

# 직렬화할 정규 attr (스칼라화). attr() 가 alias 처리(padding→pad, groups→group, kernel_size→kernel).
_ATTR_KEYS = ["stride", "padding", "dilation", "groups", "kernel_size",
             "dim", "output_size", "start_dim", "num_groups", "negative_slope", "scale"]
# 리스트 그대로 실어보낼 attr (decode: strided_slice begin/dims, permute order).
_LIST_KEYS = ["begin", "slice_dims", "order"]


def _attrs_str(n):
    out = {}
    for k in _ATTR_KEYS:
        v = attr(n, k, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = scalarize(v, 0)  # 정사각 kernel/stride/pad 가정 → 첫 요소
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    for k in _LIST_KEYS:
        v = attr(n, k, None)
        if v is None:
            continue
        if not isinstance(v, (list, tuple)):
            v = [v]
        vals = [int(x) for x in v if x is not None]
        if vals:
            out[k] = ",".join(str(x) for x in vals)   # 리스트는 콤마 join
    return ";".join(f"{k}={v}" for k, v in out.items())


def serialize(graph):
    """Graph IR → (node_lines, input_ids, output_ids). 활성 입력만 노드 참조로, 가중치는 prefix."""
    nodes = list(graph.nodes)
    tid2nid = {}
    for i, n in enumerate(nodes):
        for t in (n.out_tensors or []):
            if t is not None:
                tid2nid[id(t)] = i

    # 스칼라 const 노드(예: 0.5) 값 수집 → 참조를 #val 로 인라인(별도 노드 불필요).
    const_val = {}
    for i, n in enumerate(nodes):
        if op_value(n) == "const":
            outs = n.out_tensors or []
            d = getattr(outs[0], "data", None) if outs and outs[0] is not None else None
            if d is not None:
                arr = np.asarray(d)
                if arr.size == 1:
                    const_val[i] = float(arr.reshape(-1)[0])

    lines, input_ids = [], []
    for i, n in enumerate(nodes):
        op = op_value(n)
        if op == "return":
            continue
        if op == "input":
            input_ids.append(i)
            lines.append(f"{i}|input||||" + ",".join(str(d) for d in (out_shape(n) or [])))
            continue
        # 입력: producer 노드=id / 자유파라미터(anchor buffer 등)=@key / 스칼라 const=#val.
        # (_WEIGHTED op 의 가중치는 아래 wkey 로 별도 처리하므로 @ 참조 안 함.)
        ins = []
        for t in (n.in_tensors or []):
            if t is None:
                continue
            pid = tid2nid.get(id(t))
            if pid is not None:
                ins.append("#" + repr(const_val[pid]) if pid in const_val else str(pid))
                continue
            if op in _WEIGHTED:
                continue
            data = getattr(t, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.size == 1:
                ins.append("#" + repr(float(arr.reshape(-1)[0])))     # 스칼라 const
            else:
                key = str(getattr(t, "name", "") or "").split("::")[-1]
                if key:
                    ins.append("@" + key)                            # 자유파라미터(gguf 텐서)
        wkey = weight_key(n) if op in _WEIGHTED else ""
        osh = out_shape(n) or []
        lines.append(
            f"{i}|{op}|{','.join(ins)}|{wkey}|{_attrs_str(n)}|{','.join(str(d) for d in osh)}"
        )

    # 출력 = return 노드의 in_tensor 를 만든 producer 노드 id
    output_ids = []
    for t in graph.return_node.in_tensors:
        pid = tid2nid.get(id(t))
        if pid is not None:
            output_ids.append(pid)
    return lines, input_ids, output_ids


def write_gguf(state_dict, path, node_lines, input_ids, output_ids, eps=1e-5):
    """BN-fold fp16 가중치 + 그래프 데이터 KV 로 gguf 작성 (compile_compose._write_folded_gguf 미러)."""
    import gguf

    sd = {k: v.detach().cpu().numpy() for k, v in state_dict.items()
          if not k.endswith("num_batches_tracked")}
    # BN fold: <p>.running_mean/var → <p>.weight/bias 로 접고 stats 제거 (batch_norm_2d 는 fused 기대)
    drop = set()
    for k in list(sd.keys()):
        if not k.endswith(".running_mean"):
            continue
        p = k[: -len(".running_mean")]
        mean, var = sd[p + ".running_mean"], sd[p + ".running_var"]
        gamma, beta = sd[p + ".weight"], sd[p + ".bias"]
        nw = gamma / np.sqrt(var + eps)
        sd[p + ".weight"] = nw
        sd[p + ".bias"] = beta - mean * nw
        drop.add(p + ".running_mean")
        drop.add(p + ".running_var")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = gguf.GGUFWriter(path, arch="graph")
    w.add_architecture()
    w.add_string("graph.nodes", "\n".join(node_lines))
    w.add_string("graph.inputs", ",".join(str(i) for i in input_ids))
    w.add_string("graph.outputs", ",".join(str(i) for i in output_ids))
    n = 0
    for key, arr in sd.items():
        if key in drop:
            continue
        a = arr.astype(np.float16)
        if a.ndim == 0:
            a = a.reshape(1)
        w.add_tensor(key, a)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  → gguf: {path}  ({n} tensors fp16, {len(node_lines)} nodes, "
          f"out={output_ids}, arch=graph)")


def _defrost_bn(model):
    """torchvision 검출 백본의 FrozenBatchNorm2d → 표준 BatchNorm2d(같은 파라미터). trace 시
    rsqrt/mul/sub 분해 대신 batch_norm op 로 잡혀 인터프리터의 batch_norm 핸들러로 처리됨."""
    import torch.nn as nn
    from torchvision.ops.misc import FrozenBatchNorm2d
    for nm, mod in list(model.named_children()):
        if isinstance(mod, FrozenBatchNorm2d):
            c = mod.weight.shape[0]
            bn = nn.BatchNorm2d(c, eps=float(getattr(mod, "eps", 1e-5))).eval()
            bn.weight.data = mod.weight.data.clone(); bn.bias.data = mod.bias.data.clone()
            bn.running_mean.data = mod.running_mean.data.clone()
            bn.running_var.data = mod.running_var.data.clone()
            setattr(model, nm, bn)
        else:
            _defrost_bn(mod)


class _MMDetStatic(torch.nn.Module):
    """mmdet 검출기 → **decode 까지 포함**한 정적 forward (NMS 前 박스좌표+점수).

    anchor 를 center 형(px,py,pw,ph)으로 미리 풀어 buffer(→gguf const)로 둔다(whole-tensor →
    그래프에서 anchor 슬라이스 불필요). forward 는 delta2bbox 를 exp/mul/add/sub/slice/cat 로
    전개 → 전부 그래프 op 이라 인터프리터가 실행(decode-in-graph, host postproc.cpp 불필요).
    anchor 헤드(DeltaXYWHBBoxCoder)만 decode, 그 외(distance/rpn)는 raw 폴백. means=0/stds=1 가정."""
    def __init__(self, det, input_size=512):
        super().__init__()
        self.backbone = det.backbone
        self.neck = det.neck if getattr(det, "with_neck", False) else None
        bh = det.bbox_head
        self.bbox_head = bh
        self.decode = False
        bc = getattr(bh, "bbox_coder", None)
        pg = getattr(bh, "prior_generator", None)
        if bc is not None and pg is not None and "Delta" in type(bc).__name__:
            self.decode = True
            self.ncls = int(bh.cls_out_channels)
            strides = [s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
            feats = [(int(input_size // st), int(input_size // st)) for st in strides]
            anchors = pg.grid_priors(feats, device="cpu")
            self.n_levels = len(anchors)
            for i, a in enumerate(anchors):                       # a: (N,4) x1y1x2y2
                pw = (a[:, 2] - a[:, 0]).reshape(-1, 1)
                ph = (a[:, 3] - a[:, 1]).reshape(-1, 1)
                px = (a[:, 0] + 0.5 * pw.flatten()).reshape(-1, 1)
                py = (a[:, 1] + 0.5 * ph.flatten()).reshape(-1, 1)
                self.register_buffer(f"px_{i}", px)               # (N,1) whole-tensor const
                self.register_buffer(f"py_{i}", py)
                self.register_buffer(f"pw_{i}", pw)
                self.register_buffer(f"ph_{i}", ph)

    def forward(self, x):
        f = self.backbone(x)
        if self.neck is not None:
            f = self.neck(f)
        outs = self.bbox_head(f)
        cls_scores, bbox_preds = outs[0], outs[1]
        if not self.decode:
            return tuple(cls_scores) + tuple(bbox_preds)          # raw 폴백
        out = []
        for i in range(self.n_levels):
            box = bbox_preds[i].permute(0, 2, 3, 1).reshape(-1, 4)   # (N,4) 델타
            dx = box[:, 0:1]; dy = box[:, 1:2]; dw = box[:, 2:3]; dh = box[:, 3:4]  # (N,1) slice
            px = getattr(self, f"px_{i}"); py = getattr(self, f"py_{i}")
            pw = getattr(self, f"pw_{i}"); ph = getattr(self, f"ph_{i}")
            gx = px + pw * dx                                     # (mul, add)
            gy = py + ph * dy
            gw = pw * torch.exp(dw)                               # (exp, mul)
            gh = ph * torch.exp(dh)
            x1 = gx - 0.5 * gw; y1 = gy - 0.5 * gh                # (scale, sub)
            x2 = gx + 0.5 * gw; y2 = gy + 0.5 * gh
            boxes = torch.cat([x1, y1, x2, y2], dim=1)           # (N,4) cat (stack 아님)
            out.append(boxes.reshape(1, -1, 4))
            out.append(cls_scores[i].permute(0, 2, 3, 1).reshape(1, -1, self.ncls).sigmoid())
        return tuple(out)


def load_model(name, size):
    """모델 frontend (확장 가능). torchvision + mmdet. **g2c 무관** — mmdet 도 import 만 한다."""
    import torchvision
    x = torch.randn(1, 3, size, size)
    if name.startswith("mmdet:"):                 # mmdet config → 정적 forward
        from mmdet.apis import init_detector      # mmdet 은 g2c 아님, import 만
        det = init_detector(name[len("mmdet:"):], None, device="cpu").eval()
        m = _MMDetStatic(det, input_size=size)
        _defrost_bn(m)  # 혹 FrozenBN 있으면 표준 BN 으로 (mmdet 은 대개 BN(norm_eval), no-op)
        return m, x
    if name == "retinanet_bb":  # torchvision ResNet50 + FPN 백본
        net = torchvision.models.detection.retinanet_resnet50_fpn(weights=None).eval()
        m = net.backbone
        _defrost_bn(m)
        return m, x
    m = getattr(torchvision.models, name)(weights=None).eval()
    return m, x


def main(argv=None):
    ap = argparse.ArgumentParser(prog="serialize")
    ap.add_argument("--model", default="resnet18", help="torchvision 모델명")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", required=True, help="출력 gguf")
    ap.add_argument("--dump", action="store_true", help="노드 리스트 stdout 출력")
    ap.add_argument("--golden", default="", help="검증용 입력/출력 저장 prefix (<p>.in.bin/<p>.golden.bin)")
    a = ap.parse_args(argv)

    model, x = load_model(a.model, a.size)
    graph = TorchParser()(a.model, model, StandardInputData((x,), {}))
    node_lines, in_ids, out_ids = serialize(graph)
    if a.dump:
        print("\n".join(node_lines))
    write_gguf(model.state_dict(), a.out, node_lines, in_ids, out_ids)

    if a.golden:
        with torch.no_grad():
            y = model(x)
        # 입력을 cwhn(C,W,H,N) contiguous 로 저장 (compute_graph_input {C,W,H,1} 규약)
        x[0].permute(1, 2, 0).contiguous().numpy().astype("float32").tofile(a.golden + ".in.bin")
        # 다출력(dict/tuple) 지원. 인터프리터 out_i 는 contiguous_2d_to_cwhn = cwhn(C,W,H,N).
        outs = list(y.values()) if isinstance(y, dict) else (list(y) if isinstance(y, (list, tuple)) else [y])

        def _cwhn(t):
            return (t.permute(0, 2, 3, 1).contiguous().reshape(-1) if t.dim() == 4
                    else t.reshape(-1)).numpy().astype("float32")

        for i, t in enumerate(outs):
            _cwhn(t).tofile(f"{a.golden}.golden.{i}.bin")
        print(f"  → golden: {a.golden}.in.bin / .golden.0..{len(outs)-1}.bin  "
              f"(outs {[tuple(t.shape) for t in outs]})")
    print("완료!")


if __name__ == "__main__":
    main()
