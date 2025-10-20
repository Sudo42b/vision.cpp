import torch
from .workbench import invoke_test, to_nhwc


def make_fake_feats():
    # torch.Size([1, 144, 80, 80])
    # torch.Size([1, 144, 40, 40])
    # torch.Size([1, 144, 20, 20])
    feats = [torch.rand(1, 144, 80, 80, dtype=torch.float32), 
             torch.rand(1, 144, 40, 40, dtype=torch.float32),
             torch.rand(1, 144, 20, 20, dtype=torch.float32)]
    return feats


def test_make_anchors_cpp_matches_python():
    feats = make_fake_feats()
    # Prepare inputs to C++: workbench expects NHWC tensors
    inputs = [to_nhwc(f) for f in feats]

    # Build state dict empty
    state = {}

    # Provide strides param as string
    params = {"strides": "8.0, 16.0, 32.0", "grid_cell_offset": 0.5}

    out = invoke_test("make_anchors_cpp", inputs, state, params)
    assert out is not None

    # out is raw torch tensor(s) returned by C++: anchor_points, stride_tensor
    anchors_cpp = out[0]
    stride_cpp = out[1]

    # The workbench C++ raw conversion returns shape (ne3, ne2, ne1, ne0),
    # which for our tensors becomes (1,1,N,2) and (1,1,N,1). Normalize to (N,2) and (N,1).
    anchors_cpp = anchors_cpp.reshape(-1, anchors_cpp.shape[-1])
    stride_cpp = stride_cpp.reshape(-1, stride_cpp.shape[-1])

    # Now compute Python anchors using the same function imported from scripts.modules
    from scripts.modules import make_anchors
    # Build effective strides list the same way the C++ code does: parse provided
    # strides string and then trim or extend to match the number of feature maps.
    strides_s = params.get("strides", "")
    strides_list = []
    if strides_s:
        for tok in strides_s.split(','):
            try:
                strides_list.append(float(tok.strip()))
            except Exception:
                pass
    # Ensure the strides_list matches the number of feats: extend with defaults if needed
    while len(strides_list) < len(feats):
        i = len(strides_list)
        strides_list.append(8.0 * (2 ** i))

    anchors_py, stride_py = make_anchors(feats, strides_list, grid_cell_offset=0.5)

    # Compare shapes
    assert anchors_cpp.shape == anchors_py.shape
    assert stride_cpp.shape == stride_py.shape

    # Compare a few entries
    assert torch.allclose(anchors_cpp, anchors_py)
    assert torch.allclose(stride_cpp.squeeze(1), stride_py.squeeze(1))
