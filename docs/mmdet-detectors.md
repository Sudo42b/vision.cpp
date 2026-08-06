# Object Detection with MMDetection Models

This guide describes how to run detectors from
[MMDetection](https://github.com/open-mmlab/mmdetection) with _vision_.cpp.
If you are new to the library, read the [Overview](overview.md) first.

MMDetection defines hundreds of detectors as compositions of a backbone, a neck and a head.
The backbone and neck are plain feed-forward networks and translate directly into a ggml graph.
The head does not: it contains data-dependent control flow (NMS, dynamic offsets, variable
proposal counts) that cannot be captured by tracing. _vision_.cpp therefore uses a **hybrid**
structure — the feature extractor runs as a compiled ggml graph, and the head, decoding and
post-processing are hand-written C++ built from library primitives.

```
 image ──▶ backbone + neck (compiled ggml graph) ──▶ FPN features
                                                          │
                                                    head  │  tools/detect/head.cpp
                                                          ▼
                                                    raw cls / bbox
                                                          │
                                        decode + NMS      │  visp/postproc.h
                                                          ▼
                                                     detections
```

Everything below the graph is ordinary CPU code in `visp/postproc.h`, so it is reusable
outside MMDetection: the decoders are parameterised by plain structs, not by framework config
objects.

## Contents

- [Prerequisites](#prerequisites)
- [Pipeline](#pipeline)
- [Output](#output)
- [Detection heads](#detection-heads)
- [Post-processing API](#post-processing-api)
- [Two-stage detectors](#two-stage-detectors)
- [Instance segmentation](#instance-segmentation)
- [Multi-object tracking](#multi-object-tracking)
- [Configuration reference](#configuration-reference)


## Prerequisites

- _vision_.cpp built from source — see [Building](../README.md#building).
  The runners link against `libvisioncpp` plus the ggml libraries.
- A Python environment with MMDetection installed, for the export step only.
  Nothing in the runtime path depends on Python.

## Pipeline

Running a detector takes four steps. Steps 1 and 2 happen once per model; steps 3 and 4 are
the deployment path.

### Step 1 — Export the detector

`mmdet_to_pt.py` loads an MMDetection config, wraps the detector so that the backbone and neck
form a traceable `nn.Module`, and writes two files.

```sh
python tools/frontend/mmdet/mmdet_to_pt.py \
    --config /path/to/retinanet_r18_fpn_1x_coco.py \
    --checkpoint retinanet_r18.pth \
    --out backbone.pt \
    --size 512
```

| Option | Description |
| :--- | :--- |
| `--config` | MMDetection config file (`.py`). Required. |
| `--checkpoint` | Weights (`.pth`). If omitted, the config's initialisation is used — useful for shape checks, not for accuracy. |
| `--out` | Output path for the traceable module (`.pt`). Required. |
| `--size` | Square input resolution used for tracing. Default `512`. |

Outputs:

`backbone.pt`
:   The backbone and neck as a traceable module. The head is kept as an attribute so that its
    weights are preserved in the `state_dict`, but it does not participate in the forward pass.

`backbone.postproc.h`
:   Everything the C++ side needs to reconstruct the head and decode its output: anchor
    generator settings, bbox coder statistics, head convolution layout, and the pre-processing
    normalisation taken from the config's `data_preprocessor` — emitted as a generated
    `mmdet_params()` function. These values are constants once the architecture is chosen, so
    they are compiled into the runner rather than read at run time.
    See [Configuration reference](#configuration-reference).

The `.pt` file pickles by module name `mmdet_wrap`, so the export directory must be on
`PYTHONPATH` when it is loaded again.

### Step 2 — Compile the backbone

`backbone.pt` is a plain PyTorch module and is compiled to a _vision_.cpp arch module by a
PyTorch-to-ggml model compiler. That compiler is outside the scope of this document; what
matters is the interface the generated code must satisfy.

**Generated files** — for an architecture named `MMDetBackbone`:

| File | Contents |
| :--- | :--- |
| `MMDetBackbone.h` | Declarations (see below). |
| `MMDetBackbone.cpp` | Builds the ggml graph for backbone and neck. |
| `MMDetBackbone.gguf` | Weights for **both** the backbone and the head, under their original `state_dict` names. |

**Header contract** — the runner includes the header and reaches the graph through three
macro-expanded names:

```c++
namespace visp {

struct MMDetBackbone_params { /* ... */ };

tensor MMDetBackbone_forward(model_ref m, tensor x, MMDetBackbone_params const& p);
MMDetBackbone_params MMDetBackbone_detect_params(model_file const& f);

}  // namespace visp
```

**Graph contract**

- The input tensor is named `x`, type `f32`, shape `{3, size, size, 1}` in ggml `ne` order
  (CWHN — channels vary fastest).
- Each FPN level is exposed as a named graph tensor `out_0`, `out_1`, … `out_{L-1}`,
  ordered from the finest level to the coarsest. The runner resolves them with
  `ggml_graph_get_tensor`, so the names must match exactly.
- Head weights must be present in the GGUF under the names the config uses, for example
  `bbox_head.cls_convs.0.conv.weight` and `bbox_head.retina_cls.weight`. The head component
  looks them up by prefix.

If a level cannot be found the runner stops and reports the missing `out_<n>`, which means the
generated graph did not name its outputs as expected.

### Step 3 — Build the runner

```sh
bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone
```

The script compiles three translation units together, with the generated parameters
included as a header, and links them against `libvisioncpp`:

- `tools/verify/backbone/run_mmdet.cpp` — the runner,
- `tools/detect/head.cpp` — the head component,
- `output/MMDetBackbone/MMDetBackbone.cpp` — the generated graph,
- `backbone.postproc.h` — the generated parameters.

`build_mmdet_cpp.sh <gen_dir> [params.h] [arch_name]`
:   `gen_dir` is the directory holding the generated `.cpp`, `.h` and `.gguf`. The parameters
    header is found in `gen_dir` when it is there, and named explicitly otherwise.
    `arch_name` defaults to the base name of the `.cpp` found there.

The library is looked up in `build/`, which is where [Building](../README.md#building) puts it.
If you configured elsewhere, point `VISP_BUILD` at that directory:

```sh
VISP_BUILD=/path/to/that/directory bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone
```

The result is `<gen_dir>/run_mmdet`.

### Step 4 — Run

```sh
output/MMDetBackbone/run_mmdet \
    output/MMDetBackbone/MMDetBackbone.gguf \
    image.jpg \
    detected.png \
    512
```

```
run_mmdet <gguf> <input> <output> [size=512]
```

`<gguf>`
:   Weights produced in step 2.

`<input>`
:   An image (`.jpg`, `.jpeg`, `.png`, `.bmp`) or a pre-processed tensor (`.bin`).
    Images are resized and normalised in-process using `preprocess()`; the mean, standard
    deviation and channel order are compiled in. A `.bin` file is taken as-is and must
    contain `3 × size × size` `float32` values in CWHN order.

`<output>`
:   Where to write the result. The extension decides what is written: `.bin` gives raw
    detections, anything else gives the input image with the boxes drawn on it.

`[size]`
:   Input resolution. Must match the value passed to `--size` in step 1 and the shape the graph
    was compiled for.

### Output

The extension of the output path decides what is written.

```sh
run_mmdet model.gguf image.jpg detected.png 512   # the image, boxes drawn on it
run_mmdet model.gguf image.jpg boxes.bin     512   # raw float32, six numbers per box
```

An image is the default, which is what the rest of the command-line tools produce and what a
person looking at a result wants. Raw `float32` is what comparing against a reference
implementation needs, so it stays one extension away rather than being the only option.

Either way the highest-scoring detections are printed:

```
- detect(anchor): 100 boxes, 12 drawn at score >= 0.30 → detected.png

      #       x1       y1       x2       y2    score  label
      0    459.3    241.1    512.0    263.3    0.837     63
      1    306.4     69.4    361.2     84.5    0.835     63
    ...  (98 more)
```

The image carries where, the table carries what. Class is drawn as colour rather than text,
which keeps a font out of the runner.

| Variable | Effect |
| :--- | :--- |
| `VISP_DRAW_THRESHOLD` | Minimum score to draw. Default `0.3` |
| `VISP_PRINT_DETS` | How many rows to print. `0` turns the table off |

In the raw form each detection is six `float32` values written back to back:

```
x1  y1  x2  y2  score  label
```

Coordinates are pixels in the square input the detector ran on. There is no header and no
count — the number of detections is the file size divided by 24 bytes.

```python
import numpy as np
d = np.fromfile("boxes.bin", dtype="float32").reshape(-1, 6)
```

`tools/verify/draw_boxes.py` draws such a file afterwards, with class names and scores as text.


## Detection heads

Head components take FPN features and produce the raw per-level tensors that the decoders
expect. They are declared in `tools/detect/head.h`.

### `anchor_head_forward`

Shared convolution tower followed by classification and regression convolutions — the layout
used by RetinaNet, ATSS, GFL and other anchor-based dense heads. All levels share one set of
weights.

```c++
void anchor_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c,
                         std::vector<tensor>& cls_out, std::vector<tensor>& box_out);
```

`anchor_head_cfg`

| Field | Default | Description |
| :--- | :--- | :--- |
| `stacked_convs` | `4` | Depth of the shared cls/reg tower. |
| `feat_channels` | `256` | Channels inside the tower. |
| `num_base` | `9` | Anchors per location. |
| `num_classes` | `80` | Classification output channels. |
| `cls_convs_prefix` | `bbox_head.cls_convs` | Weight-name prefix of the cls tower. |
| `reg_convs_prefix` | `bbox_head.reg_convs` | Weight-name prefix of the reg tower. |
| `cls_head` | `bbox_head.retina_cls` | Final classification convolution. |
| `reg_head` | `bbox_head.retina_reg` | Final regression convolution. |
| `head_has_norm` | `false` | Whether the tower contains normalisation layers. |

Output shapes, per level `l`, in ggml `ne` order:

- `cls_out[l]` — `{num_base * num_classes, feat_w, feat_h, 1}`
- `box_out[l]` — `{num_base * 4, feat_w, feat_h, 1}`

Feed these to [`detect_anchor`](#detect_anchor).

### `vfnet_head_forward`

VFNet's head predicts distances rather than anchor deltas, and refines them with a
star-shaped deformable convolution whose offsets are computed from the first bbox prediction.
That offset computation is exactly the part that cannot be traced, so it is assembled here as
an explicit graph.

```c++
void vfnet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        vfnet_head_cfg const& c, tensor dcn_base,
                        std::vector<tensor>& cls_out, std::vector<tensor>& box_out);
```

`dcn_base` is the fixed 3×3 sampling grid, shape `{18, 1, 1, 1}`, supplied by the caller. The
component computes `offset = star_dcn_offset(bbox_pred) - dcn_base` and applies
`conv_2d_deform`, a library primitive. Per level, `cls_out[l]` is `{num_classes, w, h, 1}` and
`box_out[l]` is `{4, w, h, 1}`.

`vfnet_head_cfg` adds `gn_groups` (GroupNorm groups in the tower), `strides` (per level, used
to project offsets into feature scale) and `reg_denoms` (per level, `bbox_pred = exp(reg) *
reg_denom`).

### Adding a head

1. Add a `<name>_head_forward` function to `tools/detect/head.cpp` that turns FPN features
   into raw per-level tensors. Use library primitives (`conv_2d`, `group_norm`,
   `conv_2d_deform`); do not add framework-specific code to `src/visp`.
2. Extract the head's structural parameters in `mmdet_wrap.postproc_cfg`; they are emitted
   into the generated parameters header.
3. Connect the raw output to the matching decoder in `visp/postproc.h`, or add one if the
   decoding scheme is new.

## Post-processing API

Declared in `src/visp/postproc.h`, implemented as plain CPU code with no ggml dependency.
All multi-level inputs are per-level flat `float` buffers in CWHN order
(`index = (y * W + x) * C + c`), with the per-level `(feat_h, feat_w)` passed alongside.

```c++
struct detection {
    float x1, y1, x2, y2;  // pixel coordinates
    float score;
    int label;
};
```

### Pre-processing

`std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c, int out_size, float const mean[3], float const std[3], bool to_rgb, int* out_w = nullptr, int* out_h = nullptr)`
:   Resize to `out_size × out_size` and normalise to `(v - mean) / std`, optionally swapping
    channel order. Returns a CWHN `float32` tensor ready for the graph input.

### Dense heads

<a id="detect_anchor"></a>
`std::vector<detection> detect_anchor(cls_scores, bbox_preds, feat_hw, det_params const& p)`
:   Anchor-based decoding: anchor generation, delta decoding, per-level top-k, score
    thresholding and NMS. Used by RetinaNet, ATSS, GFL and RPN-style heads.

`std::vector<detection> detect_fcos(cls_scores, bbox_preds, centerness, feat_hw, fcos_params const& p)`
:   Anchor-free distance decoding with centerness weighting.

`std::vector<detection> detect_yolox(cls, box, obj, feat_hw, yolox_params const& p)`
:   Grid-based decoding with an objectness branch; score is `sigmoid(cls) * sigmoid(obj)`.

`std::vector<detection> detect_detr(float const* cls, float const* bbox, detr_params const& p)`
:   Set prediction. Takes query logits and normalised `cxcywh` boxes, applies top-k, and
    performs no NMS. Set `use_sigmoid` for Deformable-DETR-style heads.

`det_params` carries the anchor generator (`strides`, `octave_base_scale`, `octave_scales`,
`ratios`, `center_offset`), the bbox coder (`means`, `stds`), and the test-time thresholds
(`score_thr`, `nms_thr`, `nms_pre`, `max_per_img`). `input_w`/`input_h` clip boxes to the
image.

### Two-stage components

`std::vector<float> rpn_proposals(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`
:   Region proposals from RPN outputs: anchor decode, per-level top-k, NMS across levels.
    Returns `M × 4` boxes in image coordinates, `M ≤ max_per_img`.

`std::vector<float> roi_align(feats, feat_hw, float const* rois, int m, roi_align_params const& p)`
:   MMCV-compatible RoIAlign (`aligned = true`, adaptive `sampling_ratio`). Level assignment
    follows `clamp(floor(log2(sqrt(w*h) / finest_scale + 1e-6)), 0, L-1)`.
    Returns `M × C × out × out` in NCHW order.

`std::vector<detection> detect_roi(float const* scores, float const* bbox_deltas, float const* proposals, int n, roi_params const& p)`
:   Final RoI-head decoding: class-wise delta decoding and per-class NMS. `scores` are
    post-softmax with background last. Set `class_agnostic` when `bbox_pred` has four columns
    instead of `num_classes * 4`.

### Masks and keypoints

`std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw, detection const& box, float thr = 0.5f, int* out_h = nullptr, int* out_w = nullptr)`
:   Sigmoid, resize to the box, threshold. Returns a binary mask covering the box.

`std::vector<float> decode_keypoints(float const* heatmap, int k, int hm_h, int hm_w, float stride)`
:   Per-keypoint argmax over a heatmap; returns `k × 3` as `(x, y, score)`.

### Building blocks

`gen_anchors`, `gen_points`, `delta2bbox`, `distance2bbox` and `nms` are exposed individually
for building custom decoders.

## Two-stage detectors

RPN proposals and RoIAlign are data-dependent — the number of proposals is not known until the
network has run — so a two-stage detector cannot be a single graph. It is split into two
compiled sub-graphs with host code in between.

```
 image
   │ SubA (backbone + neck + RPN)          14 outputs: P2-P5, rpn_cls×5, rpn_bbox×5
   ▼
   │ rpn_proposals(host)                   decode + per-level NMS -> 1000 proposals
   │ roi_align(host)                       proposals + P2-P5 -> roi_feat (N,256,7,7)
   ▼
   │ SubB (bbox head, Shared2FC)           -> cls_score (N,81), bbox_pred (N,320)
   ▼
   │ detect_roi(host)                      softmax + delta decode + per-class NMS
   ▼ detections
```

Export both sub-graphs with `frcnn_to_pt.py`, compile each, then build and run:

```sh
python tools/frontend/mmdet/frcnn_to_pt.py \
    --config faster-rcnn_r50_fpn_1x_coco.py --checkpoint frcnn.pth --out /tmp/frcnn
# compile /tmp/frcnn/FRCNN_SubA.pt at 1,3,800,800 and /tmp/frcnn/FRCNN_SubB.pt at 4,256,7,7

bash tools/build/build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB

output/FRCNN_SubA/run_frcnn \
    output/FRCNN_SubA/FRCNN_SubA.gguf output/FRCNN_SubB/FRCNN_SubB.gguf \
    /tmp/frcnn/frcnn.json input.bin 800
```

`run_roi_verify` and `run_rpn_verify` check the two host stages in isolation against dumps
from the reference implementation.

## Instance segmentation

Mask R-CNN extends the above with a second RoIAlign at output size 14 over the final boxes,
a mask sub-graph, and host-side mask pasting.

```
 final boxes
   │ roi_align(out=14, host)     -> mask_feat (M,256,14,14)
   │ SubC (mask head FCN)        -> mask_logits (M,80,28,28)
   │ paste_mask(host)            -> per-instance binary masks
   ▼
```

```sh
run_maskrcnn <SubA.gguf> <SubB.gguf> <SubC.gguf> <mask.json> <input.bin> <out_prefix> [size=800]
```

> **Note**
> ggml's `conv_transpose_2d_p0` does not support batching. `run_maskrcnn` therefore evaluates
> the mask sub-graph one RoI at a time. Running it batched leaves only the first RoI correct.

## Multi-object tracking

Tracking is state management, not a network — there is nothing to compile. `ByteTracker`
(`src/visp/tracker.h`) keeps track state across frames and is detector-agnostic: it
consumes `std::vector<detection>` from any of the decoders above.

```c++
ByteTracker tracker;                       // byte_params overrides thresholds
for (int frame = 0; frame < n; ++frame) {
    std::vector<detection> dets = /* run the detector */;
    std::vector<track_result> tracks = tracker.track(dets, frame);
    // tracks[i].id is stable across frames
}
```

Each call performs Kalman prediction over an 8-state `cxcyah` model, two-stage IoU matching
(high-score detections first, then low-score), and track lifecycle management —
`num_tentatives` consecutive matches to confirm a track, `num_frames_retain` frames without a
match to drop it. Passing `frame_id == 0` resets the tracker.

## Configuration reference

`mmdet_params()` in `<name>.postproc.h` is generated by the export step and compiled into the
runner. Its fields are grouped here by consumer.

**Pre-processing** (`c.img_mean`, `c.img_std`, `c.to_rgb`) — used only when the runner is
given an image rather than a `.bin`.

| Field | Description |
| :--- | :--- |
| `img_mean`, `img_std` | Per-channel normalisation, taken from the config's `data_preprocessor`. |
| `to_rgb` | Whether to swap channel order before normalising. |

**Head reconstruction** (`c.head`) — maps onto `anchor_head_cfg`.

| Field | Description |
| :--- | :--- |
| `stacked_convs`, `feat_channels` | Shape of the shared tower. |
| `cls_convs_prefix`, `reg_convs_prefix` | Weight-name prefixes of the towers. |
| `cls_head`, `reg_head` | Names of the final convolutions. |
| `head_has_norm` | Whether the tower contains normalisation layers. |

**Decoding** (`c.det`) — maps onto `det_params`.

| Field | Description |
| :--- | :--- |
| `strides` | Stride per FPN level; its length defines the level count `L`. |
| `octave_base_scale`, `octave_scales`, `ratios`, `center_offset` | Anchor generator. |
| `num_base` | Anchors per location, `len(octave_scales) * len(ratios)`. |
| `means`, `stds` | Delta coder statistics. |
| `num_classes`, `use_sigmoid` | Classification output layout and activation. |

When the config's head is not recognised the generated function returns defaults and leaves
the stride list empty, and the runner stops rather than decoding with meaningless anchors. The
backbone still exports, but decoding must then be supplied by the caller.

Isolation harnesses for each stage live in `tools/verify/`: `run_vfnet_head` for a dense head,
`run_rpn_verify` and `run_roi_verify` for the two-stage host components, and
`run_bytetrack_verify` for tracking. Each one runs a single stage against a dump from the
reference implementation, which is the fastest way to locate a mismatch.
