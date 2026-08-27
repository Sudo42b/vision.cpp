# Object Detection with MMDetection Models

This guide describes how to run detectors from
[MMDetection](https://github.com/open-mmlab/mmdetection) with vision.cpp.

MMDetection defines hundreds of detectors as compositions of a backbone, a neck and a head.
The neck is usually an FPN — a feature pyramid network, which turns one backbone output into
several feature maps at different resolutions, the *levels* this document keeps referring to.
The backbone and neck are plain feed-forward networks and translate directly into a ggml graph.
vision.cpp splits the model there: the feature extractor runs as a compiled ggml graph, and the
head, decoding and post-processing are C++ built from library primitives. One head assembled
that way serves every family that shares its structure.

```
 image ──▶ backbone + neck (compiled ggml graph) ──▶ FPN levels
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
- [What decodes to boxes](#what-decodes-to-boxes)
- [Detection heads](#detection-heads)
- [Post-processing API](#post-processing-api)
- [Two-stage detectors](#two-stage-detectors)
- [Instance segmentation](#instance-segmentation)
- [Multi-object tracking](#multi-object-tracking)
- [Configuration reference](#configuration-reference)


## Prerequisites

- vision.cpp built from source — see [Building](../README.md#building).
  The runners link against `libvisioncpp` plus the ggml libraries.
- A Python environment with MMDetection installed, for the export step only.
  Nothing in the runtime path depends on Python.

## Pipeline

Running a detector takes four steps. Steps 1 and 2 happen once per model; steps 3 and 4 are
the deployment path.

Commands in this chapter run from the **vision.cpp checkout** — the paths are written
`tools/...`. The one exception is the compiler itself: `g2c` belongs to the project that
carries vision.cpp as a submodule, so *Models whose head survives tracing*, at the end, runs
from that project's root instead and says so.

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

**`backbone.pt`**

  The backbone and neck as a traceable module. The head is kept as an attribute so that its
  weights are preserved in the `state_dict`, but it does not participate in the forward pass.

**`backbone.postproc.h`**

  Everything the C++ side needs to reconstruct the head and decode its output: anchor
  generator settings, bbox coder statistics, head convolution layout, and the pre-processing
  normalisation taken from the config's `data_preprocessor` — emitted as a generated
  `mmdet_params()` function. These values are constants once the architecture is chosen, so
  they are compiled into the runner rather than read at run time.
  See [Configuration reference](#configuration-reference).

Those two are what you read. Two more land beside them, and they are not optional: saving a
module pickles its classes by module name, so whatever opens the `.pt` has to be able to
import `mmdet_wrap`. The export copies `mmdet_wrap.py` and `mmdet_compat.py` next to the `.pt`
for exactly that reason. The compiler puts the `.pt`'s own directory on `sys.path`, so the
file opens with **nothing set in the environment** — no `PYTHONPATH`.

Keep the four together when you move them. A `.pt` separated from its wrapper fails at load in
a way that reads like a model problem.

### Step 2 — Compile the backbone

`backbone.pt` is a plain PyTorch module and is compiled to a vision.cpp arch module by
**g2c**, the PyTorch-to-ggml compiler at
[GTX_Compiler](https://github.com/Sudo42b/GTX_Compiler) — the project that carries this
repository as a submodule. Its own workings are outside the scope of this document; what
matters here is the interface the generated code must satisfy. The same compiler handles the
whole-model route at the end of this document, so it is one tool, not two.

Compile at the resolution `--size` used in step 1. Tracing records the operations for one
input shape, so the graph runs at that shape and no other. A graph built for a different size
aborts at run time in `ggml_can_repeat` once a tensor of the wrong extent reaches a residual
addition.

Generated files — for an architecture named `MMDetBackbone`:

| File | Contents |
| :--- | :--- |
| `MMDetBackbone.h` | Declarations (see below). |
| `MMDetBackbone.cpp` | Builds the ggml graph for backbone and neck. |
| `MMDetBackbone.gguf` | Weights for both the backbone and the head, under their original `state_dict` names. |

Header contract — the runner includes the header and reaches the graph through three
macro-expanded names:

```c++
namespace visp {

struct MMDetBackbone_params { /* ... */ };

tensor MMDetBackbone_forward(model_ref m, tensor x, MMDetBackbone_params const& p);
MMDetBackbone_params MMDetBackbone_detect_params(model_file const& f);

}  // namespace visp
```

Graph contract

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
bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone backbone.postproc.h
```

The script compiles three translation units together, with the generated parameters
included as a header, and links them against `libvisioncpp`:

- `tools/verify/mmdet/backbone/run_mmdet.cpp` — the runner,
- `tools/detect/head.cpp` — the head component,
- `output/MMDetBackbone/MMDetBackbone.cpp` — the generated graph,
- `backbone.postproc.h` — the generated parameters.

**`build_mmdet_cpp.sh <gen_dir> [params.h] [arch_name]`**

  `gen_dir` is the directory holding the generated `.cpp`, `.h` and `.gguf`. The parameters
  header is found in `gen_dir` when it is there, and named explicitly otherwise.
  `arch_name` defaults to the base name of the `.cpp` found there.

The library is looked up in `build/`, which is where [Building](../README.md#building) puts it.
If you configured elsewhere, name that directory:

```sh
bash tools/build/build_mmdet_cpp.sh --build /path/to/that/directory \
    output/MMDetBackbone backbone.postproc.h
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

**`<gguf>`**

  Weights produced in step 2.

**`<input>`**

  An image (`.jpg`, `.jpeg`, `.png`, `.bmp`) or a pre-processed tensor (`.bin`).
  Images are resized and normalised in-process using `preprocess()`; the mean, standard
  deviation and channel order are compiled in. A `.bin` file is taken as-is and must
  contain `3 × size × size` `float32` values in CWHN order.

**`<output>`**

  Where to write the result. The extension decides what is written: `.bin` gives raw
  detections, anything else gives the input image with the boxes drawn on it.

**`[size]`**

  Input resolution. Must match the value passed to `--size` in step 1 and the shape the graph
  was compiled for.

### Output

The extension of the output path decides what is written.

```sh
run_mmdet model.gguf image.jpg detected.png 512   # the image, boxes drawn on it
run_mmdet model.gguf image.jpg boxes.bin     512   # raw float32, six numbers per box
```

An image is the default; a `.bin` extension gives raw `float32` instead, which is what
comparing against a reference implementation needs.

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
| `VISP_PRINT_DETS` | How many rows to print. Default `10`; `0` turns the table off |

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

`tools/verify/common/draw_boxes.py` draws such a file afterwards, with class names and scores as text.


## Models whose head survives tracing

Everything above splits the model because an MMDetection head cannot be traced. Most other
detectors have no such problem — an ultralytics YOLO or a torchvision model traces whole, and
then none of the steps above apply. A compiler emits the entire graph, `install_arch.py` drops
it into `src/visp/arch/` with a registration unit beside it, and `vision-cli` dispatches on the
architecture name recorded in the GGUF:

Run these from the **compiler checkout root** — your clone of
[GTX_Compiler](https://github.com/Sudo42b/GTX_Compiler), the directory that holds `vision.cpp/`
and `pyproject.toml`. `g2c` is that project's console script, not one of vision.cpp's, and `uv run`
started inside `vision.cpp/` finds no project there: it builds a second virtual environment and
then fails to spawn.

```sh
uv run g2c --model "ultralytics.YOLO('yolo26m.pt')" --name Yolo26m \
    --output out --input-shape 1,3,640,640
python vision.cpp/tools/install_arch.py out --name Yolo26m --detect-yolo
cmake --build vision.cpp/build -j4
./vision.cpp/build/bin/vision-cli yolo26m -m out/Yolo26m.gguf \
    -i vision.cpp/tests/input/cat-and-hat.jpg -o detected.png
```

> ⚠️ **Read the `→ gguf:` line, not the exit code.** A `g2c` run that prints `done` without a
> `→ gguf:` line has failed, and it still exits `0`. A script that checks the exit status will
> carry on with no weights file.
>
> ⚠️ **Registration needs a shared library.** The registration object is referenced by nothing,
> so a static link discards the whole translation unit. The build succeeds and the command
> silently disappears from `vision-cli --help`.

`--detect-yolo` supplies what the GGUF does not carry — class count, strides, whether the head
is NMS-free — so the result comes back as boxes: `vision-cli` draws them and prints their
coordinates and scores. Registered without the flag, the same command writes the graph
outputs as raw `float32` files instead, which is what a numerical comparison needs. The `.bin`
rule described under *Output* above belongs to `run_mmdet`; a registered detector writes an
image whatever the output path is called.

The full version of this route — options, failure modes, how to measure it — is
`docs/vision-cpp-mmdet-guide-en.md` in the compiler checkout that carries this repository
as a submodule.

Measured on `yolo26m` at 640, against ultralytics on the same pixels: the three detections
above 0.25 agree in class and score (cat 0.918, couch 0.771, tv 0.681) with box coordinates
within 0.1 px, and the pre-decode tensors are at relative L1 1.7e-03 on the boxes and
4.7e-04 on the class logits.

## What decodes to boxes

Every family that has been verified, the decoder each one uses, and the measured error are in
the verification report in the compiler repository,
[`docs/verification-report-en.md`](https://github.com/Sudo42b/GTX_Compiler/blob/main/docs/verification-report-en.md).
It also records what did not verify and why.

Two measures appear there and they answer different questions. **Box** — coordinates within
2 px of MMDetection's own output, scores within 0.05, no label or count mismatch. **Tensor** —
whether the compiled graph reproduces the head's output *before* decoding, at relative L1/L2
under 5e-02. Neither is a correction of the other, and they must not be put in the same table.

If you only need to know whether your family is covered, the list in the report is the answer.
If you need to know how far to trust the number, read the method section above it.

## Detection heads

Head components take FPN features and produce the raw per-level tensors that the decoders
expect. They are declared in `tools/detect/head.h`.

### `anchor_head_forward`

Shared convolution tower followed by classification and regression convolutions — the layout
used by RetinaNet, ATSS, GFL and other anchor-based **dense** heads — dense because they
predict at every cell of every level, with no proposal step to narrow the field first. All levels share one set of
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
| `reg_stacked_convs` | `0` | Depth of the regression tower when it differs from the classification tower. `0` means they match; YOLOF has 2 and 4. |
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

**`std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c, int out_size, float const mean[3], float const std[3], bool to_rgb, int* out_w = nullptr, int* out_h = nullptr)`**

  Resize to `out_size × out_size` and normalise to `(v - mean) / std`, optionally swapping
  channel order. Returns a CWHN `float32` tensor ready for the graph input.

### Dense heads

<a id="detect_anchor"></a>
**`std::vector<detection> detect_anchor(cls_scores, bbox_preds, feat_hw, det_params const& p, score_factors = nullptr)`**

  Anchor-based decoding: anchor generation, delta decoding, per-level top-k, score
  thresholding and NMS. Used by RetinaNet, ATSS, PAA and other delta-coded heads.
  `score_factors` is the optional centerness/IoU branch. MMDetection thresholds and takes
  top-k on the class score **alone** and multiplies the factor in afterwards, so passing it
  here rather than folding it into `cls_scores` keeps the surviving set the same.

**`std::vector<detection> detect_fcos(cls_scores, bbox_preds, centerness, feat_hw, fcos_params const& p)`**

  Anchor-free distance decoding. `centerness` may be empty — GFL and VFNet fold quality into
  the class score and have no such branch. `bbox_preds` are already pixel distances: the head
  component applies the DFL integral — DFL is Distribution Focal Loss, which predicts each
  box edge as a distribution over bins and recovers the distance by integrating it — plus the
  stride multiply and the exponent, so this function
  must not apply a stride again. `point_offset` is 0.5 for FCOS and 0 for the heads built on
  an `AnchorGenerator`.

**`std::vector<detection> detect_yolox(cls, box, obj, feat_hw, yolox_params const& p)`**

  Grid-based decoding with an objectness branch; score is `sigmoid(cls) * sigmoid(obj)`.

**`std::vector<detection> detect_detr(float const* cls, float const* bbox, detr_params const& p)`**

  Set prediction. Takes query logits and normalised `cxcywh` boxes, applies top-k, and
  performs no NMS. Set `use_sigmoid` for Deformable-DETR-style heads.

`det_params` carries the anchor generator (`strides`, `octave_base_scale`, `octave_scales`,
`ratios`, `center_offset`), the bbox coder (`means`, `stds`), and the test-time thresholds
(`score_thr`, `nms_thr`, `nms_pre`, `max_per_img`). `input_w`/`input_h` clip boxes to the
image.

### Two-stage components

**`std::vector<detection> detect_rpn(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`**

  Region proposals with their objectness scores. NMS runs **per level**, not per class —
  MMDetection passes level ids to `batched_nms` — and there is no pre-NMS score threshold.
  The returned `label` is the level the proposal came from.

**`std::vector<float> rpn_proposals(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`**

  The same computation with the scores dropped: `M × 4` boxes in image coordinates,
  `M ≤ max_per_img`, which is the form the RoIAlign stage takes.

**`std::vector<float> roi_align(feats, feat_hw, float const* rois, int m, roi_align_params const& p)`**

  MMCV-compatible RoIAlign (`aligned = true`, adaptive `sampling_ratio`). Level assignment
  follows `clamp(floor(log2(sqrt(w*h) / finest_scale + 1e-6)), 0, L-1)`.
  Returns `M × C × out × out` in NCHW order.

**`std::vector<detection> detect_roi(float const* scores, float const* bbox_deltas, float const* proposals, int n, roi_params const& p)`**

  Final RoI-head decoding: class-wise delta decoding and per-class NMS. `scores` are
  post-softmax with background last. Set `class_agnostic` when `bbox_pred` has four columns
  instead of `num_classes * 4`.

### Masks and keypoints

**`std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw, detection const& box, float thr = 0.5f, int* out_h = nullptr, int* out_w = nullptr)`**

  Sigmoid, resize to the box, threshold. Returns a binary mask covering the box.

**`std::vector<float> decode_keypoints(float const* heatmap, int k, int hm_h, int hm_w, float stride)`**

  Per-keypoint argmax over a heatmap; returns `k × 3` as `(x, y, score)`.

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
# compile /tmp/frcnn/FRCNN_SubA.pt at 1,3,800,800 and /tmp/frcnn/FRCNN_SubB.pt at
# rpn_max,roi_channels,roi_out,roi_out — 1000,256,7,7 for this config; the
# values are in frcnn.json. The runner feeds every proposal as one batch,
# so a smaller batch dimension aborts at the first reshape.

bash tools/build/build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB

output/FRCNN_SubA/run_frcnn \
    output/FRCNN_SubA/FRCNN_SubA.gguf output/FRCNN_SubB/FRCNN_SubB.gguf \
    /tmp/frcnn/frcnn.json input.bin out 800
```

`input.bin` is the image at the traced resolution, normalised with `frcnn.json`'s
`img_mean`/`img_std`, written as raw CWHN `float32`:

```python
x = (np.asarray(img.resize((800, 800)), dtype=np.float32) - mean) / std
np.ascontiguousarray(x).tofile("input.bin")          # HWC memory order == CWHN
```

`out` is a prefix. `out.boxes.bin` holds the final detections in the same six-`float32`
layout `run_mmdet` writes (`x1 y1 x2 y2 score label`), and the runner prints the
highest-scoring rows. Beside it go the raw stages — `out.cls.0.bin` (`rpn_max ×
num_classes+1` logits), `out.box.0.bin`, the proposals and the per-level RPN tensors — which
is what comparing against the reference at a single stage needs.

Measured on a trained Faster R-CNN R50 at 800, `cat-and-hat.jpg`, against MMDetection's own
`predict` on the same pixels: four detections above 0.30 on both sides, box coordinates within
0.10 px and scores within 0.0008 (top row: class 15 at 0.893).

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

> Note
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

Pre-processing (`c.img_mean`, `c.img_std`, `c.to_rgb`) — used only when the runner is
given an image rather than a `.bin`.

| Field | Description |
| :--- | :--- |
| `img_mean`, `img_std` | Per-channel normalisation, taken from the config's `data_preprocessor`. |
| `to_rgb` | Whether to swap channel order before normalising. |

Head reconstruction (`c.head`) — maps onto `anchor_head_cfg`.

| Field | Description |
| :--- | :--- |
| `stacked_convs`, `feat_channels` | Shape of the shared tower. |
| `cls_convs_prefix`, `reg_convs_prefix` | Weight-name prefixes of the towers. |
| `cls_head`, `reg_head` | Names of the final convolutions. |
| `head_has_norm` | Whether the tower contains normalisation layers. |
| `num_base` | Anchors per location, `len(octave_scales) * len(ratios)`. |

Decoding (`c.det`) — maps onto `det_params`.

| Field | Description |
| :--- | :--- |
| `strides` | Stride per FPN level; its length defines the level count `L`. |
| `octave_base_scale`, `octave_scales`, `ratios`, `center_offset` | Anchor generator. |
| `means`, `stds` | Delta coder statistics. |
| `num_classes`, `use_sigmoid` | Classification output layout and activation. |

When the config's head is not recognised the generated function returns defaults and leaves
the stride list empty, and the runner stops rather than decoding with meaningless anchors. The
backbone still exports, but decoding must then be supplied by the caller.

Isolation harnesses for each stage live in `tools/verify/`: `run_vfnet_head` for a dense head,
`run_rpn_verify` and `run_roi_verify` for the two-stage host components, and
`run_bytetrack_verify` for tracking. Each one runs a single stage against a dump from the
reference implementation, which is the fastest way to locate a mismatch.
