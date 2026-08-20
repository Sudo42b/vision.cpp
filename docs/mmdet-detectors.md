# Object Detection with MMDetection Models

This guide describes how to run detectors from
[MMDetection](https://github.com/open-mmlab/mmdetection) with vision.cpp.

MMDetection defines hundreds of detectors as compositions of a backbone, a neck and a head.
The backbone and neck are plain feed-forward networks and translate directly into a ggml graph.
vision.cpp splits the model there: the feature extractor runs as a compiled ggml graph, and the
head, decoding and post-processing are C++ built from library primitives. One head assembled
that way serves every family that shares its structure.

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

Those two files are the whole output. What the `.pt` does *not* carry is the class definition:
saving a module pickles its classes by module name, so whatever opens the file has to be able
to import `mmdet_wrap`. That module is not copied next to the `.pt` — it stays in
`tools/frontend/mmdet/`, and the loader has to be told where it is:

```sh
PYTHONPATH=<vision.cpp>/tools/frontend/mmdet python …
```

Keeping one copy rather than a snapshot per export is deliberate: a copy drifts from the code
that produced it, and a `.pt` loaded against a drifted wrapper fails in ways that look like a
model problem.

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

- `tools/verify/backbone/run_mmdet.cpp` — the runner,
- `tools/detect/head.cpp` — the head component,
- `output/MMDetBackbone/MMDetBackbone.cpp` — the generated graph,
- `backbone.postproc.h` — the generated parameters.

`build_mmdet_cpp.sh <gen_dir> [params.h] [arch_name]`
:   `gen_dir` is the directory holding the generated `.cpp`, `.h` and `.gguf`. The parameters
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

`tools/verify/draw_boxes.py` draws such a file afterwards, with class names and scores as text.


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

**Eighty-six families are verified against MMDetection** — forty-one single-stage and
thirty-eight two-stage agree on boxes; seven more are checked at the compiled-graph level
because they emit no boxes to compare (five mask-only families and three text-conditioned ones,
one of which is also single-stage).

That figure is on the **box** measure: coordinates within 2 px of MMDetection's own output,
scores within 0.05, no label or count mismatch. A second measure exists and answers a
different question — whether the compiled graph reproduces the head's tensors *before*
decoding, at relative L1/L2 under 5e-02. That one reads 89 of 100. Neither number is a
correction of the other, and they should not be put in the same table.

**The tables below will not add up to those figures, and that is deliberate.** A table lists
only what clears the strict bar — box under 2 px, score under 0.05, no label or count
mismatch — so the single-stage table holds forty rows and the two-stage table thirty-five.
The remainder are verified but sit outside the bar for a stated reason, and each is accounted
for in the sections that follow: `free_anchor` and `yolact` land on a score-cut boundary,
`dynamic_rcnn`, `pafpn` and `res2net` are several pixels out from fp16 weights alone (0.00 px
when recompiled in fp32), and `double_heads` agrees at 0.16 px. Counting rows and expecting the
summary is the mistake; the rows are the strict set, not the verified set.

Assembling a head and decoding its output are
separate steps, and a family can pass the first and fail the second. The runner picks a decoder from what the box prediction *is* — a delta
against an anchor, a distance from a grid point, a normalised `cxcywh` query — not from the
shape of the tower that produced it. YOLOX and RPN build the same tower as RetinaNet and decode
nothing like it.

Each row below was measured against MMDetection's own `predict_by_feat` on the same pixels at
512 (`cat-and-hat.jpg`), with the trained checkpoint the harness pairs with that config.
The harness is `tools/verify/dense_head/verify_postproc.py`; it passes at 2 px, 0.05 score,
no label mismatch and no difference in how many boxes survive.

| Family | Decoder | Worst box | Worst score |
| :--- | :--- | ---: | ---: |
| `cornernet` | `detect_corner` (embedding pairing) | 0.03 px | 0.003 |
| `bytetrack` | `detect_yolox` (tracker wrapper) | 0.06 px | 0.007 |
| `ddq` | `detect_detr` (distinct queries) | 0.06 px | 0.018 |
| `yolof` | `detect_anchor` (ctr_clamp) | 0.12 px | 0.002 |
| `ocsort` | `detect_yolox` (tracker wrapper) | 0.15 px | 0.041 |
| `centernet` | `detect_centernet` (heatmap peaks) | 0.13 px | 0.001 |
| `atss` | `detect_anchor` | 0.14 px | 0.000 |
| `dyhead` | `detect_anchor` (center_offset 0.5) | 0.21 px | 0.000 |
| `efficientnet` | `detect_anchor` | 0.15 px | 0.005 |
| `pisa` | `detect_anchor` | 0.18 px | 0.005 |
| `nas_fcos` | `detect_fcos` | 0.19 px | 0.001 |
| `condinst` | `detect_fcos` (mask branch ignored) | 0.19 px | 0.003 |
| `gfl` | `detect_fcos` | 0.23 px | 0.004 |
| `yolo` | `detect_yolov3` | 0.23 px | 0.005 |
| `boxinst` | `detect_fcos` (mask branch ignored) | 0.25 px | 0.001 |
| `strongsort` | `detect_yolox` (tracker wrapper) | 0.25 px | 0.001 |
| `reppoints` | `detect_fcos` (xyxy offset) | 0.25 px | 0.006 |
| `retinanet` | `detect_anchor` | 0.27 px | 0.000 |
| `conditional_detr` | `detect_detr` | 0.27 px | 0.002 |
| `dab_detr` | `detect_detr` | 0.28 px | 0.001 |
| `dino` | `detect_detr` | 0.28 px | 0.004 |
| `deformable_detr` | `detect_detr` | 0.26 px | 0.002 |
| `autoassign` | `detect_fcos` | 0.30 px | 0.006 |
| `ld` | `detect_fcos` | 0.30 px | 0.006 |
| `fcos` | `detect_fcos` | 0.32 px | 0.004 |
| `foveabox` | `detect_fcos` (base_edge) | 0.40 px | 0.002 |
| `ddod` | `detect_anchor` | 0.41 px | 0.001 |
| `yolox` | `detect_yolox` | 0.41 px | 0.002 |
| `ssd` | `detect_anchor` (per-level anchors) | 0.42 px | 0.003 |
| `ghm` | `detect_anchor` | 0.46 px | 0.005 |
| `vfnet` | `detect_fcos` | 0.46 px | 0.003 |
| `paa` | `detect_paa` (score voting) | 0.54 px | 0.011 |
| `pvt` | `detect_anchor` (PVT-Tiny) | 0.55 px | 0.005 |
| `sabl` | `detect_sabl` (buckets) | 0.55 px | 0.003 |
| `fsaf` | `detect_anchor` (TBLR coder) | 0.56 px | 0.003 |
| `tood` | `detect_tood` (decoded in-graph) | 0.63 px | 0.003 |
| `rtmdet` | `detect_fcos` | 0.68 px | 0.002 |
| `lad` | `detect_paa` (score voting) | 0.74 px | 0.001 |
| `nas_fpn` | `detect_anchor` | 1.04 px | 0.004 |
| `detr` | `detect_detr` | 1.85 px | 0.044 |

Every row above was re-measured together in one run, so the numbers are comparable with each
other. The harness picks a representative config per family from a hand-written override
list, and comparing against the config `metafile.yml` would have chosen instead silently pairs
a compiled graph with someone else's checkpoint. Thirteen
families differ between those two choices, and the mismatch reads as a decode failure —
`retinanet` looked 20 px out and `dino` looked like a regression until the pairing was fixed.

`pvt` is measured on PVT-Tiny, the variant the table has always used. Selecting the family's
representative config from `metafile.yml` instead picks PVTv2-B5, a different architecture
(overlapping patch embedding, linear spatial reduction), which lands at 15.74 px and is not
covered. Two variants of one family can disagree completely; naming the variant is not
optional.

Six of those rows were added after the first pass, and every one of them had been recorded as
out of scope. **"Needs its own decoder" is not the same as "cannot be done"** — the six closed
in a day. What they needed was reading the family's own
config rather than assuming defaults. `ssd` lays down a different number of anchors per level;
`yolact` gives base sizes and centres separately from the strides, so the usual
stride-times-scale reconstruction lands 0.859× small and half a cell off; `paa` and `lad` rank
by anchor rather than by (anchor, class) and re-average boxes by score voting, which is
selected by the `with_score_voting` attribute and not by the class name — `lad` subclasses
`PAAHead`, so matching on the name misses it.

`ld` needs its command run from the MMDetection root. Distillation configs name the teacher as
`teacher_config='configs/gfl/...'`, relative to the working directory rather than to the config
file, so running from anywhere else fails to find it and the family looks broken.

`condinst` was added without writing a line of code, and finding it was an accounting exercise
rather than an engineering one. Every table in this chapter classifies the families the two
harnesses *run* — thirty-four single-stage plus forty two-stage — but `configs/` holds a
hundred. Subtracting gives twenty-six that neither harness has ever touched, and a family that
is never run cannot appear as a failure. Most of the twenty-six are legitimately outside the
question: seven trackers that wrap a detector rather than being one, five mask-only families
that emit no boxes at all, the three text-conditioned families, `reid`, and the five already
discussed above. Two were not: `condinst` and `boxinst` both have an ordinary box branch, and
both were already listed in `verify_heads.py` with their checkpoints downloaded. Nobody had
run them.

`condinst` passes because `CondInstBboxHead` extends `FCOSHead` and leaves the box path alone.
Its `forward_single` adds a fourth branch — a `controller` convolution predicting 169 mask
parameters — and its `_predict_by_feat_single` carries `param_pred`, `points` and `strides`
alongside the boxes, but every one of those feeds the mask head. The decode is the FCOS decode:
`DistancePointBBoxCoder` against grid points, sigmoid centerness as the score factor,
`filter_scores_and_topk`, the standard `_bbox_post_process`. The harness classified it as
`kind fcos` on its own and `detect_fcos` was already right.

`boxinst` followed for free, at 0.25 px — `BoxInstBboxHead` subclasses `CondInstBboxHead` and
overrides neither `forward_single` nor either `predict_by_feat`, so it runs the decoder above
verbatim; only `num_params` changes, from 169 to 593. What had blocked it was not the family
but `BoxInstDataPreprocessor.__init__`, which raises unconditionally when `scikit-image` is
absent. The only use of `skimage` in that class is inside an `if training:` branch, computing
LAB colour similarity for the pseudo-masks that box-supervised *training* needs; inference
never reaches it. Installing the package was the whole fix — no code changed. A constructor
guard on a training-only dependency reads exactly like an unsupported architecture in a results
table, and the two deserve different columns.

Five families produce no boxes at all and are measured differently. `solo` and `solov2` predict
masks by location, `maskformer` and `mask2former` classify mask embeddings, and
`mask2former_vis` does the same over video; none of them has a `bbox_head`, so a box comparison
has nothing to compare. What they do have is a compiled graph, and that is what the harness
checks: the
harness compiles the backbone and neck as usual, runs `tools/verify/backbone/run_dump.cpp` —
which emits the graph's `out_*` tensors with no head and no decoding — and compares them against
torch at the same relative-L1 threshold the box families use.

| Family | Head attribute | Worst rel L1 |
| :--- | :--- | ---: |
| `solov2` | `mask_head` | 6.30e-04 |
| `solo` | `mask_head` | 6.99e-04 |
| `mask2former_vis` | `track_head` | 1.87e-03 |
| `maskformer` | `panoptic_head` | 1.94e-03 |
| `mask2former` | `panoptic_head` | 2.19e-03 |

Read that table for what it is: **the compiled portion agrees with torch**, not "the family runs
end to end". The mask heads were never ported to C++, and `maskformer`/`mask2former` declare no
neck at all, so for those two the compiled portion is the backbone alone. Recording this as
"masks verified" would be the same mistake as recording a harness limitation as a model
limitation. `mask2former_vis` is measured on a single frame; multi-frame tracking is untested,
not unsupported.

These families were failing as `HEAD_NONE` before, which read like a defect and was not: their
heads are simply named `mask_head`, `panoptic_head` or `track_head` rather than `bbox_head`, and
the export produced a perfectly good `bb.pt` the whole time. A missing classification is not a
missing capability.

Two-stage families are measured separately, at 800 and against the detector's own `predict`
rather than a head's `predict_by_feat`, because the boxes do not exist until RPN proposals,
RoIAlign and the RoI head have run. The harness is `tools/verify/roi/verify_postproc_roi.py`
and the thresholds are the same. Of the forty-five families with a `roi_head`, thirty-seven
agree — the thirty-five in the table below, plus `cascade_rpn` and `guided_anchoring`, which
need their proposals supplied from outside and are discussed after it. Forty-five rather than
forty because the harness now looks one layer inside wrapper detectors (see below):

| Family | Decoder | Worst box | Worst score |
| :--- | :--- | ---: | ---: |
| `detectors` | `detect_roi` (SAC) | 0.03 px | 0.0006 |
| `panoptic_fpn` | `detect_roi` | 0.04 px | 0.0001 |
| `qdtrack` | `detect_roi` (tracker wrapper) | 0.03 px | 0.0003 |
| `deepsort` | `detect_roi` (tracker wrapper) | 0.04 px | 0.0000 |
| `sort` | `detect_roi` (tracker wrapper) | 0.04 px | 0.0000 |
| `dcnv2` | `detect_roi` | 0.05 px | 0.0006 |
| `carafe` | `detect_roi` | 0.06 px | 0.0007 |
| `hrnet` | `detect_roi` | 0.06 px | 0.0002 |
| `mask_rcnn` | `detect_roi` | 0.06 px | 0.0009 |
| `crowddet` | `detect_roi` (set-NMS, 2 instances) | 0.07 px | 0.0001 |
| `ms_rcnn` | `detect_roi` (+ mask-IoU rescoring) | 0.07 px | 0.0015 |
| `gn+ws` | `detect_roi` | 0.08 px | 0.0008 |
| `masktrack_rcnn` | `detect_roi` (tracker wrapper) | 0.09 px | 0.0012 |
| `tridentnet` | `detect_roi` (C4, no neck) | 0.09 px | 0.0013 |
| `cascade_rcnn` | `detect_roi` (3 stages) | 0.09 px | 0.0025 |
| `gn` | `detect_roi` | 0.09 px | 0.0003 |
| `empirical_attention` | `detect_roi` | 0.09 px | 0.0003 |
| `seesaw_loss` | `detect_roi` (NormedLinear, custom activation) | 0.09 px | 0.0002 |
| `faster_rcnn` | `detect_roi` | 0.10 px | 0.0008 |
| `libra_rcnn` | `detect_roi` | 0.11 px | 0.0007 |
| `htc` | `detect_roi` (3 stages + semantic) | 0.12 px | 0.0011 |
| `resnest` | `detect_roi` | 0.12 px | 0.0002 |
| `gcnet` | `detect_roi` | 0.14 px | 0.0010 |
| `albu_example` | `detect_roi` | 0.14 px | 0.0008 |
| `grid_rcnn` | `detect_roi` (grid heatmap, no reg branch) | 0.15 px | 0.0007 |
| `point_rend` | `detect_roi` | 0.15 px | 0.0012 |
| `regnet` | `detect_roi` | 0.16 px | 0.0012 |
| `scnet` | `detect_roi` (+ global context) | 0.18 px | 0.0020 |
| `simple_copy_paste` | `detect_roi` | 0.19 px | 0.0007 |
| `swin` | `detect_roi` | 0.22 px | 0.0003 |
| `resnet_strikes_back` | `detect_roi` | 0.24 px | 0.0023 |
| `fpg` | `detect_roi` (at 1024) | 0.28 px | 0.0036 |
| `dcn` | `detect_roi` | 0.37 px | 0.0002 |
| `instaboost` | `detect_roi` | 0.39 px | 0.0079 |
| `soft_teacher` | `detect_roi` (semi-supervised wrapper) | 0.42 px | 0.0017 |

`panoptic_fpn` is back in the table, and the round trip it took is the useful part. It was
recorded at 0.04 px, then removed when the harness started deleting each stage's outputs before
that stage ran: the old run's export had failed and a stale `frcnn.json` from an earlier run had
carried it through, so the number could no longer be trusted. It was marked unverified rather
than wrong. Installing `panopticapi` and re-measuring returns **0.04 px** — the original number
was right all along. "Unverified" and "wrong" are different claims, and only one of them
survived contact with the measurement.

What blocked the re-measurement was the interpreter, not the environment, and the distinction
matters because two virtualenvs on this machine disagreed: the one the harness uses had a
working `import mmdet.models` but no `panopticapi`, while the other had `panopticapi` and could
not import `mmdet.models` at all — a stale `mmpretrain` install makes its
`reid_data_preprocessor` raise `TypeError` at class-definition time. The harness resolves child
processes through `sys.executable`, so which Python starts it decided the answer. Two sessions
measuring the same package reached opposite conclusions, each correct about its own interpreter.

Checking this needs the failing call, not an import. `import panopticapi` and even
`import mmdet.datasets.coco_panoptic` both succeed without the package, because the check is
deferred to `LoadPanopticAnnotations.__init__` (`mmdet/datasets/transforms/loading.py:572`).
The discriminating command is
`python -c "from mmdet.datasets.transforms.loading import LoadPanopticAnnotations as L; L()"`.

All eight **wrapper** families are now measured: seven trackers and one semi-supervised
detector. Trackers and semi-supervised
detectors are not detectors themselves: they put one inside `model.detector` and keep only their
own machinery at the top level, so reading `model.backbone` raises
`'ConfigDict' object has no attribute 'backbone'` and the export stops. Unwrapping the config
fixes the export, and the harness now also looks one layer inside when it decides which families
are two-stage — otherwise a family that passes is never swept again.

Unwrapping the config is only half of it, and the other half fails silently. The checkpoint is
keyed by the **attribute name the wrapper class created**, which is not the config key: both
write `model.detector`, but `SoftTeacher` builds `self.student` and `self.teacher`, so its
weights are stored under `teacher.` / `student.` and the config decides which one inference uses
(`semi_test_cfg.predict_on`). Stripping `detector.` from such a checkpoint matches nothing,
`load_state_dict` reports it and carries on, and the graph is built on random weights — which
reads as a decode bug, not a loading bug: boxes 562 px out, 91 labels wrong, 100 detections
against MMDetection's 5. Check the count of unloaded tensors, not whether loading raised.
Trackers mostly use `detector.`, but not all of them: MMDetection ships some tracker weights as
the **detector alone**, already flat (`deepsort`, `sort` and `strongsort` are keyed
`backbone.` / `neck.` / `bbox_head.`). So a prefix that matches nothing means one of two things,
and they need opposite handling — already-flat, which should be used as-is, or a wrong guess,
which must stop. Every detector has a `backbone`, and that is the signature that separates them.

The other half of a wrapper is its `data_preprocessor`, and it is easy to drop. Six of the seven
trackers declare it on the **wrapper** and give the inner detector none (`soft_teacher` is the
exception, which is why it worked first). Unwrap without carrying it down and normalisation
disappears — mean 0, std 1 — and the detector returns nothing at all. The reference side reads
the same unwrapped config, so both sides return nothing; the harness reports `EMPTY` rather than
a pass, so the failure is visible, but the family cannot be measured until the preprocessor comes
down with it. The wrapper's preprocessor is a `TrackDataPreprocessor`, which expects video-shaped
batches, so it is rewritten to `DetDataPreprocessor` on the way down — same numbers, no trap for
whoever opens the dumped config with `inference_detector`.

Trackers are trained on pedestrians with `num_classes=1`, so the harness picks a test image per
family (`mmdet_families.test_image`); the cat photo the other families use yields no detections
on either side, which reports as `EMPTY`. The chosen image is printed beside the numbers
whenever it differs from the default, so a zero can be told apart from a wrong photo later.

`fpg` is measured at 1024 rather than 800 because it builds levels below P5, where 800 stops
dividing evenly — a 25-wide map meets a 26-wide one and the export
aborts. That is a property of the resolution, not of the family.

The rest split three ways, and the split matters more than the count. Only the first group is
unsolved; the other two are a deliberate precision choice and a boundary case:

- **The RPN is not a standard anchor RPN**, so host `rpn_proposals` cannot lay down the priors.
  `queryinst` and `sparse_rcnn` learn proposals outright, and `groie` is refused for the
  neighbouring reason — `GenericRoIExtractor` aggregates every level through per-level
  convolutions, which host RoIAlign cannot express. All three stop at export with a stated
  reason rather than a wrong number.

  `cascade_rpn` and `guided_anchoring` were in this group and are no longer. Their RPNs are
  indeed non-standard — one refines across stages, the other predicts anchor shapes — but
  **their RoI heads are ordinary**, so taking the proposals from outside leaves the rest of the
  path measurable. Each is given the proposals its own `rpn_head` produced in torch, which
  `frcnn.json` selects with `own_rpn`. Measured that way: `guided_anchoring` 0.02 px,
  `cascade_rpn` 0.03 px. Do not reuse the fixed grid that `fast_rcnn` gets — that family has no
  RPN to ask, and these two do.
- **FP16 weights, not a defect.** `dynamic_rcnn` (10.24 px), `pafpn` (8.77 px) and `res2net`
  (6.81 px) return the right count and the right labels with the coordinates several pixels
  out. Recompiling with fp32 weights makes all three exact at **0.00 px**, so the gap is the
  half-precision the compiler deliberately stores — the NPU this targets is FP16-native.
  Isolating it took swapping one tensor at a time: substituting the C++ RPN *class* scores
  into torch changed nothing, while substituting the *box deltas* reproduced the full 10.25 px
  from a maximum delta error of 0.0023. Do not replace these numbers with their fp32 twins;
  they are what the deployment precision produces.
- **Neither the graph nor the decoder is at fault.** `double_heads` agrees at 0.16 px and
  differs by exactly one box: mmdet scores it 0.2954 and the compiled graph 0.3010, on either
  side of the 0.30 cut the harness itself applies. That is the fp16 score error landing on a
  threshold, not a decode difference, so the harness now prints how many boxes sit in the
  0.30–0.35 band beside every count mismatch. `fast_rcnn` is not measured at all — its
  metafile lists no weights, and random initialisation cannot judge a decoder, so it is
  recorded as untried rather than failing.
- **The family post-processed its own way.** This group is now empty, and none of the three
  was visible in the tensors. `ms_rcnn` multiplies every
  score by a predicted mask IoU, which needs the whole mask branch — RoIAlign at 14, the mask
  head, then a second head over the features concatenated with the chosen mask channel.
  `seesaw_loss` is two things at once: a `NormedLinear` classifier (the weight normalisation is
  constant at inference and folds away; the input normalisation stays) and a custom activation
  over `num_classes + 2` channels, so reading the class count as "output size minus one" shifts
  every label by one. `crowddet` needed set-NMS — boxes from the same proposal do not suppress
  each other — but three of its four differences were RPN settings that are not the defaults:
  fixed anchor `centers`, `clip_border=False`, and an objectness head with two channels rather
  than one because its `loss_cls` omits `use_sigmoid`. Its RPN tensors matched at 3e-04 from
  the start, which is what said the problem was in the host code and not in the graph.
- **An operator was missing, approximated, or silently reduced along the wrong axis.** This
  group is now empty. `carafe` rendered
  `pixel_shuffle` as a pass-through identity, skipping CARAFE's upsampling outright
  (29 px → 0.06 px). `libra_rcnn` approximated with a fixed kernel the non-integer
  `adaptive_max_pool2d` that BFP uses to scatter back to P6 (12 px → 0.11 px). Both announced
  themselves as `TODO` comments in the generated `.cpp`, so grep for those before reading
  anything else.

  `gcnet` (37 px → 0.14 px) had no such marker. `ContextBlock` uses
  `nn.LayerNorm([planes, 1, 1])` — three normalised axes — but the renderer always emitted
  `ggml_norm`, which reduces `ne0` alone. At that point the tensor is `ne [1, 1, C, N]`, so
  `ne0` is 1: normalising a single element gives `x - mean(x) = 0`, and the whole channel
  branch collapses to a constant bias. A renderer that cannot express an operation still emits
  shape-correct code, which passes compilation and every shape assertion while returning wrong
  values.

  The three cascade families joined them later, and each was a different missing piece rather
  than a shared cascade bug. `htc` (94 px → 0.12 px) feeds a semantic segmentation branch back
  into every RoI: RoIAlign at stride 8 onto a 14×14 grid, average-pooled to 7×7 and added to
  the box features **at every stage**, not only the first. `scnet` (29 px → 0.18 px) adds a
  global-context vector to all RoI positions the same way. `detectors` (30 px → 0.03 px) was
  not a fusion at all — its switchable atrous convolution ran at one dilation. `swin`
  (0.22 px) failed earlier still, at load: `GGML_MAX_NAME` is 64 and its tensor names are
  longer, and separately the shifted-window attention mask is built by slice assignment that
  tracing drops, which silently zeroes the mask instead of crashing. The length is resolved
  by **shortening the names, not by patching ggml** — a patched submodule commit lives on no
  remote we can push to, so it would break every fresh clone. What made this one hard to see
  is that the shortener already existed and still let the name through: it folds the module
  prefix against a budget that assumes a short suffix (`running_mean`, twelve characters),
  and `relative_position_bias_table` is twenty-eight. The prefix here is short enough to pass
  untouched, and nothing checked the finished length, so the bake wrote a GGUF that could not
  be loaded and said so only much later, as one line from the runner. Counting the longest
  weight name is not the same as checking what the shortener does with it.

Nothing is left unsorted. `tridentnet` used to return no boxes at all, and it took two
different C4-only faults to explain that. Its RPN puts five scales on **one** level
(`scales=[2,4,8,16,32]`, stride 16) where an FPN RPN puts one scale on each of five, so
reading only `scales[0]` built three anchors instead of fifteen and then read a
fifteen-channel objectness map as if it had three — every proposal landed somewhere else.
Fixing that moved the crash rather than removing it: with one level, NMS left 107 proposals
where the RoI graph had been compiled for 1000, and the `flatten` inside it bakes the row
count into a reshape. Proposals are now padded up to the cap and only the real rows are
decoded. A first fix that does not make the symptom go away usually means a second cause,
not a wrong first fix.

`rpn` decodes proposals rather than detections, so a worst case over the whole set says little:
of 185 proposals the median is 0.09 px and 182 are within 1 px, with one proposal of 186
falling on the other side of the score cut. `free_anchor` agrees to 0.35 px and 0.025 with one
box likewise on the boundary.

Three groups do not decode, and they fail for different reasons:

- **Something before the decoder already disagrees with torch.** This group is now empty, and
  emptying it took no new code — the three families recorded here (`tood`, `deformable_detr`,
  `dyhead`) were re-measured after the compiler fixes landed, and two of them simply passed:
  `tood` at 0.63 px and `deformable_detr` at 0.26 px. The note that `tood`'s "box branch blows
  up" and that `dyhead`'s neck sat at 0.7 relative L1 described a tree that no longer exists.
  Re-running a recorded failure after unrelated fixes is cheaper than reading it.

  `dyhead` closed too, and it hid well. Its neck agreed at 2e-03 relative L1
  and its head at 1e-04, yet the boxes were 112 px out with the score identical to four digits —
  the right cell won and was placed at the wrong pixel. The cause was in anchor generation:
  `AnchorGenerator` keeps `base_sizes` equal to the strides and puts `octave_base_scale` into
  `scales`, but the host folded the two together into `base_size = stride * octave_base_scale`.
  Anchor *sizes* come out the same either way, which is why no shape check ever complained; the
  *centre* does not, because it is `center_offset * base_size`. At stride 32 with
  `octave_base_scale` 8 that is 128 against MMDetection's 16 — exactly the 112 px observed.
  Families with `center_offset` 0 (retinanet, atss, gfl, …) are immune, since both readings give
  zero, so a single family carried the defect for the whole anchor decoder. Only `dyhead` and
  `glip` set it non-zero. Now 0.21 px, with the anchor families re-measured unchanged.

  `MMDET_DUMP_HEAD` writes the neck output beside the head output for exactly this split — a
  family whose `feat` dumps match and whose `cls`/`box` dumps do not is a head problem, and the
  reverse is a compiler problem. When both match, as here, what is left is the decoder.

- **Beware the pairing when you re-measure by hand.** `tood` first re-measured at 13.89 px with
  a count mismatch, which looked like a live defect and was not: `verify_heads.py` exports from
  its hand-written override list (`tood_r50_fpn_1x_coco.py`, the anchor-free variant) while the
  config that `mmdet_families.resolve()` returns is the anchor-based one. Compiling from one and
  judging against the other compares a graph with someone else's weights. Take the pair from the
  harness, never from the config directory.
- **The family post-processes its own way.** This group is down to `yolact`, and it is a
  boundary case rather than a decode failure: fast NMS is implemented and the boxes agree to
  0.74 px, but one box sits either side of the harness's own 0.30 cut (mmdet 0.3013, compiled
  0.2951). It is counted with `free_anchor` and `double_heads`, not with the failures.
- **The priors or the coder are outside `det_params`.** This group is now empty. `ssd`,
  `fsaf`, `paa`, `lad`, `cornernet` and `centernet` all decode, and `yolov3` did earlier.
  `centripetalnet` decodes too but lands at 3.80 px on one of two boxes, where a single
  top-k rank flips between two nearly-equal heatmap peaks; running mmdet's own
  `_decode_heatmap` on our tensors returns the same answer (0.4387 against 0.4393), so the
  formula is equivalent and what remains is half-precision, not the decoder.

A family in the first group is not silently wrong: without anchor parameters `detect_anchor`
generates no candidates and the runner reports zero boxes.

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

`std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c, int out_size, float const mean[3], float const std[3], bool to_rgb, int* out_w = nullptr, int* out_h = nullptr)`
:   Resize to `out_size × out_size` and normalise to `(v - mean) / std`, optionally swapping
    channel order. Returns a CWHN `float32` tensor ready for the graph input.

### Dense heads

<a id="detect_anchor"></a>
`std::vector<detection> detect_anchor(cls_scores, bbox_preds, feat_hw, det_params const& p, score_factors = nullptr)`
:   Anchor-based decoding: anchor generation, delta decoding, per-level top-k, score
    thresholding and NMS. Used by RetinaNet, ATSS, PAA and other delta-coded heads.
    `score_factors` is the optional centerness/IoU branch. MMDetection thresholds and takes
    top-k on the class score **alone** and multiplies the factor in afterwards, so passing it
    here rather than folding it into `cls_scores` keeps the surviving set the same.

`std::vector<detection> detect_fcos(cls_scores, bbox_preds, centerness, feat_hw, fcos_params const& p)`
:   Anchor-free distance decoding. `centerness` may be empty — GFL and VFNet fold quality into
    the class score and have no such branch. `bbox_preds` are already pixel distances: the head
    component applies the DFL integral, the stride multiply and the exponent, so this function
    must not apply a stride again. `point_offset` is 0.5 for FCOS and 0 for the heads built on
    an `AnchorGenerator`.

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

`std::vector<detection> detect_rpn(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`
:   Region proposals with their objectness scores. NMS runs **per level**, not per class —
    MMDetection passes level ids to `batched_nms` — and there is no pre-NMS score threshold.
    The returned `label` is the level the proposal came from.

`std::vector<float> rpn_proposals(rpn_cls, rpn_bbox, feat_hw, rpn_params const& p)`
:   The same computation with the scores dropped: `M × 4` boxes in image coordinates,
    `M ≤ max_per_img`, which is the form the RoIAlign stage takes.

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
