# tools — detection frontends, components and runners

Everything needed to run a detector with _vision_.cpp that does not belong in the library
itself. [docs/mmdet-detectors.md](../docs/mmdet-detectors.md) explains the pipeline end to end;
this file describes how the directory is arranged.

## Layout

```
tools/
  frontend/
    mmdet/            MMDetection-specific. The only place that imports mmdet.
      mmdet_wrap.py     traceable module + config extraction
      mmdet_to_pt.py    CLI: config -> backbone.pt + <name>.postproc.h
      frcnn_wrap.py     two-stage / mask sub-graphs
      frcnn_to_pt.py    CLI for the above
  detect/             Framework-neutral head components, compiled into the runner
    head.h  head.cpp
  verify/             Runners and inspection, grouped by task
    backbone/    run_mmdet.cpp  run_dump.cpp
    dense_head/  run_vfnet_head.cpp
    roi/         run_frcnn.cpp  run_roi_verify.cpp  run_rpn_verify.cpp
    seg/         run_maskrcnn.cpp
    tracking/    run_bytetrack_verify.cpp
    draw_boxes.py
  build/              Build scripts
```

A detector runs as a hybrid: the backbone and neck are a compiled ggml graph, while the head,
decoding and post-processing are C++ assembled from library primitives. Detection heads carry
control flow that depends on the data — suppression counts that are not known in advance,
deformable offsets derived from an earlier prediction, proposal counts that vary per image — and
tracing records only the path one input happened to take.

Two arrangements keep framework knowledge from spreading:

- **`detect/head.cpp` is not part of `libvisioncpp`.** It is compiled together with the runner,
  so detector-specific structure never enters the core library.
- **Decoding lives in the library, not in the frontend.** `detect_anchor`, `roi_align` and
  `rpn_proposals` in `src/visp/postproc.h` take numbers, not configuration objects, which is why
  they are reusable for detectors that never went through MMDetection.

## Single-stage detectors

```sh
# 1. Export. Writes backbone.pt and backbone.postproc.h.
python tools/frontend/mmdet/mmdet_to_pt.py \
    --config retinanet_r18_fpn_1x_coco.py --checkpoint retinanet_r18.pth \
    --out backbone.pt --size 512

# 2. Compile backbone.pt to a vision.cpp arch module: <Arch>.cpp, <Arch>.h, <Arch>.gguf.
#    The interface the generated code must satisfy is in docs/mmdet-detectors.md.

# 3. Build the runner: the generated graph, head.cpp and run_mmdet.cpp together.
bash tools/build/build_mmdet_cpp.sh output/MMDetBackbone

# 4. Run.
output/MMDetBackbone/run_mmdet output/MMDetBackbone/MMDetBackbone.gguf image.jpg boxes.bin 512
```

`backbone.postproc.h` holds a generated `mmdet_params()` — anchor scales, head convolution
layout, normalisation values. Once an architecture is fixed those are constants, so they are
compiled into the runner rather than read at run time, and the deployed set is the executable
and the weights.

`mmdet_wrap.postproc_cfg` is the only code that reads an MMDetection configuration. It also
extracts `img_mean` / `img_std` / `to_rgb` from `data_preprocessor`, so pre-processing is the
library's `preprocess()` driven by extracted values rather than anything hand-written.

## Looking at the output

`run_mmdet` writes detections as raw `float32` — six numbers per box, `x1 y1 x2 y2 score label`
— because that is what comparing against a reference implementation needs. It also prints the
highest-scoring ones:

```
- detect(anchor): 100 boxes → boxes.bin (x1,y1,x2,y2,score,label f32*6)

      #       x1       y1       x2       y2    score  label
      0    459.3    241.1    512.0    263.3    0.837     63
      1    306.4     69.4    361.2     84.5    0.835     63
    ...  (98 more)
```

`VISP_PRINT_DETS` changes how many are printed, or turns the table off with `0`.

To see the boxes on the image:

```sh
python tools/verify/draw_boxes.py image.jpg boxes.bin -o annotated.png -t 0.3
```

`--size` if the detector ran at something other than 512, `--labels` for a class list other than
COCO 80.

`run_dump` does the same for any generated graph — it prints each output tensor's shape and
writes it as raw `float32`, which is the way to inspect a backbone that has no head attached.

## Heads

`detect/head.h` declares the components that turn FPN features into raw per-level tensors.

`anchor_head_forward`
:   Shared convolution tower followed by classification and regression convolutions — RetinaNet,
    ATSS, GFL and other anchor-based dense heads. The differences between them are values in
    `anchor_head_cfg`, not code.

`vfnet_head_forward`
:   Distances rather than anchor deltas, refined by a star-shaped deformable convolution whose
    sampling offsets are computed from the first bbox prediction. Those offsets are values
    produced during the forward pass, so the component builds that computation explicitly.

Adding a head means a new `<name>_head_forward` here using library primitives (`conv_2d`,
`group_norm`, `conv_2d_deform`), the matching parameters emitted by `mmdet_wrap.postproc_cfg`,
and a decoder in `postproc.h` if the decoding scheme is new.

## Two-stage detectors

RPN proposals and RoIAlign are data-dependent — the number of proposals is not known until the
network has run — so a two-stage detector cannot be a single graph.

```
 image
   │ SubA (backbone + neck + RPN)     14 outputs: P2-P5, rpn_cls×5, rpn_bbox×5
   ▼
   │ rpn_proposals (host)             decode + per-level NMS -> proposals
   │ roi_align (host)                 proposals + P2-P5 -> roi_feat (N,256,7,7)
   ▼
   │ SubB (bbox head, Shared2FC)      -> cls_score, bbox_pred
   ▼
   │ detect_roi (host)                softmax + delta decode + per-class NMS
   ▼ detections
```

```sh
python tools/frontend/mmdet/frcnn_to_pt.py \
    --config faster-rcnn_r50_fpn_1x_coco.py --checkpoint frcnn.pth --out /tmp/frcnn
# compile FRCNN_SubA.pt at 1,3,800,800 and FRCNN_SubB.pt at 4,256,7,7

bash tools/build/build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB
output/FRCNN_SubA/run_frcnn output/FRCNN_SubA/FRCNN_SubA.gguf \
    output/FRCNN_SubB/FRCNN_SubB.gguf /tmp/frcnn/frcnn.json input.bin 800
```

The host operations are `rpn_proposals`, `roi_align` and `detect_roi` in the library.

## Instance segmentation

Mask R-CNN adds a second RoIAlign at output size 14 over the final boxes, a mask sub-graph, and
host-side mask pasting.

```
 final boxes -> roi_align(out=14) -> SubC (mask head FCN) -> paste_mask -> instance masks
```

```
run_maskrcnn <SubA.gguf> <SubB.gguf> <SubC.gguf> <mask.json> <input.bin> <out_prefix> [size=800]
```

> ggml's `conv_transpose_2d_p0` does not support batching, so `run_maskrcnn` evaluates the mask
> sub-graph one RoI at a time. Running it batched leaves only the first RoI correct.

## Tracking

Tracking is state management, not a network — there is nothing to compile. `ByteTracker` in
`src/visp/tracker.h` holds track state across frames and is detector-agnostic: it consumes
`std::vector<detection>` from any of the decoders.

## Isolation harnesses

Each takes one stage and compares it against a dump from the reference implementation, which
locates a mismatch to a single stage rather than to the pipeline as a whole.

| Harness | Stage |
| :--- | :--- |
| `run_vfnet_head` | A dense head in isolation, from FPN features to raw cls/box |
| `run_rpn_verify` | RPN proposal generation |
| `run_roi_verify` | RoIAlign |
| `run_bytetrack_verify` | Frame-to-frame association |
| `run_dump` | Any generated graph — every output tensor as raw `float32` |
