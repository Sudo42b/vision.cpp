# Overview

vision.cpp is a C++ library for running computer-vision neural networks. It loads weights
from a GGUF file, builds a compute graph with [ggml](https://github.com/ggml-org/ggml), and
executes it on CPU or GPU. The result is a single native binary with no Python, no framework
runtime, and no interchange-format interpreter.

It is the same idea as [llama.cpp](https://github.com/ggml-org/llama.cpp), applied to vision
models instead of language models.

## The idea: a model is code, not a file

Most inference stacks treat a model as data. You export a graph to ONNX or TorchScript, and
a general-purpose runtime loads that graph, matches its operators against a kernel library, and
interprets it. The runtime has to support every operator anyone might export, so it is large,
and it has to plan the graph at startup, so loading takes time.

vision.cpp splits the model in two:

| Part | Form | Where it lives |
| :--- | :--- | :--- |
| Structure | C++ that builds a ggml graph | compiled into your binary |
| Weights | GGUF tensors | a `.gguf` file loaded at run time |

Nothing interprets a graph description at run time, because there is no graph description — the
graph is the code you compiled. That is what makes the deployment small and start-up fast, and
it is the trade-off at the centre of the project: adding a model that isn't supported yet
means writing or generating code, not exporting a file.

Weights stay external, so swapping checkpoints, changing precision, or quantising does not
require rebuilding.

## What you get

- Self-contained. The only dependencies are ggml, `stb` for image I/O, and optionally
  `fmt`. There is no Python in the runtime path.
- CPU and GPU. CPU works everywhere; Vulkan covers NVIDIA, AMD and Intel from one build.
- Small and quick to start. Deployment size and model-load time are explicit goals of the
  project — see the [Performance](../README.md#performance) section for the current numbers.
- Modular. The same primitives the built-in models are made of are public, so you can
  assemble your own.

## How it fits together

The library is layered. Each layer is usable on its own; you can stop at whichever one matches
how much control you need.

| Layer | Header | What it gives you |
| :--- | :--- | :--- |
| Model APIs | `visp/vision.h` | Ready-made models — load, run, get a result. |
| Image I/O | `visp/image.h` | Load, save, resize, tile, convert. |
| Graph and backends | `visp/ml.h` | GGUF loading, weight transfer, graph construction, execution. |
| Vectors and small utilities | `visp/util.h` | `i32x2`, spans, and the shared scalar types. |

Those four are what installing puts under `include/visp/`, and they are the whole public
surface. Three more headers are part of the build but **not installed**, so a program compiled
against an installed SDK cannot include them. They are listed because the rest of this guide
refers to them:

| Layer | Header (in-tree only) | What it gives you |
| :--- | :--- | :--- |
| Neural network layers | `src/visp/nn.h` | `conv_2d`, `group_norm`, attention, and other building blocks. |
| Detection post-processing | `src/visp/postproc.h` | Anchors, decoding, NMS, RoIAlign, masks. |
| Tracking | `src/visp/tracker.h` | ByteTrack association across frames. |

Code that needs those is built inside the tree — which is what registering an architecture does.

Two front-ends are built on top:

- `vision-cli` — a command-line tool for the built-in models
  (`vision-cli sam -m MobileSAM-F16.gguf -i image.jpg -p 100 200 -o mask.png`).
- Python bindings — `bindings/python`, for scripting and comparison against reference
  implementations.

## Running a model

Most of the time there is nothing to add. The models in the
[README](../README.md#features) are already implemented, so running one means downloading its
weights and pointing at them:

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i photo.jpg -o mask.png
```

Or from your own program, in three calls — pick a device, load the weights, compute. See
[using the command line](using-the-cli.md) and [using the library](using-the-library.md).

If you have your own checkpoint for one of those architectures, convert it with
`scripts/convert.py`. The structure is already in the library; only the weights change.

## Adding a model

When the architecture is not implemented yet, it has to be written. The
[model implementation guide](model-implementation-guide.md) walks through it: describe the
network with the `nn.h` primitives, and provide a conversion function that turns the original
checkpoint into GGUF. Every built-in model was added this way, and it gives the most control
over layout and precision.

The result is what the library loads: a `<Arch>_forward` function that builds a graph, plus a
GGUF file of weights.

For model families with hundreds of variants, hand-writing each one is not realistic and the
C++ can be generated from a traced PyTorch module instead. The
[MMDetection guide](mmdet-detectors.md) covers that case — the interface generated code must
satisfy, and how to handle what tracing cannot capture.

## Scope

vision.cpp is an inference library. There is no training, no autograd, and no optimizer.

It is also not a general model runtime: it does not aim to execute arbitrary exported graphs.
Supported models are the ones that have been implemented or generated, which is why the model
list in the [README](../README.md#features) is finite and why growing it is a code change.

## Next

- [Getting started](getting-started.md) — run a model end to end in five minutes.
- [Using the command line](using-the-cli.md) — every built-in model, no code.
- [Using the library](using-the-library.md) — the same models from C++ or Python.
- [README](../README.md) — install, build, supported models, performance.
- [Model implementation guide](model-implementation-guide.md) — write a model by hand.
- [MMDetection detectors](mmdet-detectors.md) — run detectors from a compiled backbone.
