# Using the library

Everything `vision-cli` does is a few calls into `libvisioncpp`. This page shows the shape of
those calls so you can put the models inside your own program.

## The pattern

Every built-in model follows the same three steps: pick a device, load the weights, compute.

```c++
#include <visp/vision.h>
using namespace visp;

int main() {
    backend_device dev = backend_init();                       // 1. device
    birefnet_model model = birefnet_load_model("BiRefNet-lite-F16.gguf", dev);   // 2. weights

    image_data input = image_load("photo.jpg");
    image_data mask = birefnet_compute(model, input);          // 3. compute

    image_save(mask, "mask.png");
}
```

The structure of the network is already in the library — that is why loading takes only the
weights file. Nothing is parsed or planned at start-up beyond reading the tensors.

## Devices

```c++
backend_device dev = backend_init();                 // best available
backend_device cpu = backend_init(backend_type::cpu);
backend_device gpu = backend_init(backend_type::gpu);
```

`backend_init()` with no argument picks the GPU when the build has Vulkan and a device is
present, and falls back to CPU. One device is used for the whole model; pass it to the load
function and it decides where weights and computation live.

## The models

Each model has a `_load_model` and a `_compute`. The differences are only in what goes in and
what comes out.

| Model | Load | Compute |
| :--- | :--- | :--- |
| BiRefNet | `birefnet_load_model(path, dev)` | `birefnet_compute(m, image)` → alpha mask |
| Depth-Anything | `depthany_load_model(path, dev)` | `depthany_compute(m, image)` → depth, f32 in [0, 1] |
| MI-GAN | `migan_load_model(path, dev)` | `migan_compute(m, image, mask)` → filled image |
| ESRGAN | `esrgan_load_model(path, dev)` | `esrgan_compute(m, image)` → upscaled image |
| MobileSAM | `sam_load_model(path, dev)` | see below — two calls |

SAM is split because the expensive part does not depend on the prompt. Encode the image once,
then ask for as many objects as you like:

```c++
sam_model sam = sam_load_model("MobileSAM-F16.gguf", dev);

sam_encode(sam, image);                                  // once per image

image_data a = sam_compute(sam, i32x2{300, 200});                    // by point
image_data b = sam_compute(sam, box_2d{{420, 120}, {650, 430}});     // by box
```

Prompt coordinates are pixels with the origin in the top-left corner.

## Images

`image_data` owns its pixels; `image_view` refers to pixels someone else owns. Functions take
views, so you can pass data you already have without copying it.

```c++
image_data img = image_load("photo.jpg");     // from disk
image_save(img, "out.png");                   // to disk

image_view v{extent, image_format::rgba_u8, my_buffer};   // wrap your own memory
```

That last form is the one to reach for when frames come from a camera, a decoder, or another
part of your application — nothing needs to go through a file.

## Going lower

The one-call functions above are compositions. Each model also exposes the steps separately:
parameter detection, pre-processing, graph construction, post-processing.

```c++
birefnet_params p = birefnet_detect_params(file);   // read shape/variant from the GGUF
image_data in = birefnet_process_input(image, p);   // resize, normalise
tensor out = birefnet_predict(m, input_tensor, p);  // build the graph
image_data mask = birefnet_process_output(data, target_extent, p);
```

Use these when you need to batch work, keep tensors on the device between stages, run
pre-processing somewhere else, or share a compute graph across calls. `visp/ml.h` has the
pieces underneath — `model_load`, `model_transfer`, `compute_graph_init`, `compute`.

## Detection post-processing

If you are building a detector rather than using a built-in model, `visp/postproc.h` has the
parts that are not neural networks: anchor generation, box decoding, NMS, RoIAlign, mask
pasting. `visp/tracker.h` has ByteTrack for keeping identities across frames. Both are plain
CPU code and take structs, not framework config.

The [MMDetection guide](mmdet-detectors.md) shows them assembled into a working detector.

## Python

The bindings cover the same models for scripting and comparison work.

```python
from visioncpp import Device, Model, Backend

device = Device.init(Backend.auto)
model = Model.load("BiRefNet-lite-F16.gguf", device)
mask = model.compute(image)
```

They live in `bindings/python`. The C++ API is the reference; the bindings follow it.

## Next

- [Using the command line](using-the-cli.md) — the same models without writing code.
- [Model implementation guide](model-implementation-guide.md) — adding a model the library does
  not have.
