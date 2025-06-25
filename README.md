# _vision_.cpp

Computer Vision ML inference in C++

* Self-contained C++ library
* Efficient inference on consumer CPU and GPUs
* Lightweight deployment on many platforms (Windows, Linux,)
* Simple modular design for experimentation and extension

Follows the spirit of the popular [llama.cpp]() project and is based on the same [infrastructure]().

### Features

| Model         | Task             | Backends    |
| :------------ | :--------------- | :---------- |
| [MobileSAM]() | Segmentation     | CPU, Vulkan |
| [BiRefNet]()  | Segmentation     | CPU, Vulkan |
| [MI-GAN]()    | Inpainting       | CPU, Vulkan |
| [ESRGAN]()    | Super-resolution | CPU, Vulkan |

## Get Started

See [Building]() to build from source.

### Example: Select an object in an image

Let's use MobileSAM to generate a segmentation mask for the <object>
at pixel position (320, 240).

First, download the model from [huggingface]().

#### CLI

```
vision-cli -m MobileSAM-F16.gguf -i input.png -p "[320, 240]" -o mask.png
```

#### API

```c++
#include <visp/vision.hpp>
using namespace visp;

void main() {
  backend cpu     = init_backend(backend_type::cpu);
  sam_model model = load_sam_model(cpu, "MobileSAM-F16.gguf");
  
  image_data input     = load_image("input.png");
  sam_embedding embeds = embed(cpu, model, input);
  image_data mask      = predict(cpu, model, embeds, point{320, 240});
  save_image(mask, "mask.png");
}
```
There is also a low-level API with a high degree of flexibility and control over
allocations and performance.

> ![WARNING]
> The API is not considered stable at the moment and may change without notice.

#### UI


## Models



### Converting models


## Building

Building requires CMake and a compiler with C++20 support.

**Get the sources**
```
git clone --recursive
cd vision.cpp
```

**Configure and build**
```
cmake . -B build
cmake --build build --config Release
```

### Vulkan

Vulkan GPU support requires the [Vulkan SDK]() to be installed.

```
cmake . -B build -DVISP_VULKAN=ON
cmake --build build --config Release
```

### Tests

Run all tests with the following command:
```
ctest build -C Release
```

Some tests require a Python environment. It can be set up with [uv]():
```
# Setup venv and install dependencies
uv sync

# Run only python tests
uv run pytest
```