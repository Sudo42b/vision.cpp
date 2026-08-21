# Using the command line

`vision-cli` runs every built-in model without writing any code. All you need is the executable
and a `.gguf` weights file.

If you have not run anything yet, start with [Getting started](getting-started.md).

## The shape of a command

```sh
vision-cli <command> -m <weights.gguf> -i <input> -o <output>
```

The command selects the model, `-m` says which weights to load, `-i` and `-o` are files.

| Command | Task | Input | Output |
| :--- | :--- | :--- | :--- |
| `birefnet` | Background removal | image | mask |
| `sam` | Segment one object you point at | image + prompt | mask |
| `depthany` | Depth estimation | image | depth map |
| `migan` | Inpainting — fill a region | image + mask | image |
| `esrgan` | Upscaling | image | larger image |

## Options

`-m, --model <file>`
:   The `.gguf` weights. Omit it and each command looks for its own default name —
    `MobileSAM-F16.gguf`, `BiRefNet-lite-F16.gguf`, and so on — under `models/`,
    `$VISION_MODEL_DIR`, `$XDG_DATA_HOME/visioncpp`, `~/.local/share/visioncpp` and the
    install directory, in that order.

`-i, --input <image> [<image> ...]`
:   Input image. `migan` takes two — the image and the mask.

`-o, --output <file>`
:   Output file. Defaults to `output.png`. Images are always written as **PNG**, whatever
    the name says — `-o out.jpg` produces a PNG file called `out.jpg`, which some viewers
    refuse to open. Give it a `.png` name.

`-p, --prompt <x> [<y> ...]`
:   Prompt for models that take one. `sam` accepts a point (`x y`) or a box
    (`x1 y1 x2 y2`) in pixels, origin top-left.

`-b, --backend <cpu|gpu>`
:   Which device to run on. Defaults to automatic — GPU if the build has Vulkan and a device is
    available, CPU otherwise.

`--composite <file>`
:   Also write the input image combined with the resulting mask, instead of the mask alone.

`--tile <size>`
:   Split large inputs into tiles of this size. Used by `esrgan` to keep memory bounded.

`-h, --help`
:   Print the command list and exit.

## Getting weights

Each model has its own GGUF repository. Download the file and pass it with `-m`.

| Model | Weights |
| :--- | :--- |
| MobileSAM | [Acly/MobileSAM-GGUF](https://huggingface.co/Acly/MobileSAM-GGUF) |
| BiRefNet | [Acly/BiRefNet-GGUF](https://huggingface.co/Acly/BiRefNet-GGUF) |
| Depth-Anything V2 | [Acly/Depth-Anything-V2-GGUF](https://huggingface.co/Acly/Depth-Anything-V2-GGUF) |
| MI-GAN | [Acly/MIGAN-GGUF](https://huggingface.co/Acly/MIGAN-GGUF) |
| Real-ESRGAN | [Acly/Real-ESRGAN-GGUF](https://huggingface.co/Acly/Real-ESRGAN-GGUF) |

Several variants are usually available per model — different sizes or resolutions. The
executable reads which one it got from the file's metadata, so no extra flag is needed.

## Remove a background

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i photo.jpg -o mask.png --composite cutout.png
```

`mask.png` is white where the subject is. `cutout.png` is the photo with the background removed.

## Segment one object

Unlike background removal, this needs to be told which object. Give a point inside it:

```sh
vision-cli sam -m MobileSAM-F16.gguf -i photo.jpg -p 300 200 -o mask.png
```

or a box around it:

```sh
vision-cli sam -m MobileSAM-F16.gguf -i photo.jpg -p 420 120 650 430 -o mask.png
```

A box is usually more reliable when the object touches others.

## Estimate depth

```sh
vision-cli depthany -m Depth-Anything-V2-Small-F16.gguf -i photo.jpg -o depth.png
```

The output is a single-channel image — bright is near, dark is far. Values are relative to the
image, not metric distances.

## Fill a region

Inpainting takes two inputs: the image, and a mask marking what to replace.

```sh
vision-cli migan -m MIGAN-512-places2-F16.gguf -i photo.jpg mask.png -o filled.png
```

White in the mask is the region to fill. You can produce that mask with `birefnet` or `sam`,
which makes removing an object a two-step operation.

## Upscale

```sh
vision-cli esrgan -m RealESRGAN-x4plus_anime-6B-F16.gguf -i photo.jpg -o large.png
```

The scale factor comes from the weights — the model above is 4×. Large inputs are processed in
tiles; you will see them counted off, and the whole run takes considerably longer than the other
models.

## Choosing a device

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i photo.jpg -o mask.png -b gpu
```

GPU requires a build with Vulkan enabled — see [Building](../README.md#building). Without it,
`-b gpu` has nothing to select and the run stays on CPU. The first two lines of output always
name the device that was actually used.

## Using your own weights

If you have a checkpoint for an architecture the library already implements, convert it to GGUF
rather than looking for a pre-made file.

```sh
uv run scripts/convert.py <arch> MyModel.pth
```

`<arch>` is one of `sam`, `sam3`, `birefnet`, `depth-anything`, `migan`, `esrgan`.
The result lands in `models/`.

Two of those names do not carry over to the command line unchanged. `depth-anything` here is
`depthany` there — same model, two spellings. And `sam3` converts but has no `vision-cli`
subcommand yet, so the GGUF it writes can only be reached from the library API.

| Option | Description |
| :--- | :--- |
| `-o, --output` | Output directory or file. Default `models`. |
| `-q, --quantize f16` | Store float weights as f16 — roughly half the file size. |
| `-l, --layout whcn\|cwhn` | Tensor layout for 2D operations. Leave unset unless you know you need the other one. |
| `--model-name` | Name recorded in the file's metadata. |
| `-v, --verbose` | Print every tensor as it is converted. |

Conversion also rearranges and precomputes tensors, so it is not a pure format change — this is
why a checkpoint cannot be loaded directly.

This route only covers architectures that exist in the library. For anything else, see the
[model implementation guide](model-implementation-guide.md).

## Next

- [Using the library](using-the-library.md) — the same models from your own C++ or Python code.
- [Overview](overview.md) — why weights and structure are separate files.
