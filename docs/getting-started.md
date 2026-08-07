# Getting Started

In this tutorial you will cut an object out of a photo using vision.cpp. It takes about five
minutes and needs nothing but the release package, one model file and one image — no build, no
Python, no conversion.

At the end you will have this:

| | |
| :--- | :--- |
| `mask.png` | a black-and-white mask of the object |
| `object.png` | the original photo with the background dimmed away |

## Step 1 — Get the executable

Download a [release package](https://github.com/Acly/vision.cpp/releases) and extract it. You
will find `vision-cli` in the `bin` folder.

Check that it runs:

```sh
vision-cli --help
```

You should see a list of commands: `sam`, `birefnet`, `depthany`, `migan`, `esrgan`.

> If you would rather build from source, follow [Building](../README.md#building) first, then
> come back here. `vision-cli` ends up in `build/bin`.

## Step 2 — Get a model and an image

The executable contains the network structure, but not the weights. Download them:

```sh
curl -L -O https://huggingface.co/Acly/BiRefNet-GGUF/resolve/main/BiRefNet-lite-F16.gguf
```

This is BiRefNet, a model that separates a subject from its background. The file is a
[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — the weights and nothing else.

For the input, use any photo with a clear subject. If you cloned the repository, there is one
at `docs/media/input.jpg`. Put it next to the model file and call it `input.jpg`.

## Step 3 — Run it

```sh
vision-cli birefnet -m BiRefNet-lite-F16.gguf -i input.jpg -o mask.png --composite object.png
```

The output tells you what it is doing:

```
Initializing backend... done (1.1 ms)
- device: CPU - Intel(R) Core(TM) i3-14100
Loading model weights from 'BiRefNet-lite-F16.gguf'... done (151.3 ms)
- float type: f16
- tensor layout: cwhn
- model image size: 1024
- inference image size: 1024x1024
- flash attention: off
Running inference... complete (5372.6 ms)
-> mask saved to mask.png
-> image composited and saved to object.png
```

Inference takes a few seconds on a desktop CPU. Loading the weights takes a fraction of a
second — that number is the point of the project, and it is the same on any machine.

## Step 4 — Look at the result

Open `object.png`. The subject is untouched and the background has faded away.

`mask.png` is what the model actually produced: white where the subject is, black elsewhere.
Everything in `object.png` was computed from it.

That is the whole loop. An executable that already knows the network, a `.gguf` that carries
the weights, an image in, a result out.

## Try one more

The same executable runs the other built-in models. Only the command and the weights change:

```sh
curl -L -O https://huggingface.co/Acly/Real-ESRGAN-GGUF/resolve/main/RealESRGAN-x4plus_anime-6B-F16.gguf

vision-cli esrgan -m RealESRGAN-x4plus_anime-6B-F16.gguf -i input.jpg -o upscaled.png
```

This one upscales the image four times. It works on tiles and takes noticeably longer — you will
see it count them off.

## Where to go next

- [Overview](overview.md) — what the library is and why weights and structure are separate.
- [Using the command line](using-the-cli.md) — every option, every built-in model.
- [Using the library](using-the-library.md) — the same models from your own code.
- [README](../README.md#features) — the other built-in models, and what each one does.
- [Model implementation guide](model-implementation-guide.md) — when the model you want is not
  in the list, and you want to add it.
- [MMDetection detectors](mmdet-detectors.md) — running detectors whose structure is generated
  rather than hand-written.
