# Model Implementation Guide

This describes my way of implementing new model architectures in C++/ggml from a
PyTorch reference. It also contains brief introductions to some concepts in
vision.cpp and ggml.

* **Prerequisites:** Proficient in C++ & Python, basic knowledge of ML / neural networks
* **Time investment:** Really depends on model complexity and experience. Can be
  few hours. Can be several weeks for implementing kernels, profiling and
  optimization.

## Overview

1. Inspect the model architecture and weights
2. Write a script that converts the weights to GGUF format
4. Implement the compute graph
   * Copy a module/layer from the reference into a Python test file
   * Implement the `forward` function in C++ and expose it
   * Run the reference and C++ implementation on dummy data from Python and compare
   * Repeat until everything is implemented (and tested)
5. Implement pre-/post-processing steps in C++
6. Add the model to the CLI
7. Add the model to the API
8. Add the model to `test-models`

This might sound like a lot, but most of the steps are pretty straight-forward.
The process has a pretty good chance to result in something that works at the
end. And _that_ saves a lot of time, because finding small bugs hiding somewhere
while debugging the model in its entirety is painful. For small models someone
with experience using ggml can probably get away with a less diligent method.

## Introduction

### ggml

[ggml](https://github.com/ggml-org/ggml) is conceptually simple, but it's fairly
low-level and has some peculiarities. It's worth reading the
[introduction](https://huggingface.co/blog/introduction-to-ggml) if you haven't
used it before. Checking out the
[examples](https://github.com/ggml-org/ggml/tree/master/examples) also helps.

Perhaps most strikingly, it enumerates tensor dimensions from most contiguous in
memory to least. That is `[rows, columns, matrices, batches]`, which is in
reverse order compared to PyTorch and most other frameworks. _vision.cpp_
follows the ggml convention in all C++ files. Python files use PyTorch
convention.

Permutation (`ggml_permute`) also works differently and can't be copied 1:1 from
PyTorch code.

The great thing about ggml is, you can always follow-reference in your IDE and
see almost immediately how things are implemented. It is small enough to be
compiled along-side, so you can step into functions, add prints, etc. If some
functionality is missing, you can quickly hack it in. Make sure to use.

### vision.cpp

_vision.cpp_ adds some infrastructure on top of ggml to reduce boilerplate for
common tasks. It's designed to amend functionality, not wrap or replace it. The
[include/visp/ml.hpp](/include/visp/ml.hpp) public header contains all the
interesting bits.

If you take a look at the existing model implementations in `src/visp/arch`, you
will notice they are all built from functions like this:
```c++
tensor some_module(model_ref m, tensor x, ...)
```
Here `tensor` is short for `ggml_tensor *`, which can be a weight or the result
of an operation. The `model_ref` is used to build a compute graph by passing it
to ggml functions as replacement for `ggml_context *`. It keeps track of parent
modules and provides a way to access model weights. 

`some_module` typically represents the forward function of a PyTorch
`nn.Module`. The whole model can be defined with reusable functions.

## Guide

### 1. Analysis

I usually start by taking a closer look at the model. There's various ways to do
that (including tools and visualizers for this purpose), but I tend to just run
the model from a Python Notebook.

* Printing the top-level `nn.Module` gives a good summary (but misses pure
  function calls). 
* Dumping the state dict gives some idea about the weights that need to be
  converted.
* Reading the source code can be useful, but it depends. Researchy code is
  actually often better in this case, while it can be messy, the low level of
  abstraction helps. Production code tends to have more layers and dependencies.
  Here it can help to convert to an inference format like onnx and inspect that
  instead.
* Being able to run the code allows to add some prints when it's unclear which
  modules are actually executed, which ones are only relevant for training,
  typical parameters passed in, etc.

The most important part is to identify _weird custom_ stuff that might be going
on. Lookup and index tables being constructed from convoluted PyTorch calls.
Niche operations that aren't supported by ggml and will have to be implemented.
Tensors with more than 4 dimensions can be an issue (maximum in ggml is 4). This
is the stuff that will decide if it's an afternoon's work, or 3 weeks.

### 2. Weights Conversion

The library reads `.gguf` files, which is ggml's binary format to store tensors.
There's currently no way to load other formats directly, so the weights have to
be converted. It's usually a good opportunity to optimize for inference, throw
away training-only stuff, maybe fuse some operations, or convert to a faster
memory layout.

If you haven't already, setup a Python environment (just running `uv sync` will do).

Open `scripts/convert.py` and add a conversion function similar to the existing
ones. A 1:1 conversion is very simple:
```
model = torch.load(input_filepath, weights_only=True)

for name, tensor in model.state_dict().items():
    writer.add_tensor(name, tensor)
```

You might have to shorten weight names to fit the 64-characters limit.

### 3. The Compute Graph

This is the part where PyTorch modules (or equivalent) are translated to ggml
function calls which build up an execution graph. You can just wing it, but in
my experience going step-by-step means less headache later. So here's an example
to demonstrate the workflow.

### Reference network

```py
class Piong(nn.Module):
    def __init__(self, feat, hidden):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat)
        self.linear1 = nn.Linear(feat, hidden)
        self.linear2 = nn.Linear(hidden, feat)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.linear2(x)
        return x
```
Let's say you encounter this super revolutionary module, it could be the whole
model or a building block. There are 3 operations to implement here:
`LayerNorm`, `Linear` and `Piong`. The `gelu` is a basic operation in ggml,
`LayerNorm` and `Linear` are not.

### Testing

Starting with `LayerNorm`, the first thing to do is to pull it into a simple
test:
```py
def test_layer_norm():
    norm = nn.LayerNorm(4)
    state = workbench.randomize(norm.state_dict())
    norm.load_state_dict(state)
    norm.eval()

    x = torch.rand(1, 6, 4)
    expected = norm(x)

    result = workbench.invoke_test("piong_layer_norm", x, state)
    assert torch.allclose(result, expected)
```

This performs layer-norm on some tiny random input. Then it uses `workbench` to
execute my C++ implementation (which doesn't exist yet) and compares the result.
If there's a bug it's easy to spot because tensors are small, numbers simple,
and complexity low.

This is how you run it. It will print "Error: Test case not found:
piong_layer_norm".
```sh
uv run pytest tests/test_piong.py
```

### C++ Implementation

I'd usually put something as basic as layer-norm in
[src/visp/nn.cpp](/src/visp/nn.cpp). And in fact, it's already there. But for
this example, let's pretend it's a more model-specific operation, and put it
into a new file [src/visp/arch/piong.cpp]().

```c++
tensor layer_norm(model_ref m, tensor x, float eps = 1e-5) {
    x = ggml_norm(m, x, eps);
    x = ggml_mul_inplace(m, x, m.weights("weight"));
    x = ggml_add_inplace(m, x, m.weights("bias"));
    return x;
}
```

The [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#layernorm)
accurately describes what layer-norm does. Checking existing ggml operations in
`ggml.h` I find the norm part is already covered by `ggml_norm`. The weight
names come from the "Variables" section of the docs, but you can also see them
by printing the state-dict in the test.

### Workbench

To make the test work, we're missing some glue. It's tempting to export and use
the function directly, but having a separate "invoker" function has proven to be
more flexible. So I go to [tests/workbench.cpp](/tests/workbench.cpp) and add a
little bit of boilerplate:

```c++
DEF(piong_layer_norm)(model_ref m, span<tensor> input, param_dict const& p) {
    return {layer_norm(m, input[0])};
}
```

Notes:
* I also added `layer_norm` to a `piong.hpp` header file and included it in `workbench.cpp`
* The identifier inside `DEF()` becomes the test case name
* `m` already holds the state-dict passed in on python side
* You can have multiple input and output tensors
* `p` allows to pass arbitrary int/float/strings for different scenarios

I now compile and run the test again, and it passes, there is some green text,
and a modicum of happiness is injected into the brain. Should it fail however,
it's very easy to inspect, iterate, fix, and feel confident about the next steps
which build on top. (I'm not someone to preach TDD, but sometimes it's nice.)

### Repeat

Now I do the same for `Linear` (except it also already exists in `nn.cpp`) and
then the `Piong` operation itself:

```py
def test_piong():
    # ... can hit Tab here and let LLM do the rest
    piong = Piong(4, 8)
    state = workbench.randomize(piong.state_dict())
    piong.load_state_dict(state)
    piong.eval()

    x = torch.rand(1, 6, 4)
    expected = piong(x)

    result = workbench.invoke_test("piong_piong", x, state)
    assert torch.allclose(result, expected)
```
```c++
tensor piong(model_ref m, tensor x) {    // # from above's forward(x):
    x = layer_norm(m["input_norm"], x);  // x = self.input_norm(x)
    x = linear(m["linear1"], x);         // x = self.linear1(x)
    x = ggml_gelu(m, x);                 // x = nn.functional.gelu(x)
    x = linear(m["linear2"], x);         // x = self.linear2(x)
    return x;
}
```
The `m["module_name"]` part takes care of chaining weight names, so the C++
functions are reusable and end up looking just like the forward functions in
Python. Neat.

And now I've implemented the illustrious "Piong" model and it's tested and works
(probably).

### Extensions

Since this is a dumb example, the whole process is somewhat overkill. But it's
actually not a lot of overhead, for simple cases the repetitive stuff can be
auto-completed, and for more complex cases the process usually pays off
immediately.

Some examples where this helped:
* the reference code is confusing and I'm 99% sure my implementation is probably
  wrong
* there is no 1:1 correspondence of operations and it needs an alternative
  solution
* I want to change memory layout, or fuse some weights, or try quantization, or
  other inference optimizations
* there are some custom buffers that need to be pre-computed
* the operation doesn't exist in ggml and I need to write a kernel from scratch

### Common Problems

1. **Precision** - If results don't match exactly, it might be necessary to pass
   a higher tolerance threshold to `torch.allclose`. Low-level operations on
   small tensors should match very closely. `gelu` is a common exception, since
   by default ggml uses a 16-bit look-up table. It can make sense to disable
   `GGML_GELU_FP16` during development to get more accurate results.
2. **Input numbers** - Random input can result in extreme numbers that are
   difficult to compare. Using a fixed range can make it easier, or
   scaling/changing the distribution to include negative numbers.
3. **Complexity** - For the most top-level operation it's sometimes not practical
   to come up with dummy input, especially if it includes downsample/upsample steps
   which require large input tensors. So just skip the test and hope for the
   best :)

## 4. Pre- and Postprocessing

It's common for vision models to process images and masks with a wild mix of
PIL/numpy/OpenCV/torchvision/whatever. The
[include/visp/image.hpp](/include/visp/image.hpp) header has a collection of
common transformations. If that doesn't cover it, it also has some tools to
implement custom per-pixel operations.

It usually boils down to writing some nested for-loops. The fun part is figuring
out what the for-loop should be when the reference is implemented as some insane
chain of numpy-arange-permuted-linspace-meshgrid-slicing gymnastics. Good way to
keep those brain cells from falling asleep. (Or you could try feeding it to an
LLM. Good luck.)

I put pre/post-processing into the same file as the model compute graph and make
them public functions. This allows users to plug in their own, or supply data
that's already been processed, and reuse the compute graph only.

### Parameter Detection

By convention, all model architectures (like `piong`) define a struct with
parameters (`piong_params`). This stores hyper-parameters which can influence
the compute graph, such as input resolution or encoder size. If the compute
graph supports multiple variants of a model, it should also provide a function
`piong_detect_params` to derive those parameters from the loaded weights.

### Pre-computed Buffers

Some models pre-compute tensors as `nn.parameter.Buffer` or similar. They can be quite
large, or depend on input parameters like resolution, which makes it impractical
to store them with the weights. This has to be migrated to C++ also.

```c++
std::array<tensor_data> piong_precompute(model_ref m, piong_params const&);
```
A function like this can allocate and pre-compute the tensors, and the caller
can take care to transfer them to the backend device. See `birefnet_precompute`
for an example.

## 5. CLI

The CLI is a nice way to try out a model on some real input. It's commonly
invoked like this:
```sh
vision-cli <arch> -m <model-file> -i <input1> [<input2>...] -o <output>
```
Adding a new model arch in [src/cli/cli.cpp](/src/cli/cli.cpp) is pretty
straight-forward by following one of the existing implementations. It usually
includes some practical post-processing too.

## 6. API

Models are exported in [include/visp/vision.hpp](/include/visp/vision.hpp). This
includes a high-level API which represents the most common use cases. It should
be simple, and does not need to support configuration options. Typically that
means a function to load the model, and one to run inference. These are
implemented in [src/visp/vision.cpp](/src/visp/vision.cpp).

Below there is space for a more modular API, which directly exports the
functions specific to the model: parameter detection, pre-/post processing, and
graph building.

## 7. Test

Finally, it is good to have a test that actually runs the entire model on some
sensible input (an image!) and spits out something nice to look at and go "yep,
it works". This is what [tests/test-models.cpp](/tests/test-models.cpp) is for.
With all the previous work, those tests are really simple to implement: load an
image, call the high level API, compare the result to a reference and store it.

_Note on reference images:_ Those aren't checked into the repository to avoid
bloat. GitHub's LFS support is kinda bullshit, so it currently involves me
invoking a script to upload new images. If you're making a PR, just leave them
out.

## Afterword

This is my process, it works for me. I don't expect anyone to follow it by
heart. All contributions are welcome as long as the results are good!

I don't actually value all the Python tests much as an end result, and may
decide to scrap them if they increase maintenance burden. For now they're
included in the repository, but their main purpose is to make the implementation
faster and finding bugs less painful. And increasing the chance that things just
work!
