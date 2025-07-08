# Convert ESRGAN model checkpoint to gguf format
#

import sys
import torch
from pathlib import Path

import gguf
from spandrel import ModelLoader


def conv_2d_kernel_to_nhwc(kernel: torch.Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


def convert(
    input_filepath: Path, output_filepath: Path, quantize: str | None, verbose: bool
):
    # Load the model using spandrel
    # - it converts the various versions of ESRGAN checkpoints to a common format
    model = ModelLoader().load_from_file(input_filepath)

    if model.model.shuffle_factor is not None:
        raise ValueError("RealESRGAN models with pixel shuffle are not supported yet.")
    if getattr(model.model, "plus", False):
        raise ValueError("RealESRGAN+ (plus) models are not supported yet.")

    writer = gguf.GGUFWriter(output_filepath, "esrgan")
    writer.add_name("ESRGAN")

    for name, tensor in model.model.state_dict().items():
        if len(name) >= 64:
            print("Warning: name too long", len(name), name)

        # Conv2d: convert to NHWC format
        is_conv = (
            tensor.ndim == 4
            and tensor.shape[2] == tensor.shape[3]
            and tensor.shape[2] in (1, 3, 4, 7)
            and name.endswith("weight")
        )
        if is_conv:
            tensor = conv_2d_kernel_to_nhwc(tensor)

        if quantize == "f16" and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)

        tensor_data = tensor.numpy()
        if verbose:
            print("⇄" if is_conv else "○", name, tensor.shape, tensor_data.dtype)
        writer.add_tensor(name, tensor_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
