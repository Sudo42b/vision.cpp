#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "torchvision",
#   "timm",
#   "pytest",
#   "opencv-python",
#   "ruff",
#   "einops>=0.8.1",
#   "spandrel>=0.4.1",
#   "gguf>=0.17.1",
# ]
# ///
import argparse
import itertools
import torch
import safetensors
import numpy as np

from pathlib import Path
from gguf import GGUFWriter, Metadata, GGML_QUANT_VERSION
from torch import Tensor

#
# Common

batch_norm_eps = 1e-5


def is_conv_2d(name: str, tensor: Tensor):
    return (
        tensor.ndim == 4
        and tensor.shape[2] == tensor.shape[3]
        and tensor.shape[2] in (1, 3, 4, 7)
        and name.endswith("weight")
    )


def conv_2d_to_nhwc(kernel: Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


def conv_transpose_2d_to_nhwc(kernel: Tensor):
    # C_in C_out H W -> C_out H W C_in
    return kernel.permute(1, 2, 3, 0)


def add_tensor(
    writer: GGUFWriter, name: str, tensor: Tensor, quantize: str, verbose: bool
):
    if len(name) >= 64:
        print("Warning: name too long", len(name), name)

    if quantize == "f16" and tensor.dtype == torch.float32:
        tensor = tensor.to(torch.float16)

    tensor_data = tensor.numpy()
    if verbose:
        print(name, tensor.shape, tensor_data.dtype)
    writer.add_tensor(name, tensor_data)


#
# MobileSAM


def convert_sam(
    input_filepath: Path, writer: GGUFWriter, quantize: str | None, verbose: bool
):
    assert quantize is None, "MobileSAM does not support quantization"

    model: dict[str, Tensor] = torch.load(
        input_filepath, map_location="cpu", weights_only=True
    )

    for name, tensor in model.items():
        name = name.replace("image_encoder.", "enc.")
        name = name.replace("mask_decoder.", "dec.")
        name = name.replace("_image_to_token.", "_i2t.")
        name = name.replace("_token_to_image.", "_t2i.")

        if name.endswith("attention_biases"):
            num_heads = tensor.shape[0]
            resolution = {4: 7, 5: 14, 10: 7}[num_heads]
            attention_bias_idxs = build_attention_bias_indices(resolution)
            name = name + "_indexed"
            tensor = tensor[:, attention_bias_idxs]

        if name.endswith("running_var"):
            tensor = torch.sqrt(tensor + batch_norm_eps)

        if (
            name.endswith("c.weight")
            or name.endswith("neck.0.weight")
            or name.endswith("neck.2.weight")
        ):
            assert tensor.shape[2] == tensor.shape[3] and tensor.shape[2] <= 3
            tensor = conv_2d_to_nhwc(tensor)

        # Precompute dense positional embeddings from random matrix stored in the model
        if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            pe = build_dense_positional_embeddings(tensor)
            writer.add_tensor("dec.dense_positional_embedding", pe.numpy())

        add_tensor(writer, name, tensor, quantize, verbose)


def build_attention_bias_indices(resolution: int):
    points = list(itertools.product(range(resolution), range(resolution)))
    N = len(points)
    attention_offsets = {}
    idxs = []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            idxs.append(attention_offsets[offset])

    return torch.LongTensor(idxs).view(N, N)


def build_dense_positional_embeddings(
    positional_encoding_gaussian_matrix: torch.Tensor, image_embedding_size=64
):
    # from sam/modeling/prompt_encoder.py - PositionEmbeddingRandom
    h, w = image_embedding_size, image_embedding_size
    grid = torch.ones((h, w), dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    coords = torch.stack((x_embed, y_embed), dim=-1)
    coords = 2 * coords - 1
    coords = coords @ positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    return pe


#
# BirefNet


def convert_birefnet(
    input_filepath: Path, writer: GGUFWriter, quantize: str | None, verbose: bool
):
    model: dict[str, Tensor] = safetensors.safe_open(input_filepath, "pt")

    for name in model.keys():
        tensor = model.get_tensor(name)

        # Shorten some names to fit into 64 chars
        name = name.replace("decoder_block", "block")
        name = name.replace("atrous_conv", "conv")
        name = name.replace("modulator_conv", "modulator")
        name = name.replace("offset_conv", "offset")
        name = name.replace("regular_conv", "conv")

        if name.endswith("relative_position_index"):
            continue  # precomputed in c++ code

        # BatchNorm2d: precompute sqrt(var + eps)
        if name.endswith("running_var"):
            tensor = torch.sqrt(tensor + batch_norm_eps)

        if is_conv_2d(name, tensor):
            tensor = conv_2d_to_nhwc(tensor)

        add_tensor(writer, name, tensor, quantize, verbose)


#
# MI-GAN


def convert_migan(
    input_filepath: Path, writer: GGUFWriter, quantize: str | None, verbose: bool
):
    model: dict[str, Tensor] = torch.load(input_filepath, weights_only=True)

    for name, tensor in model.items():
        if is_conv_2d(name, tensor):
            tensor = conv_2d_to_nhwc(tensor)

        add_tensor(writer, name, tensor, quantize, verbose)


#
# ESRGAN


def convert_esrgan(
    input_filepath: Path, writer: GGUFWriter, quantize: str | None, verbose: bool
):
    from spandrel import ModelLoader

    # Load the model using spandrel
    # - it converts the various versions of ESRGAN checkpoints to a common format
    model = ModelLoader().load_from_file(input_filepath)

    if model.model.shuffle_factor is not None:
        raise ValueError("RealESRGAN models with pixel shuffle are not supported yet.")
    if getattr(model.model, "plus", False):
        raise ValueError("RealESRGAN+ (plus) models are not supported yet.")

    for name, tensor in model.model.state_dict().items():
        if is_conv_2d(name, tensor):
            tensor = conv_2d_to_nhwc(tensor)

        add_tensor(writer, name, tensor, quantize, verbose)


#
# Main
#######

arch_names = {
    "sam": "mobile-sam",
    "birefnet": "birefnet",
    "migan": "migan",
    "esrgan": "esrgan",
}

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert model weights (.pt/.pth/.safetensors) to GGUF format.")
    parser.add_argument("arch", choices=["sam", "birefnet", "migan", "esrgan"], help="Model architecture")
    parser.add_argument("input", type=str, help="Path to the input model file")
    parser.add_argument("--output", "-o", type=str, default="models", help="Path to the output directory or file")
    parser.add_argument("--quantize", "-q", choices=["f16"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for metadata")
    parser.add_argument("--metadata", type=Path, help="Specify the path for an authorship metadata override file")
    # fmt: on
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    quant_suffix = f"-{args.quantize.upper()}" if args.quantize else ""
    if output_path.is_dir() or output_path.suffix != ".gguf":
        output_path = output_path / f"{input_path.stem}{quant_suffix}.gguf"

    print(f"Converting {args.arch}")
    print("* input: ", input_path)
    print("* output:", output_path)

    try:
        writer = GGUFWriter(output_path, arch_names.get(args.arch, args.arch))
        metadata = Metadata.load(args.metadata, input_path.with_suffix(""), args.model_name)

        match args.arch:
            case "sam":
                convert_sam(input_path, writer, args.quantize, args.verbose)
            case "birefnet":
                convert_birefnet(input_path, writer, args.quantize, args.verbose)
            case "migan":
                convert_migan(input_path, writer, args.quantize, args.verbose)
            case "esrgan":
                convert_esrgan(input_path, writer, args.quantize, args.verbose)
            case _:
                raise ValueError(f"Unknown architecture: {args.arch}")

        metadata.set_gguf_meta_model(writer)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    except ValueError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        exit(-1)

    print("")
