# Convert MobileSAM (Tiny-ViT) model checkpoint to gguf format
#

import itertools
import torch
import numpy as np
from pathlib import Path

import gguf


def convert(
    input_filepath: Path, output_filepath: Path, quantize: str | None, verbose: bool
):
    assert quantize is None, "MobileSAM does not support quantization"

    model: dict[str, torch.Tensor] = torch.load(
        input_filepath, map_location="cpu", weights_only=True
    )

    writer = gguf.GGUFWriter(output_filepath, "sam")
    writer.add_name("MobileSAM")

    batch_norm_eps = 1e-5

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
            tensor = conv_2d_kernel_to_nhwc(tensor)

        # Precompute dense positional embeddings from random matrix stored in the model
        if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            pe = build_dense_positional_embeddings(tensor)
            writer.add_tensor("dec.dense_positional_embedding", pe.numpy())

        tensor_data = tensor.numpy()
        if verbose:
            print(name, tensor.shape, tensor_data.dtype)
        writer.add_tensor(name, tensor_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


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


def conv_2d_kernel_to_nhwc(kernel: torch.Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


def conv_transpose_2d_kernel_to_nhwc(kernel: torch.Tensor):
    # C_in C_out H W -> C_out H W C_in
    return kernel.permute(1, 2, 3, 0)
