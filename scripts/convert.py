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

from enum import Enum
from pathlib import Path
from gguf import GGUFWriter, Metadata, GGML_QUANT_VERSION
from torch import Tensor

#
# Common


class TensorLayout(Enum):
    unknown = "unknown"
    nchw = "whcn"
    nhwc = "cwhn"

    @staticmethod
    def parse(s: str):
        if s == "whcn" or s == "nchw":
            return TensorLayout.nchw
        if s == "cwhn" or s == "nhwc":
            return TensorLayout.nhwc
        return TensorLayout.unknown


class Writer(GGUFWriter):
    def __init__(self, path: Path, arch_name: str, float_type: str, verbose: bool):
        super().__init__(path, arch_name)
        self.arch = arch_name
        self.float_type = float_type
        self.tensor_layout = TensorLayout.unknown
        self.verbose = verbose
        self.conv2d_weights: list[int] = []
        self._index = 0

    def add_tensor(self, name: str, tensor: Tensor, float_type: str | None = None):
        if len(name) >= 64:
            print("Warning: name too long", len(name), name)

        float_type = float_type or self.float_type
        if float_type == "f16" and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)

        tensor_data = tensor.numpy()
        if self.verbose:
            print(name, tensor.shape, tensor_data.dtype)
        super().add_tensor(name, tensor_data)
        self._index += 1

    def convert_tensor_2d(self, tensor: Tensor):
        # assume tensor is NCHW layout (PyTorch default)
        if self.tensor_layout is TensorLayout.nhwc:
            return conv_2d_to_nhwc(tensor)
        else:
            # add tensor index to list to optionally convert layout on the fly later
            self.conv2d_weights.append(self._index)
            return tensor

    def add_int32(self, name: str, value: int):
        print("*", name, "=", value)
        super().add_int32(name, value)

    def set_tensor_layout(self, layout: TensorLayout):
        print("*", f"{self.arch}.tensor_data_layout", "=", layout.value)
        self.tensor_layout = layout
        self.add_tensor_data_layout(layout.value)

    def set_tensor_layout_default(self, layout: TensorLayout):
        if self.tensor_layout is TensorLayout.unknown:
            self.set_tensor_layout(layout)

    def add_conv2d_weight_indices(self):
        if self.conv2d_weights:
            self.add_array(f"{self.arch}.conv2d_weights", self.conv2d_weights)


def load_model(path: Path) -> dict[str, Tensor]:
    if path.suffix in [".safetensors", ".safetensor"]:
        weights = safetensors.safe_open(path, "pt")
        return {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        return torch.load(path, map_location="cpu", weights_only=True)


batch_norm_eps = 1e-5


def is_conv_2d(name: str, tensor: Tensor):
    return (
        tensor.ndim == 4
        and tensor.shape[2] == tensor.shape[3]
        and tensor.shape[2] in (1, 3, 4, 7, 14)
        and name.endswith("weight")
    )


def conv_2d_to_nhwc(kernel: Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0).contiguous()  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1).contiguous()  # C_out, C_in, Height, Width -> C_out H W C_in # c++ ggml에서 보이는 차원은, C_in, W, H, C_out


def conv_transpose_2d_to_nhwc(kernel: Tensor):
    # C_in C_out H W -> C_out H W C_in
    return kernel.permute(1, 2, 3, 0)


def fuse_batch_norm(model: dict[str, Tensor], key: str, key_bn: str):
    suffix_weight = f"{key_bn}.weight"
    suffix_bias = f"{key_bn}.bias"

    if key.endswith(suffix_weight):
        base = key.removesuffix(suffix_weight)
        weight = model[key]
        var = model[f"{base}{key_bn}.running_var"]
        return weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        bias = model[key]
        weight = model[f"{base}{key_bn}.weight"]
        mean = model[f"{base}{key_bn}.running_mean"]
        var = model[f"{base}{key_bn}.running_var"]
        return bias - mean * weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(f"{key_bn}.running_mean") or key.endswith(f"{key_bn}.running_var"):
        return None

    return model[key]


def fuse_conv_2d_batch_norm(
    model: dict[str, Tensor],
    key: str,
    name: str,
    key_module: str,
    key_conv: str,
    key_norm: str,
    writer: Writer,
):
    suffix_conv = f"{key_module}{key_conv}.weight"
    suffix_bias = f"{key_module}{key_conv}.bias"
    suffix_norm = f"{key_module}{key_norm}."

    if key.endswith(suffix_conv):
        conv_weight = model[key]
        base = key.removesuffix(suffix_conv)
        bn_weight = model.get(f"{base}{suffix_norm}weight")
        if bn_weight is None:
            return False
        bn_bias = model[f"{base}{suffix_norm}bias"]
        bn_mean = model[f"{base}{suffix_norm}running_mean"]
        bn_var = model[f"{base}{suffix_norm}running_var"]
        conv_bias = model.get(f"{base}{suffix_bias}", torch.zeros_like(bn_bias))

        bn_weight = bn_weight / torch.sqrt(bn_var + batch_norm_eps)
        fused_weight = conv_weight * bn_weight[:, None, None, None]
        fused_bias = (conv_bias - bn_mean) * bn_weight + bn_bias

        fused_weight = writer.convert_tensor_2d(fused_weight)
        writer.add_tensor(name, fused_weight)
        writer.add_tensor(name.replace("weight", "bias"), fused_bias)
        return True

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        return f"{base}{suffix_norm}weight" in model

    elif suffix_norm in key:
        return True  # batch norm was fused above

    return False  # tensor is not part of conv2d+batch-norm


#
# MobileSAM


def convert_sam(input_filepath: Path, writer: Writer):
    writer.add_license("apache-2.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = torch.load(input_filepath, map_location="cpu", weights_only=True)

    for key, tensor in model.items():
        name = key
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

        if "local_conv" in key:  # always convert to nhwc
            original_tensor_layout = writer.tensor_layout
            writer.tensor_layout = TensorLayout.nhwc
            fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer)
            writer.tensor_layout = original_tensor_layout
            continue

        if fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer):
            continue

        if name.endswith("neck.0.weight") or name.endswith("neck.2.weight"):
            assert tensor.shape[2] == tensor.shape[3] and tensor.shape[2] <= 3
            tensor = writer.convert_tensor_2d(tensor)

        # Precompute dense positional embeddings from random matrix stored in the model
        if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            pe = build_dense_positional_embeddings(tensor)
            writer.add_tensor("dec.dense_positional_embedding", pe, "f32")

        if name in ["dec.iou_token.weight", "dec.mask_tokens.weight"]:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


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
def convert_birefnet(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    weights = safetensors.safe_open(input_filepath, "pt")
    model: dict[str, Tensor] = {k: weights.get_tensor(k) for k in weights.keys()}

    x = model["bb.layers.0.blocks.0.attn.proj.bias"]
    if x.shape[0] == 96:
        writer.add_string("swin.config", "tiny")
        writer.add_int32("swin.embed_dim", 96)
    elif x.shape[0] == 192:
        writer.add_string("swin.config", "large")
        writer.add_int32("swin.embed_dim", 192)
    else:
        raise ValueError(f"Unsupported Swin Transformer embed dim: {x.shape[0]}")

    image_size = 1024
    if "HR" in input_filepath.name or "2K" in input_filepath.name:
        image_size = 2048  # actually 2K should rather be 2560x1440
    elif "dynamic" in input_filepath.name:
        image_size = -1
    writer.add_int32("birefnet.image_size", image_size)
    writer.add_int32("birefnet.image_multiple", 128)

    for key, tensor in model.items():
        # Shorten some names to fit into 64 chars
        name = key
        name = name.replace("decoder_block", "block")
        name = name.replace("atrous_conv", "conv")
        name = name.replace("modulator_conv", "modulator")
        name = name.replace("offset_conv", "offset")
        name = name.replace("regular_conv", "conv")

        if name.endswith("relative_position_index"):
            continue  # precomputed in c++ code

        # Fuse all regular conv + batch norm pairs into a single conv with bias
        if fuse_conv_2d_batch_norm(model, key, name, "global_avg_pool.", "1", "2", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "dec_att.", "conv1", "bn1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_in", "bn_in", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_out", "bn_out", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_4.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_3.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_2.", "0", "1", writer):
            continue

        # Fuse batch norm into multiply+add for deformable conv
        tensor = fuse_batch_norm(model, key, "bn")
        if tensor is None:
            continue  # batch norm was fused

        if is_conv_2d(name, tensor):
            if "patch_embed" in name:  # part of SWIN, always store as NHWC
                tensor = conv_2d_to_nhwc(tensor)
            else:  # store rest in requested tensor layout
                tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# Depth-Anything


def convert_depth_anything(input_filepath: Path, writer: Writer):
    if "small" in input_filepath.name.lower():
        writer.add_license("apache-2.0")
    else:
        writer.add_license("cc-by-nc-4.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = load_model(input_filepath)

    if "pretrained.cls_token" in model:
        print("The converter is written for the transformers (.safetensors) version of the model.")
        print("The original weights (.pth) are currently not supported.")
        raise ValueError("Weights not supported")

    shape = model["backbone.embeddings.patch_embeddings.projection.weight"].shape
    writer.add_int32("dino.patch_size", shape[2])
    writer.add_int32("dino.embed_dim", shape[0])
    writer.add_int32("depthanything.image_size", 518)
    match shape[0]:
        case 384:  # Small
            writer.add_int32("dino.n_heads", 6)
            writer.add_int32("dino.n_layers", 12)
            writer.add_array("depthanything.feature_layers", [2, 5, 8, 11])
        case 768:  # Base
            writer.add_int32("dino.n_heads", 12)
            writer.add_int32("dino.n_layers", 12)
            writer.add_array("depthanything.feature_layers", [2, 5, 8, 11])
        case 1024:  # Large
            writer.add_int32("dino.n_heads", 16)
            writer.add_int32("dino.n_layers", 24)
            writer.add_array("depthanything.feature_layers", [4, 11, 17, 23])

    for key, tensor in model.items():
        name = key

        if is_conv_2d(name, tensor):
            if "patch_embeddings" in name or ("projection" in name and "fusion" not in name):
                tensor = conv_2d_to_nhwc(tensor)
            elif "0.resize" in name or "1.resize" in name:
                pass  # ConvTranspose2D, don't change layout
            else:
                tensor = writer.convert_tensor_2d(tensor)

        if "position_embeddings" in name or "cls_token" in name:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


#
# MI-GAN
def convert_migan(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model: dict[str, Tensor] = torch.load(input_filepath, weights_only=True)

    if "encoder.b512.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 512)
    elif "encoder.b256.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 256)

    for name, tensor in model.items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# ESRGAN
def convert_esrgan(input_filepath: Path, writer: Writer):
    from spandrel import ModelLoader

    # Load the model using spandrel
    # - it converts the various versions of ESRGAN checkpoints to a common format
    model = ModelLoader().load_from_file(input_filepath)

    if model.model.shuffle_factor is not None:
        raise ValueError("RealESRGAN models with pixel shuffle are not supported yet.")
    if getattr(model.model, "plus", False):
        raise ValueError("RealESRGAN+ (plus) models are not supported yet.")

    writer.set_tensor_layout_default(TensorLayout.nchw)
    writer.add_int32("esrgan.scale", model.scale)
    for tag in model.tags:
        if tag.endswith("nb"):
            writer.add_int32("esrgan.block_count", int(tag[:-2]))
        if tag.endswith("nf"):
            writer.add_int32("esrgan.filter_count", int(tag[:-2]))

    for name, tensor in model.model.state_dict().items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)
        writer.add_tensor(name, tensor)


# YoloV9t
def find_conv_bn_pairs(model: dict[str, Tensor]) -> list[tuple[str, str, str]]:
    """
    모델에서 Conv-BatchNorm 쌍을 자동으로 찾습니다.
    
    Returns:
        List of (base_key, conv_suffix, bn_suffix) tuples
        예: [("model.0", "conv.weight", "bn."), ("model.1.cv1", "conv.weight", "bn.")]
    """
    conv_bn_pairs = []
    conv_keys = {}
    
    # 1. 모든 conv weight 찾기
    for key in model.keys():
        if key.endswith("conv.weight"):
            conv_keys[key] = model[key]

    # 2. 각 conv weight에 대해 대응하는 batch norm 찾기
    for conv_key in conv_keys.keys():
        # model.4.cv2.0.m.0.cv1.conv1.conv.weight 
        
        bn = conv_key.replace(".conv.weight", ".bn")
        # BatchNorm 파라미터 확인
        bn_weight_key = f"{bn}.weight"
        bn_bias_key = f"{bn}.bias"
        bn_mean_key = f"{bn}.running_mean"
        bn_var_key = f"{bn}.running_var"
        conv = conv_key.replace(".conv.weight", ".conv")
        if all(k in model for k in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
            conv_bn_pairs.append((conv, bn))
        else:
            print(bn, conv + " does not have corresponding BatchNorm parameters.")
        
    return conv_bn_pairs


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, eps=1e-3):
    # https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html
    # Ensure None, not tensor ambiguity
    try:
        if conv_b is None:
            conv_b = torch.zeros_like(bn_rm)
    except:
        print(conv_b)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    # bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
    bn_w = bn_w / torch.sqrt(bn_rv + eps)
    conv_w = conv_w * bn_w[:, None, None, None]
    conv_b = (conv_b - bn_rm) * bn_w + bn_b

    return conv_w, conv_b

def fuse_conv_and_bn(conv_weight:torch.Tensor, 
                     conv_bias:torch.Tensor, bn_weight:torch.Tensor, bn_bias:torch.Tensor, running_mean:torch.Tensor, running_var:torch.Tensor, eps=1e-3):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    # Prepare filters
    w_conv = conv_weight.clone().view(conv_weight.size(0), -1)  # C_out x (C_in*H*W)
    w_bn = torch.diag(bn_weight.div(torch.sqrt(eps + running_var)))
    fused_conv = (torch.mm(w_bn, w_conv))
    
    # Prepare spatial bias
    b_conv = torch.zeros(conv_weight.size(0)) if conv_bias is None else conv_bias
    b_bn = bn_bias - bn_weight.mul(running_mean).div(torch.sqrt(running_var + eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    return fused_conv, fused_bias

# YOLOv9t 변환 함수 수정
def convert_yolov9t(input_filepath: Path, writer: Writer, bn_fuse: bool = True):
    writer.add_license("gpl-3.0")
    writer.set_tensor_layout_default(TensorLayout.nhwc)

    checkpoint = torch.load(input_filepath, map_location="cpu", weights_only=False)
    
    # Extract state dict from checkpoint
    if isinstance(checkpoint, dict) and not any(k.startswith('model.') or k.startswith('detect.') for k in checkpoint.keys()):
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
            if hasattr(checkpoint, 'state_dict'):
                model = checkpoint.state_dict()
            else:
                model = checkpoint
        elif 'state_dict' in checkpoint:
            model = checkpoint['state_dict']
        else:
            model = checkpoint
    else:
        model = checkpoint

    # Detect number of classes
    num_classes = 80  # default

    writer.add_int32("yolov9.num_classes", num_classes)
    # writer.add_int32("yolov9.num_anchors_per_scale", 3)
    # writer.add_string("yolov9.variant", "tiny")
    writer.add_int32("yolov9.input_size", 640)
    
    # Conv-BN 쌍 찾기
    print("\n=== Finding Conv-BN pairs ===")
    conv_bn_pairs = find_conv_bn_pairs(model)
    print(f"Found {len(conv_bn_pairs)} Conv-BN pairs")
    
    
    # Convert weights
    from tqdm import tqdm
    processed_keys = set()
    for idx, (conv_suffix, bn_suffix) in tqdm(enumerate(conv_bn_pairs)):
        # List[conv_suffix, bn_suffix]
        # 220 detect.cv3.2.1.conv detect.cv3.2.1.bn
        # Regular batch norm fusion (fallback)
        if 'detect' in conv_suffix:
            continue
        conv_weight = model.get(f"{conv_suffix}.weight", None)
        conv_bias = model.get(f"{conv_suffix}.bias", None)
        if bn_fuse:
            bn_w = model.get(f"{bn_suffix}.weight", None)
            bn_b = model.get(f"{bn_suffix}.bias", None)
            bn_rm = model.get(f"{bn_suffix}.running_mean", None)
            bn_rv = model.get(f"{bn_suffix}.running_var", None)
            eps = model.get(f"{bn_suffix}.eps", 1e-3)
        else:
            # batch norm fusion을 하지 않는 경우 저장해야함. add_tensor를 사용해서 저장.
            bn_w = model.get(f"{bn_suffix}.weight", None)
            bn_b = model.get(f"{bn_suffix}.bias", None)
            bn_rm = model.get(f"{bn_suffix}.running_mean", None)
            bn_rv = model.get(f"{bn_suffix}.running_var", None)
            bn_rm = None
            bn_rv = None
            eps = model.get(f"{bn_suffix}.eps", 1e-3)
            writer.add_tensor(f"{bn_suffix}.weight", bn_w) if bn_w is not None else None
            writer.add_tensor(f"{bn_suffix}.bias", bn_b) if bn_b is not None else None
            writer.add_tensor(f"{bn_suffix}.running_mean", bn_rm) if bn_rm is not None else None
            writer.add_tensor(f"{bn_suffix}.running_var", bn_rv) if bn_rv is not None else None

            bn_w = None
            bn_b = None
            bn_rm = None
            bn_rv = None
            eps = 1e-3

        if bn_w is None:
            # Fusing이 불가능한 경우.
            
            tensor = writer.convert_tensor_2d(conv_weight)
            writer.add_tensor(f"{conv_suffix}.weight", tensor)
            if conv_bias is not None:
                writer.add_tensor(f"{conv_suffix}.bias", conv_bias)
                processed_keys.add(f"{conv_suffix}.bias")
            processed_keys.add(f"{conv_suffix}.weight")
            print("Cannot fuse:", conv_suffix, bn_suffix)
        else:
            # Fusing이 가능한 경우.
            # print(conv_weight.shape, conv_bias.shape, bn_rm.shape, bn_rv.shape, bn_w.shape, bn_b.shape)
            fused_conv, fused_bias = fuse_conv_bn_weights(conv_weight, 
                                                      conv_bias, 
                                                      bn_rm, 
                                                      bn_rv, 
                                                      bn_w, 
                                                      bn_b, 
                                                      eps)
        
            # fused_conv , fused_bias = fuse_conv_and_bn(conv_weight= conv_weight, 
            #                                            conv_bias= conv_bias,
            #                                            bn_weight= bn_w, 
            #                                            bn_bias= bn_b,
            #                                            running_mean= bn_rm, 
            #                                            running_var= bn_rv,  
            #                                            eps= eps)
            
            tensor = writer.convert_tensor_2d(fused_conv)

            writer.add_tensor(f"{conv_suffix}.weight", tensor)
            writer.add_tensor(f"{conv_suffix}.bias", fused_bias)
            
            processed_keys.add(f"{conv_suffix}.weight")
            processed_keys.add(f"{conv_suffix}.bias")

            # print(f"{conv_suffix}, {bn_suffix}")
    # === 추가: Detect head의 마지막 레이어만 저장 ===
    print("\n=== Finding Detect head final layers (nn.Conv2d) ===")
    
    
    # detect와 weight가 들어가는 패턴만 매칭
    detect_layers = {}
    for idx, key in enumerate(model.keys()):
        if "detect" in key and ".conv.weight" in key:
            if key in detect_layers.keys():
                continue
            
            conv_suffix = key.replace(".conv.weight", ".conv")
            detect_layers[key] = model[key]
            conv_weight = model.get(f"{conv_suffix}.weight", None)
            conv_bias = model.get(f"{conv_suffix}.bias", None)
            
            if bn_fuse:
                bn_suffix = conv_suffix.replace(".conv", ".bn")
                bn_w = model.get(f"{bn_suffix}.weight", None)
                bn_b = model.get(f"{bn_suffix}.bias", None)
                bn_rm = model.get(f"{bn_suffix}.running_mean", None)
                bn_rv = model.get(f"{bn_suffix}.running_var", None)
                eps = model.get(f"{bn_suffix}.eps", 1e-3)
            else:
                bn_suffix = conv_suffix.replace(".conv", ".bn")
                bn_w = model.get(f"{bn_suffix}.weight", None)
                bn_b = model.get(f"{bn_suffix}.bias", None)
                bn_rm = model.get(f"{bn_suffix}.running_mean", None)
                bn_rv = model.get(f"{bn_suffix}.running_var", None)
                bn_rm = None
                bn_rv = None
                eps = model.get(f"{bn_suffix}.eps", 1e-3)
                writer.add_tensor(f"{bn_suffix}.weight", bn_w) if bn_w is not None else None
                writer.add_tensor(f"{bn_suffix}.bias", bn_b) if bn_b is not None else None
                writer.add_tensor(f"{bn_suffix}.running_mean", bn_rm) if bn_rm is not None else None
                writer.add_tensor(f"{bn_suffix}.running_var", bn_rv) if bn_rv is not None else None
                bn_suffix = None
                bn_w = None
                bn_b = None
                bn_rm = None
                bn_rv = None
                eps = None
            
            if bn_w is None or bn_b is None or bn_rm is None or bn_rv is None:
                # Fusing이 불가능한 경우.
                tensor = writer.convert_tensor_2d(conv_weight)
                processed_keys.add(f"{conv_suffix}.weight")
                writer.add_tensor(f"{conv_suffix}.weight", tensor)
                if conv_bias is not None:
                    writer.add_tensor(f"{conv_suffix}.bias", conv_bias)
                    processed_keys.add(f"{conv_suffix}.bias")
            else:
                # Fusing이 가능한 경우.
                fused_conv, fused_bias = fuse_conv_bn_weights(conv_weight, 
                                                    conv_bias, 
                                                    bn_rm, 
                                                    bn_rv, 
                                                    bn_w, 
                                                    bn_b, 
                                                    eps)
                tensor = writer.convert_tensor_2d(fused_conv)
                writer.add_tensor(f"{conv_suffix}.weight", tensor)
                processed_keys.add(f"{conv_suffix}.weight")
                if fused_bias is not None:
                    writer.add_tensor(f"{conv_suffix}.bias", fused_bias)
                    processed_keys.add(f"{conv_suffix}.bias")
                
            
            
            
        elif 'detect' in key and\
            ".conv" not in key and\
            '.bn' not in key and\
            '.weight' in key:
            print("Only weight:", key)
            detect_layers[key] = model[key]
            conv_weight = model.get(f"{key}", None)
            
            bias = key.replace('.weight', '.bias')
            conv_bias = model.get(f"{bias}", None)
            
            tensor = writer.convert_tensor_2d(conv_weight)
            writer.add_tensor(key, tensor)
            if conv_bias is not None:
                writer.add_tensor(bias, conv_bias)
            # detect.cv2.0.2.weight
            # detect.cv2.0.2.bias
            # detect.cv2.1.2.weight
            # detect.cv2.1.2.bias
            # detect.cv2.1.2.weight
            # detect.cv2.1.2.bias
            # detect.cv2.2.2.weight
            # detect.cv2.2.2.bias
            # detect.cv3.0.2.weight
            # detect.cv3.0.2.bias
            # detect.cv3.2.2.weight
            # detect.cv3.2.2.bias
            # detect.dfl.conv.weight

    print(f"Found {len(detect_layers)} detect head final layer parameters")
    
    
    total_detect_layers = len([k for k in detect_layers.keys() if k.endswith('.weight')])
    
    print(f"\n✓ Conversion complete:")
    print(f"  - {len(conv_bn_pairs)} Conv-BN pairs (fused)")
    print(f"  - {total_detect_layers} Detect head final Conv2d layers")
    print(f"  - Total: {len(conv_bn_pairs) + total_detect_layers} layers")
#
# Main
#######

arch_names = {
    "sam": "mobile-sam",
    "birefnet": "birefnet",
    "depth-anything": "depthanything",
    "migan": "migan",
    "esrgan": "esrgan",
    "yolov9t": "yolov9t",
}

file_types = {None: 0, "f32": 0, "f16": 1}

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert model weights (.pt/.pth/.safetensors) to GGUF format.")
    parser.add_argument("--arch", choices=list(arch_names.keys()), help="Model architecture")
    parser.add_argument("--input", type=str, help="Path to the input model file")
    parser.add_argument("--output", "-o", type=str, default="models", help="Path to the output directory or file")
    parser.add_argument("--quantize", "-q", choices=["f32"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--layout", "-l", choices=["whcn", "cwhn"], default=None, help="Tensor data layout for 2D operations like convolution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for metadata")
    parser.add_argument("--metadata", type=Path, help="Specify the path for an authorship metadata override file")
    parser.add_argument("--bn-fuse", action="store_true", help="Enable BatchNorm fusion for YOLOv9t conversion")
    # fmt: on
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    quant_suffix = f"-{args.quantize.upper()}" if args.quantize else ""
    layout_suffix = f"-{args.layout.upper()}" if args.layout else ""
    if output_path.is_dir() or output_path.suffix != ".gguf":
        output_path = output_path / f"{input_path.stem}{quant_suffix}{layout_suffix}.gguf"

    print(f"Converting {args.arch}")
    print("* input: ", input_path)
    print("* output:", output_path)

    try:
        writer = Writer(
            output_path,
            arch_names.get(args.arch, args.arch),
            args.quantize,
            args.verbose,
        )
        metadata = Metadata.load(args.metadata, input_path.with_suffix(""), args.model_name)

        if args.layout is not None:
            writer.set_tensor_layout(TensorLayout.parse(args.layout))

        match args.arch:
            case "sam":
                convert_sam(input_path, writer)
            case "birefnet":
                convert_birefnet(input_path, writer)
            case "depthany" | "depth-anything":
                convert_depth_anything(input_path, writer)
            case "migan":
                convert_migan(input_path, writer)
            case "esrgan":
                convert_esrgan(input_path, writer)
            case "yolov9t":
                convert_yolov9t(input_path, writer, bn_fuse=args.bn_fuse)
            case _:
                raise ValueError(f"Unknown architecture: {args.arch}")

        metadata.set_gguf_meta_model(writer)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_types[args.quantize])
        writer.add_conv2d_weight_indices()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    except ValueError as e:
        print("\033[31mError:\033[0m", e)
        exit(1)
    except Exception as e:
        print("\033[31mError:\033[0m", e)
        exit(-1)

    print("")
