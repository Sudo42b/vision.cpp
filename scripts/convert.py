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
        if key.endswith(".weight") and "bn" not in key:
            tensor = model[key]
            # Conv2d weight는 4차원 (out_channels, in_channels, height, width)
            if len(tensor.shape) == 4:
                conv_keys[key] = tensor
    
    # 2. 각 conv weight에 대해 대응하는 batch norm 찾기
    for conv_key in conv_keys.keys():
        # conv.weight에서 base와 suffix 분리
        if ".conv.weight" in conv_key:
            base = conv_key.replace(".conv.weight", "")
            conv_suffix = "conv.weight"
            bn_suffix = "bn."
        elif ".weight" in conv_key:
            # 다른 패턴들 (예: model.0.weight -> model.0.bn.)
            base = conv_key.replace(".weight", "")
            conv_suffix = "weight"
            bn_suffix = "bn."
        else:
            continue
        
        # BatchNorm 파라미터 확인
        bn_weight_key = f"{base}.{bn_suffix}weight"
        bn_bias_key = f"{base}.{bn_suffix}bias"
        bn_mean_key = f"{base}.{bn_suffix}running_mean"
        bn_var_key = f"{base}.{bn_suffix}running_var"
        
        if all(k in model for k in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
            conv_bn_pairs.append((base, conv_suffix, bn_suffix))
    
    return conv_bn_pairs

def fuse_conv_2d_batch_norm_dynamic(
    model: dict[str, Tensor],
    key: str,
    writer: Writer,
    batch_norm_eps: float = 1e-5,
) -> bool:
    """
    동적으로 Conv-BatchNorm fusion을 수행합니다.
    
    Args:
        model: 모델 state dict
        key: 현재 처리 중인 키
        writer: GGUF writer
        batch_norm_eps: BatchNorm epsilon 값
    
    Returns:
        True if the key was part of a conv-bn pair and was handled
    """
    # BatchNorm 관련 키는 skip (이미 fusion됨)
    if any(bn_key in key for bn_key in [".bn.weight", ".bn.bias", ".bn.running_mean", 
                                          ".bn.running_var", ".bn.num_batches_tracked"]):
        return True
    
    # Conv weight가 아니면 skip
    if not key.endswith(".weight"):
        return False
    
    tensor = model[key]
    if len(tensor.shape) != 4:  # Conv2d weight가 아님
        return False
    
    # RepConv 패턴 특별 처리 (conv1.conv.weight, conv2.conv.weight)
    if ".conv1.conv.weight" in key or ".conv2.conv.weight" in key:
        # RepConv conv1 또는 conv2 처리
        base = key.replace(".conv.weight", "")
        
        # BatchNorm 파라미터 확인
        bn_weight_key = f"{base}.bn.weight"
        bn_bias_key = f"{base}.bn.bias"
        bn_mean_key = f"{base}.bn.running_mean"
        bn_var_key = f"{base}.bn.running_var"
        
        if all(k in model for k in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
            # Fusion 수행
            conv_weight = model[key]
            bn_weight = model[bn_weight_key]
            bn_bias = model[bn_bias_key]
            bn_mean = model[bn_mean_key]
            bn_var = model[bn_var_key]
            
            # Conv bias 확인 (보통 없음)
            conv_bias_key = f"{base}.conv.bias"
            conv_bias = model.get(conv_bias_key, torch.zeros_like(bn_bias))
            
            # Fuse
            bn_weight = bn_weight / torch.sqrt(bn_var + batch_norm_eps)
            fused_weight = conv_weight * bn_weight[:, None, None, None]
            fused_bias = (conv_bias - bn_mean) * bn_weight + bn_bias
            
            # GGUF에 저장 (RepConv용 이름으로)
            output_name = base.replace(".conv", "")  # .conv 제거
            writer.add_tensor(f"{output_name}.weight", writer.convert_tensor_2d(fused_weight))
            writer.add_tensor(f"{output_name}.bias", fused_bias)
            
            print(f"✓ Fused RepConv: {key} + {bn_weight_key} -> {output_name}")
            return True
    
    # Conv-BN 쌍 찾기 (일반 패턴)
    # 여러 패턴 시도
    patterns = [
        (".conv.weight", ".conv.bias", ".bn."),
        (".weight", ".bias", ".bn."),
    ]
    
    for weight_suffix, bias_suffix, bn_prefix in patterns:
        if not key.endswith(weight_suffix):
            continue
            
        base = key.removesuffix(weight_suffix)
        
        # BatchNorm 파라미터 확인
        bn_weight_key = f"{base}{bn_prefix}weight"
        bn_bias_key = f"{base}{bn_prefix}bias"
        bn_mean_key = f"{base}{bn_prefix}running_mean"
        bn_var_key = f"{base}{bn_prefix}running_var"
        
        if not all(k in model for k in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
            continue
        
        # Fusion 수행
        conv_weight = model[key]
        bn_weight = model[bn_weight_key]
        bn_bias = model[bn_bias_key]
        bn_mean = model[bn_mean_key]
        bn_var = model[bn_var_key]
        
        conv_bias_key = f"{base}{bias_suffix}"
        conv_bias = model.get(conv_bias_key, torch.zeros_like(bn_bias))
        
        # Fuse
        bn_weight = bn_weight / torch.sqrt(bn_var + batch_norm_eps)
        fused_weight = conv_weight * bn_weight[:, None, None, None]
        fused_bias = (conv_bias - bn_mean) * bn_weight + bn_bias
        
        # GGUF에 저장
        fused_weight = writer.convert_tensor_2d(fused_weight)
        output_name = key  # 또는 원하는 이름 변환
        writer.add_tensor(output_name, fused_weight)
        writer.add_tensor(output_name.replace("weight", "bias"), fused_bias)
        
        print(f"✓ Fused: {key} + {bn_weight_key}")
        return True
    
    # Conv-BN 쌍이 아닌 일반 Conv
    return False

# YOLOv9t 변환 함수 수정
def convert_yolov9t(input_filepath: Path, writer: Writer):
    writer.add_license("gpl-3.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

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
    for key in model.keys():
        if 'detect.cv2' in key or 'detect.cv3' in key:
            if '.bias' in key and 'bn' not in key:
                bias_size = model[key].shape[0]
                nc = (bias_size // 3) - 5
                if nc > 0:
                    num_classes = nc
                    break
    
    writer.add_int32("yolov9.num_classes", num_classes)
    writer.add_int32("yolov9.num_anchors_per_scale", 3)
    writer.add_string("yolov9.variant", "tiny")
    writer.add_int32("yolov9.input_size", 640)
    
    # Anchors
    anchors_p3 = [10, 13, 16, 30, 33, 23]
    anchors_p4 = [30, 61, 62, 45, 59, 119]
    anchors_p5 = [116, 90, 156, 198, 373, 326]
    
    writer.add_array("yolov9.anchors_p3", anchors_p3)
    writer.add_array("yolov9.anchors_p4", anchors_p4)
    writer.add_array("yolov9.anchors_p5", anchors_p5)
    
    # Conv-BN 쌍 찾기
    print("\n=== Finding Conv-BN pairs ===")
    conv_bn_pairs = find_conv_bn_pairs(model)
    print(f"Found {len(conv_bn_pairs)} Conv-BN pairs")
    
    # Convert weights
    processed_keys = set()
    
    for key, tensor in model.items():
        if key in processed_keys:
            continue
            
        # Skip unnecessary tensors
        if any(skip in key for skip in ["anchor_grid", "anchors", "stride", "num_batches_tracked"]):
            continue
        
        # 동적 Conv-BN fusion 시도
        if fuse_conv_2d_batch_norm_dynamic(model, key, writer):
            processed_keys.add(key)
            # 관련 BN 키들도 processed로 표시
            for bn_suffix in [".bn.weight", ".bn.bias", ".bn.running_mean", ".bn.running_var"]:
                if key.endswith(".weight"):
                    base = key.replace(".weight", "").replace(".conv.weight", "")
                    processed_keys.add(f"{base}{bn_suffix}")
            continue
        
        # Regular batch norm fusion (fallback)
        tensor = fuse_batch_norm(model, key, "bn")
        if tensor is None:
            processed_keys.add(key)
            continue
        
        # Convert 2D convolutions
        if is_conv_2d(key, tensor):
            tensor = writer.convert_tensor_2d(tensor)
        
        writer.add_tensor(key, tensor)
        processed_keys.add(key)
    
    print(f"\n✓ Conversion complete. Processed {len(processed_keys)} tensors.")
#
# Main
#######

arch_names = {
    "sam": "mobile-sam",
    "birefnet": "birefnet",
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
    parser.add_argument("--quantize", "-q", choices=["f16"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--layout", "-l", choices=["whcn", "cwhn"], default=None, help="Tensor data layout for 2D operations like convolution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for metadata")
    parser.add_argument("--metadata", type=Path, help="Specify the path for an authorship metadata override file")
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
            case "migan":
                convert_migan(input_path, writer)
            case "esrgan":
                convert_esrgan(input_path, writer)
            case "yolov9t":
                convert_yolov9t(input_path, writer)
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
