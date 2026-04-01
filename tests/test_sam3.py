from transformers import Sam3Config, Sam3Processor, Sam3Model, Sam3VisionModel
from transformers.masking_utils import create_causal_mask
from transformers.models.sam3.modeling_sam3 import (
    Sam3SinePositionEmbedding,
    Sam3ViTEmbeddings,
    Sam3ViTModel,
    Sam3ViTRotaryEmbedding,
    Sam3ViTLayer,
    Sam3VisionNeck,
    apply_rotary_pos_emb_2d,
)
from transformers.models.sam3.configuration_sam3 import Sam3ViTConfig, Sam3VisionConfig
from pathlib import Path
import pytest
import torch
from PIL import Image

from . import workbench
from .workbench import tensors_match, images_match

model_path = Path("/mnt/share/ml/vision/sam3")
test_dir = Path(__file__).parent
image_path = test_dir / "input" / "wardrobe.jpg"
results_dir = test_dir / "results"
vocab_path = test_dir / "data" / "sam3-vocab.gguf"
tmp_dir = test_dir.parent / ".tmp"
tmp_dir.mkdir(exist_ok=True)


def test_transformers():
    model = Sam3Model.from_pretrained(model_path)
    processor = Sam3Processor.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")

    # Segment using text prompt
    inputs = processor(images=image, text="shirt", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    print(f"Found {len(results['masks'])} objects")

    for i, (mask, box, score) in enumerate(
        zip(results["masks"], results["boxes"], results["scores"])
    ):
        print(f"Object {i}: Box={box.cpu().numpy()}, Score={score.item():.4f}")
        mask_image = Image.fromarray((mask.cpu().numpy() * 255).astype("uint8"))
        mask_image.save(results_dir / f"sam3_mask_{i}.png")


def test_processor():
    processor = Sam3Processor.from_pretrained(model_path)
    config = Sam3Config.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")
    text = "shirt cow H7!"

    expected = processor(images=image, text=text, return_tensors="pt")
    expected_pixels = expected["pixel_values"]
    expected_ids = expected["input_ids"]
    expected_mask = expected["attention_mask"]
    assert isinstance(expected_pixels, torch.Tensor)
    assert isinstance(expected_ids, torch.Tensor)
    assert isinstance(expected_mask, torch.Tensor)

    # In transformers this happens later in CLIPTextTransformer
    # vision.cpp creates the causal mask directly in pre-process
    dummy_embeds = torch.randn(1, expected_ids.shape[1], 1024)
    dummy_cache = torch.arange(expected_ids.shape[1])
    config._attn_implementation = "sdpa"
    causal_mask = create_causal_mask(
        config.text_config, dummy_embeds, expected_mask, dummy_cache, None
    )
    # boolean mask true/false -> 0/-inf additive mask for FA
    assert isinstance(causal_mask, torch.Tensor)
    causal_mask_f16 = torch.zeros_like(causal_mask, dtype=torch.float16)
    causal_mask_f16.masked_fill_(~causal_mask, float("-inf"))

    result_pixels = workbench.invoke_test("sam3_process_image", [], {}, {"image": str(image_path)})
    assert isinstance(result_pixels, torch.Tensor)
    result_pixels = result_pixels.permute(0, 3, 1, 2)

    assert images_match(result_pixels, expected_pixels, tol=0.05)

    result = workbench.invoke_test(
        "sam3_process_text", [], {}, {"text": text, "vocab": str(vocab_path)}
    )
    assert isinstance(result, list)
    assert tensors_match(result[0], expected_ids.to(torch.int32))
    assert tensors_match(result[1], causal_mask_f16)


@pytest.mark.parametrize("normalize", [False, True])
def test_sine_position_embedding(normalize):
    embedding = Sam3SinePositionEmbedding(4, normalize=normalize)
    expected = embedding(torch.Size([1, 8, 3, 3]), "cpu", torch.float32, mask=None)

    params = {"width": 3, "height": 3, "n_pos_feats": 4, "normalize": int(normalize)}
    result = workbench.invoke_test("sam3_sine_position_embedding", [], {}, params)
    assert isinstance(result, torch.Tensor)
    assert tensors_match(result, expected)


def test_rotary_embedding():
    config = Sam3ViTConfig(hidden_size=16, num_attention_heads=2)
    head_dim = config.hidden_size // config.num_attention_heads
    width = 3
    height = 3
    scale = 0.5
    embedding = Sam3ViTRotaryEmbedding(config, end_x=width, end_y=height, scale=scale)
    embedding.eval()
    cos, sin = embedding()
    q = torch.ones(1, width * height, config.num_attention_heads, head_dim)
    k = torch.ones(1, width * height, config.num_attention_heads, head_dim)

    # transformers' apply_rotary_pos_emb_2d expects [head_dim, n_head, n_pos, batch]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    expected_q, expected_k = apply_rotary_pos_emb_2d(q_t, k_t, cos, sin)
    expected_q = expected_q.transpose(1, 2)
    expected_k = expected_k.transpose(1, 2)

    # Compare with C++ implementation (works on non-transposed q/k)
    params = {"scale": scale, "width": width, "height": height}
    result = workbench.invoke_test("sam3_rotary_embedding", [q, k], {}, params)
    assert isinstance(result, list)

    assert tensors_match(result[0], expected_q)
    assert tensors_match(result[1], expected_k)


def test_vision_embed():
    config = Sam3ViTConfig(pretrain_image_size=8, patch_size=2, num_channels=3, hidden_size=6)
    patch_embed = Sam3ViTEmbeddings(config)
    state = patch_embed.state_dict()
    state = workbench.generate_state(state)
    patch_embed.load_state_dict(state)
    patch_embed.eval()

    x = workbench.input_tensor(1, 3, 8, 8)
    expected = patch_embed(x)
    result = workbench.invoke_test(
        "sam3_vision_embed", [x], state, {"patch_size": config.patch_size}
    )
    assert tensors_match(result, expected)


@pytest.mark.parametrize("window_size", [4, 0])
def test_vision_layer(window_size):
    config = Sam3ViTConfig(
        hidden_size=16, num_attention_heads=2, image_size=16, patch_size=2, window_size=4
    )
    n_row = window_size if window_size > 0 else config.image_size // config.patch_size
    params = {
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "window_size": config.window_size,
        "is_global": int(window_size == 0),
    }
    layer = Sam3ViTLayer(config, window_size)
    layer.eval()

    x = torch.randn(1, n_row, n_row, config.hidden_size)
    expected = layer(x)
    result = workbench.invoke_test("sam3_vision_layer", [x], layer.state_dict(), params)
    assert tensors_match(result, expected, atol=1e-2)


def test_vision_neck():
    backbone_config = Sam3ViTConfig(hidden_size=16)
    config = Sam3VisionConfig(
        backbone_config=backbone_config, fpn_hidden_size=128, scale_factors=[4.0, 2.0, 1.0, 0.5]
    )
    neck = Sam3VisionNeck(config)
    neck.eval()

    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        fpn_hidden_states, _ = neck(x)

    result = workbench.invoke_test("sam3_vision_neck", [x], neck.state_dict())
    assert isinstance(result, list)
    assert len(result) == 4
    for i in range(4):
        assert tensors_match(result[i], fpn_hidden_states[i], atol=1e-3)


def _convert_tensor_names(state: dict[str, torch.Tensor]):
    # transformer names -> gguf names (see convert.py)
    new_state = {}
    for key in state.keys():
        name = key
        name = name.replace("detector_model", "det")
        name = name.replace("text_encoder", "te")
        name = name.replace("vision_encoder", "ve")
        name = name.replace("tracker_model", "trk")
        name = name.replace("mask_decoder.", "decoder.")
        name = name.replace("_image_to_token.", "_i2t.")
        name = name.replace("_token_to_image.", "_t2i.")
        new_state[f"det.{name}"] = state[key]
    return new_state


def fwd_wrapper(layer, name: str, captures: dict[str, torch.Tensor]):
    old_fwd = layer.forward

    def fwd(x):
        result = old_fwd(x)
        captures[name] = result
        return result

    return fwd


def test_model():
    text = "cat"
    image = Image.open(test_dir / "input" / "cat-and-hat.jpg")

    model = Sam3Model.from_pretrained(model_path)
    processor = Sam3Processor.from_pretrained(model_path)
    inputs = processor(images=image, text=text, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    assert isinstance(pixel_values, torch.FloatTensor)

    model.eval()
    state = _convert_tensor_names(model.state_dict())

    # Text encoder

    text_features_cache = tmp_dir / "sam3_text_features.pt"
    if text_features_cache.exists():
        print("Loading cached text features...")
        expected_text_embeds = torch.load(text_features_cache)
    else:
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)  # type: ignore
        expected_text_embeds: torch.Tensor = text_features.pooler_output  # type: ignore
        torch.save(expected_text_embeds, text_features_cache)
    assert expected_text_embeds is not None

    result_text_embeds = workbench.invoke_test(
        "sam3_text_embeds", [], state, {"text": "cat", "model": str(vocab_path)}
    )
    assert isinstance(result_text_embeds, torch.Tensor)
    assert tensors_match(result_text_embeds, expected_text_embeds, atol=1e-2)  # f16 gelu

    # Vision encoder

    vision_embeds_cache = tmp_dir / "sam3_vision_embeds.pt"
    vision_pos_cache = tmp_dir / "sam3_vision_pos.pt"
    captures = {}
    if vision_embeds_cache.exists():
        print("Loading cached vision embeds...")
        expected_vision_embeds = torch.load(vision_embeds_cache)
        expected_vision_pos = torch.load(vision_pos_cache)
        for i in range(32):
            capture_name = f"sam3_vision_layer_{i}"
            capture_file = tmp_dir / f"{capture_name}.pt"
            if capture_file.exists():
                captures[capture_name] = torch.load(capture_file)
    else:
        bb: Sam3ViTModel = model.vision_encoder.backbone
        for i, layer in enumerate(bb.layers):
            layer.forward = fwd_wrapper(layer, f"sam3_vision_layer_{i}", captures)  # type: ignore
        with torch.no_grad():
            vision_outputs = model.get_vision_features(pixel_values)
        expected_vision_embeds = vision_outputs.fpn_hidden_states[:-1]
        expected_vision_pos = vision_outputs.fpn_position_encoding[:-1]
        torch.save(expected_vision_embeds, vision_embeds_cache)
        torch.save(expected_vision_pos, vision_pos_cache)
        for name, tensor in captures.items():
            torch.save(tensor, tmp_dir / f"{name}.pt")

    print("Invoking vision encoder with captures...")

    capture_names = [f"det.ve.backbone.layers.{i}" for i in range(32)]
    result_vision = workbench.invoke_test(
        "sam3_vision_encoder", [pixel_values], state, capture=capture_names
    )
    assert isinstance(result_vision, list)

    print("Layer output comparison:")
    for i in range(32):
        expected_layer = captures[f"sam3_vision_layer_{i}"]
        result_layer = result_vision[2 + i]
        if not tensors_match(expected_layer, result_layer, atol=1e-2):
            print(f"Layer {i} outputs do not match!")
            diff = torch.abs(expected_layer - result_layer)
            print(f"Max difference: {diff.max().item():.4f}")
            print(f"Mean difference: {diff.mean().item():.4f}")
            print(f"Non-matching elements: {(diff > 0.01).sum().item()} / {diff.numel()}")

    workbench.print_results(result_vision[0], expected_vision_embeds[-1])

    assert tensors_match(result_vision[0], expected_vision_embeds[-1], atol=1e-2)
    assert tensors_match(result_vision[1], expected_vision_pos[-1])
