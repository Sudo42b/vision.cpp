from transformers import Sam3Config, Sam3Processor, Sam3Model
from transformers.masking_utils import create_causal_mask
from pathlib import Path
import torch
from PIL import Image

from . import workbench
from .workbench import tensors_match, images_match

model_path = Path("/mnt/share/ml/vision/sam3")
test_dir = Path(__file__).parent
image_path = test_dir / "input" / "wardrobe.jpg"
results_dir = test_dir / "results"
vocab_path = test_dir / "data" / "sam3-vocab.gguf"


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


def test_text_embeds():
    model = Sam3Model.from_pretrained(model_path)
    processor = Sam3Processor.from_pretrained(model_path)
    text_inputs = processor(text="cat", return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)  # type: ignore
    expected_embeds: torch.Tensor = text_features.pooler_output  # type: ignore
    assert expected_embeds is not None

    model.eval()
    state = _convert_tensor_names(model.state_dict())

    result_embeds = workbench.invoke_test(
        "sam3_text_embeds", [], state, {"text": "cat", "model": str(vocab_path)}
    )
    assert isinstance(result_embeds, torch.Tensor)
    assert tensors_match(result_embeds, expected_embeds, atol=1e-2)  # f16 gelu
