from transformers import Sam3Processor, Sam3Model
from pathlib import Path
import torch
from PIL import Image

from . import workbench
from .workbench import tensors_match, images_match

torch.set_printoptions(precision=2, linewidth=100, sci_mode=False)

model_path = Path("/mnt/share/ml/vision/sam3")
image_path = Path(__file__).parent / "input" / "wardrobe.jpg"
results_dir = Path(__file__).parent / "results"


def test_transformers():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam3Model.from_pretrained(model_path).to(device)
    processor = Sam3Processor.from_pretrained(model_path)
    image = Image.open(image_path).convert("RGB")

    # Segment using text prompt
    inputs = processor(images=image, text="shirt", return_tensors="pt").to(device)

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
    image = Image.open(image_path).convert("RGB")

    expected = processor(images=image, text="shirt cow H7", return_tensors="pt")
    expected_pixels = expected["pixel_values"]
    expected_ids = expected["input_ids"]
    expected_mask = expected["attention_mask"]
    assert isinstance(expected_pixels, torch.Tensor)
    assert isinstance(expected_ids, torch.Tensor)
    assert isinstance(expected_mask, torch.Tensor)

    result_pixels = workbench.invoke_test("sam3_process_image", [], {}, {"image": str(image_path)})
    assert isinstance(result_pixels, torch.Tensor)
    result_pixels = result_pixels.permute(0, 3, 1, 2)

    assert images_match(result_pixels, expected_pixels, tol=0.05)

    result = workbench.invoke_test(
        "sam3_process_text", [], {}, {"text": "shirt cow H7", "vocab": "tests/data/sam3-vocab.gguf"}
    )
    assert isinstance(result, list)
    assert tensors_match(result[0], expected_ids)
    assert tensors_match(result[1], expected_mask)
