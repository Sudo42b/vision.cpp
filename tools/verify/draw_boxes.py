#!/usr/bin/env python3
"""Draw detections onto the image they came from.

The runners write detections as raw float32 -- six numbers per box -- because that is what
comparing against a reference implementation needs. This turns such a file into something to
look at.

    python tools/verify/draw_boxes.py image.jpg boxes.bin -o annotated.png

Boxes are in the coordinate space of the resized square input the detector ran on, so the
script scales them back to the original image. Pass --size if the detector ran at something
other than 512.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO 80. Detectors trained on something else take --labels.
COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
    "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork "
    "knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake "
    "chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
    "hair_drier toothbrush"
).split()

# Distinct enough to tell classes apart without a legend.
PALETTE = [
    (230, 60, 60), (60, 160, 230), (70, 190, 110), (240, 160, 40), (170, 100, 220),
    (40, 200, 200), (230, 110, 170), (150, 160, 60), (110, 130, 240), (200, 90, 60),
]


def load(path: Path) -> np.ndarray:
    d = np.fromfile(path, dtype="float32")
    if d.size % 6:
        raise SystemExit(f"{path}: {d.size} floats is not a multiple of 6")
    return d.reshape(-1, 6)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Draw raw float32 detections onto an image.")
    ap.add_argument("image", type=Path, help="the image the detector was given")
    ap.add_argument("boxes", type=Path, help="detections written by the runner (.bin)")
    ap.add_argument("-o", "--output", type=Path, default=Path("annotated.png"))
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="skip detections below this score. Default 0.3")
    ap.add_argument("--size", type=int, default=512,
                    help="square input resolution the detector ran at. Default 512")
    ap.add_argument("--labels", type=Path, default=None,
                    help="class names, one per line. Default: COCO 80")
    a = ap.parse_args(argv)

    names = (a.labels.read_text(encoding="utf-8").split() if a.labels else COCO)
    img = Image.open(a.image).convert("RGB")
    dets = load(a.boxes)
    keep = dets[dets[:, 4] >= a.threshold]

    # The detector saw a square of --size; put the boxes back on the original.
    sx, sy = img.width / a.size, img.height / a.size

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(12, img.height // 45))
    except TypeError:  # Pillow < 9.2 has no size argument
        font = ImageFont.load_default()

    for x1, y1, x2, y2, score, label in keep:
        label = int(label)
        colour = PALETTE[label % len(PALETTE)]
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        draw.rectangle(box, outline=colour, width=max(2, img.height // 300))

        name = names[label] if label < len(names) else str(label)
        text = f"{name} {score:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        ty = max(0.0, box[1] - th - 2)
        draw.rectangle((box[0], ty, box[0] + tw + 6, ty + th + 4), fill=colour)
        draw.text((box[0] + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    img.save(a.output)
    print(f"  → {a.output}  ({len(keep)} of {len(dets)} detections at score >= {a.threshold})")


if __name__ == "__main__":
    main()
