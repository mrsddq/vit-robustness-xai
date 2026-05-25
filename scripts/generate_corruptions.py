from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def generate(input_dir: Path, output_dir: Path) -> None:
    try:
        from imagecorruptions import corrupt
    except ImportError as exc:
        raise ImportError("Install imagecorruptions to generate ImageNet-C-style data.") from exc
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for image_path in images:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        rel_parent = image_path.relative_to(input_dir).parent
        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                corrupted = corrupt(image, corruption_name=corruption, severity=severity)
                target = output_dir / corruption / str(severity) / rel_parent / image_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(corrupted).save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ImageNet-C-style corruptions locally.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corrupted"))
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
