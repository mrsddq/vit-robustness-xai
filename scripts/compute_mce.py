from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALEXNET_REFERENCE_ERRORS = {
    "gaussian_noise": 0.886,
    "shot_noise": 0.894,
    "impulse_noise": 0.922,
    "defocus_blur": 0.819,
    "glass_blur": 0.826,
    "motion_blur": 0.785,
    "zoom_blur": 0.798,
    "snow": 0.866,
    "frost": 0.826,
    "fog": 0.819,
    "brightness": 0.564,
    "contrast": 0.853,
    "elastic_transform": 0.646,
    "pixelate": 0.717,
    "jpeg_compression": 0.606,
}


def compute_mce(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        corruption = row["corruption"]
        accuracy = float(row["accuracy"])
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be between 0 and 1 for {corruption}")
        grouped.setdefault(corruption, []).append(1.0 - accuracy)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["corruption", "error", "alexnet_error", "ce"])
        writer.writeheader()
        ce_values = []
        for corruption, errors in sorted(grouped.items()):
            if corruption not in ALEXNET_REFERENCE_ERRORS:
                continue
            error = sum(errors) / len(errors)
            ce = error / ALEXNET_REFERENCE_ERRORS[corruption]
            ce_values.append(ce)
            writer.writerow(
                {
                    "corruption": corruption,
                    "error": error,
                    "alexnet_error": ALEXNET_REFERENCE_ERRORS[corruption],
                    "ce": ce,
                }
            )
        if not ce_values:
            raise ValueError("No recognized ImageNet-C corruption rows found")
        writer.writerow({"corruption": "mean", "error": "", "alexnet_error": "", "ce": sum(ce_values) / len(ce_values)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean corruption error from robustness CSV.")
    parser.add_argument("--input", type=Path, default=Path("outputs/metrics/robustness_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/mce_results.csv"))
    args = parser.parse_args()
    compute_mce(args.input, args.output)


if __name__ == "__main__":
    main()
