"""Evaluate ViT robustness under ImageNet-C style corruptions."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from scripts.common import accuracy, imagenet_transform, load_torchvision_model, load_yaml


def main(model_name, data_dir, cfg_path):
    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchvision_model(model_name, pretrained=True).to(device)
    transform = imagenet_transform()
    results = []
    data_dir = Path(data_dir)

    for corruption in cfg["evaluation"]["corruption_types"]:
        for severity in cfg["evaluation"]["severity_levels"]:
            corrupt_path = data_dir / corruption / str(severity)
            if not corrupt_path.exists():
                print(f"Skipping {corruption} severity {severity}: path not found")
                continue

            loader = DataLoader(
                ImageFolder(str(corrupt_path), transform=transform),
                batch_size=cfg["evaluation"]["batch_size"],
                num_workers=4,
            )
            acc = accuracy(model, loader, device)
            results.append({"corruption": corruption, "severity": severity, "accuracy": acc})
            print(f"{corruption} severity {severity}: {acc:.4f}")

    if results:
        frame = pd.DataFrame(results)
        output_dir = Path(cfg["logging"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "robustness_results.csv"
        frame.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"Mean accuracy across corruptions: {frame.accuracy.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_b_16")
    parser.add_argument("--data", default="data/corrupted/")
    parser.add_argument("--config", default="configs/robustness.yaml")
    args = parser.parse_args()
    main(args.model, args.data, args.config)
