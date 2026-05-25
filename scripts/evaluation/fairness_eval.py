"""Evaluate per-subgroup accuracy for fairness analysis."""
import argparse
from pathlib import Path

import torch

from scripts.common import load_torchvision_model, subgroup_gap


def main(model_name, data_dir, groups_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_torchvision_model(model_name, pretrained=True).to(device).eval()
    print(f"Model: {model_name} | Data: {data_dir}")

    if groups_path and not Path(groups_path).exists():
        raise FileNotFoundError(f"Group annotation file not found: {groups_path}")

    print("Provide subgroup-labelled data to compute per-group accuracy and report the gap.")
    print("Example gap calculation:", subgroup_gap({"group_a": 0.91, "group_b": 0.84}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_b_16")
    parser.add_argument("--data", default="data/clean/")
    parser.add_argument("--groups")
    args = parser.parse_args()
    main(args.model, args.data, args.groups)
