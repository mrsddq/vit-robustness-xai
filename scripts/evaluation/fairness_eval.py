"""Evaluate per-subgroup accuracy for fairness analysis.
Usage: python scripts/evaluation/fairness_eval.py --model vit_b_16 --data data/clean/
"""
import argparse, torch, json
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, ImageFolder
import numpy as np
from pathlib import Path


def main(model_name, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_name)(pretrained=True).to(device).eval()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    print(f"Model: {model_name}  |  Data: {data_dir}")
    print("Plug in subgroup-labelled dataset (e.g. FairFace or CelebA) and compute per-group accuracy.")
    # group_accs = {}
    # for group, indices in subgroup_map.items():
    #     subset = Subset(dataset, indices)
    #     loader = DataLoader(subset, batch_size=64)
    #     group_accs[group] = accuracy(model, loader, device)
    # max_gap = max(group_accs.values()) - min(group_accs.values())
    # print(f"Fairness gap: {max_gap:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_b_16")
    p.add_argument("--data", default="data/clean/")
    a = p.parse_args()
    main(a.model, a.data)
