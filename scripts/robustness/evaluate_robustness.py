"""Evaluate ViT robustness under corruptions.
Usage: python scripts/robustness/evaluate_robustness.py --model vit_b_16 --data data/corrupted/
"""
import argparse, yaml, torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, ImageFolder
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def main(model_name, data_dir, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, model_name)(pretrained=True).to(device)
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    results = []
    data_dir = Path(data_dir)
    for corruption in cfg["evaluation"]["corruption_types"]:
        for severity in cfg["evaluation"]["severity_levels"]:
            corrupt_path = data_dir / corruption / str(severity)
            if not corrupt_path.exists():
                print(f"  Skipping {corruption} s{severity} — path not found")
                continue
            loader = DataLoader(ImageFolder(str(corrupt_path), transform=transform),
                                batch_size=cfg["evaluation"]["batch_size"], num_workers=4)
            acc = accuracy(model, loader, device)
            results.append({"corruption": corruption, "severity": severity, "accuracy": acc})
            print(f"  {corruption} s{severity}: {acc:.4f}")
    if results:
        df = pd.DataFrame(results)
        out = Path(cfg["logging"]["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "robustness_results.csv", index=False)
        print(f"\nSaved: {out}/robustness_results.csv")
        print(f"Mean accuracy across all corruptions: {df.accuracy.mean():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_b_16")
    p.add_argument("--data", default="data/corrupted/")
    p.add_argument("--config", default="configs/robustness.yaml")
    a = p.parse_args()
    main(a.model, a.data, a.config)
