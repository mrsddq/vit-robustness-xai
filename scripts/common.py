from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as T
import yaml


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_yaml(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    return config


def imagenet_transform(image_size=224):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_torchvision_model(model_name, pretrained=True):
    constructor = getattr(models, model_name)
    if pretrained:
        weights_enum_name = f"{model_name.upper()}_Weights"
        weights_enum = getattr(models, weights_enum_name, None)
        weights = weights_enum.DEFAULT if weights_enum else "DEFAULT"
        return constructor(weights=weights)
    return constructor(weights=None)


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def subgroup_gap(group_scores):
    if not group_scores:
        raise ValueError("group_scores must not be empty")
    return max(group_scores.values()) - min(group_scores.values())
