from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str = "vit_b_16", pretrained: bool = True) -> None:
        super().__init__()
        if model_name != "vit_b_16":
            raise ValueError("ViTClassifier currently supports model_name='vit_b_16'.")
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

    def forward(self, x):
        return self.backbone(x)


def load_model(name: str = "vit_b_16", pretrained: bool = True) -> nn.Module:
    if name == "vit_b_16":
        return ViTClassifier(name, pretrained)
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    raise ValueError(f"Unsupported model: {name}")
