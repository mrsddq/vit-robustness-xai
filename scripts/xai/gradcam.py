from __future__ import annotations

import numpy as np
import torch


def gradcam_heatmap(model: torch.nn.Module, input_tensor: torch.Tensor, target_category: int | None = None) -> np.ndarray:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError("pytorch-grad-cam is required for GradCAM heatmaps.") from exc

    target_layers = _target_layers(model)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)
    return grayscale[0]


def _target_layers(model: torch.nn.Module):
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
        return [backbone.encoder.layers[-1]]
    if hasattr(backbone, "layer4"):
        return [backbone.layer4[-1]]
    raise ValueError("No supported target layer found for GradCAM.")
