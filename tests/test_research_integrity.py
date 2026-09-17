import csv
import json
import numpy as np
import pytest
import torch
from torch import nn
from scripts.xai.generate_heatmaps import attention_rollout
from scripts.evaluation.fairness_eval import evaluate_predictions
from scripts.compute_mce import compute_mce


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(4, 2, batch_first=True)
    def forward(self, x):
        return self.self_attention(x, x, x, need_weights=False)[0]


class TinyVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Block(), Block()])
    def forward(self, x):
        for block in self.encoder.layers:
            x = block(x)
        return x


def test_attention_rollout_uses_weights_and_removes_hooks():
    torch.manual_seed(1)
    model = TinyVisionTransformer().train()
    mask = attention_rollout(model, torch.rand(5, 4))
    assert mask.shape == (2, 2)
    assert np.isfinite(mask).all() and mask.max() <= 1 and mask.min() >= 0
    assert model.training
    for block in model.encoder.layers:
        assert not block.self_attention._forward_hooks
        assert not block.self_attention._forward_pre_hooks


def test_attention_hooks_cleaned_on_failure():
    model = TinyVisionTransformer()
    with pytest.raises(Exception):
        attention_rollout(model, torch.rand(5, 3))
    assert not model.encoder.layers[0].self_attention._forward_hooks


def test_subgroup_counts_intervals_and_duplicate_rejection(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("sample_id,group,label,prediction\n1,a,yes,yes\n2,a,no,yes\n3,b,no,no\n")
    report = evaluate_predictions(path)
    assert report["accuracy_gap"] == 0.5
    assert report["groups"]["a"]["samples"] == 2
    low, high = report["groups"]["a"]["accuracy_wilson_95"]
    assert low < 0.5 < high
    path.write_text(path.read_text() + "1,a,yes,yes\n")
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_predictions(path)


def test_partial_mce_is_labelled_and_duplicates_fail(tmp_path):
    source, output = tmp_path / "in.csv", tmp_path / "out.csv"
    source.write_text("corruption,severity,accuracy\ngaussian_noise,1,0.5\n")
    compute_mce(source, output)
    rows = list(csv.DictReader(output.open()))
    assert rows[-1]["benchmark_complete"] == "False"
    source.write_text(source.read_text() + "gaussian_noise,1,0.4\n")
    with pytest.raises(ValueError, match="Duplicate"):
        compute_mce(source, output)



def test_rollout_with_real_tiny_torchvision_vit_wrapper():
    from torchvision.models.vision_transformer import VisionTransformer
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = VisionTransformer(image_size=16, patch_size=8, num_layers=2, num_heads=2, hidden_dim=8, mlp_dim=16, num_classes=2)
        def forward(self, images):
            return self.backbone(images)
    mask = attention_rollout(Wrapper(), torch.rand(3, 16, 16))
    assert mask.shape == (2, 2)
    assert np.isfinite(mask).all()
