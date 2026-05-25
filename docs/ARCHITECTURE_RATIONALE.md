# Architecture Rationale

## Why ViT

Vision Transformers are useful for robustness studies because their global attention mechanism behaves differently from convolutional inductive bias under corruptions, blur, noise, and compression.

## Model Factory

`models/vit.py` provides a `load_model()` factory so experiments can swap model names without rewriting evaluation scripts. The default remains `vit_b_16`, but the config can be extended to compare DeiT, Swin, or other torchvision/timm backbones.

## Evaluation Axes

- robustness: accuracy under corruption type and severity
- fairness: subgroup accuracy gap
- explainability: attention rollout overlays for qualitative inspection

## Known Limitation

Attention rollout is diagnostic, not causal evidence. Interpret heatmaps beside quantitative metrics and failure cases.
