# Ablation Plan

| Experiment | Variable | Fixed Controls | Expected Signal |
|---|---|---|---|
| baseline | clean validation data | model, preprocessing | reference accuracy |
| corruption severity | severity 1-5 | corruption type, model | degradation curve |
| corruption type | ImageNet-C category | severity, model | sensitivity profile |
| model family | ViT vs CNN/Swin | dataset, preprocessing | architecture robustness comparison |
| explanation method | rollout vs GradCAM/SHAP | image set, model | qualitative agreement/disagreement |

## Reporting Standard

Each ablation should include config path, command, hardware, seed, metrics CSV, and at least one qualitative failure case.
