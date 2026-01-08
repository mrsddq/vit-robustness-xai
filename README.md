# ViT Robustness & Explainability (XAI)

Research code for evaluating robustness and fairness of Vision Transformers (ViTs), with explainability analysis using Grad-CAM, SHAP, and attention rollout.

## Related Publications

| Paper | Role |
|---|---|
| Continuous Robustness and Fairness Evaluation for Deployed Vision Transformers | _[add contribution]_ |
| Interpretable Visual Reasoning through Human Feedback and Symbolic Explanation | _[add contribution]_ |
| RLHF for Trustworthy Vision-Language Models | _[add contribution]_ |

> Add paper links/DOIs and your specific contribution per paper above.

## Results

| Evaluation | Value |
|---|---|
| Baseline accuracy (clean ImageNet val) | _add_ |
| Accuracy under Gaussian noise (severity 3) | _add_ |
| Accuracy under blur corruption (severity 3) | _add_ |
| Mean corruption robustness (mCE) | _add_ |
| Fairness gap (subgroup accuracy delta) | _add_ |

## Quickstart

```bash
git clone https://github.com/your-username/vit-robustness-xai
cd vit-robustness-xai
pip install -r requirements.txt
```

## Robustness Evaluation

Tests ViT performance under 15 corruption types at 5 severity levels (ImageNet-C protocol).

```bash
python scripts/robustness/evaluate_robustness.py \
  --model vit_b_16 \
  --data data/corrupted/ \
  --config configs/robustness.yaml
```

## XAI — Attention & Grad-CAM Heatmaps

```bash
python scripts/xai/generate_heatmaps.py \
  --model vit_b_16 \
  --input data/clean/ \
  --output outputs/heatmaps/ \
  --method gradcam
```

Supported methods: `gradcam`, `attention_rollout`, `shap`

## Fairness Evaluation

```bash
python scripts/evaluation/fairness_eval.py \
  --model vit_b_16 \
  --data data/clean/ \
  --groups configs/subgroups.yaml
```

## Sample Outputs

| File | Contents |
|---|---|
| `assets/01_paper_title_page.png` | Title page screenshot of core paper(s) |
| `assets/02_gradcam_heatmap.png` | ViT image + Grad-CAM attention overlay |
| `assets/03_robustness_chart.png` | Accuracy vs corruption severity |
| `assets/04_contribution_note.png` | Your contribution statement per paper |

## Architecture Note

All experiments use pretrained `vit_b_16` (ViT-B/16) from `torchvision.models`. Fine-tuning is optional — robustness is evaluated on the pretrained checkpoint unless noted.

## Limitations

- Robustness evaluation on ImageNet-C requires the full corrupted dataset (~30GB) — not included
- SHAP explanations are slow at full image resolution; subsample for exploration
- Fairness metrics require labelled demographic subgroups — use FairFace or CelebA

## Environment

```
Python 3.10
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
grad-cam==1.4.8
shap==0.43.0
numpy==1.26.0
matplotlib==3.8.0
```
