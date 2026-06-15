# ViT Robustness and Explainability

[![CI](https://github.com/mrsddq/vit-robustness-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/vit-robustness-xai/actions/workflows/ci.yml)

Research-oriented toolkit for evaluating Vision Transformer robustness, subgroup performance, and visual explanations.

This repository is designed as a portfolio-quality ML research engineering project. It provides reusable scripts and experiment structure without claiming unverified results or shipping large datasets.

## Highlights

- ImageNet-C style corruption robustness evaluation
- Subgroup accuracy gap helper for fairness analysis
- Attention rollout heatmap generation
- Shared model, transform, config, and metric utilities
- Experiment card template for reproducible reporting
- Pytest checks for config and utility behavior

## Structure

```text
configs/
  robustness.yaml
docs/
  ABLATION_PLAN.md
  ARCHITECTURE_RATIONALE.md
  EXPERIMENT_CARD.md
  REPRODUCIBILITY.md
  DEPLOYMENT_NOTES.md
models/
  vit.py
scripts/
  common.py
  robustness/evaluate_robustness.py
  evaluation/fairness_eval.py
  xai/generate_heatmaps.py
tests/
  test_common.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Robustness Evaluation

Expected ImageNet-C style layout:

```text
data/corrupted/
  gaussian_noise/
    1/
    2/
    3/
    4/
    5/
```

Run:

```bash
python -m scripts.robustness.evaluate_robustness ^
  --model vit_b_16 ^
  --data data/corrupted ^
  --config configs/robustness.yaml
```

Metrics are saved to `outputs/metrics/robustness_results.csv`.

## Explainability

Generate attention rollout overlays:

```bash
python -m scripts.xai.generate_heatmaps ^
  --model vit_b_16 ^
  --input data/clean ^
  --output outputs/heatmaps ^
  --method attention_rollout
```

## Fairness Evaluation

```bash
python -m scripts.evaluation.fairness_eval ^
  --model vit_b_16 ^
  --data data/clean ^
  --groups configs/subgroups.yaml
```

Connect a subgroup-labelled dataset such as FairFace or CelebA before reporting fairness metrics.

## Testing

```bash
pytest
```

## Results

No verified public metrics are committed yet. Use [docs/EXPERIMENT_CARD.md](docs/EXPERIMENT_CARD.md) to record real experiment runs.

Research support docs:

- [Portfolio Evidence Plan](docs/PORTFOLIO_EVIDENCE.md)
- [Reproducibility Plan](docs/REPRODUCIBILITY.md)
- [Architecture Rationale](docs/ARCHITECTURE_RATIONALE.md)
- [Ablation Plan](docs/ABLATION_PLAN.md)
- [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

`outputs/metrics/smoke_test_results.csv` is a schema artifact only, not a benchmark.

Recommended artifacts:

- `assets/robustness-chart.png`
- `assets/attention-rollout.png`
- `assets/fairness-gap-table.png`
- `assets/failure-case.png`

## Limitations

- ImageNet-C and subgroup-labelled datasets are not included.
- Full robustness evaluation can require substantial storage and GPU time.
- Attention rollout is not a causal explanation; use it as a diagnostic signal, not proof of model reasoning.
