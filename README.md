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
source .venv/bin/activate
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
python -m scripts.robustness.evaluate_robustness --model vit_b_16 --data data/corrupted --config configs/robustness.yaml
```

Metrics are saved to `outputs/metrics/robustness_results.csv`.

## Explainability

Generate attention rollout overlays:

```bash
python -m scripts.xai.generate_heatmaps --model vit_b_16 --input data/clean --output outputs/heatmaps --method attention_rollout
```

## Subgroup evaluation

Record held-out predictions in a CSV with `sample_id,group,label,prediction`.
Each sample has one group; use a compound group for intersectional analysis.

```bash
python -m scripts.evaluation.fairness_eval --predictions data/predictions.csv --output outputs/metrics/subgroups.json
```

This implemented evaluator reports counts, correct predictions, accuracy, Wilson
95% intervals, and maximum-minus-minimum accuracy gap. It rejects duplicate IDs
and missing values. A gap describes the observed sample, not proof of causality,
discrimination, or performance on unrepresented populations.

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

## Executable research-integrity checks

Install `requirements-test.txt` and run `python -m pytest -q` for CPU/offline tests.
Attention rollout captures per-head weights from PyTorch tuple outputs, restores
model mode, and removes hooks even after failure. It normalizes residual attention
row-wise and supports the repository's wrapped ViT. Displayed overlays use the
same resize/center-crop geometry as classifier inputs. Tests use a tiny randomly
initialized attention model; they do not download pretrained weights.

The corruption-error utility rejects unknown corruptions, repeated severities,
and nonfinite/out-of-range accuracy. CSV output includes coverage and
`benchmark_complete`: false means a subset diagnostic, not full ImageNet-C mCE.
The fixed reference errors apply to the documented ImageNet-C/AlexNet convention;
do not interpret ratios on different datasets or class mappings as comparable
benchmark results. Full robustness evaluation and real-data explanations remain
unrun; no quality score is inferred from synthetic tests.
