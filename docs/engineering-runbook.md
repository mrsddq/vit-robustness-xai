# ViT robustness and explainability engineering runbook

This repository provides corruption evaluation, attention rollout and subgroup
prediction analysis. Follow the [README setup](../README.md#setup),
[robustness commands](../README.md#robustness-evaluation),
[explainability commands](../README.md#explainability), and
[subgroup CSV workflow](../README.md#subgroup-evaluation).

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests exercise real attention hooks and a tiny torchvision ViT without pretrained
weights, including wrapper support, failure cleanup and mode restoration. They
also check subgroup counts/intervals and corruption-row coverage/validation. They
do not establish pretrained-model robustness, explanation quality or fairness.

## Data and artifact contract

- Real image evaluation requires an appropriate dataset and matching classifier
  labels. The full dependency stack and first pretrained-weight retrieval are
  separate from offline verification; do not substitute arbitrary class-folder
  indices and treat the result as an ImageNet benchmark.
- Keep source images, downloaded weights, predictions and generated heatmaps out
  of git. Retain model identity, transform settings, data split and command with
  each report. Overlays use the classifier's resize/center-crop geometry.
- Subgroup input CSV columns are `sample_id,group,label,prediction`; each row has
  one group and sample IDs must be unique. Reports include sample counts, accuracy,
  Wilson intervals and observed maximum-minus-minimum gap. Inspect small groups
  and uncertainty; a descriptive gap is not a causal fairness conclusion.
- Corruption CSV input must identify valid corruption/severity pairs with finite
  accuracies. `benchmark_complete: false` marks a subset diagnostic. The fixed
  AlexNet reference is only meaningful under the matching ImageNet-C convention.
- Attention rollout is a diagnostic, not proof of causal model reasoning. Full
  ImageNet-C evaluation and real-data explanation assessment remain unrun.

Use the [experiment card](EXPERIMENT_CARD.md) to record real runs, including
limitations and failure cases; do not publish synthetic tests as benchmark scores.
