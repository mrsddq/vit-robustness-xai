# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Hardware: record GPU/CPU model, CUDA version, RAM, and OS for each run

## Data Contract

Expected datasets are not committed. Each experiment should record:

- dataset name and version
- corruption benchmark source
- clean validation split
- subgroup label source, if fairness analysis is used
- checksum or DVC hash for any local dataset snapshot

## Run Order

1. Validate `configs/robustness.yaml`.
2. Run clean baseline accuracy.
3. Run corruption sweep for all configured severities.
4. Run subgroup gap evaluation when labels are available.
5. Generate attention rollout overlays for qualitative review.
6. Fill `docs/EXPERIMENT_CARD.md` and commit metrics under `outputs/metrics/`.

## Non-Benchmark Artifact

`outputs/metrics/smoke_test_results.csv` is a schema example only. Do not cite it as model performance.
