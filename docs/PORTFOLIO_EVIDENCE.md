# Portfolio Evidence Plan

This project should be shown as robustness and explainability engineering around ViT-style classifiers. Do not claim robustness gains or fairness improvements without a documented run.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.generate_corruptions --config configs/robustness.yaml
python -m scripts.robustness.evaluate_robustness --config configs/robustness.yaml
python -m scripts.xai.generate_heatmaps --config configs/robustness.yaml --output outputs/heatmaps
python -m scripts.compute_mce --input outputs/metrics
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/corruption-grid.png` | Shows clean vs corrupted examples across severity levels. |
| `assets/heatmap-grid.png` | Shows explainability overlays for correct and incorrect predictions. |
| `outputs/metrics/robustness.csv` | Records accuracy by corruption and severity. |
| `outputs/metrics/mce.json` | Records mean corruption error or comparable robustness summary. |
| `docs/RESULTS.md` | Summarizes only verified experiment runs. |

## Demo Narrative

1. Start with clean-image baseline behavior.
2. Show corruption families and severity levels from `configs/robustness.yaml`.
3. Compare model performance under corruption.
4. Use heatmaps to explain where the model attends in success and failure cases.
5. Discuss limitations around dataset bias and subgroup evaluation.

## Evidence Checklist Before Pinning

- [ ] Clean and corrupted sample grid added to `assets/`.
- [ ] Heatmap grid added to `assets/`.
- [ ] Real robustness table added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
- [ ] Dataset and subgroup limitations documented.
