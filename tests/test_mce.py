import csv

import pytest

from scripts.compute_mce import compute_mce


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["corruption", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def test_compute_mce_writes_mean_row(tmp_path):
    input_csv = tmp_path / "robustness.csv"
    output_csv = tmp_path / "mce.csv"
    _write_rows(
        input_csv,
        [
            {"corruption": "gaussian_noise", "severity": 1, "accuracy": 0.50},
            {"corruption": "gaussian_noise", "severity": 2, "accuracy": 0.40},
        ],
    )

    compute_mce(input_csv, output_csv)

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert rows[-1]["corruption"] == "mean"
    assert float(rows[-1]["ce"]) > 0


def test_compute_mce_rejects_invalid_accuracy(tmp_path):
    input_csv = tmp_path / "robustness.csv"
    output_csv = tmp_path / "mce.csv"
    _write_rows(input_csv, [{"corruption": "gaussian_noise", "severity": 1, "accuracy": 1.2}])

    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_mce(input_csv, output_csv)


def test_compute_mce_rejects_unknown_only_input(tmp_path):
    input_csv = tmp_path / "robustness.csv"
    output_csv = tmp_path / "mce.csv"
    _write_rows(input_csv, [{"corruption": "custom_shift", "severity": 1, "accuracy": 0.9}])

    with pytest.raises(ValueError, match="No recognized"):
        compute_mce(input_csv, output_csv)
