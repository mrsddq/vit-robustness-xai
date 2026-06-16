import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compute_mce import compute_mce


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["corruption", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


class MCETests(unittest.TestCase):
    def test_compute_mce_writes_mean_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
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

            with output_csv.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[-1]["corruption"], "mean")
            self.assertGreater(float(rows[-1]["ce"]), 0)

    def test_compute_mce_rejects_invalid_accuracy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_csv = tmp_path / "robustness.csv"
            output_csv = tmp_path / "mce.csv"
            _write_rows(input_csv, [{"corruption": "gaussian_noise", "severity": 1, "accuracy": 1.2}])

            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                compute_mce(input_csv, output_csv)

    def test_compute_mce_rejects_unknown_only_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_csv = tmp_path / "robustness.csv"
            output_csv = tmp_path / "mce.csv"
            _write_rows(input_csv, [{"corruption": "custom_shift", "severity": 1, "accuracy": 0.9}])

            with self.assertRaisesRegex(ValueError, "No recognized"):
                compute_mce(input_csv, output_csv)


if __name__ == "__main__":
    unittest.main()
