"""Audit subgroup accuracy from recorded predictions with sample counts and uncertainty."""
import argparse
import csv
import json
import math
from pathlib import Path


def wilson_interval(correct, total, z=1.96):
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "group", "label", "prediction"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CSV requires sample_id, group, label, prediction")
        rows = list(reader)
    if not rows:
        raise ValueError("No predictions to evaluate")
    seen, groups = set(), {}
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("Prediction fields must not be empty")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id would bias subgroup counts")
        seen.add(row["sample_id"])
        counts = groups.setdefault(row["group"], {"samples": 0, "correct": 0})
        counts["samples"] += 1
        counts["correct"] += row["label"] == row["prediction"]
    for counts in groups.values():
        counts["accuracy"] = counts["correct"] / counts["samples"]
        counts["accuracy_wilson_95"] = wilson_interval(counts["correct"], counts["samples"])
    scores = [value["accuracy"] for value in groups.values()]
    return {"samples": len(rows), "groups": groups, "accuracy_gap": max(scores) - min(scores),
            "interpretation": "Observed subgroup accuracy gap; not a causal fairness conclusion. Review sample counts and intervals."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default="outputs/metrics/subgroups.json")
    a = p.parse_args()
    report = evaluate_predictions(a.predictions)
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))
