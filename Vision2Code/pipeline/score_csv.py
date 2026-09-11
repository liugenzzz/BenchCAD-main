"""Export final Vision2Code score summaries as CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path


SCORE_CSV_NAME = "iou_accruary.csv"


def write_score_summary(rows: Iterable[dict], output_path: Path) -> None:
    """Write one aggregate score row per (model, score type), excluding failures."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["model"]), str(row.get("score_type", "iou")))
        buckets.setdefault(key, []).append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "score_type", "records", "mean_iou", "mean_score"),
        )
        writer.writeheader()
        for (model, score_type), score_rows in sorted(buckets.items()):
            successful_rows = [row for row in score_rows if row.get("status") == "ok"]
            count = len(successful_rows)
            if count:
                mean_iou = sum(
                    float(row.get("iou", 0.0)) for row in successful_rows
                ) / count
                mean_score = sum(
                    float(row.get("score", row.get("iou", 0.0)))
                    for row in successful_rows
                ) / count
            else:
                mean_iou = None
                mean_score = None
            writer.writerow(
                {
                    "model": model,
                    "score_type": score_type,
                    "records": count,
                    "mean_iou": "" if mean_iou is None else f"{mean_iou:.6f}",
                    "mean_score": "" if mean_score is None else f"{mean_score:.6f}",
                }
            )
