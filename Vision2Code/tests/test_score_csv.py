"""Tests for the final Vision2Code CSV score summary."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.score_csv import SCORE_CSV_NAME, write_score_summary  # noqa: E402


def test_write_score_summary_groups_models_and_score_types(tmp_path):
    rows = [
        {
            "model": "model-b",
            "score_type": "iou",
            "status": "exec_fail",
            "iou": 0.0,
            "score": 0.0,
        },
        {
            "model": "model-a",
            "score_type": "iou",
            "status": "ok",
            "iou": 0.5,
            "score": 0.5,
        },
        {
            "model": "model-a",
            "score_type": "iou",
            "status": "ok",
            "iou": 1.0,
            "score": 1.0,
        },
        {
            "model": "model-a",
            "score_type": "iou",
            "status": "score_fail",
            "iou": 0.0,
            "score": 0.0,
        },
        {
            "model": "model-a",
            "score_type": "composite",
            "status": "ok",
            "iou": 0.5,
            "score": 0.8,
        },
    ]
    output_path = tmp_path / SCORE_CSV_NAME

    write_score_summary(rows, output_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    assert output_path.name == "iou_accruary.csv"
    assert result == [
        {
            "model": "model-a",
            "score_type": "composite",
            "records": "1",
            "mean_iou": "0.500000",
            "mean_score": "0.800000",
        },
        {
            "model": "model-a",
            "score_type": "iou",
            "records": "2",
            "mean_iou": "0.750000",
            "mean_score": "0.750000",
        },
        {
            "model": "model-b",
            "score_type": "iou",
            "records": "0",
            "mean_iou": "",
            "mean_score": "",
        },
    ]
