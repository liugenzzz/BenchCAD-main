from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sample_custom_jsonl.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sample_custom_jsonl", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_is_reproducible_and_without_replacement():
    module = _load_module()
    records = [{"id": str(index)} for index in range(20)]

    first = module.sample_records(records, count=10, seed=42)
    second = module.sample_records(records, count=10, seed=42)

    assert first == second
    assert len({record["id"] for record in first}) == 10


def test_jsonl_round_trip_preserves_custom_fields(tmp_path):
    module = _load_module()
    source = tmp_path / "source.jsonl"
    output = tmp_path / "sampled.jsonl"
    records = [
        {
            "id": str(index),
            "images": [f"/renders/{index}/composite.png"],
            "conversations": [{"from": "human", "value": "<image>\nGenerate CAD."}],
            "trace": {"step": f"/steps/{index}.step"},
        }
        for index in range(5)
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = module.read_jsonl(source)
    sampled = module.sample_records(loaded, count=3, seed=7)
    module.write_jsonl(output, sampled, overwrite=False)

    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert result == sampled
    assert all("images" in record and "trace" in record for record in result)


def test_rewrite_image_paths_changes_only_matching_prefix():
    module = _load_module()
    records = [
        {
            "id": "163670",
            "images": [
                "/trainspace/renders/0099/00994712/composite.png",
                "/trainspace_backup/renders/unchanged.png",
            ],
        },
        {"id": "without-images"},
    ]

    replacements = module.rewrite_image_paths(
        records,
        source_prefix="/trainspace",
        target_prefix="/app/llamafactory/trainspace",
    )

    assert replacements == 1
    assert records[0]["images"] == [
        "/app/llamafactory/trainspace/renders/0099/00994712/composite.png",
        "/trainspace_backup/renders/unchanged.png",
    ]


def test_split_sampled_records_removes_exact_rows_and_keeps_remaining_order():
    module = _load_module()
    records = [
        {"id": "duplicate", "value": 1},
        {"id": "duplicate", "value": 1},
        {"id": "unique", "value": 2},
    ]

    sampled, remaining = module.split_sampled_records(records, count=1, seed=1)

    assert len(sampled) == 1
    assert len(remaining) == 2
    assert len(sampled) + len(remaining) == len(records)
    assert remaining == [records[index] for index in range(3) if records[index] is not sampled[0]]


def test_replace_jsonl_creates_backup_and_rewrites_source(tmp_path):
    module = _load_module()
    source = tmp_path / "source.jsonl"
    original = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    remaining = [{"id": "a"}, {"id": "c"}]
    source.write_text(
        "\n".join(json.dumps(record) for record in original) + "\n",
        encoding="utf-8",
    )

    backup = module.replace_jsonl_with_backup(source, remaining)

    assert module.read_jsonl(source) == remaining
    assert module.read_jsonl(backup) == original
    assert backup.name == "source.jsonl.bak"
