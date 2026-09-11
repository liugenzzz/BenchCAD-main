from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "eval_local_model.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_local_model", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_loading_transformers():
    module = _load_module()

    assert module.ROOT == ROOT


def test_resolve_path_keeps_absolute_and_roots_relative_paths():
    module = _load_module()
    absolute = ROOT / "test_data"

    assert module.resolve_path(absolute) == absolute
    assert module.resolve_path(Path("test_data")) == ROOT / "test_data"


def test_select_records_filters_then_limits_with_seed():
    module = _load_module()
    records = [
        {"record_id": "a"},
        {"record_id": "b"},
        {"record_id": "c"},
        {"record_id": "d"},
    ]

    selected = module.select_records(
        records,
        record_ids=["a", "b", "c"],
        limit=2,
        seed=7,
    )

    assert [record["record_id"] for record in selected] == ["b", "a"]


def test_fallback_prompt_contains_system_user_and_generation_hint():
    module = _load_module()

    prompt = module.fallback_prompt_text("system text", "user text")

    assert "system text" in prompt
    assert "user text" in prompt
    assert "CadQuery Python code" in prompt


def test_discover_parquet_files_returns_sorted_subset(tmp_path):
    module = _load_module()
    (tmp_path / "code_gen-00002-of-00018.parquet").write_text("")
    (tmp_path / "code_gen-00000-of-00018.parquet").write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "code_gen-00001-of-00018.parquet").write_text("")

    files = module.discover_parquet_files(tmp_path, max_shards=2)

    assert [file.name for file in files] == [
        "code_gen-00000-of-00018.parquet",
        "code_gen-00001-of-00018.parquet",
    ]


def test_records_path_detection(tmp_path):
    module = _load_module()
    data_dir = tmp_path / "data"

    assert not module.has_materialized_records(data_dir)

    data_dir.mkdir()
    (data_dir / "records.jsonl").write_text("{}\n")

    assert module.has_materialized_records(data_dir)


def test_output_paths_include_raw_response_file(tmp_path):
    module = _load_module()

    paths = module.output_paths(tmp_path, "local/checkpoint-4596", "washer_000001")

    assert paths["raw"] == tmp_path / "outputs" / "local_checkpoint-4596" / "washer_000001.raw.txt"


def test_code_defines_result_detects_required_binding():
    module = _load_module()

    assert module.code_defines_result(
        "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)"
    )
    assert not module.code_defines_result(
        "import cadquery as cq\npart = cq.Workplane('XY').box(1, 1, 1)"
    )


def test_normalize_record_accepts_custom_training_schema():
    module = _load_module()
    record = {
        "id": "163670",
        "images": ["renders/0099/00994712/composite.png"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nGenerate the CADQuery code. Just the code.",
            },
            {
                "from": "gpt",
                "value": "import cadquery as cq\nsolid = cq.Workplane('XY').box(1, 2, 3)",
            },
        ],
        "trace": {"step": "steps/0099/00994712.step"},
    }

    normalized = module.normalize_record(record, 1)

    assert normalized["record_id"] == "163670"
    assert normalized["step_path"] == "steps/0099/00994712.step"
    assert normalized["user_text"] == "Generate the CADQuery code. Just the code."
    assert normalized["gt_code"].endswith("box(1, 2, 3)")


def test_normalize_record_accepts_minimal_llamafactory_sharegpt_schema():
    module = _load_module()
    record = {
        "conversations": [
            {"from": "human", "value": "<image>\nGenerate CadQuery code."},
            {
                "from": "gpt",
                "value": "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)",
            },
        ],
        "images": ["cad_images/dome_cap_009388_s20260505.png"],
    }

    normalized = module.normalize_record(record, 1)

    assert normalized["record_id"] == "dome_cap_009388_s20260505"
    assert normalized["gt_code"].startswith("import cadquery")
    assert "step_path" not in normalized


def test_ground_truth_step_is_generated_once_and_cached(tmp_path, monkeypatch):
    module = _load_module()
    import benchcad_core.scoring.exec_cq as exec_cq

    calls = []

    def fake_execute(code, step_path, timeout):
        calls.append((code, step_path, timeout))
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_bytes(b"STEP")

    monkeypatch.setattr(exec_cq, "execute_cq_to_step", fake_execute)
    record = {
        "record_id": "dome_cap_009388_s20260505",
        "gt_code": (
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').box(1, 2, 3)  # UTF-8: ↔"
        ),
    }

    first = module.ground_truth_step_for_record(
        record,
        data_dir=tmp_path,
        results_root=tmp_path / "results",
        exec_timeout=123,
    )
    second = module.ground_truth_step_for_record(
        record,
        data_dir=tmp_path,
        results_root=tmp_path / "results",
        exec_timeout=123,
    )

    assert first == second
    assert first.read_bytes() == b"STEP"
    assert len(calls) == 1
    assert calls[0][2] == 123


def test_read_records_file_accepts_json_array_and_jsonl(tmp_path):
    module = _load_module()
    json_file = tmp_path / "train.json"
    jsonl_file = tmp_path / "train.jsonl"
    json_file.write_text('[{"id": "a"}, {"id": "b"}]', encoding="utf-8")
    jsonl_file.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")

    assert [row["id"] for row in module.read_records_file(json_file)] == ["a", "b"]
    assert [row["id"] for row in module.read_records_file(jsonl_file)] == ["a", "b"]


def test_ensure_result_binding_adapts_deepcad_solid_name():
    module = _load_module()
    code = "import cadquery as cq\nsolid = cq.Workplane('XY').box(1, 2, 3)"

    adapted = module.ensure_result_binding(code)

    assert adapted.endswith("result = solid\n")
    assert module.code_defines_result(adapted)


def test_resume_skips_completed_and_optionally_reruns_failed():
    module = _load_module()
    records = [
        {"record_id": "ok"},
        {"record_id": "failed"},
        {"record_id": "different-score"},
        {"record_id": "new"},
    ]
    existing = {
        ("local/model", "ok"): {"status": "ok", "score_type": "iou"},
        ("local/model", "failed"): {
            "status": "exec_fail",
            "score_type": "iou",
        },
        ("local/model", "different-score"): {
            "status": "ok",
            "score_type": "composite",
        },
    }

    pending, skipped = module.pending_records_for_resume(
        records,
        existing,
        model_name="local/model",
        score_type="iou",
        rerun_failed=False,
    )
    retry_pending, retry_skipped = module.pending_records_for_resume(
        records,
        existing,
        model_name="local/model",
        score_type="iou",
        rerun_failed=True,
    )

    assert [record["record_id"] for record in pending] == [
        "different-score",
        "new",
    ]
    assert skipped == 2
    assert [record["record_id"] for record in retry_pending] == [
        "failed",
        "different-score",
        "new",
    ]
    assert retry_skipped == 1


def test_write_results_checkpoint_round_trip(tmp_path):
    module = _load_module()
    results_path = tmp_path / "results.jsonl"
    rows = {
        ("local/model", "a"): {
            "model": "local/model",
            "record_id": "a",
            "status": "ok",
        }
    }

    module.write_results(results_path, rows)

    assert module.read_results(results_path) == rows
    assert not list(tmp_path.glob("*.tmp"))
