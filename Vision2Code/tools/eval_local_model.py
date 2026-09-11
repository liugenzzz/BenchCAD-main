#!/usr/bin/env python
"""Evaluate a locally trained HuggingFace VLM on Vision2Code.

This script intentionally lives outside the existing cloud-model runner. It
loads a local model directory, generates CadQuery code from the same Vision2Code
prompt/images, then reuses the official execution and scoring utilities.

Example:
    python tools/eval_local_model.py ^
        --model-path D:\models\my-vlm ^
        --data-dir data ^
        --out-dir results_local_my_vlm ^
        --limit 10 ^
        --dtype bfloat16 ^
        --device-map auto
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from benchcad_core.run_config import DEFAULT_EXEC_TIMEOUT, DEFAULT_MAX_TOKENS  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "test_data"
DEFAULT_OUT_DIR = ROOT / "results_local"


def resolve_path(path: str | Path) -> Path:
    """Resolve paths relative to Vision2Code/ to match the existing runner."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""
    if not path.exists():
        raise SystemExit(f"records.jsonl not found at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_records_file(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array/object or a JSONL custom-data file."""
    if not path.exists():
        raise SystemExit(f"data file not found at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return payload
        raise SystemExit(f"JSON data file must contain an object or object array: {path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def conversation_value(record: dict[str, Any], roles: set[str]) -> str | None:
    """Return the first non-empty conversation value for one of ``roles``."""
    for turn in record.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        value = turn.get("value", turn.get("content"))
        if role in roles and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_record_id(record: dict[str, Any]) -> str | None:
    """Infer a stable ID from a ShareGPT image path when no ID was exported."""
    images = record.get("images")
    candidates: list[Any] = []
    if isinstance(images, list):
        candidates.extend(images)
    elif isinstance(images, (str, Path)):
        candidates.append(images)
    candidates.append(record.get("composite_png"))
    for candidate in candidates:
        if isinstance(candidate, (str, Path)) and str(candidate).strip():
            stem = Path(candidate).stem.strip()
            if stem:
                return stem
    return None


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Map the training-data schema and native schema to one internal record."""
    normalized = dict(record)
    record_id = (
        normalized.get("record_id")
        or normalized.get("id")
        or infer_record_id(normalized)
    )
    if record_id is None or not str(record_id).strip():
        raise SystemExit(
            f"record {index} has no record_id or id and no usable image filename; "
            "cannot identify the sample"
        )
    normalized["record_id"] = str(record_id)

    if not normalized.get("step_path"):
        trace = normalized.get("trace")
        if isinstance(trace, dict) and trace.get("step"):
            normalized["step_path"] = trace["step"]
    if not normalized.get("user_text"):
        human_text = conversation_value(normalized, {"human", "user"})
        if human_text:
            # Image placeholders are supplied separately to AutoProcessor.
            normalized["user_text"] = human_text.replace("<image>", "").strip()
    if not normalized.get("gt_code"):
        gt_code = conversation_value(normalized, {"gpt", "assistant"})
        if gt_code:
            normalized["gt_code"] = gt_code
    if not normalized.get("step_path") and not normalized.get("gt_code"):
        raise SystemExit(
            f"record {normalized['record_id']} has neither step_path/trace.step nor "
            "ground-truth code in conversations; cannot build scoring geometry"
        )
    return normalized


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and validate all records read from disk."""
    return [normalize_record(record, index) for index, record in enumerate(records, 1)]


def record_path(data_dir: Path, value: str | Path) -> Path:
    """Resolve a record-owned asset path, preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def has_materialized_records(data_dir: Path) -> bool:
    """Return whether a Vision2Code data dir has records.jsonl."""
    return (data_dir / "records.jsonl").exists()


def discover_parquet_files(
    parquet_dir: Path,
    max_shards: int | None = None,
) -> list[Path]:
    """Return sorted local code_gen parquet shards."""
    if not parquet_dir.exists():
        raise SystemExit(f"parquet directory not found: {parquet_dir}")
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet files found under {parquet_dir}")
    if max_shards is not None:
        files = files[:max_shards]
    return files


def materialize_parquet_dataset(
    *,
    parquet_dir: Path,
    out_dir: Path,
    max_shards: int | None = None,
    limit: int | None = None,
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    overwrite_steps: bool = False,
) -> list[dict[str, Any]]:
    """Convert local HF parquet shards into records.jsonl/codes/steps."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Materializing parquet data requires pandas in the active Python "
            "environment."
        ) from exc

    from benchcad_core.scoring.exec_cq import execute_cq_to_step

    parquet_files = discover_parquet_files(parquet_dir, max_shards=max_shards)
    print(f"materialize: reading {len(parquet_files)} parquet shard(s) from {parquet_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    if limit is not None:
        frame = frame.head(limit)
        print(f"materialize: limiting to first {len(frame)} parquet row(s)")

    (out_dir / "codes").mkdir(parents=True, exist_ok=True)
    (out_dir / "steps").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    skipped = 0
    for row in frame.itertuples(index=False):
        record_id = str(row.stem)
        code_path = f"codes/{record_id}.py"
        step_path = f"steps/{record_id}.step"
        code = str(row.code)
        code_file = out_dir / code_path
        step_file = out_dir / step_path
        code_file.write_text(code, encoding="utf-8")

        if overwrite_steps or not step_file.exists():
            try:
                execute_cq_to_step(code, step_file, timeout=exec_timeout)
            except Exception as exc:  # noqa: BLE001 - skip invalid GT rows.
                skipped += 1
                print(
                    f"  ! skip {record_id}: GT code failed to exec "
                    f"({type(exc).__name__})"
                )
                continue

        records.append(
            {
                "record_id": record_id,
                "family": getattr(row, "family", "") or "",
                "code_path": code_path,
                "step_path": step_path,
            }
        )

    records_jsonl = out_dir / "records.jsonl"
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"materialize: wrote {len(records)} record(s) -> {records_jsonl}")
    if skipped:
        print(f"materialize: skipped {skipped} row(s) whose GT code failed")
    return records


def ensure_data_records(
    *,
    data_dir: Path,
    parquet_dir: Path | None,
    rebuild_data: bool,
    max_shards: int | None,
    materialize_limit: int | None,
    materialize_exec_timeout: int,
) -> list[dict[str, Any]]:
    """Load records.jsonl, or build it from local parquet shards when requested."""
    if has_materialized_records(data_dir) and not rebuild_data:
        return normalize_records(read_jsonl(data_dir / "records.jsonl"))
    if parquet_dir is None:
        raise SystemExit(
            f"records.jsonl not found at {data_dir / 'records.jsonl'}; "
            "pass --parquet-dir to materialize local code_gen parquet shards."
        )
    return normalize_records(
        materialize_parquet_dataset(
            parquet_dir=parquet_dir,
            out_dir=data_dir,
            max_shards=max_shards,
            limit=materialize_limit,
            exec_timeout=materialize_exec_timeout,
            overwrite_steps=rebuild_data,
        )
    )


def select_records(
    records: list[dict[str, Any]],
    record_ids: list[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by record_id, then optionally cap or sample records."""
    selected = list(records)
    if record_ids:
        wanted = set(record_ids)
        selected = [record for record in selected if record["record_id"] in wanted]
    if limit is not None:
        limit = min(limit, len(selected))
        if seed is None:
            selected = selected[:limit]
        else:
            selected = random.Random(seed).sample(selected, limit)
    return selected


def fallback_prompt_text(system: str, user_text: str) -> str:
    """Prompt used when a processor does not provide a chat template."""
    return (
        f"{system}\n\n"
        "<image>\n"
        f"{user_text}\n\n"
        "Return only a fenced Python block containing CadQuery Python code."
    )


def chat_messages(
    system: str,
    user_text: str,
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    """Build a common chat-template message structure for multimodal processors."""
    user_content: list[dict[str, str]] = [
        {"type": "image", "image": str(path)} for path in image_paths
    ]
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def safe_model_name(model_name: str) -> str:
    """Return a filesystem-safe model label."""
    return (
        model_name.replace(":", "_")
        .replace("=", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


_WARNED: set[str] = set()


def warn_once(message: str) -> None:
    """Print a warning the first time it happens, then stay quiet."""
    if message in _WARNED:
        return
    _WARNED.add(message)
    print(f"\n[warn] {message}", file=sys.stderr, flush=True)


def output_paths(results_root: Path, model_name: str, record_id: str) -> dict[str, Path]:
    """Create and return per-record output paths."""
    base = results_root / "outputs" / safe_model_name(model_name)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "raw": base / f"{record_id}.raw.txt",
        "code": base / f"{record_id}.py",
        "step": base / f"{record_id}.step",
        "png": base / f"{record_id}.png",
    }


def _target_binds_name(target: ast.expr, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_binds_name(item, name) for item in target.elts)
    return False


def code_binds_name(code: str, name: str) -> bool:
    """Return whether Python code assigns a value to ``name``."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(_target_binds_name(target, name) for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if _target_binds_name(node.target, name):
                return True
        elif isinstance(node, ast.AugAssign):
            if _target_binds_name(node.target, name):
                return True
    return False


def code_defines_result(code: str) -> bool:
    """Return whether Python code assigns the required final solid to result."""
    return code_binds_name(code, "result")


def ensure_result_binding(code: str) -> str:
    """Adapt DeepCAD-style output whose final object is named ``solid``."""
    try:
        if code_defines_result(code):
            return code
        if code_binds_name(code, "solid"):
            return f"{code.rstrip()}\n\nresult = solid\n"
    except SyntaxError:
        pass
    return code


def record_image_paths(record: dict[str, Any], data_dir: Path) -> list[Path] | None:
    """Prompt images a record declares, or None when they are rendered lazily.

    Mirrors the resolution order in pipeline/prompt.py:build without rendering
    anything, so a preflight can check for missing files cheaply.
    """
    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else data_dir / path

    image_values = record.get("images") or []
    if isinstance(image_values, (str, Path)):
        image_values = [image_values]
    if image_values:
        return [resolve(value) for value in image_values]
    if record.get("composite_png"):
        return [resolve(record["composite_png"])]
    views = [
        resolve(record[f"view_{index}_png"])
        for index in range(4)
        if record.get(f"view_{index}_png")
    ]
    return views or None


def missing_image_paths(record: dict[str, Any], data_dir: Path) -> list[Path]:
    """Declared prompt images that are not on disk."""
    declared = record_image_paths(record, data_dir)
    if not declared:
        return []
    return [path for path in declared if not path.is_file()]


def preflight_images(
    records: list[dict[str, Any]],
    data_dir: Path,
) -> list[tuple[str, Path]]:
    """Report records whose prompt images are missing, before any model loads.

    A missing image raises inside generate() and is otherwise indistinguishable
    from a model failure -- and with --resume it gets checkpointed as one. Far
    better to say so up front than to spend a GPU-hour writing 400 identical
    FileNotFoundErrors.
    """
    missing: list[tuple[str, Path]] = []
    for record in records:
        for path in missing_image_paths(record, data_dir):
            missing.append((record["record_id"], path))
    return missing


def ground_truth_step_for_record(
    record: dict[str, Any],
    *,
    data_dir: Path,
    results_root: Path,
    exec_timeout: int,
    cache_root: Path | None = None,
) -> Path:
    """Resolve a supplied GT STEP or build and cache one from ShareGPT GT code."""
    step_path = record.get("step_path")
    if step_path:
        resolved = record_path(data_dir, step_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"ground-truth STEP not found for {record['record_id']}: {resolved}"
            )
        return resolved

    gt_code = str(record.get("gt_code") or "")
    from benchcad_core.scoring.exec_cq import execute_cq_to_step, extract_code

    code = ensure_result_binding(extract_code(gt_code))
    if not code.strip() or not code_defines_result(code):
        raise RuntimeError(
            f"ground-truth code for {record['record_id']} has no executable result"
        )

    digest = hashlib.sha256(gt_code.encode("utf-8")).hexdigest()[:12]
    safe_record_id = safe_model_name(str(record["record_id"]))
    cached_step = (
        (cache_root or results_root) / "ground_truth_steps" / f"{safe_record_id}_{digest}.step"
    )
    if not cached_step.is_file():
        execute_cq_to_step(code, cached_step, timeout=exec_timeout)
    return cached_step


def read_results(jsonl: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read existing result rows keyed by (model, record_id)."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["model"], row["record_id"])] = row
    return rows


def write_results(jsonl: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Atomically write result rows so interruption cannot corrupt the checkpoint."""
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=jsonl.name + ".",
            suffix=".tmp",
            dir=jsonl.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, jsonl)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def pending_records_for_resume(
    records: list[dict[str, Any]],
    existing_rows: dict[tuple[str, str], dict[str, Any]],
    *,
    model_name: str,
    score_type: str,
    rerun_failed: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Filter records already checkpointed for this model and score type."""
    pending: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        row = existing_rows.get((model_name, record["record_id"]))
        same_score = row is not None and row.get("score_type", "iou") == score_type
        failed = row is not None and row.get("status") != "ok"
        if same_score and not (rerun_failed and failed):
            skipped += 1
        else:
            pending.append(record)
    return pending, skipped


def torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Translate a CLI dtype name into a torch dtype or transformers' auto value."""
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def choose_device(torch_module: Any, device: str) -> str:
    """Choose a concrete torch device string."""
    if device != "auto":
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def first_parameter_device(model: Any, fallback: str) -> str:
    """Find the device where inputs should be placed."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def move_inputs_to_device(inputs: Any, device: str) -> Any:
    """Move tensor-like processor outputs to the target device."""
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


def load_images(image_paths: list[Path]) -> list[Any]:
    """Load prompt images lazily so importing this script does not need Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for local Vision2Code inference. "
            "Install it in the active environment."
        ) from exc

    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def load_processor_and_model(
    model_path: Path,
    *,
    device: str,
    dtype_name: str,
    device_map: str | None,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any, str]:
    """Load a local HuggingFace processor/model pair."""
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Local model evaluation requires torch and transformers in the "
            "active Python environment."
        ) from exc

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype(torch, dtype_name),
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    errors: list[str] = []
    class_names = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )
    model = None
    for class_name in class_names:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            model = model_cls.from_pretrained(str(model_path), **model_kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - report all loader attempts.
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    if model is None:
        detail = "\n".join(errors[-3:])
        raise SystemExit(f"could not load local model from {model_path}\n{detail}")

    target_device = choose_device(torch, device)
    if not device_map:
        model.to(target_device)
    model.eval()
    input_device = first_parameter_device(model, target_device)
    return processor, model, torch, input_device


def format_prompt(
    processor: Any,
    system: str,
    user_text: str,
    image_paths: list[Path],
) -> str:
    """Use the processor chat template when available, otherwise fall back."""
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                chat_messages(system, user_text, image_paths),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 - templates vary across local models.
            pass
    return fallback_prompt_text(system, user_text)


def decode_generated_text(
    processor: Any,
    output_ids: Any,
    input_ids: Any | None,
) -> tuple[str, dict[str, int | None]]:
    """Decode only newly generated tokens when input_ids are available."""
    generated_ids = output_ids
    prompt_tokens = None
    completion_tokens = None
    if input_ids is not None and hasattr(output_ids, "shape"):
        prompt_tokens = int(input_ids.shape[-1])
        generated_ids = output_ids[:, input_ids.shape[-1] :]
        completion_tokens = int(generated_ids.shape[-1])

    decoder = processor
    if not hasattr(decoder, "batch_decode") and hasattr(processor, "tokenizer"):
        decoder = processor.tokenizer
    text = decoder.batch_decode(generated_ids, skip_special_tokens=True)[0]
    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": None,
        "total_tokens": total_tokens,
    }
    return text, usage


class LocalVisionLanguageModel:
    """Small wrapper around a locally loaded HuggingFace vision-language model."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        dtype_name: str,
        device_map: str | None,
        trust_remote_code: bool,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        (
            self.processor,
            self.model,
            self.torch,
            self.input_device,
        ) = load_processor_and_model(
            model_path,
            device=device,
            dtype_name=dtype_name,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    def generate(
        self,
        *,
        system: str,
        user_text: str,
        image_paths: list[Path],
    ) -> tuple[str, dict[str, int | None]]:
        """Generate raw model text for one Vision2Code prompt."""
        images = load_images(image_paths)
        prompt = format_prompt(self.processor, system, user_text, image_paths)
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        inputs = move_inputs_to_device(inputs, self.input_device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        return decode_generated_text(
            self.processor,
            output_ids,
            inputs.get("input_ids"),
        )


def run_record(
    *,
    record: dict[str, Any],
    data_dir: Path,
    results_root: Path,
    model_name: str,
    local_model: LocalVisionLanguageModel,
    score: str,
    exec_timeout: int,
    render_png: bool = True,
    iou_timeout: float | None = None,
    gt_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run and score one local-model Vision2Code record."""
    from benchcad_core.scoring.exec_cq import execute_cq_to_step, extract_code
    from benchcad_core.scoring.iou import iou_step_vs_step
    from pipeline.prompt import build as build_prompt

    record_id = record["record_id"]
    paths = output_paths(results_root, model_name, record_id)

    # Everything before generation gets its own status instead of crashing the
    # whole run or masquerading as a model failure: one unbuildable ground
    # truth used to take all remaining records down with it.
    blocked: tuple[str, str] | None = None
    gt_step: Path | None = None
    try:
        gt_step = ground_truth_step_for_record(
            record,
            data_dir=data_dir,
            results_root=results_root,
            exec_timeout=exec_timeout,
            cache_root=gt_cache_root,
        )
    except Exception as exc:  # noqa: BLE001 - one bad GT must not end the run.
        blocked = ("gt_fail", f"{type(exc).__name__}: {exc}")

    system = user_text = ""
    image_paths: list[Path] = []
    if blocked is None:
        try:
            system, user_text, image_paths = build_prompt(record, data_dir)
            missing = missing_image_paths(record, data_dir)
            if missing:
                blocked = (
                    "missing_image",
                    "prompt image not found: "
                    + ", ".join(str(path) for path in missing[:3]),
                )
        except Exception as exc:  # noqa: BLE001 - bad record, not a bad model.
            blocked = ("prompt_fail", f"{type(exc).__name__}: {exc}")

    usage: dict[str, int | None] = {}
    raw = ""
    model_error = None
    latency = 0.0
    if blocked is None:
        start = time.time()
        try:
            raw, usage = local_model.generate(
                system=system,
                user_text=user_text,
                image_paths=image_paths,
            )
        except Exception as exc:  # noqa: BLE001 - keep benchmark running.
            raw = ""
            model_error = f"{type(exc).__name__}: {exc}"
        latency = time.time() - start

    code = ensure_result_binding(extract_code(raw)) if raw else ""
    paths["raw"].write_text(raw or "", encoding="utf-8")
    paths["code"].write_text(code or raw or "", encoding="utf-8")
    if blocked is not None:
        status, err_msg = blocked
    elif model_error:
        status = "model_fail"
        err_msg = model_error
    elif not code.strip():
        status = "no_code"
        err_msg = "no parseable code in response"
    else:
        try:
            try:
                has_result = code_defines_result(code)
            except SyntaxError:
                has_result = True
            if not has_result:
                status = "exec_fail"
                err_msg = "RuntimeError: missing required `result` binding"
            else:
                execute_cq_to_step(code, paths["step"], timeout=exec_timeout)
                status = "ok"
                err_msg = None
        except Exception as exc:  # noqa: BLE001 - failed code scores zero.
            status = "exec_fail"
            err_msg = f"{type(exc).__name__}: {exc}"

    if status == "ok" and paths["step"].exists():
        try:
            iou = iou_step_vs_step(
                paths["step"],
                gt_step,
                timeout=iou_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - failed scoring scores zero.
            iou = 0.0
            status = "score_fail"
            err_msg = f"iou_fail: {type(exc).__name__}: {exc}"
    else:
        iou = 0.0

    primary = float(iou)
    if score == "composite" and status == "ok":
        from scoring.composite import composite_score

        gt_code = ""
        code_path = record.get("code_path")
        if code_path and record_path(data_dir, code_path).exists():
            gt_code = record_path(data_dir, code_path).read_text(errors="ignore")
        elif record.get("gt_code"):
            gt_code = str(record["gt_code"])
        try:
            primary = composite_score(
                gt_step=gt_step,
                gen_step=paths["step"],
                gen_code=code,
                gt_code=gt_code,
                family=record.get("family", ""),
            )
        except Exception as exc:  # noqa: BLE001 - keep row with explicit error.
            primary = 0.0
            status = "score_fail"
            err_msg = f"composite_fail: {type(exc).__name__}: {exc}"

    if render_png and status == "ok" and paths["step"].exists():
        try:
            from benchcad_core.scoring.views import composite_for_step

            composite_for_step(paths["step"], paths["png"])
        except Exception as exc:  # noqa: BLE001 - previews never fail a row.
            warn_once(f"PNG preview disabled: {type(exc).__name__}: {exc}")

    row = {
        "record_id": record_id,
        "model": model_name,
        "status": status,
        "iou": round(float(iou), 4),
        "score": round(float(primary), 4),
        "score_type": score,
        "lat_s": round(latency, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": None,
        "err": err_msg,
        "raw_path": str(paths["raw"].relative_to(results_root)),
        "code_path": str(paths["code"].relative_to(results_root)),
        "step_path": (
            str(paths["step"].relative_to(results_root))
            if paths["step"].exists()
            else None
        ),
        "png_path": (
            str(paths["png"].relative_to(results_root))
            if paths["png"].exists()
            else None
        ),
    }
    jsonl = results_root / "results.jsonl"
    rows = read_results(jsonl)
    rows[(model_name, record_id)] = row
    write_results(jsonl, rows)
    return row


def print_summary(out_dir: Path) -> None:
    """Print mean score per model from results.jsonl."""
    jsonl = out_dir / "results.jsonl"
    if not jsonl.exists():
        return
    rows = read_jsonl(jsonl)
    if not rows:
        return
    from pipeline.score_csv import SCORE_CSV_NAME, write_score_summary

    score_csv = out_dir / SCORE_CSV_NAME
    write_score_summary(rows, score_csv)
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["model"], [])
        if row.get("status") == "ok":
            buckets[row["model"]].append(float(row.get("score", 0.0)))
    score_type = rows[-1].get("score_type", "iou")
    print(f"\nresults  -> {jsonl}  ({len(rows)} rows)")
    print(f"csv      -> {score_csv}")
    print(f"summary  -> mean {score_type} per model:")
    width = max(len(model) for model in buckets) + 2
    for model, values in sorted(buckets.items()):
        mean_score = f"{sum(values) / len(values):.3f}" if values else "N/A"
        print(f"    {model:<{width}}n={len(values):<3} mean_{score_type}={mean_score}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a locally trained HuggingFace VLM on Vision2Code.",
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Result label. Defaults to local/<model directory name>.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON/JSONL file in the custom training-data schema. "
            "Relative image/STEP paths are resolved from the file's directory."
        ),
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help=(
            "Optional local BenchCAD code_gen/data parquet directory. Used to "
            "create records.jsonl/codes/steps under --data-dir."
        ),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Only read the first N parquet shards while materializing data.",
    )
    parser.add_argument(
        "--materialize-limit",
        type=int,
        default=None,
        help="Only materialize the first N parquet rows into --data-dir.",
    )
    parser.add_argument(
        "--materialize-exec-timeout",
        type=int,
        default=DEFAULT_EXEC_TIMEOUT,
        help="CadQuery execution timeout for building ground-truth STEP files.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild --data-dir from --parquet-dir even if records.jsonl exists.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from <out-dir>/results.jsonl, skipping rows already saved "
            "for the same model and score type."
        ),
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume, rerun checkpointed rows whose status is not ok.",
    )
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score", choices=["iou", "composite"], default="iou")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc. Ignored when --device-map is set.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional transformers device_map, commonly 'auto' for large models.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help=(
            "Skip the 4-view PNG preview of generated STEPs. The preview is "
            "cosmetic; scoring never uses it. Useful on headless machines "
            "where VTK has no display."
        ),
    )
    parser.add_argument(
        "--iou-timeout",
        type=float,
        default=120.0,
        help=(
            "Wall-clock budget for scoring one pair of STEPs (default: 120). "
            "Voxelizing a dense or non-watertight mesh can run for hours; "
            "exceeding this scores the record score_fail instead of hanging "
            "the run. 0 disables the limit."
        ),
    )
    parser.add_argument(
        "--gt-cache-dir",
        type=Path,
        default=None,
        help=(
            "Where to cache ground-truth STEPs built from GT code "
            "(default: <out-dir>/ground_truth_steps). Point it outside --out-dir "
            "so deleting results does not throw the cache away too."
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=25,
        help=(
            "Abort after this many records fail in a row (default: 25, 0 to "
            "disable). A dead CUDA context or a wrong data dir fails every "
            "remaining record; stopping early keeps --resume checkpoints clean."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not check prompt images before loading the model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    model_path = resolve_path(args.model_path)
    data_file = resolve_path(args.data_file) if args.data_file else None
    data_dir = data_file.parent if data_file else resolve_path(args.data_dir)
    parquet_dir = resolve_path(args.parquet_dir) if args.parquet_dir else None
    out_dir = resolve_path(args.out_dir)
    model_name = args.model_name or f"local/{model_path.name}"
    if args.rerun_failed and not args.resume:
        raise SystemExit("--rerun-failed requires --resume")

    if data_file:
        if parquet_dir or args.rebuild_data:
            raise SystemExit("--data-file cannot be combined with parquet materialization")
        all_records = normalize_records(read_records_file(data_file))
    else:
        all_records = ensure_data_records(
            data_dir=data_dir,
            parquet_dir=parquet_dir,
            rebuild_data=args.rebuild_data,
            max_shards=args.max_shards,
            materialize_limit=args.materialize_limit,
            materialize_exec_timeout=args.materialize_exec_timeout,
        )
    records = select_records(
        all_records,
        record_ids=args.records,
        limit=args.limit,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_count = len(records)
    resumed_count = 0
    if args.resume:
        records, resumed_count = pending_records_for_resume(
            records,
            read_results(out_dir / "results.jsonl"),
            model_name=model_name,
            score_type=args.score,
            rerun_failed=args.rerun_failed,
        )

    print(f"model:  {model_path}")
    print(f"label:  {model_name}")
    print(f"data:   {data_dir}")
    if data_file:
        print(f"file:   {data_file}")
    if parquet_dir:
        print(f"source: {parquet_dir}")
    print(f"out:    {out_dir}")
    print(f"score:  {args.score}")
    if args.resume:
        print(f"resume: skipped {resumed_count}/{selected_count} checkpointed record(s)")
    print(f"runs:   {len(records)} record(s) remaining")

    if not records:
        print("resume: no pending records; evaluation is already complete")
        print_summary(out_dir)
        return

    if not args.skip_preflight:
        missing = preflight_images(records, data_dir)
        if missing:
            print(
                f"\npreflight: {len(missing)} prompt image(s) missing for "
                f"{len({record_id for record_id, _ in missing})}/{len(records)} record(s)",
                file=sys.stderr,
            )
            for record_id, path in missing[:5]:
                print(f"  {record_id}: {path}", file=sys.stderr)
            if len({record_id for record_id, _ in missing}) == len(records):
                raise SystemExit(
                    "preflight: every record is missing its prompt image. Check "
                    "--data-file / --data-dir before spending a GPU on this."
                )
            print(
                "preflight: those records will be recorded as missing_image; "
                "pass --skip-preflight to silence this check.",
                file=sys.stderr,
            )

    local_model = LocalVisionLanguageModel(
        model_path,
        device=args.device,
        dtype_name=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    gt_cache_root = resolve_path(args.gt_cache_dir) if args.gt_cache_dir else None
    consecutive_failures = 0
    for index, record in enumerate(records, 1):
        print(
            f"  [{model_name}] {index}/{len(records)} {record['record_id']}",
            end=" ... ",
            flush=True,
        )
        row = run_record(
            record=record,
            data_dir=data_dir,
            results_root=out_dir,
            model_name=model_name,
            local_model=local_model,
            score=args.score,
            exec_timeout=args.exec_timeout,
            render_png=not args.no_png,
            iou_timeout=args.iou_timeout or None,
            gt_cache_root=gt_cache_root,
        )
        print(f"{row['status']:10s} {row['score_type']}={row['score']:.3f}")
        if row["status"] == "ok":
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        if (
            args.max_consecutive_failures
            and consecutive_failures >= args.max_consecutive_failures
        ):
            print(
                f"\nabort: {consecutive_failures} record(s) failed in a row; "
                f"last error on {record['record_id']}: {row['err']}\n"
                "Fix the cause and re-run with --resume --rerun-failed.",
                file=sys.stderr,
            )
            break
    print_summary(out_dir)


if __name__ == "__main__":
    main()
