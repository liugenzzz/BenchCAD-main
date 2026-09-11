#!/usr/bin/env python
"""Randomly sample records from a custom Vision2Code JSONL file.

Examples:
    python tools/sample_custom_jsonl.py input.jsonl sampled_400.jsonl
    python tools/sample_custom_jsonl.py input.jsonl sampled.jsonl --count 400 --seed 42

By default, paths in each record's ``images`` field are rewritten from
``/trainspace/...`` to ``/app/llamafactory/trainspace/...``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Read and validate non-empty JSON objects from a JSONL file."""
    if not path.is_file():
        raise SystemExit(f"input JSONL does not exist or is not a file: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise SystemExit(
                    f"invalid record at {path}:{line_number}: "
                    "each JSONL line must be a JSON object"
                )
            records.append(record)
    return records


def split_sampled_records(
    records: list[dict], count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Return sampled records and unselected records using exact row positions."""
    if count <= 0:
        raise SystemExit("--count must be greater than 0")
    if len(records) < count:
        raise SystemExit(
            f"not enough records: requested {count}, but input contains "
            f"only {len(records)} valid records"
        )
    sampled_indices = random.Random(seed).sample(range(len(records)), count)
    sampled_index_set = set(sampled_indices)
    sampled = [records[index] for index in sampled_indices]
    remaining = [
        record for index, record in enumerate(records) if index not in sampled_index_set
    ]
    return sampled, remaining


def sample_records(records: list[dict], count: int, seed: int) -> list[dict]:
    """Sample ``count`` records without replacement using a reproducible seed."""
    sampled, _ = split_sampled_records(records, count, seed)
    return sampled


def rewrite_image_paths(
    records: list[dict],
    source_prefix: str,
    target_prefix: str,
) -> int:
    """Rewrite matching paths in ``images`` and return the replacement count."""
    source = source_prefix.rstrip("/")
    target = target_prefix.rstrip("/")
    if not source or not target:
        raise SystemExit("image path prefixes must not be empty")

    replacements = 0
    for record_index, record in enumerate(records, 1):
        images = record.get("images")
        if images is None:
            continue
        if not isinstance(images, list):
            raise SystemExit(
                f"invalid images field in sampled record {record_index}: "
                "expected a list of paths"
            )

        rewritten: list[str] = []
        for image_index, image_path in enumerate(images, 1):
            if not isinstance(image_path, str):
                raise SystemExit(
                    f"invalid images[{image_index}] in sampled record "
                    f"{record_index}: expected a string path"
                )
            if image_path == source:
                image_path = target
                replacements += 1
            elif image_path.startswith(source + "/"):
                image_path = target + image_path[len(source) :]
                replacements += 1
            rewritten.append(image_path)
        record["images"] = rewritten
    return replacements


def write_jsonl(path: Path, records: list[dict], overwrite: bool) -> None:
    """Write sampled records as UTF-8 JSONL."""
    if path.exists() and not overwrite:
        raise SystemExit(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_backup_path(path: Path) -> Path:
    """Choose a backup name without overwriting an earlier backup."""
    candidate = path.with_name(path.name + ".bak")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{suffix}")
        suffix += 1
    return candidate


def replace_jsonl_with_backup(path: Path, records: list[dict]) -> Path:
    """Back up ``path`` and atomically replace it with the supplied records."""
    backup_path = next_backup_path(path)
    shutil.copy2(path, backup_path)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample records without replacement from JSONL.",
    )
    parser.add_argument("input", type=Path, help="Source custom-format JSONL file.")
    parser.add_argument("output", type=Path, help="Destination sampled JSONL file.")
    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="Number of records to sample (default: 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--image-prefix-from",
        default="/trainspace",
        help="Prefix to replace in images paths (default: /trainspace).",
    )
    parser.add_argument(
        "--image-prefix-to",
        default="/app/llamafactory/trainspace",
        help=(
            "Replacement prefix for images paths "
            "(default: /app/llamafactory/trainspace)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument(
        "--remove-from-input",
        action="store_true",
        help=(
            "Remove sampled rows from the input JSONL after writing the output. "
            "A non-overwriting .bak backup is created first."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise SystemExit("input and output must be different files")

    records = read_jsonl(input_path)
    sampled, remaining = split_sampled_records(records, args.count, args.seed)
    rewritten_count = rewrite_image_paths(
        sampled,
        source_prefix=args.image_prefix_from,
        target_prefix=args.image_prefix_to,
    )
    write_jsonl(output_path, sampled, args.overwrite)
    backup_path = None
    if args.remove_from_input:
        backup_path = replace_jsonl_with_backup(input_path, remaining)

    print(f"input:   {input_path}")
    print(f"records: {len(records)}")
    print(f"sampled: {len(sampled)} (without replacement, seed={args.seed})")
    print(
        f"images:  rewrote {rewritten_count} path(s): "
        f"{args.image_prefix_from} -> {args.image_prefix_to}"
    )
    print(f"output:  {output_path}")
    if args.remove_from_input:
        print(f"source:  removed {len(sampled)} row(s); {len(remaining)} remain")
        print(f"backup:  {backup_path}")


if __name__ == "__main__":
    main()
