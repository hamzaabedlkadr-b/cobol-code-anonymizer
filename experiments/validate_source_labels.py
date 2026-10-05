#!/usr/bin/env python3
"""Validate human labels against unchanged source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cobol_code_anonymizer.scanner import read_text  # noqa: E402

ALLOWED_LABELS = {"PERSON", "PARTIAL", "NOT_PERSON", "UNSURE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(source_root: Path, file_name: str) -> Path:
    if source_root.is_file():
        return source_root
    return source_root / file_name


def validate_source_labels(labels_path: Path, source_root: Path) -> list[str]:
    """Return all validation errors; an empty list means the file is valid."""
    errors: list[str] = []
    seen_rows: set[tuple[str, int, int]] = set()
    for row_number, raw in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"row {row_number}: invalid JSON ({exc.msg})")
            continue
        prefix = f"row {row_number}"
        required = {"file", "file_sha256", "line_start", "line_end", "source_text", "labels"}
        missing = sorted(required - set(row)) if isinstance(row, dict) else sorted(required)
        if missing or not isinstance(row, dict):
            errors.append(f"{prefix}: missing required fields {missing}")
            continue

        path = source_path(source_root, str(row["file"]))
        if not path.exists():
            errors.append(f"{prefix}: source file does not exist: {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row["file_sha256"]:
            errors.append(f"{prefix}: source hash changed")
            continue
        text = read_text(path)
        try:
            line_start = int(row["line_start"])
            line_end = int(row["line_end"])
        except (TypeError, ValueError):
            errors.append(f"{prefix}: line_start and line_end must be integers")
            continue
        if not 0 <= line_start <= line_end <= len(text):
            errors.append(f"{prefix}: source offsets are outside the file")
            continue
        if text[line_start:line_end] != row["source_text"]:
            errors.append(f"{prefix}: source_text does not match source offsets")
        key = (str(row["file"]), line_start, line_end)
        if key in seen_rows:
            errors.append(f"{prefix}: duplicate source occurrence")
        seen_rows.add(key)

        labels = row["labels"]
        if not isinstance(labels, list) or not labels:
            errors.append(f"{prefix}: labels must be a non-empty list")
            continue
        ranges: list[tuple[int, int]] = []
        for label_index, label in enumerate(labels, start=1):
            label_prefix = f"{prefix} label {label_index}"
            if not isinstance(label, dict):
                errors.append(f"{label_prefix}: must be an object")
                continue
            if label.get("label") not in ALLOWED_LABELS:
                errors.append(f"{label_prefix}: label must be one of {sorted(ALLOWED_LABELS)}")
                continue
            try:
                start = int(label["start"])
                end = int(label["end"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label_prefix}: start and end must be integers")
                continue
            if not line_start <= start < end <= line_end:
                errors.append(f"{label_prefix}: span must be inside the sampled line")
                continue
            if text[start:end] != label.get("text"):
                errors.append(f"{label_prefix}: text does not match source offsets")
            ranges.append((start, end))
        ranges.sort()
        if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
            errors.append(f"{prefix}: labels overlap")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate manually completed source-position labels.")
    parser.add_argument("source", type=Path, help="Original source file or folder used for sampling.")
    parser.add_argument("labels", type=Path, help="Completed sample JSONL file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.source.exists():
        parser.error(f"Source path does not exist: {args.source}")
    if not args.labels.exists():
        parser.error(f"Labels file does not exist: {args.labels}")
    errors = validate_source_labels(args.labels.resolve(), args.source.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated source labels: {args.labels.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
