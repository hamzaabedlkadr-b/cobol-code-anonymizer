#!/usr/bin/env python3
"""Create a detector-blind, reproducible sample of source lines.

The output is an evaluation template, not a production finding file.  It uses
only simple source-shape heuristics to stratify lines and never loads a
watchlist, spaCy, an LLM, or any detector result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cobol_code_anonymizer.scanner import (  # noqa: E402
    iter_text_files,
    read_text,
)

IDENTIFICATION_RE = re.compile(
    r"\b(?:IDENTIFICATION\s+DIVISION|PROGRAM-ID|AUTHOR|INSTALLATION|DATE-(?:WRITTEN|COMPILED)|SECURITY)\b",
    re.IGNORECASE,
)
JCL_RE = re.compile(r"^\s*(?://|/\*|\*/)")
COBOL_LEVEL_RE = re.compile(r"^\s*(?:0[1-9]|[1-7][0-9]|8[08])\s+[A-ZÀ-ÖØ-Þ][A-Z0-9À-ÖØ-Þ-]*\b", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"^\s*[A-ZÀ-ÖØ-Þ][A-Z0-9À-ÖØ-Þ-]{2,}\s*\.\s*(?:$|\*)", re.IGNORECASE)
HYPHEN_IDENTIFIER_RE = re.compile(r"\b[A-ZÀ-ÖØ-Þ][A-Z0-9À-ÖØ-Þ]*-[A-Z0-9À-ÖØ-Þ-]+\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(path: Path, input_path: Path) -> str:
    if input_path.is_file():
        return path.name
    return str(path.relative_to(input_path))


def classify_line(line: str) -> str:
    """Assign one source-only stratum, with no detector or watchlist lookup."""
    if JCL_RE.match(line):
        return "jcl"
    if len(line) > 6 and line[6] in "*/":
        return "comment"
    if line.lstrip().startswith(("*", "*>")):
        return "comment"
    if IDENTIFICATION_RE.search(line):
        return "identification"
    if '"' in line or "'" in line:
        return "literal"
    if COBOL_LEVEL_RE.search(line) or PARAGRAPH_RE.match(line) or HYPHEN_IDENTIFIER_RE.search(line):
        return "identifier_code"
    return "code"


def line_records(path: Path, input_path: Path, file_hash: str) -> list[dict[str, object]]:
    text = read_text(path)
    records: list[dict[str, object]] = []
    offset = 0
    for line_number, line_with_newline in enumerate(text.splitlines(keepends=True), start=1):
        line = line_with_newline.rstrip("\r\n")
        line_start = offset
        line_end = line_start + len(line)
        records.append(
            {
                "file": relative_name(path, input_path),
                "file_sha256": file_hash,
                "line": line_number,
                "line_start": line_start,
                "line_end": line_end,
                "source_text": line,
                "stratum": classify_line(line),
                "labels": [],
            }
        )
        offset += len(line_with_newline)
    if text and not records:
        records.append(
            {
                "file": relative_name(path, input_path),
                "file_sha256": file_hash,
                "line": 1,
                "line_start": 0,
                "line_end": len(text),
                "source_text": text,
                "stratum": classify_line(text),
                "labels": [],
            }
        )
    return records


def sample_source_lines(
    input_path: Path,
    output_path: Path,
    *,
    per_stratum: int = 10,
    seed: int = 17,
) -> int:
    """Write a deterministic random sample and return its row count."""
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1")
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    files = iter_text_files(input_path)
    all_records: list[dict[str, object]] = []
    for path in files:
        all_records.extend(line_records(path, input_path, sha256_file(path)))

    buckets: dict[str, list[dict[str, object]]] = {}
    for record in all_records:
        buckets.setdefault(str(record["stratum"]), []).append(record)
    randomizer = random.Random(seed)
    selected: list[dict[str, object]] = []
    for stratum in sorted(buckets):
        bucket = list(buckets[stratum])
        randomizer.shuffle(bucket)
        selected.extend(bucket[:per_stratum])
    selected.sort(key=lambda row: (str(row["file"]), int(row["line_start"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in selected:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return len(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a detector-blind random source-line sample for manual labeling."
    )
    parser.add_argument("input", type=Path, help="Input COBOL/JCL file or folder.")
    parser.add_argument("--output", required=True, type=Path, help="Output sample JSONL path.")
    parser.add_argument("--per-stratum", type=int, default=10, help="Rows per source stratum.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    if args.per_stratum < 1:
        parser.error("--per-stratum must be at least 1")
    if input_path.is_dir() and output_path.is_relative_to(input_path):
        parser.error("--output must be outside the input folder")
    count = sample_source_lines(
        input_path,
        output_path,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    print(f"Sampled {count} source line(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
