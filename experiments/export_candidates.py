#!/usr/bin/env python3
"""Export current NAME candidates without changing anonymizer behavior.

This is an evaluation utility.  It deliberately does not initialize an LLM
extractor or judge, does not anonymize input, and does not change production
scanner policy.  Each JSONL row represents one detected occurrence and retains
the scanner's decoded character offsets.

Example:

    .venv/bin/python experiments/export_candidates.py batch \
        --watchlist private_watchlists/names.txt \
        --output experiments/candidate_exports/baseline/candidates.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cobol_code_anonymizer.scanner import Finding, scan_path  # noqa: E402

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the original source bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, directory: Path) -> bool:
    """Compatibility helper for Python versions before Path.is_relative_to."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def source_path(input_path: Path, finding: Finding) -> Path:
    if input_path.is_file():
        return input_path
    return input_path / finding.file


def finding_sort_key(finding: Finding) -> tuple[object, ...]:
    return (
        finding.file,
        finding.start,
        finding.end,
        finding.entity_type,
        finding.text,
        finding.source,
        finding.confidence,
    )


def candidate_record(
    input_path: Path,
    finding: Finding,
    file_hashes: dict[Path, str],
) -> dict[str, object]:
    path = source_path(input_path, finding)
    if path not in file_hashes:
        file_hashes[path] = sha256_file(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "export_stage": "pre_judge",
        "file": finding.file,
        "file_sha256": file_hashes[path],
        "entity_type": finding.entity_type,
        "text": finding.text,
        "start": finding.start,
        "end": finding.end,
        "line": finding.line,
        "column": finding.column,
        "confidence": finding.confidence,
        "source": finding.source,
        "context": finding.context,
    }


def export_candidates(
    input_path: Path,
    output_path: Path,
    *,
    watchlists: Iterable[Path] = (),
    include_default_names: bool = True,
    use_presidio: bool = True,
    presidio_model: str = "it_core_news_sm",
    name_scope: str = "context",
    detect_unknown_names: bool = False,
    unknown_name_min_length: int = 4,
) -> tuple[int, list[str]]:
    """Run the current NAME candidate scanners and write deterministic JSONL."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    watchlist_paths = [path.resolve() for path in watchlists]
    diagnostics: list[str] = []
    findings = scan_path(
        input_path=input_path,
        entities={"NAME"},
        extra_watchlists=watchlist_paths,
        include_default_names=include_default_names,
        detect_unknown_names=detect_unknown_names,
        unknown_name_min_length=unknown_name_min_length,
        name_scope=name_scope,
        use_presidio=use_presidio,
        presidio_model=presidio_model,
        diagnostics=diagnostics,
    )

    ordered = sorted(findings, key=finding_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[Path, str] = {}
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for finding in ordered:
            record = candidate_record(input_path, finding, file_hashes)
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")

    return len(ordered), diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export current pre-judge NAME candidates as JSONL. "
            "This command never anonymizes source files or calls an LLM."
        )
    )
    parser.add_argument("input", type=Path, help="Input COBOL/JCL file or folder.")
    parser.add_argument("--output", required=True, type=Path, help="Output candidates.jsonl path.")
    parser.add_argument(
        "--watchlist",
        action="append",
        type=Path,
        default=[],
        help="Text file containing one watchlist value per line. May be supplied more than once.",
    )
    parser.add_argument(
        "--no-default-name-watchlist",
        action="store_true",
        help="Do not include the bundled Italian-name list.",
    )
    parser.add_argument("--no-presidio", action="store_true", help="Disable Presidio/spaCy.")
    parser.add_argument(
        "--presidio-model",
        default="it_core_news_sm",
        help="spaCy model used by the existing Presidio scanner.",
    )
    parser.add_argument(
        "--name-scope",
        choices=("context", "all"),
        default="context",
        help="Use the scanner's existing context or all-text name scope.",
    )
    parser.add_argument(
        "--detect-unknown-names",
        action="store_true",
        help="Enable the existing unknown-name candidate detector.",
    )
    parser.add_argument(
        "--unknown-name-min-length",
        type=int,
        default=4,
        help="Minimum letters for unknown-name candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    if input_path.is_dir() and is_within(output_path, input_path):
        parser.error("--output must be outside the input folder")
    if args.unknown_name_min_length < 1:
        parser.error("--unknown-name-min-length must be at least 1")

    count, diagnostics = export_candidates(
        input_path,
        output_path,
        watchlists=args.watchlist,
        include_default_names=not args.no_default_name_watchlist,
        use_presidio=not args.no_presidio,
        presidio_model=args.presidio_model,
        name_scope=args.name_scope,
        detect_unknown_names=args.detect_unknown_names,
        unknown_name_min_length=args.unknown_name_min_length,
    )
    for message in diagnostics:
        print(f"Warning: {message}", file=sys.stderr)
    print(f"Exported {count} pre-judge NAME candidate occurrence(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
