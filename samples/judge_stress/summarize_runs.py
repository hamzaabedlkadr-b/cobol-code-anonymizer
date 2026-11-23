#!/usr/bin/env python3
"""Write the compact cross-fixture table produced by ``run_all.sh``.

This helper reads completed local reports only.  It never starts the scanner or
contacts a model.  Keeping aggregation here makes the shell runner small and
ensures the table uses the same case classifier as each ``analysis.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_results import analyze, model_error_counts, review_cost, released_file_count, decisions_file, load_payload


def review_group_count(path: Path) -> int:
    """Count actual review groups, excluding the CSV header."""

    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def summary_row(label: str, manifest_path: Path, run_dir: Path) -> dict[str, object]:
    """Aggregate the release-gate counters for one completed fixture run."""

    decisions_path = decisions_file(run_dir)
    payload = load_payload(decisions_path)
    category_rows = analyze(manifest_path, decisions_path, run_dir)
    withheld = analyze(manifest_path, decisions_path, run_dir, "needs_review")
    review_items, review_minutes = review_cost(run_dir)
    judge_errors, verifier_errors = model_error_counts(payload)
    verifier_calls = sum(
        str(row.get("verifier_outcome", "")) not in {"", "not_called"}
        for row in payload.get("decisions", [])
        if isinstance(row, dict)
    )
    runtime_seconds_path = run_dir / "runtime_seconds.txt"
    runtime_seconds = (
        float(runtime_seconds_path.read_text(encoding="utf-8").strip())
        if runtime_seconds_path.is_file()
        else 0.0
    )
    return {
        "run": label,
        "leaks": sum(int(row["name_leaks"]) for row in category_rows),
        "fp_review": sum(int(row["fp_review"]) for row in category_rows),
        "fp_hidden": sum(int(row["fp_hidden"]) for row in category_rows),
        "review_groups": review_group_count(run_dir / "review_queue.csv"),
        "judge_errors": judge_errors,
        "verifier_errors": verifier_errors,
        "calls": int(payload.get("llm_calls", 0)) + (verifier_calls if decisions_path.suffix != ".csv" else 0),
        "minutes": runtime_seconds / 60.0,
        "released_files": released_file_count(run_dir),
        "withheld_leaks": sum(int(row["name_leaks"]) for row in withheld),
        "withheld_fp_review": sum(int(row["fp_review"]) for row in withheld),
        "withheld_fp_hidden": sum(int(row["fp_hidden"]) for row in withheld),
        "review_items": review_items, "review_minutes": review_minutes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "MANIFEST", "RUN_DIR"),
        required=True,
        help="One completed primary run; repeat for each fixture.",
    )
    args = parser.parse_args()

    rows = [
        summary_row(label, Path(manifest), Path(run_dir))
        for label, manifest, run_dir in args.run
    ]
    fields = (
        "run", "leaks", "fp_review", "fp_hidden", "review_groups",
        "judge_errors", "verifier_errors", "calls", "minutes",
        "released_files", "withheld_leaks", "withheld_fp_review", "withheld_fp_hidden", "review_items", "review_minutes",
    )
    print(" | ".join(fields))
    print(" | ".join("---" for _ in fields))
    for row in rows:
        values = [
            f"{float(row[field]):.1f}" if field in {"minutes", "review_minutes"} else str(row[field])
            for field in fields
        ]
        print(" | ".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
