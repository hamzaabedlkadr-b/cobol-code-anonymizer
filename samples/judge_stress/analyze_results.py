#!/usr/bin/env python3
"""Summarize a judge/policy run against the generated stress-fixture truth."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.json"
DEFAULT_DECISIONS = Path("/tmp/js_judge/judge_decisions.json")
ANONYMIZE = {"anonymize_whole", "anonymize_part", "anonymize_and_review"}
LEAVE = {"leave_unchanged"}
REVIEW = {"review_required"}


def decisions_file(root: Path) -> Path:
    current = root / "decisions.csv"
    return current if current.is_file() else root / "judge_decisions.json"


def load_payload(path: Path) -> dict[str, object]:
    if path.suffix != ".csv":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{**row, "text": row["word"], "policy_outcome": {"hide": "anonymize_whole", "show": "leave_unchanged", "review": "review_required"}[row["action"]],
                 "judge_outcome": row["judge"], "verifier_outcome": row["verifier"]} for row in csv.DictReader(handle)]
    payload = {"decisions": rows}
    summary = path.parent / "scan_summary.txt"
    if summary.is_file():
        match = re.search(r"Model calls (\d+)", summary.read_text(encoding="utf-8"))
        if match:
            payload["llm_calls"] = int(match.group(1))
    return payload


def load_decisions(path: Path):
    result = defaultdict(list)
    for row in load_payload(path).get("decisions", []):
        try:
            result[(str(row.get("file", "")), int(row["line"]))].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def row_outcome(row: dict[str, object]) -> str:
    return str(row.get("policy_outcome") or row.get("decision") or "")


def rows_for_case(case: dict[str, object], decisions: dict[int, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows = decisions.get((_source_file(case), int(case["line"])), []) or decisions.get(("", int(case["line"])), [])
    truths = case.get("truth_names", [])
    if not truths:
        return rows

    matched = []
    for row in rows:
        row_text = str(row.get("text", ""))
        if any(
            row_text.casefold() == str(item["text"]).casefold()
            or str(item["text"]).casefold() in row_text.casefold()
            for item in truths
        ):
            matched.append(row)
    return matched


def _source_file(case: dict[str, object]) -> str:
    return str(case.get("source_file", "PAYROLL.CBL"))


def load_output_lines(run_dir: Path | None, source_file: str, tree: str = "out") -> list[str] | None:
    """Read only the selected released or withheld output tree."""
    if run_dir is None:
        return None
    root = run_dir.parent if run_dir.name in {"out", "needs_review"} else run_dir
    if tree == "needs_review":
        candidate = root / tree / source_file
    elif (root / "out").exists() or (root / "needs_review").exists():
        candidate = root / "out" / source_file
    else:
        candidate = root / source_file
    manifest = root / "out_manifest.csv"
    if tree == "out" and manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = {row["file"]: row for row in csv.DictReader(handle)}
        row = rows.get(source_file)
        if row is not None:
            if row["status"].lower() != "released":
                return None
            summary = root / "scan_summary.txt"
            output_root = root / "out"
            if summary.is_file():
                share = next((line[12:].strip() for line in summary.read_text(encoding="utf-8").splitlines() if line.startswith("SHARE ONLY: ")), "")
                if share:
                    output_root = Path(share)
            candidate = output_root / (row.get("output_file") or source_file)
    if candidate.is_file():
        from cobol_code_anonymizer.source_reader import read_source
        from cobol_code_anonymizer.source_reader import split_source_lines
        return split_source_lines(read_source(candidate).text)
    return None


def _readable_tokens(value: str) -> set[str]:
    """Whole words containing at least two letters, including noisy spellings."""
    return {token.casefold() for token in re.findall(r"[^\W_]+", value, re.UNICODE)
            if sum(character.isalpha() for character in token) >= 2}


def span_contains_leak(case: dict[str, object], output_lines: list[str] | None) -> bool | None:
    """Score truth words anywhere on their output line after variable-length edits."""
    if output_lines is None:
        return None
    for truth in case.get("truth_names", []):
        try:
            line = output_lines[int(truth["line"]) - 1]
        except (IndexError, KeyError, TypeError, ValueError):
            return True
        if _readable_tokens(line) & _readable_tokens(str(truth["text"])):
            return True
    return False


def output_keeps_case_text(case: dict[str, object], output_lines: list[str] | None) -> bool | None:
    """Check a leave-truth against bytes in the produced file, without repair."""

    if output_lines is None:
        return None
    try:
        line = output_lines[int(case["line"]) - 1]
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return str(case["text"]).casefold() in line.casefold()


def queued_review_locations(run_dir: Path | None) -> set[tuple[str, int]]:
    """Return the locations actually written to the local review queue.

    A policy result is not enough to call an item reviewable: the report a
    person will receive must exist.  The caller may supply either the report
    root or its ``out`` child, like :func:`load_output_lines`.
    """

    if run_dir is None:
        return set()
    for path in (run_dir / "review_queue.csv", run_dir.parent / "review_queue.csv"):
        if not path.is_file():
            continue
        locations: set[tuple[str, int]] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for raw in row.get("where", row.get("locations", "")).replace(" | ", "; ").split("; "):
                    file_name, separator, line = raw.rpartition(":")
                    if separator and line.isdigit():
                        locations.add((file_name, int(line)))
        return locations
    return set()


def classify_case(
    case: dict[str, object],
    rows: list[dict[str, object]],
    output_lines: list[str] | None = None,
    queued_locations: set[tuple[str, int]] = set(),
) -> str:
    expected = str(case["expected_outcome"])
    outcomes = {row_outcome(row) for row in rows}
    if expected == "anonymize":
        output_leak = span_contains_leak(case, output_lines)
        if output_leak is not None:
            return "leak" if output_leak else "handled"
        if outcomes & LEAVE:
            return "leak"
        if outcomes & (ANONYMIZE | REVIEW):
            return "handled"
        # G2 roster matches are intentionally protected before the judge and
        # therefore have no decision row.
        return "handled" if case.get("expected_route") == "G2" else "leak"
    if expected == "leave":
        kept = output_keeps_case_text(case, output_lines)
        if kept is not None:
            return "handled" if kept else "false_positive"
        return "false_positive" if outcomes & ANONYMIZE else "handled"
    if expected == "review_required":
        # These cases are privacy cases too.  Inspect written output before
        # trusting a policy record, so regex replacements count as hidden.
        kept = output_keeps_case_text(case, output_lines)
        if kept is True:
            return "review_leak"
        if kept is False:
            location = (_source_file(case), int(case["line"]))
            return "review_handled" if location in queued_locations else "review_hidden"
        return "review_missing"
    return "unknown"


def false_positive_kind(rows: list[dict[str, object]]) -> str:
    """Classify an anonymized leave-truth case by reviewer visibility.

    A case is counted once. If it has both hidden and queued occurrences, it
    belongs to ``fp_review`` because a reviewer can see and correct it.
    """

    outcomes = {row_outcome(row) for row in rows}
    if "anonymize_and_review" in outcomes:
        return "fp_review"
    if outcomes & ANONYMIZE:
        return "fp_hidden"
    # Exact output proves a false positive even when the caller did not retain
    # decision records.  It is not review-visible in this run directory.
    return "fp_hidden"


def _empty_stats(key: tuple[str, str]) -> dict[str, object]:
    return {
        "category": key[0], "line_type": key[1], "cases": 0,
        "name_truths": 0, "name_leaks": 0, "non_name_cases": 0,
        "false_positives": 0, "fp_review": 0, "fp_hidden": 0,
        "review_cases": 0,
        "review_handled": 0, "review_hidden": 0, "review_leak": 0,
        "review_missing": 0,
    }


def _add_case_stats(
    stats: dict[str, object],
    case: dict[str, object],
    case_rows: list[dict[str, object]],
    classification: str,
) -> None:
    stats["cases"] += 1
    truths = case.get("truth_names", [])
    expected = str(case["expected_outcome"])
    if expected == "anonymize":
        stats["name_truths"] += len(truths)
    elif expected == "leave":
        stats["non_name_cases"] += 1
    else:
        stats["review_cases"] += 1
    if classification == "leak":
        stats["name_leaks"] += max(1, len(truths))
    elif classification == "false_positive":
        stats["false_positives"] += 1
        stats[false_positive_kind(case_rows)] += 1
    elif classification == "review_handled":
        stats["review_handled"] += 1
    elif classification == "review_hidden":
        stats["review_hidden"] += 1
    elif classification == "review_leak":
        stats["review_leak"] += 1
    elif classification == "review_missing":
        stats["review_missing"] += 1


def analyze(
    manifest_path: Path,
    decisions_path: Path,
    run_dir: Path | None = None,
    tree: str = "out",
) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = load_decisions(decisions_path)
    output_by_file = {
        file: load_output_lines(run_dir, file, tree)
        for file in {_source_file(case) for case in manifest}
    }
    queued_locations = queued_review_locations(run_dir)
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for case in manifest:
        if run_dir is not None and output_by_file[_source_file(case)] is None:
            continue
        key = (str(case["category"]), str(case.get("line_type", "unknown")))
        case_rows = rows_for_case(case, decisions)
        classification = classify_case(
            case, case_rows, output_by_file[_source_file(case)],
            queued_locations,
        )
        _add_case_stats(grouped.setdefault(key, _empty_stats(key)), case, case_rows, classification)
    return [grouped[key] for key in sorted(grouped)]


def noise_breakdown(
    manifest_path: Path,
    decisions_path: Path,
    run_dir: Path | None = None,
    tree: str = "out",
) -> list[dict[str, object]]:
    """Return the same exact metrics partitioned by file level and noise tag."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = load_decisions(decisions_path)
    output_by_file = {
        file: load_output_lines(run_dir, file, tree)
        for file in {_source_file(case) for case in manifest}
    }
    queued_locations = queued_review_locations(run_dir)
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for case in manifest:
        if run_dir is not None and output_by_file[_source_file(case)] is None:
            continue
        level = str(case.get("file_noise_level", "clean"))
        tags = case.get("noise", []) or ["none"]
        case_rows = rows_for_case(case, decisions)
        classification = classify_case(
            case, case_rows, output_by_file[_source_file(case)],
            queued_locations,
        )
        for tag in tags:
            key = (level, str(tag))
            stats = grouped.setdefault(key, _empty_stats(key))
            stats["file_noise_level"] = level
            stats["noise_type"] = str(tag)
            _add_case_stats(stats, case, case_rows, classification)
    return [grouped[key] for key in sorted(grouped)]


def case_classifications(
    manifest_path: Path,
    decisions_path: Path,
    run_dir: Path | None = None,
    tree: str = "out",
) -> dict[str, str]:
    """Return one stable outcome class per case for regression comparison."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = load_decisions(decisions_path)
    output_by_file = {
        file: load_output_lines(run_dir, file, tree)
        for file in {_source_file(case) for case in manifest}
    }
    queued_locations = queued_review_locations(run_dir)
    results: dict[str, str] = {}
    for case in manifest:
        if run_dir is not None and output_by_file[_source_file(case)] is None:
            results[str(case["id"])] = "not_released" if tree == "out" else "not_withheld"
            continue
        rows = rows_for_case(case, decisions)
        classification = classify_case(
            case,
            rows,
            output_by_file[_source_file(case)],
            queued_locations,
        )
        # The aggregate output calls both kinds "false_positive".  The
        # comparator needs the finer class because a newly hidden false
        # positive is a regression while a queued one remains correctable.
        if classification == "false_positive":
            classification = false_positive_kind(rows)
        results[str(case["id"])] = classification
    return results


def compare_runs(
    manifest_path: Path,
    decisions_path: Path,
    run_dir: Path | None,
    old_run_dir: Path,
    *,
    model_noise_case_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[tuple[str, str, str]], bool]:
    """List changed classes and identify non-noise privacy regressions.

    ``model_noise_case_ids`` comes only from two runs of unchanged code.  The
    cases remain visible in the comparison, but their unstable model answers
    cannot reject an otherwise safe implementation change.
    """

    old_decisions = decisions_file(old_run_dir)
    if not old_decisions.is_file():
        raise FileNotFoundError(f"comparison run has no decisions file: {old_decisions}")
    current = case_classifications(manifest_path, decisions_path, run_dir)
    previous = case_classifications(manifest_path, old_decisions, old_run_dir)
    if run_dir is not None:
        current.update({"needs_review:" + key: value for key, value in
                        case_classifications(manifest_path, decisions_path, run_dir, "needs_review").items()})
        previous.update({"needs_review:" + key: value for key, value in
                         case_classifications(manifest_path, old_decisions, old_run_dir, "needs_review").items()})
    changes = [
        (case_id, previous[case_id], current[case_id])
        for case_id in sorted(current)
        if case_id in previous and previous[case_id] != current[case_id]
    ]
    failed = any(
        (after in {"leak", "review_leak"} and before not in {"leak", "review_leak"})
        or (after == "fp_hidden" and before != "fp_hidden"
            and case_id.removeprefix("needs_review:") not in model_noise_case_ids)
        for case_id, before, after in changes
    )
    return changes, failed


def model_noise_changes(
    manifest_path: Path,
    first_run_dir: Path,
    second_run_dir: Path,
) -> list[tuple[str, str, str]]:
    """Return cases whose class changes between two same-code model runs."""

    first_decisions = decisions_file(first_run_dir)
    second_decisions = decisions_file(second_run_dir)
    for decisions_path in (first_decisions, second_decisions):
        if not decisions_path.is_file():
            raise FileNotFoundError(f"noise run has no decisions file: {decisions_path}")
    first = case_classifications(manifest_path, first_decisions, first_run_dir)
    second = case_classifications(manifest_path, second_decisions, second_run_dir)
    return [
        (case_id, first[case_id], second[case_id])
        for case_id in sorted(first)
        if case_id in second and first[case_id] != second[case_id]
    ]


def model_error_counts(payload: dict[str, object]) -> tuple[int, int]:
    """Count judge and verifier errors separately from recorded decisions."""

    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        return 0, 0
    return (
        sum(row.get("judge_outcome") == "error" for row in decisions if isinstance(row, dict)),
        sum(row.get("verifier_outcome") == "error" for row in decisions if isinstance(row, dict)),
    )


def review_cost(run_dir: Path | None) -> tuple[int, float]:
    """Count distinct word/line decisions; each takes twenty seconds."""
    if run_dir is None:
        return 0, 0.0
    root = run_dir.parent if run_dir.name in {"out", "needs_review"} else run_dir
    path = root / "review_queue.csv"
    items = set()
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                word = row.get("word") or row.get("group_key") or row.get("candidate", "")
                lines = [row["line"]] if "line" in row else row.get("line_texts", "").splitlines()
                items.update((word.casefold(), row.get("key") or line) for line in lines if not row.get("answer"))
    return len(items), len(items) / 3


def released_file_count(run_dir: Path | None) -> int:
    if run_dir is None:
        return 0
    root = run_dir.parent if run_dir.name in {"out", "needs_review"} else run_dir
    manifest = root / "out_manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            return sum(row["status"].lower() == "released" for row in csv.DictReader(handle))
    return sum(path.is_file() for path in (root / "out").rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--run-dir", type=Path,
        help="Anonymized-output directory (or its out/ child) for exact span scoring.",
    )
    parser.add_argument(
        "--compare", type=Path,
        help="Previous report directory; fail on a new leak or hidden false positive.",
    )
    parser.add_argument(
        "--noise-runs", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"),
        help=(
            "Two report directories produced by unchanged code. Cases whose "
            "classes differ are reported as model noise; only false-positive "
            "changes may be exempted."
        ),
    )
    args = parser.parse_args()
    payload = load_payload(args.decisions)
    rows = analyze(args.manifest, args.decisions, args.run_dir)
    noise_rows = noise_breakdown(args.manifest, args.decisions, args.run_dir)
    withheld_rows = analyze(args.manifest, args.decisions, args.run_dir, "needs_review")
    items, minutes = review_cost(args.run_dir)
    print(f"released_files: {released_file_count(args.run_dir)}")
    print(f"review_items: {items}")
    print(f"review_minutes_at_20_seconds: {minutes:.2f}")
    root = args.run_dir.parent if args.run_dir and args.run_dir.name in {"out", "needs_review"} else args.run_dir
    if root and (root / "review_queue.csv").is_file():
        with (root / "review_queue.csv").open(newline="", encoding="utf-8") as handle:
            queue = list(csv.DictReader(handle))
        for priority in ("REQUIRED", "optional"):
            decisions = {(row.get("word", "").casefold(), row.get("key") or row.get("line", "")) for row in queue if row.get("priority") == priority and not row.get("answer")}
            print(f"{priority}_items: {len(decisions)}; minutes_at_20_seconds: {len(decisions) / 3:.2f}")

    print("--- JUDGE RUN METRICS ---")
    for key in (
        "llm_calls",
        "cache_hits",
        "person_text_retries",
        "runtime_seconds",
        "review_required",
    ):
        if key in payload:
            print(f"{key}: {payload[key]}")
    judge_errors, verifier_errors = model_error_counts(payload)
    print(f"judge_errors: {judge_errors}")
    print(f"verifier_errors: {verifier_errors}")
    print("")
    print("--- RELEASED OUTPUT BY CATEGORY AND LINE TYPE ---")
    print("category | line_type | cases | name_truths | leaks | non_name_cases | false_positives | fp_review | fp_hidden | review_cases | review_handled | review_hidden | review_leak | review_missing")
    for row in rows:
        print(" | ".join(str(row[field]) for field in (
            "category", "line_type", "cases", "name_truths", "name_leaks",
            "non_name_cases", "false_positives", "fp_review", "fp_hidden",
            "review_cases", "review_handled", "review_hidden",
            "review_leak", "review_missing")))
    print("")
    print("Leaks are truth words still readable on the output line; false positives are leave-truth cases hidden.")
    print("fp_review is anonymized-and-queued; fp_hidden is anonymized without a correction queue item.")
    print("Review handling checks written output: queued, hidden-but-not-queued, or readable.")
    print("")
    print("--- NOISE BREAKDOWN ---")
    print("file_noise_level | noise_type | cases | leaks | false_positives | fp_review | fp_hidden")
    for row in noise_rows:
        print(" | ".join(str(row[field]) for field in (
            "file_noise_level", "noise_type", "cases", "name_leaks",
            "false_positives", "fp_review", "fp_hidden")))
    print("\n--- WITHHELD OUTPUT BY CATEGORY AND LINE TYPE ---")
    print("category | line_type | cases | leaks | fp_review | fp_hidden | review_leak")
    for row in withheld_rows:
        print(" | ".join(str(row[field]) for field in (
            "category", "line_type", "cases", "name_leaks", "fp_review", "fp_hidden", "review_leak")))
    model_noise: list[tuple[str, str, str]] = []
    if args.noise_runs:
        model_noise = model_noise_changes(args.manifest, *args.noise_runs)
        print("\n--- MODEL NOISE (UNCHANGED CODE) ---")
        if model_noise:
            for case_id, first, second in model_noise:
                print(f"{case_id}: {first} <> {second}")
        else:
            print("No case classes changed between the same-code runs.")
    if args.compare:
        print("\n--- RUN COMPARISON ---")
        noise_case_ids = {case_id for case_id, _, _ in model_noise}
        changes, failed = compare_runs(
            args.manifest,
            args.decisions,
            args.run_dir,
            args.compare,
            model_noise_case_ids=noise_case_ids,
        )
        if changes:
            for case_id, before, after in changes:
                note = " [false-positive model noise]" if case_id.removeprefix("needs_review:") in noise_case_ids and after in {"fp_review", "fp_hidden"} else ""
                print(f"{case_id}: {before} -> {after}{note}")
        else:
            print("No case classes changed.")
        if failed:
            print("Comparison failed: a handled case became a leak or a new hidden false positive appeared.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
