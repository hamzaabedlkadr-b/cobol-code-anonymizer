"""Command-line interface for COBOL code anonymization."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import __version__
from .llm import (
    NAME_EXTRACT_MODEL,
    NAME_JUDGE_MODEL,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
)
from .replacements import (
    ValueGroup,
    apply_replacements,
    entity_sort_order,
    group_findings,
    group_key,
    load_mapping,
    suggested_replacement,
    write_mapping_template,
)
from .scanner import DEFAULT_ENTITIES, Finding, scan_path, write_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    apply_mode_preset(args, raw_args)
    if args.mode == "extraction-only" and args.detect_unknown_names:
        parser.error(
            "extraction-only mode cannot be combined with --detect-unknown-names "
            "because that is a deterministic name detector"
        )
    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    output_dir = args.out_dir.resolve() if args.out_dir else default_output_dir(input_path)
    if input_path.is_dir() and output_dir == input_path:
        parser.error("--out-dir must be different from the input folder")

    entities = {"NAME"} if args.names_only else set(args.entities) if args.entities else set(DEFAULT_ENTITIES)
    extra_watchlists = [path.resolve() for path in args.watchlist]
    employee_rosters = [path.resolve() for path in args.employee_roster]
    report_dir = args.report_dir.resolve() if args.report_dir else output_dir
    skip_roots = [path for path in (output_dir, report_dir) if path.exists()]
    diagnostics: list[str] = []
    print(f"Mode: {args.mode}")

    name_extractor = None
    if args.name_extract or args.name_extract_model:
        from .extractor import NameExtractor

        extract_model = args.name_extract_model or NAME_EXTRACT_MODEL
        extract_host = args.ollama_host or OLLAMA_HOST
        extract_timeout = args.llm_timeout if args.llm_timeout is not None else OLLAMA_TIMEOUT
        name_extractor = NameExtractor(
            extract_host,
            extract_model,
            timeout=extract_timeout,
            chunk_lines=args.name_extract_chunk_lines,
        )
        extract_ok, extract_reason = name_extractor.canary_ok()
        if not extract_ok:
            report_dir.mkdir(parents=True, exist_ok=True)
            audit_path = report_dir / "extraction_decisions.json"
            name_extractor.write_audit(audit_path)
            print(f"Error: name extraction startup check failed ({extract_reason}).")
            print(f"Incomplete extraction audit: {audit_path}")
            return 1
        print(f"Name extraction enabled: {extract_model} at {extract_host}")

    name_judge = None
    if args.name_judge or args.name_judge_model:
        from .judge import NameJudge

        judge_model = args.name_judge_model or NAME_JUDGE_MODEL
        judge_host = args.ollama_host or OLLAMA_HOST
        judge_timeout = args.llm_timeout if args.llm_timeout is not None else OLLAMA_TIMEOUT
        name_judge = NameJudge(
            judge_host,
            judge_model,
            timeout=judge_timeout,
            policy=args.judge_policy,
        )
        judge_ok, judge_reason = name_judge.canary_ok()
        if judge_ok:
            print(f"Name judge enabled: {judge_model} at {judge_host}")
        else:
            # Never continue with a judge that cannot be trusted: a model that
            # answers the same way to everything looks like it works while
            # adding nothing, so fall back to keeping every candidate.
            print(f"Warning: name judge disabled ({judge_reason}). Keeping all name candidates.")
            name_judge = None

    findings = scan_path(
        input_path=input_path,
        entities=entities,
        extra_watchlists=extra_watchlists,
        employee_rosters=employee_rosters,
        include_default_names=not args.no_default_name_watchlist,
        detect_unknown_names=args.detect_unknown_names,
        unknown_name_min_length=args.unknown_name_min_length,
        name_scope=args.name_scope,
        skip_root=skip_roots,
        use_presidio=not args.no_presidio,
        presidio_model=args.presidio_model,
        diagnostics=diagnostics,
        name_extractor=name_extractor,
        name_judge=name_judge,
        deterministic_names_enabled=args.mode != "extraction-only",
        progress=print_progress,
    )
    groups = group_findings(findings)

    for message in diagnostics:
        print(f"Warning: {message}")

    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "anonymization_findings.json", findings)
    summary_path = report_dir / "scan_summary.txt"
    write_scan_summary_report(
        summary_path,
        args.mode,
        input_path,
        findings,
        diagnostics,
    )

    extraction_incomplete = False
    if name_extractor is not None:
        extraction_path = report_dir / "extraction_decisions.json"
        name_extractor.write_audit(extraction_path)
        extraction_incomplete = not name_extractor.complete
        print(
            f"\nName extraction: {name_extractor.chunk_calls} calls, "
            f"{name_extractor.http_attempts} HTTP attempts, "
            f"{name_extractor.anchored} anchored, "
            f"{name_extractor.unlocatable} unlocatable, "
            f"{name_extractor.errors} errors."
        )
        print(f"Extraction audit: {extraction_path}")

    if name_judge is not None:
        judge_path = report_dir / "judge_decisions.json"
        name_judge.write_decisions(judge_path)
        rejected = sum(1 for row in name_judge.decisions if row["decision"] == "reject")
        review_rejected = sum(
            1 for row in name_judge.decisions if row["decision"] == "review_reject"
        )
        marker_rejected = sum(
            1 for row in name_judge.decisions if row["decision"] == "marker_reject"
        )
        instruction_rejected = sum(
            1 for row in name_judge.decisions if row["decision"] == "instruction_reject"
        )
        print(
            f"\nName judge ({name_judge.policy}): {name_judge.calls} calls, "
            f"{name_judge.cache_hits} cached, "
            f"{name_judge.errors} errors, {rejected} candidates rejected, "
            f"{review_rejected} review-only rejections, "
            f"{marker_rejected} overridden on person-marker lines, "
            f"{instruction_rejected} overridden on instruction-like lines."
        )
        if rejected:
            print(f"Rejected candidates were left unanonymized. Review: {judge_path}")
        if review_rejected:
            print(
                "Multi-token review rejections remained anonymized. "
                f"Review before enabling active rejection: {judge_path}"
            )
        if marker_rejected:
            print(
                f"{marker_rejected} rejection(s) on lines naming a person were overridden "
                f"and stayed anonymized. Unexpected volume here can indicate prompt "
                f"injection in the source: {judge_path}"
            )
        if instruction_rejected:
            print(
                f"{instruction_rejected} rejection(s) on instruction-like source lines were "
                f"overridden and stayed anonymized: {judge_path}"
            )
        if (
            not rejected
            and not review_rejected
            and not marker_rejected
            and not instruction_rejected
            and name_judge.calls
        ):
            # Symptom of a model that keeps everything on the cases that matter.
            # The startup canary only catches total degeneracy, not this.
            print(
                f"Warning: the judge rejected nothing across {name_judge.calls} calls, so "
                f"{name_judge.model} changed no output here. Verify it discriminates "
                "before trusting it as a precision filter."
            )

    if name_extractor is not None or name_judge is not None:
        review_path = report_dir / "llm_name_review.csv"
        write_llm_name_review_csv(review_path, name_extractor, name_judge)
        llm_text_path = report_dir / "llm_finding.txt"
        write_llm_name_review_text(
            llm_text_path,
            args.mode,
            name_extractor,
            name_judge,
        )
        print(f"LLM name review: {review_path}")
        print(f"LLM findings text: {llm_text_path}")

    if args.names_only:
        names = [finding for finding in findings if finding.entity_type == "NAME"]
        print_names_only_report(input_path, names)
        if args.explain:
            print_name_explanations(names, name_extractor, name_judge)
        write_json(report_dir / "names_findings.json", names)
        write_names_csv(report_dir / "names_findings.csv", names)
        print(f"\nNames JSON: {report_dir / 'names_findings.json'}")
        print(f"Names CSV: {report_dir / 'names_findings.csv'}")
        print(f"Scan summary: {summary_path}")
        if extraction_incomplete:
            print("\nINCOMPLETE: extraction coverage failed; these reports are diagnostic only.")
            return 1
        return 0

    print_scan_summary(input_path, findings, groups)
    print(f"\nScan summary: {summary_path}")
    if args.explain:
        print_name_explanations(findings, name_extractor, name_judge)

    if extraction_incomplete:
        print(
            "\nINCOMPLETE: one or more extraction chunks failed. "
            "No mapping or anonymized output was created."
        )
        return 1

    loaded_mapping = load_mapping(args.map_file.resolve() if args.map_file else None)
    if args.create_map:
        map_path = args.create_map.resolve()
        write_mapping_template(map_path, groups, loaded_mapping, args.salt)
        print(f"\nMapping template written to: {map_path}")
        print("Edit the replacement column, then run again with --map-file.")
        return 0

    if args.scan_only:
        print(f"\nFindings JSON written to: {report_dir / 'anonymization_findings.json'}")
        return 0

    if not findings:
        print("\nNo findings to anonymize.")
        return 0

    replacements = choose_replacements(groups, loaded_mapping, args.salt, args.auto)
    if not replacements:
        print("\nNo replacements selected; no anonymized output was written.")
        return 0

    changed_files, replacement_count = apply_replacements(input_path, output_dir, findings, replacements)
    write_mapping_template(output_dir / "replacement_map.csv", groups, replacements, args.salt)
    print(f"\nAnonymized output written to: {output_dir}")
    print(f"Changed files: {changed_files}")
    print(f"Applied replacements: {replacement_count}")
    print(f"Findings JSON: {report_dir / 'anonymization_findings.json'}")
    print(f"Replacement map: {output_dir / 'replacement_map.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cobol-anonymizer",
        description="Scan COBOL, copybook, and JCL files for names and identifiers, then anonymize reviewed values.",
    )
    parser.add_argument("input", type=Path, help="Input .cbl/.cpy/.jcl file or folder.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=("baseline", "extraction-only", "union", "union-judge"),
        default=None,
        help=(
            "Name-detection preset (default: baseline). extraction-only uses only the "
            "LLM for names; union adds LLM extraction; union-judge also enables the "
            "active LLM judge. Individual model and runtime flags remain available."
        ),
    )
    mode_group.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Extraction-only shortcut: use the default LLM to find names, while "
            "keeping deterministic structured-PII detection and replacement prompts."
        ),
    )
    mode_group.add_argument(
        "--union",
        action="store_true",
        help=(
            "Union shortcut: combine spaCy/Presidio, watchlists, the employee roster, "
            "and default LLM name extraction."
        ),
    )
    mode_group.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Judge shortcut: run union mode, then let the default LLM judge actively "
            "filter name candidates before replacement prompts."
        ),
    )
    parser.add_argument("--out-dir", type=Path, help="Folder for anonymized copies.")
    parser.add_argument("--report-dir", type=Path, help="Folder for anonymization_findings.json.")
    parser.add_argument(
        "--entities",
        nargs="+",
        choices=sorted(DEFAULT_ENTITIES),
        help="Entity types to scan. Default: all supported entities.",
    )
    parser.add_argument(
        "--name-scope",
        choices=("context", "all"),
        default="context",
        help="Scan names only in comments/literals by default, or scan all text.",
    )
    parser.add_argument(
        "--watchlist",
        action="append",
        type=Path,
        default=[],
        help="Extra text file with one name/surname or numeric matricola per line. Can be used multiple times.",
    )
    parser.add_argument(
        "--employee-roster",
        action="append",
        type=Path,
        default=[],
        help="Private employee roster file containing names and matriculas. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-default-name-watchlist",
        action="store_true",
        help="Do not load the bundled Italian name list; useful for exact roster-only scans.",
    )
    parser.add_argument(
        "--detect-unknown-names",
        action="store_true",
        help=(
            "Also report uppercase surname-like tokens in comments/contact text even when they are "
            "not in Presidio or a watchlist. Review these before anonymizing."
        ),
    )
    parser.add_argument(
        "--unknown-name-min-length",
        type=int,
        default=4,
        help="Minimum letters for --detect-unknown-names candidates after removing apostrophes.",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Print and write only detected names with folder, file, line, and column, then stop.",
    )
    parser.add_argument(
        "--no-presidio",
        action="store_true",
        help="Disable Microsoft Presidio/spaCy and use only the bundled watchlist for names.",
    )
    parser.add_argument(
        "--presidio-model",
        default="it_core_news_sm",
        help="spaCy model used by Microsoft Presidio for Italian PERSON detection.",
    )
    parser.add_argument(
        "--name-judge",
        action="store_true",
        help=(
            "Enable the Ollama name judge using NAME_JUDGE_MODEL and OLLAMA_HOST from llm.py. "
            "Use --name-judge-model or --ollama-host to override either value for this run."
        ),
    )
    parser.add_argument(
        "--name-extract",
        action="store_true",
        help=(
            "Enable direct Ollama name extraction using NAME_EXTRACT_MODEL from llm.py. "
            "Use --name-extract-model to choose a different model for this run."
        ),
    )
    parser.add_argument(
        "--name-extract-model",
        help="Ollama model used for direct name extraction; also enables extraction.",
    )
    parser.add_argument(
        "--name-extract-chunk-lines",
        type=extraction_chunk_size,
        default=25,
        help="Maximum scoped records per extraction request (default: 25, minimum: 2).",
    )
    parser.add_argument(
        "--name-judge-model",
        help=(
            "Override the Ollama model configured in llm.py and enable the name judge. "
            "Judges whether NAME candidates "
            "are people or ordinary words. Rejections are review-only unless --judge-policy "
            "active is selected; protected roster names and model failures always stay."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        help="Override the Ollama server address configured in llm.py for this run.",
    )
    parser.add_argument(
        "--judge-policy",
        choices=("conservative", "active"),
        default="conservative",
        help=(
            "How LLM rejections affect findings. conservative keeps and logs every rejection; "
            "active applies guarded single-token rejections."
        ),
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        help="Override the per-request timeout configured in llm.py for this run.",
    )
    parser.add_argument(
        "--create-map",
        type=Path,
        help="Write a CSV mapping template and stop. Edit replacement values, then rerun with --map-file.",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        help="CSV mapping file with entity_type, key, original, and replacement columns.",
    )
    parser.add_argument("--scan-only", action="store_true", help="Only scan and write findings JSON.")
    parser.add_argument("--auto", action="store_true", help="Accept suggested replacements without prompts.")
    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Show which detector found each name and, when enabled, how the LLM "
            "extractor and judge handled it. This makes no additional LLM calls."
        ),
    )
    parser.add_argument("--salt", default="cobol-code-anonymizer", help="Salt for deterministic suggestions.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def apply_mode_preset(args: argparse.Namespace, argv: list[str]) -> None:
    """Expand a short mode name into the existing low-level CLI settings."""
    if args.llm:
        args.mode = "extraction-only"
    elif args.union:
        args.mode = "union"
    elif args.judge:
        args.mode = "union-judge"
    elif args.mode is None:
        args.mode = "baseline"

    if args.mode == "extraction-only":
        args.name_extract = True
        args.no_presidio = True
        args.no_default_name_watchlist = True
    elif args.mode == "union":
        args.name_extract = True
    elif args.mode == "union-judge":
        args.name_extract = True
        args.name_judge = True
        policy_was_explicit = any(
            item == "--judge-policy" or item.startswith("--judge-policy=")
            for item in argv
        )
        if not policy_was_explicit:
            args.judge_policy = "active"


def extraction_chunk_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}_anonymized"
    return input_path.parent / f"{input_path.name}_anonymized"


def print_progress(message: str) -> None:
    print(message, flush=True)


def print_scan_summary(input_path: Path, findings: list[Finding], groups: list[ValueGroup]) -> None:
    print(f"Input: {input_path}")
    print(f"Findings: {len(findings)}")
    display_groups = group_findings_for_summary(findings)
    if not display_groups:
        return

    current_entity = ""
    for index, group in enumerate(display_groups, start=1):
        if group.entity_type != current_entity:
            current_entity = group.entity_type
            print(f"\n[{current_entity}]")
        print(f"  {index:>3}. {group.original}  hits={group.count}  locations={group.locations}")


def group_findings_for_summary(findings: list[Finding]) -> list[ValueGroup]:
    """Group repeated values for display without changing replacement behavior."""
    groups: dict[tuple[str, str], ValueGroup] = {}
    for finding in findings:
        key = group_key(finding.entity_type, finding.text)
        if key not in groups:
            groups[key] = ValueGroup(
                entity_type=finding.entity_type,
                original=" ".join(finding.text.split()),
                key=key,
            )
        groups[key].findings.append(finding)
    return sorted(
        groups.values(),
        key=lambda group: (entity_sort_order(group.entity_type), group.original.upper()),
    )


def write_scan_summary_report(
    path: Path,
    mode: str,
    input_path: Path,
    findings: list[Finding],
    diagnostics: list[str],
) -> None:
    """Write the terminal-style grouped findings summary as a text file."""
    lines = [f"Mode: {mode}"]
    lines.extend(f"Warning: {message}" for message in diagnostics)
    lines.extend([f"Input: {input_path}", f"Findings: {len(findings)}"])

    current_entity = ""
    for index, group in enumerate(group_findings_for_summary(findings), start=1):
        if group.entity_type != current_entity:
            current_entity = group.entity_type
            lines.extend(["", f"[{current_entity}]"])
        lines.append(
            f"  {index:>3}. {group.original}  hits={group.count}  "
            f"locations={group.locations}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_names_only_report(input_path: Path, findings: list[Finding]) -> None:
    print(f"Input: {input_path}")
    print(f"Names found: {len(findings)}")
    if not findings:
        return

    print("\nName | Folder | File | Line | Column | Source")
    print("-" * 78)
    for finding in findings:
        file_path = Path(finding.file)
        folder = "" if str(file_path.parent) == "." else str(file_path.parent)
        print(
            f"{finding.text} | {folder} | {finding.file} | "
            f"{finding.line} | {finding.column} | {finding.source or 'unknown'}"
        )


SOURCE_LABELS = {
    "presidio_spacy": "spaCy/Presidio",
    "watchlist": "name watchlist",
    "employee_roster": "employee roster",
    "unknown_name_heuristic": "unknown-name heuristic",
    "llm_extraction": "LLM extractor",
    "mixed": "multiple baseline detectors",
}

JUDGE_REASON_LABELS = {
    "common_word": "common word",
    "place": "place",
    "organization": "organization",
    "technical_term": "technical term",
}


def print_name_explanations(
    findings: list[Finding],
    name_extractor: object | None,
    name_judge: object | None,
) -> None:
    names = [finding for finding in findings if finding.entity_type == "NAME"]
    decisions = list(getattr(name_judge, "decisions", [])) if name_judge is not None else []
    if not names and not decisions:
        print("\nName explanations: no name candidates were retained or rejected.")
        return

    comparisons = (
        list(getattr(name_extractor, "comparisons", []))
        if name_extractor is not None
        else []
    )
    decision_by_location = {
        (row["file"], row["line"], row["column"], row["text"]): row
        for row in decisions
    }

    print("\nName explanations:")
    for finding in names:
        matching_extractions = [
            row for row in comparisons
            if row.get("status") in {"agreement", "extractor_only"}
            and row.get("file") == finding.file
            and int(row.get("start", -1)) < finding.end
            and finding.start < int(row.get("end", -1))
        ]
        detector_sources = {
            source
            for row in matching_extractions
            for source in row.get("detector_sources", [])
            if source
        }
        if finding.source and finding.source != "llm_extraction":
            detector_sources.add(finding.source)

        if matching_extractions and detector_sources:
            found_by = "LLM extractor + " + format_sources(detector_sources)
        elif matching_extractions or finding.source == "llm_extraction":
            found_by = "LLM extractor only"
        elif name_extractor is not None:
            found_by = (
                f"{format_sources(detector_sources)} only; LLM returned no overlapping name "
                "(not a rejection)"
            )
        else:
            found_by = format_sources(detector_sources)

        location = (finding.file, finding.line, finding.column, finding.text)
        judge_note = format_judge_decision(decision_by_location.get(location))
        suffix = f"; {judge_note}" if judge_note else ""
        print(
            f"  {finding.file}:{finding.line}:{finding.column} "
            f"{finding.text!r} -> {found_by}{suffix}"
        )

    rejected = [row for row in decisions if row.get("decision") == "reject"]
    if rejected:
        print("\nRemoved by the LLM judge:")
        for row in rejected:
            reason = JUDGE_REASON_LABELS.get(str(row.get("reason_code", "")), "not a person name")
            print(
                f"  {row['file']}:{row['line']}:{row['column']} "
                f"{row['text']!r} -> rejected as {reason}"
            )


def format_sources(sources: set[str]) -> str:
    if not sources:
        return "baseline detector"
    return " + ".join(sorted(SOURCE_LABELS.get(source, source) for source in sources))


def format_judge_decision(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    decision = str(row.get("decision", ""))
    reason = JUDGE_REASON_LABELS.get(str(row.get("reason_code", "")), "not a person name")
    if row.get("error"):
        return "judge failed, so the candidate was safely kept"
    if decision == "keep":
        return "judge classified it as a person name"
    if decision == "uncertain":
        return "judge was uncertain, so it was kept"
    if decision == "protected":
        return "judge was not called because the roster protects it"
    if decision == "review_reject":
        return f"judge suggested {reason}, but the safety policy kept it"
    if decision == "marker_reject":
        return f"judge suggested {reason}, but the person marker protected it"
    if decision == "instruction_reject":
        return f"judge suggested {reason}, but instruction-like text protected it"
    return ""


def write_names_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "folder",
                "file",
                "line",
                "column",
                "confidence",
                "source",
                "context",
            ],
        )
        writer.writeheader()
        for finding in findings:
            file_path = Path(finding.file)
            folder = "" if str(file_path.parent) == "." else str(file_path.parent)
            writer.writerow(
                {
                    "name": finding.text,
                    "folder": folder,
                    "file": finding.file,
                    "line": finding.line,
                    "column": finding.column,
                    "confidence": f"{finding.confidence:.4f}",
                    "source": finding.source,
                    "context": finding.context,
                }
            )


def build_llm_name_review_rows(
    name_extractor: object | None,
    name_judge: object | None,
) -> list[dict[str, object]]:
    """Build occurrence-level rows shared by the CSV and text reports."""
    comparisons = list(getattr(name_extractor, "comparisons", []))
    decisions = list(getattr(name_judge, "decisions", []))
    decision_by_key = {
        (
            str(row.get("file", "")),
            int(row.get("line", 0)),
            int(row.get("column", 0)),
            str(row.get("text", "")),
        ): row
        for row in decisions
    }
    matched_decisions: set[int] = set()
    rows: list[dict[str, object]] = []

    for comparison in comparisons:
        key = (
            str(comparison.get("file", "")),
            int(comparison.get("line", 0)),
            int(comparison.get("column", 0)),
            str(comparison.get("text", "")),
        )
        decision = decision_by_key.get(key)
        if decision is not None:
            matched_decisions.add(id(decision))
        rows.append(llm_review_row(comparison, decision))

    for decision in decisions:
        if id(decision) not in matched_decisions:
            rows.append(llm_review_row({}, decision))

    for chunk in list(getattr(name_extractor, "chunks", [])):
        record_lines = chunk.get("record_lines", {})
        if not isinstance(record_lines, dict):
            record_lines = {}
        for item in chunk.get("unlocatable", []):
            if not isinstance(item, dict):
                continue
            record_id = item.get("record_id")
            rows.append(
                {
                    "name": item.get("text", ""),
                    "file": chunk.get("file", ""),
                    "line": record_lines.get(str(record_id), ""),
                    "column": "",
                    "extraction_status": "discarded_unlocatable",
                    "judge_status": "not_judged",
                    "final_action": "discarded",
                    "reason": item.get("reason", "could not anchor in source"),
                    "detector_sources": "",
                    "context": "",
                }
            )

    return rows


def write_llm_name_review_csv(
    path: Path,
    name_extractor: object | None,
    name_judge: object | None,
) -> None:
    """Write one machine-readable report for all LLM name modes."""
    rows = build_llm_name_review_rows(name_extractor, name_judge)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "file",
        "line",
        "column",
        "extraction_status",
        "judge_status",
        "final_action",
        "reason",
        "detector_sources",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_llm_name_review_text(
    path: Path,
    mode: str,
    name_extractor: object | None,
    name_judge: object | None,
) -> None:
    """Write grouped LLM name decisions in the terminal summary style."""
    rows = build_llm_name_review_rows(name_extractor, name_judge)
    lines = [f"Mode: {mode}", f"LLM findings: {len(rows)}"]
    if name_judge is not None:
        lines.append(
            "REJECT means the judge said the candidate is not a person name; "
            "check action to see whether the rejection was applied."
        )
    else:
        lines.append(
            "The extractor finds names but does not reject candidates. "
            "BASELINE_ONLY means the LLM did not return that name."
        )

    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        status = llm_text_status(row)
        name = " ".join(str(row.get("name", "")).split()) or "<empty output>"
        action = str(row.get("final_action", ""))
        reason = llm_text_reason(row)
        key = (status, name.casefold(), action, reason)
        group = grouped.setdefault(
            key,
            {
                "status": status,
                "name": name,
                "action": action,
                "reason": reason,
                "locations": [],
                "count": 0,
            },
        )
        group["count"] = int(group["count"]) + 1
        file = str(row.get("file", ""))
        line = str(row.get("line", ""))
        location = f"{file}:{line}" if line else file
        if location and location not in group["locations"]:
            group["locations"].append(location)

    status_order = {
        "KEEP": 0,
        "UNCERTAIN": 1,
        "PROTECTED": 2,
        "REJECT": 3,
        "AGREEMENT": 4,
        "LLM_ONLY": 5,
        "BASELINE_ONLY": 6,
        "DISCARDED": 7,
        "NOT_JUDGED": 8,
    }
    ordered = sorted(
        grouped.values(),
        key=lambda group: (
            status_order.get(str(group["status"]), 99),
            str(group["name"]).casefold(),
        ),
    )
    current_status = ""
    section_index = 0
    for group in ordered:
        status = str(group["status"])
        if status != current_status:
            current_status = status
            section_index = 0
            lines.extend(["", f"[{status}]"])
        section_index += 1
        locations = ", ".join(str(item) for item in group["locations"]) or "unknown"
        lines.append(
            f"  {section_index:>3}. {group['name']}  hits={group['count']}  "
            f"locations={locations}  action={group['action']}  reason={group['reason']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def llm_text_status(row: dict[str, object]) -> str:
    judge_status = str(row.get("judge_status", "not_judged"))
    if judge_status in {"review_reject", "marker_reject", "instruction_reject"}:
        return "REJECT"
    if judge_status != "not_judged":
        return judge_status.upper()
    extraction_status = str(row.get("extraction_status", "not_available"))
    return {
        "agreement": "AGREEMENT",
        "extractor_only": "LLM_ONLY",
        "detector_only": "BASELINE_ONLY",
        "discarded_unlocatable": "DISCARDED",
    }.get(extraction_status, "NOT_JUDGED")


def llm_text_reason(row: dict[str, object]) -> str:
    judge_status = str(row.get("judge_status", "not_judged"))
    reason_code = str(row.get("reason", ""))
    reason = JUDGE_REASON_LABELS.get(reason_code, reason_code or "no reason supplied")
    if judge_status == "keep":
        if reason_code:
            return f"judge error ({reason_code}); candidate was safely kept"
        return "judge classified it as a person name"
    if judge_status == "uncertain":
        return "judge was uncertain, so the candidate was kept"
    if judge_status == "protected":
        return "employee roster protection; judge was not called"
    if judge_status == "reject":
        return f"judge rejected it as {reason}"
    if judge_status == "review_reject":
        return f"judge suggested {reason}; safety policy kept the candidate"
    if judge_status == "marker_reject":
        return f"judge suggested {reason}; person marker kept the candidate"
    if judge_status == "instruction_reject":
        return f"judge suggested {reason}; instruction-like text kept the candidate"

    extraction_status = str(row.get("extraction_status", "not_available"))
    sources = str(row.get("detector_sources", "")).replace(";", ", ")
    if extraction_status == "agreement":
        return f"found by LLM extractor and baseline ({sources or 'detector'})"
    if extraction_status == "extractor_only":
        return "found only by the LLM extractor"
    if extraction_status == "detector_only":
        return "found only by baseline; LLM absence is not a rejection"
    if extraction_status == "discarded_unlocatable":
        return reason
    return reason


def llm_review_row(
    comparison: dict[str, object],
    decision: dict[str, object] | None,
) -> dict[str, object]:
    extraction_status = str(comparison.get("status", "not_available"))
    judge_status = str(decision.get("decision", "not_judged")) if decision else "not_judged"
    reason_code = str(decision.get("reason_code", "")) if decision else ""
    error = str(decision.get("error", "")) if decision else ""

    if judge_status == "reject":
        final_action = "removed_from_findings"
    elif judge_status in {
        "keep",
        "uncertain",
        "protected",
        "review_reject",
        "marker_reject",
        "instruction_reject",
    }:
        final_action = "kept_as_finding"
    else:
        final_action = "kept_as_finding"

    source_row = decision or comparison
    detector_sources = comparison.get("detector_sources", [])
    if isinstance(detector_sources, list):
        detector_sources = ";".join(str(source) for source in detector_sources)
    return {
        "name": source_row.get("text", ""),
        "file": source_row.get("file", ""),
        "line": source_row.get("line", ""),
        "column": source_row.get("column", ""),
        "extraction_status": extraction_status,
        "judge_status": judge_status,
        "final_action": final_action,
        "reason": error or reason_code,
        "detector_sources": detector_sources,
        "context": decision.get("context", "") if decision else "",
    }


def choose_replacements(
    groups: list[ValueGroup],
    loaded_mapping: dict[tuple[str, str], str],
    salt: str,
    auto: bool,
) -> dict[tuple[str, str], str]:
    replacements: dict[tuple[str, str], str] = {}
    total = len(groups)
    interactive = not auto
    print("\nChoose replacements.")
    print("Press Enter to use the suggestion, type your own value, or type 'skip' to leave it unchanged.")

    print("Type 'all' to accept all remaining suggestions, or 'skip-all' to leave all remaining values unchanged.")

    for index, group in enumerate(groups, start=1):
        suggestion = loaded_mapping.get(group.key) or suggested_replacement(group, index, salt)
        if auto:
            replacements[group.key] = suggestion
            continue

        prompt = (
            f"{index}/{total} {group.entity_type} {group.original!r} "
            f"(hits={group.count}) [{suggestion}]: "
        )
        try:
            answer = input(prompt).strip()
        except EOFError:
            interactive = False
            answer = ""

        if answer.lower() == "skip-all":
            break
        if answer.lower() == "all":
            auto = True
            answer = ""
        if answer.lower() in {"skip", "s"}:
            continue
        replacements[group.key] = answer or suggestion

    if not interactive and not auto:
        print("Input ended; remaining blank answers used suggestions.")
    return replacements


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
