"""Command-line interface for COBOL code anonymization."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import __version__
from .replacements import (
    ValueGroup,
    apply_replacements,
    group_findings,
    load_mapping,
    suggested_replacement,
    write_mapping_template,
)
from .scanner import DEFAULT_ENTITIES, Finding, scan_path, write_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    output_dir = args.out_dir.resolve() if args.out_dir else default_output_dir(input_path)
    if input_path.is_dir() and output_dir == input_path:
        parser.error("--out-dir must be different from the input folder")

    entities = {"NAME"} if args.names_only else set(args.entities) if args.entities else set(DEFAULT_ENTITIES)
    extra_watchlists = [path.resolve() for path in args.watchlist]
    employee_rosters = [path.resolve() for path in args.employee_roster]
    skip_root = output_dir if output_dir.exists() else None
    diagnostics: list[str] = []

    findings = scan_path(
        input_path=input_path,
        entities=entities,
        extra_watchlists=extra_watchlists,
        employee_rosters=employee_rosters,
        include_default_names=not args.no_default_name_watchlist,
        name_scope=args.name_scope,
        skip_root=skip_root,
        use_presidio=not args.no_presidio,
        presidio_model=args.presidio_model,
        diagnostics=diagnostics,
    )
    groups = group_findings(findings)

    for message in diagnostics:
        print(f"Warning: {message}")

    report_dir = args.report_dir.resolve() if args.report_dir else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "anonymization_findings.json", findings)

    if args.names_only:
        names = [finding for finding in findings if finding.entity_type == "NAME"]
        print_names_only_report(input_path, names)
        write_json(report_dir / "names_findings.json", names)
        write_names_csv(report_dir / "names_findings.csv", names)
        print(f"\nNames JSON: {report_dir / 'names_findings.json'}")
        print(f"Names CSV: {report_dir / 'names_findings.csv'}")
        return 0

    print_scan_summary(input_path, findings, groups)

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
        help="Extra text file with one name/surname per line. Can be used multiple times.",
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
    parser.add_argument("--salt", default="cobol-code-anonymizer", help="Salt for deterministic suggestions.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}_anonymized"
    return input_path.parent / f"{input_path.name}_anonymized"


def print_scan_summary(input_path: Path, findings: list[Finding], groups: list[ValueGroup]) -> None:
    print(f"Input: {input_path}")
    print(f"Findings: {len(findings)}")
    if not groups:
        return

    current_entity = ""
    for index, group in enumerate(groups, start=1):
        if group.entity_type != current_entity:
            current_entity = group.entity_type
            print(f"\n[{current_entity}]")
        print(f"  {index:>3}. {group.original}  hits={group.count}  locations={group.locations}")


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

        if answer.lower() in {"skip", "s"}:
            continue
        replacements[group.key] = answer or suggestion

    if not interactive and not auto:
        print("Input ended; remaining blank answers used suggestions.")
    return replacements


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
