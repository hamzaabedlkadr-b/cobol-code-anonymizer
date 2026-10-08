"""Command-line interface for COBOL code anonymization."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
import re
import sys
import unicodedata
from pathlib import Path

from . import __version__
from .llm import (
    NAME_EXTRACT_MODEL,
    NAME_JUDGE_MODEL,
    NAME_VERIFIER_MODEL,
    NAME_VERIFIER_ENABLED,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    load_model_digests,
)
from .replacements import ValueGroup, apply_replacements, copy_file_atomically, group_findings, finding_key, load_mapping, suggested_replacement, replacement_spans, write_mapping_template
from .pipeline import scan_path, policy_row
from .preflight import run_preflight, write_frequent_words
from .review_queue import write_review_queue_csv
from .review_decisions import import_review_answers, load_review_decisions, folded_word, source_line
from .residual import scan_written_output
from .policy import apply_name_policy, NAME_POLICY_TABLE
from .text_matching import name_word_spans, source_occurrence_id, IDENTIFIER_PART_RE, physical_replacement, review_key
from .scanner import (
    DEFAULT_ENTITIES,
    Finding,
    iter_all_files,
    is_text_candidate,
    fold_watchlist_value,
    load_names,
    relative_name,
)


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    if args.sample_unchanged < 0:
        parser.error("--sample-unchanged must be zero or greater")
    apply_mode_preset(args, raw_args)
    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    output_dir = args.out_dir.resolve() if args.out_dir else default_output_dir(input_path)
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir
        else Path(f"{output_dir}_reports")
    )
    if any(is_path_inside(input_path, root) for root in (output_dir, report_dir)):
        parser.error("Output and reports must not contain the input")
    supplied = [*args.watchlist, *([args.approved_words] if args.approved_words else []),
                *([args.map_file] if args.map_file else [])]
    if any(is_path_inside(path.resolve(), output_dir) for path in supplied):
        parser.error("Output must not contain supplied watchlists or maps")
    if is_path_inside(report_dir, output_dir):
        parser.error("--report-dir must be outside --out-dir")
    for path in args.watchlist:
        if not path.is_file():
            parser.error(f"Watchlist file does not exist: {path}")
    if args.approved_words is not None and not args.approved_words.is_file():
        parser.error(f"Approved words file does not exist: {args.approved_words}")
    if args.preflight:
        return run_preflight_cli(
            parser,
            args,
            input_path,
            output_dir,
            report_dir,
        )
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--out-dir is not a folder: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            parser.error(
                f"Output folder is not empty: {output_dir}; use --overwrite to replace it"
            )

    entities = {"NAME"} if args.names_only else set(args.entities) if args.entities else set(DEFAULT_ENTITIES)
    extra_watchlists = [path.resolve() for path in args.watchlist]
    approved_words = frozenset(fold_watchlist_value(word) for word in
                              load_names([args.approved_words] if args.approved_words else [], include_default=False))
    skip_roots = [output_dir, report_dir, *[path.resolve() for path in (args.approved_words, args.map_file) if path]]
    source_hashes: dict[str, str] = {}
    file_times: dict[str, float] = {}
    diagnostics: list[str] = []
    not_complete_files: list[dict[str, str]] = []
    skipped_files: list[dict[str, str]] = []
    path_review_files: list[dict[str, str]] = []
    identifier_review_decisions: list[dict[str, object]] = []
    resolved_review_files: list[str] = []
    import_review_answers(report_dir / "review_queue.csv", report_dir / "review_decisions.csv")
    clear_previous_reports(report_dir, [input_path, *extra_watchlists,
                                       *([args.approved_words.resolve()] if args.approved_words else []),
                                       *([args.map_file.resolve()] if args.map_file else [])])
    review_answers = load_review_decisions(report_dir / "review_decisions.csv")
    print(f"Input: {input_path}   Output: {output_dir}   Reports: {report_dir}")
    print(f"Watchlist: {len(load_names(extra_watchlists, include_default=not args.no_default_name_watchlist))} words   Approved words: {len(approved_words)}")
    for field in ("name_extract_model", "name_judge_model", "name_verifier_model"):
        value = getattr(args, field)
        if value is not None:
            value = value.strip()
            if not value:
                parser.error(f"--{field.replace('_', '-')} must not be blank")
            setattr(args, field, value)
    if not (args.names_only or args.scan_only or args.create_map) and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    requested_models = []
    if args.name_extract or args.name_extract_model:
        requested_models.append(("extractor", args.name_extract_model or NAME_EXTRACT_MODEL))
    if args.name_judge or args.name_judge_model:
        requested_models.append(("judge", args.name_judge_model or NAME_JUDGE_MODEL))
        if NAME_VERIFIER_ENABLED:
            requested_models.append(("verifier", args.name_verifier_model or NAME_VERIFIER_MODEL))
    model_digests = {}
    if requested_models:
        role = requested_models[0][0]
        try:
            model_digests = load_model_digests(args.ollama_host or OLLAMA_HOST,
                args.llm_timeout if args.llm_timeout is not None else OLLAMA_TIMEOUT)
            for role, model in requested_models:
                model_digests[model]
        except (OSError, ValueError, KeyError) as exc:
            available = ", ".join(sorted(model_digests)) or "none available"
            message = f"cannot read requested model: {exc}; available models: {available}"
            write_failure_manifest(report_dir, model_startup_failures(input_path, role, message))
            print(f"NOT_COMPLETE: {message}")
            return 1

    name_extractor = None
    if args.name_extract or args.name_extract_model:
        from .extractor import NameExtractor

        extract_model = args.name_extract_model or NAME_EXTRACT_MODEL
        extract_host = args.ollama_host or OLLAMA_HOST
        extract_timeout = args.llm_timeout if args.llm_timeout is not None else OLLAMA_TIMEOUT
        name_extractor = NameExtractor(
            extract_host, extract_model, timeout=extract_timeout,
            chunk_lines=args.name_extract_chunk_lines,
            model_digest=model_digests[extract_model], cache_dir=report_dir,
        )
        extract_ok, extract_reason = name_extractor.canary_ok()
        if not extract_ok:
            report_dir.mkdir(parents=True, exist_ok=True)
            startup_failures = model_startup_failures(input_path, "extractor", extract_reason)
            write_failure_manifest(report_dir, startup_failures)
            print(f"Error: name extraction startup check failed ({extract_reason}).")
            return 1
        print(f"Name extraction enabled: {extract_model} at {extract_host}")

    name_judge = None
    name_verifier = None
    if args.name_judge or args.name_judge_model:
        from .judge import NameJudge

        judge_model = args.name_judge_model or NAME_JUDGE_MODEL
        judge_host = args.ollama_host or OLLAMA_HOST
        judge_timeout = args.llm_timeout if args.llm_timeout is not None else OLLAMA_TIMEOUT
        judge_digest = model_digests[judge_model]
        name_judge = NameJudge(
            judge_host,
            judge_model,
            timeout=judge_timeout,
            cache_dir=report_dir,
            model_digest=judge_digest,
        )
        judge_ok, judge_reason = name_judge.canary_ok()
        if judge_ok:
            if judge_reason:
                print(f"Warning: name judge startup check: {judge_reason}")
            print(f"Name judge enabled: {judge_model} at {judge_host}")
            if NAME_VERIFIER_ENABLED:
                from .verifier import NameVerifier

                verifier_model = args.name_verifier_model or NAME_VERIFIER_MODEL
                name_verifier = NameVerifier(
                    judge_host,
                    verifier_model,
                    timeout=judge_timeout,
                    cache_dir=report_dir,
                    model_digest=model_digests[verifier_model],
                )
                print(f"Name verifier enabled: {verifier_model} at {judge_host}")
            else:
                print("Name verifier: off")
        else:
            # An enabled model is part of the safety boundary.  If it cannot
            # start, no source is released with a partial decision pipeline.
            report_dir.mkdir(parents=True, exist_ok=True)
            startup_failures = model_startup_failures(input_path, "judge", judge_reason)
            status_path = write_failure_manifest(report_dir, startup_failures)
            print(f"Error: name judge startup check failed ({judge_reason}).")
            print(f"File status report: {status_path}")
            return 1

    try:
        findings = scan_path(
            input_path=input_path,
            entities=entities,
            extra_watchlists=extra_watchlists,
            include_default_names=not args.no_default_name_watchlist,
            name_scope=args.name_scope,
            skip_root=skip_roots,
            source_hashes=source_hashes,
            file_times=file_times,
            use_presidio=not args.no_presidio,
            presidio_model=args.presidio_model,
            diagnostics=diagnostics,
            not_complete_files=not_complete_files,
            identifier_review_decisions=identifier_review_decisions,
            resolved_review_files=resolved_review_files,
            review_answers=review_answers,
            name_extractor=name_extractor,
            name_judge=name_judge,
            name_verifier=name_verifier,
            approved_words=approved_words,
            verifier_enabled=NAME_VERIFIER_ENABLED,
            deterministic_names_enabled=args.mode != "extraction-only",
            progress=print_progress,
        )
    except RuntimeError as exc:
        failures = model_startup_failures(input_path, "detector", str(exc))
        write_failure_manifest(report_dir, failures)
        print(f"NOT_COMPLETE: {exc}")
        return 1
    # Only extraction failures prevent a complete detection pass.
    if name_extractor is not None and not name_extractor.complete:
        failed_chunks = [row for row in getattr(name_extractor, "chunks", []) if row.get("status") != "ok"]
        failures = sorted({str(row["file"]) for row in failed_chunks})
        if not failures:
            failures = [row["file"] for row in model_startup_failures(input_path, "extractor", "extraction failed")]
        not_complete_files.extend({"file": file, "status": "NOT_COMPLETE", "reason": "extractor failed",
                                   "message": name_extractor.failure_reason} for file in failures)

    path_findings = find_path_names(input_path, extra_watchlists, skip_roots, approved_words)
    path_review_files = []
    path_rows = [policy_row(finding, source_hashes.get(finding.file, "0" * 64), code=True,
                            approved=fold_watchlist_value(finding.text) in approved_words,
                            answer=review_answers.answer_for(finding.text, finding.review_line, finding.review_key))
                 for finding in path_findings]
    hidden_paths = [finding for finding, row in zip(path_findings, path_rows) if row["policy_outcome"] == "anonymize_whole"]
    groups = group_findings([*findings, *hidden_paths])
    rows = [*identifier_review_decisions, *(name_judge.decisions if name_judge else []), *path_rows]
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.names_only or args.scan_only or args.create_map:
        write_decisions(report_dir / "decisions.csv", rows)
        write_review_queue_csv(report_dir / "review_queue.csv", rows)
        write_llm_findings(report_dir / "llm_finding.txt", rows, name_extractor, name_judge, name_verifier)
        if args.create_map:
            write_mapping_template(report_dir / args.create_map.name, groups, load_mapping(args.map_file), args.salt)
        summary = f"Scan only: {len(findings)} findings. No files released.\nReports: {report_dir}\n"
        (report_dir / "scan_summary.txt").write_text(summary, encoding="utf-8")
        print(summary, end="")
        if not_complete_files:
            write_failure_manifest(report_dir, not_complete_files)
        return 1 if not_complete_files else 0

    replacements = choose_replacements(groups, load_mapping(args.map_file), args.salt, args.auto) if groups else {}
    aliases = {key[1]: value for key, value in replacements.items() if key[0] == "NAME"}
    path_aliases = {}
    for finding in hidden_paths:
        path_aliases.setdefault(finding.file, {})[fold_watchlist_value(finding.text)] = replacements[finding_key(finding)]
    file_names = {file: rename_path(file, path_aliases.get(file, {})) for file in source_hashes}
    normalized_paths = {unicodedata.normalize("NFC", file).casefold() for file in file_names.values()}
    if len(normalized_paths) != len(file_names):
        parser.error("renamed output paths collide; no output written")
    for finding, row in zip(path_findings, path_rows):
        if row["policy_outcome"] == "anonymize_whole":
            row["replacement"] = replacements[finding_key(finding)]
    staging_dir = report_dir / "staging"
    write_failures, staged_files, line_counts = [], [], {}
    blocked_files = {row["file"] for row in not_complete_files}
    for path in extra_watchlists:
        if path == input_path or input_path.is_dir() and is_path_inside(path, input_path):
            file = relative_name(path, input_path)
            blocked_files.add(file)
            skipped_files.append({"file": file, "reason": "watchlist input"})
    apply_replacements(input_path, staging_dir, findings, replacements, not_complete_files=write_failures,
                       skipped_files=skipped_files, blocked_files=blocked_files, written_files=staged_files,
                       line_counts=line_counts, skip_roots=skip_roots, source_hashes=source_hashes)
    import random
    sample_seed = args.sample_seed if args.sample_seed is not None else random.SystemRandom().randrange(2 ** 63)
    residual = scan_written_output(staging_dir, load_names(extra_watchlists, include_default=not args.no_default_name_watchlist),
                                   findings, review_answers=review_answers, replaced_spans=replacement_spans(findings, replacements),
                                   shown_decisions=[row for row in rows if row.get("policy_outcome") == "leave_unchanged"],
                                   decisions=rows, input_path=input_path, source_hashes=source_hashes,
                                   sample_count=args.sample_unchanged, sample_seed=sample_seed)
    write_failures.extend(residual.errors)
    for finding in (*residual.replaceable, *residual.code_sensitive):
        occurrence_id = source_occurrence_id(finding, source_hashes.get(finding.file, "0" * 64))
        decision = apply_name_policy(occurrence_id=occurrence_id, model_context=finding.context,
                                     judge_decision=None, code_sensitive_identifier=finding in residual.code_sensitive,
                                     residual_unresolved=True)
        rows.append({**finding.to_dict(), "logical_candidate": finding.logical_candidate, "logical_context": finding.logical_context,
                     "review_line": finding.review_line, "review_key": finding.review_key,
                     "code_sensitive_identifier": finding in residual.code_sensitive,
                     "policy_outcome": decision.outcome, "policy_reading": decision.reading})
    not_complete_files.extend(write_failures)
    write_failures.clear()
    review_files = {str(row["file"]) for row in rows if row.get("policy_outcome") == "review_required"}
    review_files.update(row["file"] for row in path_review_files)
    failed_files = {row["file"] for row in not_complete_files}
    review_files -= failed_files
    passed_files = set(staged_files) - failed_files - review_files
    _move_checked_files(staging_dir, report_dir / "needs_review", review_files, write_failures)
    _move_checked_files(staging_dir, output_dir, passed_files, write_failures, file_names=file_names)
    not_complete_files.extend(write_failures)
    for failure in not_complete_files:
        for directory in (staging_dir, output_dir, report_dir / "needs_review"):
            path = directory / (file_names.get(failure["file"], failure["file"]) if directory == output_dir else failure["file"])
            if path.is_file():
                path.unlink()
    if staging_dir.exists():
        import shutil
        shutil.rmtree(staging_dir)
    for finding in findings:
        if finding.entity_type != "NAME":
            hidden = finding_key(finding) in replacements
            rows.append({**finding.to_dict(), "policy_outcome": "anonymize_whole" if hidden else "leave_unchanged",
                         "policy_reading": NAME_POLICY_TABLE["other_hide" if hidden else "other_show"][1]})
    # One audit row per source occurrence, including repeated final-check hits.
    rows = list({(row["file"], row["line"], row.get("column", 0), row["text"]): row for row in rows}.values())
    aliases = {(finding.file, finding.line, finding.column, finding.text): physical_replacement(finding, replacements[finding_key(finding)])
               for finding in findings if finding_key(finding) in replacements}
    for row in rows:
        replacement = aliases.get((row["file"], row["line"], row.get("column", 0), row["text"]))
        if replacement is not None:
            row["replacement"] = replacement
    manifest = write_out_manifest(report_dir / "out_manifest.csv", input_path, output_dir,
                                  path_review_files, skipped_files, not_complete_files, review_files,
                                  skip_roots=skip_roots, decisions=rows, line_counts=line_counts, file_times=file_times, file_names=file_names, applied_spans=replacement_spans(findings, replacements))
    statuses = {row["file"]: row["status"] for row in manifest}
    for row in rows:
        row["file_status"] = statuses.get(row["file"], "not_complete")
    write_decisions(report_dir / "decisions.csv", rows)
    required, optional = write_review_queue_csv(report_dir / "review_queue.csv", rows)
    write_llm_findings(report_dir / "llm_finding.txt", rows, name_extractor, name_judge, name_verifier)
    write_mapping_template(report_dir / "replacement_map.csv", groups, replacements, args.salt)
    totals = Counter(row["status"] for row in manifest)
    verifier_state = f"on ({args.name_verifier_model or NAME_VERIFIER_MODEL})" if name_verifier is not None else "off"
    calls = sum(getattr(model, "calls", 0) if model is not None else 0 for model in (name_judge, name_verifier))
    calls += getattr(name_extractor, "chunk_calls", 0) if name_extractor else 0
    attempts = sum(value for model in (name_extractor, name_judge, name_verifier)
                   if isinstance(value := getattr(model, "http_attempts", 0), (int, float)))
    hits = sum(getattr(model, "cache_hits", 0) if model is not None else 0 for model in (name_extractor, name_judge, name_verifier))
    latencies = [getattr(model, "latency_s", 0) for model in (name_extractor, name_judge, name_verifier)]
    model_seconds = sum(value for value in latencies if isinstance(value, (int, float)))
    summary = (f"Released {totals['released']}   Withheld {totals['withheld']}   Not complete {totals['not_complete']}   Skipped {totals['skipped']}\n"
               f"Required review: {required} items (~{required / 3:.1f} min)   Optional corrections: {optional} items\n"
               f"Verifier: {verifier_state}\nRuntime {time.perf_counter() - started:.1f} s   Model calls {calls}   Cache hits {hits}   HTTP attempts (incl. startup/retries) {attempts}\n"
               f"Model request time (includes retries and startup): {model_seconds:.1f} s\n"
               f"Lines newly past column 72: {sum(line_counts.values())}\n"
               f"Applied spans: released {sum(row['hidden'] for row in manifest if row['status'] == 'released')}, withheld {sum(row['hidden'] for row in manifest if row['status'] == 'withheld')}. Unwritten files have zero applied spans.\n"
               f"Queue answers: show = not a name; hide = name. Optional corrections do not block upload.\nSHARE ONLY: {output_dir}\nNever share: {report_dir}, .lst/.lis listings\n"
               f"Review: {report_dir / 'review_queue.csv'}   Model report: {report_dir / 'llm_finding.txt'}\n")
    summary += "Model failures: " + ", ".join(f"{role} {getattr(model, 'errors', 0) if model else 0}" for role, model in (("extractor", name_extractor), ("judge", name_judge), ("verifier", name_verifier))) + "\n"
    for message in diagnostics:
        summary += f"Warning: {message}\n"
    terminal_summary = summary
    if args.sample_unchanged:
        samples = residual.samples if not write_failures else ()
        sample_line = (f"Human sample: {len(samples)} of {residual.sample_population} released lines (all kinds); "
                       f"seed {sample_seed}; manual check pending.\n") if not write_failures else "Human sample not prepared: release write failed; rerun.\n"
        terminal_summary += sample_line
        summary += "\n" + sample_line
        summary += "Do not upload until the sample has been checked. Any name found: fix and check a new sample.\n"
        summary += "".join(f"{file_names.get(row['file'], row['file'])}:{row['line']}  {row['text']}\n"
                           for row in sorted(samples, key=lambda row: (row['file'], row['line'])))
    else:
        summary += "Human sample not prepared; use --sample-unchanged 300 for the first batch or 40 for later batches.\n"
        terminal_summary = summary
    (report_dir / "scan_summary.txt").write_text(summary, encoding="utf-8")
    print(terminal_summary, end="")
    return 1 if totals["withheld"] or totals["not_complete"] else 0


def clear_previous_reports(report_dir: Path, inputs: list[Path]) -> None:
    """Remove known generated outputs, never caches, answers or supplied inputs."""
    import shutil

    files = (
        "anonymization_findings.json", "extraction_decisions.json", "hidden_pairs.csv",
        "judge_decisions.json", "layout_summary.json", "llm_finding.txt",
        "llm_name_review.csv", "names_findings.csv", "names_findings.json",
        "not_complete_files.json", "out_manifest.csv", "path_review.csv",
        "preflight.txt", "frequent_words.csv", "replacement_map.csv", "residual_findings.json",
        "review_queue.csv", "sample_check.csv", "scan_summary.txt", "skipped_files.csv",
        "verifier_summary.json", "decisions.csv",
    )
    protected = [path.resolve() for path in inputs]
    for name in (*files, "staging", "needs_review"):
        path = report_dir / name
        if any(path.resolve() == item or is_path_inside(item, path.resolve()) for item in protected):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif name in {"staging", "needs_review"} and path.is_dir():
            shutil.rmtree(path)




def write_failure_manifest(report_dir: Path, failures: list[dict[str, str]]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "out_manifest.csv"
    rows = [{"file": row["file"], "status": "not_complete", "reason": row.get("message") or row["reason"],
             "hidden": 0, "shown": 0, "review_required": 0, "past_col_72": 0} for row in failures]
    write_audit_rows_csv(path, rows, fieldnames=("file", "status", "reason", "hidden", "shown", "review_required", "past_col_72"), path_key="file")
    return path


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
            "Union shortcut: combine spaCy/Presidio, watchlists, "
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
    parser.add_argument("--report-dir", type=Path, help="Local reports and review answers.")
    parser.add_argument("--sample-unchanged", type=int, default=0,
                        help="Sample this many random released lines of all kinds; 300 first batch, 40 later.")
    parser.add_argument("--sample-seed", type=int,
                        help="Seed for a repeatable released-line sample.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Count deterministic and spaCy candidates without writing anonymized output or calling Ollama.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing non-empty output folder; old contents are removed first.",
    )
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
    parser.add_argument("--approved-words", type=Path, help="Watchlist words that models may show in clear context.")
    parser.add_argument(
        "--watchlist",
        action="append",
        type=Path,
        default=[],
        help="Extra text file with one name/surname or numeric matricola per line. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-default-name-watchlist",
        action="store_true", default=True,
        help="Do not load the bundled Italian name list; useful when --watchlist is the complete name list.",
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
            "The judge proposes semantic outcomes; only the privacy policy may approve "
            "leaving a candidate readable."
        ),
    )
    parser.add_argument(
        "--name-verifier-model",
        help=(
            "Override the independent verifier model. It is called only after a "
            "non-person judge proposal that could leave text readable."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        help="Override the Ollama server address configured in llm.py for this run.",
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


def extraction_chunk_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def run_preflight_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> int:
    """Run aggregate candidate measurement without touching the output folder."""

    approved_words = frozenset(fold_watchlist_value(word) for word in
                              load_names([args.approved_words] if args.approved_words else [], include_default=False))
    report_dir.mkdir(parents=True, exist_ok=True)
    skip_roots = [path.resolve() for path in (output_dir, report_dir, args.approved_words, args.map_file) if path is not None]
    try:
        counts = run_preflight(
            input_path,
            watchlist_paths=[path.resolve() for path in args.watchlist],
            include_default_names=not args.no_default_name_watchlist,
            name_scope=args.name_scope,
            use_presidio=not args.no_presidio,
            presidio_model=args.presidio_model,
            skip_roots=skip_roots,
            approved_words=approved_words, verifier_enabled=NAME_VERIFIER_ENABLED,
            chunk_lines=args.name_extract_chunk_lines,
            judge_enabled=bool(args.name_judge or args.name_judge_model),
            extractor_enabled=bool(args.name_extract or args.name_extract_model),
        )
    except RuntimeError as exc:
        print(f"NOT_COMPLETE: {exc}")
        return 1

    write_frequent_words(report_dir / "frequent_words.csv", counts)
    report_path = report_dir / "preflight.txt"
    report = counts.report_text()
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"preflight_report={report_path}")
    return 1 if counts.files_failed else 0


def is_path_inside(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is the parent itself or is nested below it."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def model_startup_failures(input_path: Path, stage: str, reason: str) -> list[dict[str, str]]:
    """Mark every source that would have been scanned when a model cannot start."""

    files = [path for path in iter_all_files(input_path) if is_text_candidate(path)]
    if not files:
        files = [input_path]
    return [
        {
            "file": relative_name(path, input_path) if path != input_path else path.name,
            "status": "NOT_COMPLETE",
            "reason": f"{stage}_startup_failure",
            "message": str(reason),
        }
        for path in files
    ]


_PATH_TOKEN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _path_tokens(value: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", value.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return {token for token in _PATH_TOKEN_RE.findall(folded)}


def find_path_names(input_path: Path, watchlist_paths: list[Path], skip_roots, approved_words) -> list[Finding]:
    """Watchlist parts in exported relative paths, with no model calls."""
    words = {fold_watchlist_value(word) for word in load_names(watchlist_paths)}
    findings = []
    for path in iter_all_files(input_path, skip_root=skip_roots):
        if not is_text_candidate(path) or path.resolve() in watchlist_paths:
            continue
        file = relative_name(path, input_path)
        for match in IDENTIFIER_PART_RE.finditer(file):
            if fold_watchlist_value(match.group()) in words:
                findings.append(Finding(file, "NAME", match.group(), match.start(), match.end(), 0,
                                        match.start() + 1, 1, file[:match.start()] + "[[" + match.group() + "]]" + file[match.end():],
                                        "path_watchlist", review_line=file, review_key=review_key("path", file)))
    return findings


def rename_path(file: str, aliases: dict[str, str]) -> str:
    return IDENTIFIER_PART_RE.sub(lambda match: aliases.get(fold_watchlist_value(match.group()), match.group()), file)


def write_audit_rows_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    fieldnames: tuple[str, ...],
    path_key: str,
) -> None:
    """Write a small local-only CSV for skipped or path-review files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(path_key if field == "path" else field, "") for field in fieldnames})


def _move_checked_files(
    output_dir: Path,
    needs_review_dir: Path,
    files: set[str],
    write_failures: list[dict[str, str]],
    *, file_names: dict[str, str] | None = None,
) -> None:
    """Promote checked files atomically to their final release or review tree."""

    for relative in sorted(files):
        source = output_dir / relative
        if not source.exists():
            continue
        try:
            target = needs_review_dir / (file_names or {}).get(relative, relative)
            copy_file_atomically(source, target)
            source.unlink()
        except OSError as exc:
            write_failures.append(
                {
                    "file": relative,
                    "status": "NOT_COMPLETE",
                    "reason": "review_move_error",
                    "message": str(exc),
                }
            )


def write_out_manifest(
    path: Path, input_path: Path, output_dir: Path, path_review_files: list[dict[str, str]],
    skipped_files: list[dict[str, str]], not_complete_files: list[dict[str, str]],
    review_required_files: set[str] | None = None, skip_roots: list[Path] | None = None,
    decisions: list[dict[str, object]] | None = None, line_counts: dict[str, int] | None = None,
    file_times: dict[str, float] | None = None,
    file_names: dict[str, str] | None = None,
    applied_spans: dict[str, list[tuple[int, int, int]]] | None = None,
) -> list[dict[str, object]]:
    """Record every input file's final release status and counts."""
    paths = {row["file"] for row in path_review_files}
    skipped = {row["file"]: row["reason"] for row in skipped_files}
    failed = {row["file"]: row.get("message") or row["reason"] for row in not_complete_files}
    reviews = review_required_files or set()
    by_file = {}
    for row in decisions or []:
        by_file.setdefault(str(row["file"]), []).append(row)
    rows = []
    for source in iter_all_files(input_path, skip_root=skip_roots):
        file = relative_name(source, input_path)
        entries = by_file.get(file, [])
        required = {(folded_word(str(row["text"])[start:end]), source_line(str(row.get("context", "")), str(row["text"])))
                    for row in entries if row["policy_outcome"] == "review_required"
                    for start, end in name_word_spans(str(row["text"]))}
        if file in failed:
            status, reason = "not_complete", failed[file]
        elif file in paths:
            status, reason = "withheld", "name in file or folder name"
        elif file in reviews:
            status, reason = "withheld", "; ".join(sorted({str(row["policy_reading"]) for row in entries if row["policy_outcome"] == "review_required"}))
        elif file in skipped or not is_text_candidate(source):
            status, reason = "skipped", skipped.get(file, "compiler listing" if source.suffix.lower() in {".lst", ".lis"} else "unsupported file type")
        elif (output_dir / (file_names or {}).get(file, file)).is_file():
            status, reason = "released", ""
        else:
            status, reason = "not_complete", "file not written"
        rows.append({"file": file, "output_file": (file_names or {}).get(file, file) if status == "released" else "",
                     "status": status, "reason": reason,
                     "hidden": len((applied_spans or {}).get(file, [])) if status in {"released", "withheld"} else 0,
                     "shown": sum(row["policy_outcome"] == "leave_unchanged" for row in entries),
                     "review_required": len(required), "past_col_72": (line_counts or {}).get(file, 0) if status in {"released", "withheld"} else 0})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "output_file", "status", "reason", "hidden", "shown", "review_required", "past_col_72"))
        writer.writeheader()
        writer.writerows(rows)
    for index, row in enumerate(rows, start=1):
        if row["status"] != "skipped":
            print(f"[{index}/{len(rows)}] {row['file']}  {row['status']}  hidden {row['hidden']}, shown {row['shown']}  {row['reason']}  {(file_times or {}).get(str(row['file']), 0) / 60:.1f} min scan")
    return rows


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "anonymized"
    return input_path.parent / "anonymized"


def print_progress(message: str) -> None:
    print(message, flush=True)


def write_decisions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "line", "col", "word", "detector", "judge", "verifier", "action", "reason", "context", "replacement", "review_answer", "key"))
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (str(row["file"]), int(row["line"]), int(row.get("column", 0)))):
            writer.writerow({"file": row["file"], "line": row["line"], "col": row.get("column", 0),
                             "word": row["text"], "detector": row.get("source", ""),
                             "judge": row.get("judge_outcome", "not_called"), "verifier": row.get("verifier_outcome", "not_called"),
                             "action": {"anonymize_whole": "hide", "leave_unchanged": "show", "review_required": "review"}[row["policy_outcome"]],
                             "reason": row["policy_reading"], "context": row.get("review_line") or row.get("context", ""),
                             "replacement": row.get("replacement", ""), "review_answer": row.get("review_answer") or "",
                             "key": row.get("review_key", "")})


def write_llm_findings(path: Path, rows: list[dict[str, object]], extractor, judge, verifier) -> None:
    lines = ["LLM findings", "Detected candidates only. Undetected names, wrong human answers and unlisted CODE/path parts can still be readable. Check the random sample before upload.", f"Verifier: {'on' if verifier is not None else 'off'} ({getattr(verifier, 'model', NAME_VERIFIER_MODEL)})"]
    for role, model, field in (("Extractor", extractor, "chunk_calls"), ("Judge", judge, "calls"), ("Verifier", verifier, "calls")):
        if model is None:
            lines.append(f"{role}: off")
            continue
        lines.append(f"{role} ({getattr(model, 'model', '')}): {getattr(model, field, 0)} calls, "
                     f"{getattr(model, 'cache_hits', 0)} cache hits, {getattr(model, 'canary_calls', 0)} startup calls, {getattr(model, 'errors', 0)} failures")
        outcome_field = "judge_outcome" if role == "Judge" else "verifier_outcome"
        if role != "Extractor":
            lines.append(f"  Audit outcomes (include reused answers): {dict(Counter(row.get(outcome_field) for row in rows if row.get(outcome_field) not in {None, 'not_called'}))}")
    if extractor is not None:
        lines.append(f"Extractor: {getattr(extractor, 'anchored', 0)} anchored, {getattr(extractor, 'unlocatable', 0)} unmatched answers (affected files not complete)")
    only_extractor = {(row["file"], row["start"], row["end"]) for row in getattr(extractor, "comparisons", []) if row.get("status") == "extractor_only"}
    sections = (
        ("[1] SHOWN approved words (check for leaks)", lambda row: row.get("approved") and row["policy_outcome"] == "leave_unchanged"),
        ("[2] SHOWN non-watchlist candidates (judge alone)", lambda row: not row.get("watchlist") and row.get("judge_outcome") == "propose_unchanged" and row["policy_outcome"] == "leave_unchanged"),
        ("[3] Names found ONLY by the extractor", lambda row: any(file == row["file"] and start < int(row.get("end", 0)) and int(row.get("start", 0)) < end for file, start, end in only_extractor)),
        ("[4] Approved words HIDDEN because the models disagreed", lambda row: row.get("approved") and row["policy_outcome"] == "anonymize_whole" and row.get("judge_outcome") == "propose_unchanged"),
    )
    for title, selected in sections:
        groups = {}
        for row in rows:
            if row.get("entity_type", "NAME") == "NAME" and selected(row):
                groups.setdefault(folded_word(str(row["text"])), []).append(row)
        lines.extend(["", title])
        for word, entries in sorted(groups.items()):
            row = entries[0]
            count = len({source_line(str(entry.get("context", "")), str(entry["text"])) for entry in entries})
            lines.append(f"  {word}  {count} lines  {row['file']}:{row['line']}  {row.get('context', '')!r}  "
                         f"judge: {row.get('judge_outcome', 'not_called')}, verifier: {row.get('verifier_outcome', 'not_called')}  "
                         f"{row['policy_reading']}")
    lines.extend(["", "[5] SHOWN by reviewer or CODE/path approval (also check for leaks)"])
    for row in rows:
        if row["policy_outcome"] == "leave_unchanged" and (row.get("review_answer") or row.get("code_sensitive_identifier")):
            lines.append(f"  {row['file']}:{row['line']} {row['text']!r}  {row['policy_reading']}  {row.get('context', '')!r}")
    lines.extend(["", f"Audited decisions by file status: {dict(Counter(row.get('file_status', 'scan_only') for row in rows))}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_replacements(
    groups: list[ValueGroup],
    loaded_mapping: dict[tuple[str, str], str],
    salt: str,
    auto: bool,
) -> dict[tuple[str, str], str]:
    replacements: dict[tuple[str, str], str] = {}
    name_aliases = {}
    name_counts = Counter()
    for group in sorted((group for group in groups if group.entity_type == "NAME"),
                        key=lambda group: (-len(group.original.split()), group.original)):
        for part, word in enumerate(group.original.split()):
            key = fold_watchlist_value(word)
            if key not in name_aliases:
                prefix = "Nome" if part == 0 else "Cognome"
                name_counts[prefix] += 1
                name_aliases[key] = f"{prefix}{name_counts[prefix]:03d}"
        replacements[group.key] = " ".join(name_aliases[fold_watchlist_value(word)] for word in group.original.split())
    total = len(groups)
    interactive = not auto
    if not auto:
        print("\nChoose replacements.")
        print("Press Enter to use the suggestion or type your own value.")
        print("Type 'all' to accept all remaining suggestions.")
        print("For non-NAME entities only, 'skip' leaves one unchanged and 'skip-all' stops prompts.")

    for index, group in enumerate(groups, start=1):
        if group.entity_type == "NAME":
            continue
        suggestion = loaded_mapping.get(group.key) or suggested_replacement(group, index, salt)
        if auto:
            replacements[group.key] = suggestion
            continue

        prompt = (
            f"{index}/{total} {group.entity_type} {group.original!r} "
            f"(hits={group.count}) [{suggestion}]: "
        )
        while True:
            try:
                answer = input(prompt).strip()
            except EOFError:
                interactive = False
                answer = ""

            command = answer.lower()
            if command == "skip-all":
                return replacements
            if command == "all":
                auto = True
                answer = ""
            if command in {"skip", "s"}:
                break
            replacements[group.key] = answer or suggestion
            break

    if not interactive and not auto:
        print("Input ended; remaining blank answers used suggestions.")
    return replacements


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
