"""Measure candidate density and write local watchlist examples without models."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
import time
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .cobol_layout import COMMENT, LITERAL, classify_source
from .scanner import (
    Finding,
    build_presidio_analyzer,
    compile_name_regex,
    find_watchlist_pair_spans,
    iter_all_files,
    iter_text_files,
    load_names,
    read_text,
    relative_name,
    scan_text,
    fold_watchlist_value,
)
from .extractor import build_extraction_records, chunk_records
from .policy import apply_name_policy, NAME_POLICY_TABLE
from .review_decisions import source_line
from .text_matching import prepare_watchlist, span_is_code, name_word_spans, IDENTIFIER_PART_RE
from .source_reader import (
    SourceDecodingError,
    UnsupportedSourceEncodingError,
    read_source,
    split_source_lines,
)


# This is intentionally a conservative vocabulary used only to flag suspicious
# watchlist rows.  It does not alter detection or acceptance decisions.
COBOL_KEYWORDS = frozenset(
    "ACCEPT ADD ALTER CALL CANCEL CLOSE COMPUTE CONFIGURATION CONTINUE COPY "
    "DATA DELETE DISPLAY DIVISION ELSE END-IF EVALUATE EXEC EXIT GO GOBACK "
    "IF INITIALIZE INSPECT INSTALLATION INTO INVOKE OPEN PERFORM PIC PICTURE "
    "MOVE PROCEDURE PROGRAM-ID READ REWRITE SECTION SELECT SET STOP STRING SUBTRACT "
    "THEN UNSTRING VALUE WHEN WORKING-STORAGE WRITE"
    .split()
)

def _normalise_word(value: str) -> str:
    value = value.replace("’", "'").replace("''", "'")
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


@dataclass
class PreflightCounts:
    files_scanned: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    code_hidden: int = 0
    code_shown: int = 0
    path_hidden: int = 0
    lines_total: int = 0
    lines_comment: int = 0
    lines_literal: int = 0
    lines_code: int = 0
    watchlist_single: int = 0
    watchlist_pair: int = 0
    spacy: int = 0
    candidate_lines: set[tuple[str, int]] = field(default_factory=set)
    watchlist_entry_counts: Counter[int] = field(default_factory=Counter)
    short_entries: int = 0
    duplicate_entries: int = 0
    keyword_entries: int = 0
    runtime_seconds: float = 0.0
    frequent_words: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    approved_lines: set[tuple[str, str]] = field(default_factory=set)
    other_lines: set[tuple[str, str]] = field(default_factory=set)
    extractor_chunks: int = 0
    verifier_enabled: bool = True

    @property
    def judge_calls(self) -> int:
        return len(self.approved_lines) + len(self.other_lines)

    @property
    def lines_per_thousand(self) -> float:
        return self.lines_total / 1000.0 if self.lines_total else 0.0

    def rate(self, count: int) -> float:
        return count / self.lines_per_thousand if self.lines_per_thousand else 0.0

    def report_text(self) -> str:
        lines = [
            f"files_scanned={self.files_scanned}",
            f"files_skipped={self.files_skipped}",
            f"files_failed={self.files_failed}",
            f"code_parts_hidden={self.code_hidden}", f"code_parts_shown={self.code_shown}",
            f"path_parts_hidden={self.path_hidden}",
            f"lines_total={self.lines_total}",
            f"lines_comment={self.lines_comment}",
            f"lines_literal={self.lines_literal}",
            f"lines_code={self.lines_code}",
            f"watchlist_single_word={self.watchlist_single} "
            f"per_1000={self.rate(self.watchlist_single):.3f}",
            f"watchlist_pair={self.watchlist_pair} "
            f"per_1000={self.rate(self.watchlist_pair):.3f}",
            f"spacy={self.spacy} per_1000={self.rate(self.spacy):.3f}",
            f"distinct_candidate_lines={len(self.candidate_lines)}",
            "top_watchlist_entries=",
        ]
        for entry_id, count in self.watchlist_entry_counts.most_common(20):
            lines.append(f"entry_id={entry_id} count={count}")
        lines.extend(
            [
                f"watchlist_entries_shorter_than_3={self.short_entries}",
                f"watchlist_duplicate_entries_case_insensitive={self.duplicate_entries}",
                f"watchlist_cobol_keyword_entries={self.keyword_entries}",
                f"estimated_judge_calls={self.judge_calls}",
                f"estimated_verifier_calls_upper_bound={len(self.approved_lines) if self.verifier_enabled else 0}",
                f"estimated_extractor_calls={self.extractor_chunks}",
                "new_extractor_candidates_not_in_judge_estimate=true",
                "estimates_are_distinct_requests_before_cache_startup_and_retries=true",
                f"runtime_seconds={self.runtime_seconds:.3f}",
            ]
        )
        return "\n".join(lines) + "\n"


def watchlist_entries(paths: list[Path], include_default: bool) -> list[tuple[int, str]]:
    """Return non-comment entries with their one-based line IDs."""

    selected = list(paths)
    if include_default and not selected:
        selected = [Path(__file__).parent / "data" / "italian_names.txt"]
    entries: list[tuple[int, str]] = []
    for path in selected:
        for line_number, line in enumerate(read_text(path).splitlines(), start=1):
            value = " ".join(line.strip().split())
            if value and not value.startswith("#") and not value.isdecimal():
                entries.append((line_number, value))
    return entries


def validate_watchlist(entries: list[tuple[int, str]]) -> tuple[int, int, int]:
    values = [_normalise_word(value) for _, value in entries]
    short = sum(len(value.replace(" ", "")) < 3 for value in values)
    duplicates = len(values) - len(set(values))
    keywords = sum(
        all(token.upper().rstrip(".") in COBOL_KEYWORDS for token in value.split())
        for value in values
    )
    return short, duplicates, keywords


def _line_kind_counts(layout, text: str) -> tuple[int, int, int, int]:
    source_lines = split_source_lines(text)
    by_line: dict[int, set[str]] = {}
    for region in layout.regions:
        if region.kind == "line_ending":
            continue
        by_line.setdefault(region.line, set()).add(region.kind)
    comments = literals = code = 0
    for line_number in range(1, len(source_lines) + 1):
        kinds = by_line.get(line_number, set())
        if COMMENT in kinds:
            comments += 1
        elif LITERAL in kinds:
            literals += 1
        else:
            code += 1
    return len(source_lines), comments, literals, code


def run_preflight(
    input_path: Path,
    *,
    watchlist_paths: list[Path],
    include_default_names: bool,
    name_scope: str,
    use_presidio: bool,
    presidio_model: str,
    skip_roots: list[Path] | None = None,
    approved_words: frozenset[str] = frozenset(),
    verifier_enabled: bool = True,
    chunk_lines: int = 25,
    judge_enabled: bool = True,
    extractor_enabled: bool = True,
) -> PreflightCounts:
    started = time.perf_counter()
    counts = PreflightCounts(verifier_enabled=verifier_enabled)
    entries = watchlist_entries(watchlist_paths, include_default_names)
    counts.short_entries, counts.duplicate_entries, counts.keyword_entries = validate_watchlist(entries)
    entry_ids_by_value: dict[str, list[int]] = {}
    for entry_id, value in entries:
        entry_ids_by_value.setdefault(_normalise_word(value), []).append(entry_id)
    all_names = load_names(watchlist_paths, include_default=include_default_names)
    entries_lookup, pair_words = prepare_watchlist(all_names)
    name_regex = compile_name_regex([word for word in all_names if fold_watchlist_value(word) not in entries_lookup])
    analyzer = build_presidio_analyzer(presidio_model, []) if use_presidio else None
    if use_presidio and analyzer is None:
        raise RuntimeError("spaCy/Presidio is unavailable; use --no-presidio to disable it")
    print("Active preflight detectors: watchlist, pairs"
          + (f", spaCy ({presidio_model})" if analyzer is not None else "")
          , flush=True)
    excluded = {path.resolve() for path in watchlist_paths}
    text_files = [
        path
        for path in iter_text_files(input_path, skip_root=skip_roots)
        if path.resolve() not in excluded
    ]
    all_files = iter_all_files(input_path, skip_root=skip_roots)
    counts.files_skipped = max(0, len(all_files) - len(text_files))

    extraction_requests = set()
    watchlist_words = frozenset(fold_watchlist_value(word) for word in all_names)
    path_words = watchlist_words - approved_words
    for index, path in enumerate(text_files, start=1):
        relative = relative_name(path, input_path)
        print(f"Preflight file {index} of {len(text_files)}: {relative}", flush=True)
        try:
            source = read_source(path)
            layout = classify_source(source, source_path=path)
        except (SourceDecodingError, UnsupportedSourceEncodingError, OSError):
            counts.files_failed += 1
            continue
        counts.files_scanned += 1
        total, comments, literals, code = _line_kind_counts(layout, source.text)
        counts.lines_total += total
        counts.lines_comment += comments
        counts.lines_literal += literals
        counts.lines_code += code

        try:
            findings = scan_text(source.text, relative, {"NAME"}, name_regex, None, set(), name_scope,
                                 presidio_analyzer=analyzer, layout=layout, watchlist_pair_values=entries_lookup, pair_words=pair_words)
        except RuntimeError:
            counts.files_failed += 1
            counts.files_scanned -= 1
            continue
        if extractor_enabled:
            records = build_extraction_records(source.text, name_scope, layout=layout)
            extraction_requests.update(tuple(record.text for record in chunk) for chunk in chunk_records(records, chunk_lines))
        counts.path_hidden += sum(fold_watchlist_value(match.group()) in path_words
                                 for match in IDENTIFIER_PART_RE.finditer(relative))
        for finding in findings:
            is_pair = finding.source == "watchlist_pair"
            is_watchlist = finding.source in {"watchlist", "employee_roster", "watchlist_pair"}
            if is_pair:
                counts.watchlist_pair += 1
            elif is_watchlist:
                counts.watchlist_single += 1
            elif finding.source == "presidio_spacy":
                counts.spacy += 1
            line = source_line(finding.context, finding.text)
            if is_watchlist:
                words = [finding.text[a:b] for a, b in name_word_spans(finding.text)] if is_pair else [finding.text]
                for word in words:
                    normalized = _normalise_word(word)
                    for entry_id in entry_ids_by_value.get(normalized, []):
                        counts.watchlist_entry_counts[entry_id] += 1
                    word = fold_watchlist_value(word)
                    counts.frequent_words[word] += 1
                    examples = counts.examples.setdefault(word, [])
                    if line not in examples and len(examples) < 3:
                        examples.append(line)
            context = finding.logical_context or finding.context
            word = fold_watchlist_value(finding.logical_candidate or finding.text)
            code = span_is_code(layout, finding.start, finding.end)
            if code:
                counts.code_shown += word in approved_words
                counts.code_hidden += word not in approved_words
                continue
            counts.candidate_lines.add((relative, finding.line))
            if not judge_enabled:
                continue
            gate = apply_name_policy(occurrence_id="0" * 64, model_context=context, judge_decision=None,
                                     code_sensitive_identifier=False, watchlist_pair=is_pair,
                                     watchlist_single=is_watchlist or word in watchlist_words,
                                     approved_word=word in approved_words)
            if gate.reading != NAME_POLICY_TABLE["no_judge"][1]:
                continue
            key = (word, context)
            (counts.approved_lines if word in watchlist_words else counts.other_lines).add(key)

    counts.extractor_chunks = len(extraction_requests)

    counts.runtime_seconds = time.perf_counter() - started
    return counts


def write_frequent_words(path: Path, counts: PreflightCounts) -> None:
    """Write the local word counts and at most three source examples."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("word", "count", "example_1", "example_2", "example_3"))
        for word, count in sorted(counts.frequent_words.items(), key=lambda item: (-item[1], item[0]))[:100]:
            examples = counts.examples[word]
            writer.writerow([word, count, *examples, *[""] * (3 - len(examples))])
