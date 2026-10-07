"""Deterministic candidate-density measurement for a source batch.

Preflight deliberately stops before replacement, judging, extraction, or any
Ollama setup.  It uses the same source reader, watchlist matcher, and optional
spaCy detector as a normal scan, but emits only aggregate counts so the report
can be shared without exposing names or source text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
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
)
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
    lines_total: int = 0
    lines_comment: int = 0
    lines_literal: int = 0
    lines_code: int = 0
    watchlist_single: int = 0
    watchlist_pair: int = 0
    spacy: int = 0
    case_shape: int = 0
    candidate_lines: set[tuple[str, int]] = field(default_factory=set)
    watchlist_entry_counts: Counter[int] = field(default_factory=Counter)
    short_entries: int = 0
    duplicate_entries: int = 0
    keyword_entries: int = 0
    runtime_seconds: float = 0.0

    @property
    def judge_calls(self) -> int:
        return len(self.candidate_lines)

    @property
    def lines_per_thousand(self) -> float:
        return self.lines_total / 1000.0 if self.lines_total else 0.0

    def rate(self, count: int) -> float:
        return count / self.lines_per_thousand if self.lines_per_thousand else 0.0

    def report_text(self) -> str:
        lines = [
            f"files_scanned={self.files_scanned}",
            f"files_skipped={self.files_skipped}",
            f"lines_total={self.lines_total}",
            f"lines_comment={self.lines_comment}",
            f"lines_literal={self.lines_literal}",
            f"lines_code={self.lines_code}",
            f"watchlist_single_word={self.watchlist_single} "
            f"per_1000={self.rate(self.watchlist_single):.3f}",
            f"watchlist_pair={self.watchlist_pair} "
            f"per_1000={self.rate(self.watchlist_pair):.3f}",
            f"spacy={self.spacy} per_1000={self.rate(self.spacy):.3f}",
            f"case_shape={self.case_shape} per_1000={self.rate(self.case_shape):.3f}",
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
                f"estimated_verifier_calls={self.judge_calls}",
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
    case_shape_enabled: bool = True,
) -> PreflightCounts:
    started = time.perf_counter()
    counts = PreflightCounts()
    entries = watchlist_entries(watchlist_paths, include_default_names)
    counts.short_entries, counts.duplicate_entries, counts.keyword_entries = validate_watchlist(entries)
    entry_ids_by_value: dict[str, list[int]] = {}
    for entry_id, value in entries:
        entry_ids_by_value.setdefault(_normalise_word(value), []).append(entry_id)
    all_names = load_names(watchlist_paths, include_default=include_default_names)
    name_regex = compile_name_regex(all_names)
    analyzer = build_presidio_analyzer(presidio_model, []) if use_presidio else None
    excluded = {path.resolve() for path in watchlist_paths}
    text_files = [
        path
        for path in iter_text_files(input_path, skip_root=skip_roots)
        if path.resolve() not in excluded
    ]
    all_files = iter_all_files(input_path, skip_root=skip_roots)
    counts.files_skipped = max(0, len(all_files) - len(text_files))

    for path in text_files:
        relative = relative_name(path, input_path)
        try:
            source = read_source(path)
            layout = classify_source(source, source_path=path)
        except (SourceDecodingError, UnsupportedSourceEncodingError, OSError):
            counts.files_skipped += 1
            continue
        counts.files_scanned += 1
        total, comments, literals, code = _line_kind_counts(layout, source.text)
        counts.lines_total += total
        counts.lines_comment += comments
        counts.lines_literal += literals
        counts.lines_code += code

        findings = scan_text(
            source.text,
            relative,
            {"NAME"},
            name_regex,
            None,
            set(),
            False,
            4,
            name_scope,
            presidio_analyzer=analyzer,
            case_shape_enabled=case_shape_enabled,
        )
        pairs = find_watchlist_pair_spans(source.text, all_names)
        pair_finding_ids: set[int] = set()
        for index, finding in enumerate(findings):
            source_is_watchlist = finding.source in {"watchlist", "employee_roster"}
            source_is_spacy = finding.source == "presidio_spacy"
            source_is_case_shape = finding.source == "case_shape"
            overlaps_pair = any(
                finding.start < end and start < finding.end for start, end in pairs
            )
            if source_is_watchlist and overlaps_pair:
                pair_finding_ids.add(index)
            elif source_is_watchlist:
                counts.watchlist_single += 1
                normalized = _normalise_word(finding.text)
                for entry_id in entry_ids_by_value.get(normalized, []):
                    counts.watchlist_entry_counts[entry_id] += 1
            elif source_is_spacy:
                counts.spacy += 1
            elif source_is_case_shape:
                counts.case_shape += 1
            if (source_is_watchlist or source_is_spacy or source_is_case_shape) and not overlaps_pair:
                counts.candidate_lines.add((relative, finding.line))
        counts.watchlist_pair += len(pairs)
        for index in pair_finding_ids:
            finding = findings[index]
            normalized = _normalise_word(finding.text)
            for entry_id in entry_ids_by_value.get(normalized, []):
                counts.watchlist_entry_counts[entry_id] += 1

    counts.runtime_seconds = time.perf_counter() - started
    return counts
