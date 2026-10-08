"""Final model-free check: readable candidates withhold the file."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, replace
from pathlib import Path

from .review_decisions import ReviewDecisions, source_line
from .scanner import (
    Finding,
    compile_name_regex,
    fold_watchlist_value,
    iter_all_files,
    is_text_candidate,
    relative_name,
    line_column,
    scan_folded_watchlist_names,
    scan_identifier_watchlist_names,
    scan_watchlist_names, scan_text,
)
from .source_reader import SourceDecodingError, UnsupportedSourceEncodingError, read_source, split_source_lines
from .cobol_layout import classify_source
from .text_matching import prepare_watchlist, marked_source_line, span_is_code, output_span_to_source, name_word_spans
from .policy import apply_name_policy
from .logical_text import LogicalText, LogicalPiece, extraction_texts
from .text_matching import logical_bounds, source_findings, source_occurrence_id


@dataclass(frozen=True)
class ResidualScan:
    """Readable TEXT and CODE candidates; both require review."""

    replaceable: tuple[Finding, ...]
    code_sensitive: tuple[Finding, ...]
    errors: tuple[dict[str, str], ...]
    samples: tuple[dict[str, object], ...] = ()
    sample_population: int = 0


def scan_written_output(
    output_dir: Path,
    watchlist_values: list[str],
    adjudicated: list[Finding],
    review_answers: ReviewDecisions | None = None,
    replaced_spans: dict[str, list[tuple[int, int, int]]] | None = None,
    shown_decisions: list[dict[str, object]] | None = None,
    decisions: list[dict[str, object]] | None = None,
    input_path: Path | None = None,
    source_hashes: dict[str, str] | None = None,
    sample_count: int = 0, sample_seed: int | None = None,
) -> ResidualScan:
    """Scan all written text without calling a model or changing files."""

    entries, pair_words = prepare_watchlist(watchlist_values)
    name_regex = compile_name_regex([word for word in watchlist_values if fold_watchlist_value(word) not in entries])
    known = {
        (finding.file, finding.start, finding.end, fold_watchlist_value(finding.text))
        for finding in adjudicated
        if finding.entity_type == "NAME"
    }
    for row in shown_decisions or []:
        if "start" not in row or "end" not in row:
            continue
        word = str(row["text"])
        known.add((row["file"], row["start"], row["end"], fold_watchlist_value(word)))
        if row.get("review_answer") == "not_person":
            for start, end in name_word_spans(word):
                known.add((row["file"], int(row["start"]) + start, int(row["start"]) + end,
                           fold_watchlist_value(word[start:end])))
    replaceable: list[Finding] = []
    code_sensitive: list[Finding] = []
    errors: list[dict[str, str]] = []
    samples, population = [], 0
    randomizer = random.Random(sample_seed)
    for path in iter_all_files(output_dir):
        if not is_text_candidate(path):
            continue
        relative = relative_name(path, output_dir)
        try:
            source = read_source(path)
            layout = classify_source(source, source_path=path)
            original = read_source(input_path if input_path.is_file() else input_path / relative) if input_path else None
            if original is not None and source_hashes is not None and original.sha256 != source_hashes.get(relative):
                raise OSError("source changed after replacement")
        except (SourceDecodingError, UnsupportedSourceEncodingError, OSError) as exc:
            errors.append(
                {
                    "file": relative,
                    "status": "NOT_COMPLETE",
                    "reason": getattr(exc, "reason", "residual_read_error"),
                    "message": str(exc),
                }
            )
            continue
        candidates = scan_text(source.text, relative, {"NAME"}, name_regex, None, set(), "all",
                               watchlist_pair_values=entries, pair_words=pair_words, layout=layout)
        original_layout, original_records = None, ()
        error_before = len(errors)
        unresolved_before = len(replaceable) + len(code_sensitive)
        seen: set[tuple[int, int, str]] = set()
        for finding in candidates:
            finding = replace(finding, context=marked_source_line(source.text, finding.start, finding.end))
            folded = fold_watchlist_value(finding.text)
            key = (finding.start, finding.end, folded)
            if key in seen or _is_placeholder(finding.text):
                continue
            seen.add(key)
            mapped = output_span_to_source(finding.start, finding.end, (replaced_spans or {}).get(relative, []))
            if mapped is not None and (relative, *mapped, folded) in known:
                continue
            code = span_is_code(layout, finding.start, finding.end)
            if original is not None and mapped is not None:
                if original_layout is None:
                    original_layout = classify_source(original, source_path=Path(relative))
                    original_records = extraction_texts(original_layout)
                code = span_is_code(original_layout, *mapped)
                records = original_records
                if code:
                    region = next(region for region in original_layout.regions
                                  if region.decoded_start <= mapped[0] < region.decoded_end)
                    value = original.text[region.decoded_start:region.decoded_end]
                    records = (LogicalText(value, (LogicalPiece(0, len(value), region.decoded_start, region.decoded_end),), "code"),)
                output_pieces = finding.logical.source_spans(finding.logical_start, finding.logical_end) if finding.logical else ((finding.start, finding.end),)
                source_pieces = [output_span_to_source(start, end, (replaced_spans or {}).get(relative, [])) for start, end in output_pieces]
                if all(piece is not None for piece in source_pieces):
                    for record in records:
                        bounds = logical_bounds(record, source_pieces)
                        if bounds is not None:
                            hits = source_findings(replace(finding, start=bounds[0], end=bounds[1]), record, original_layout)
                            finding = next(hit for hit in hits if hit.start == mapped[0])
                            break
                    else:
                        errors.append({"file": relative, "status": "NOT_COMPLETE", "reason": "final source mapping failed", "message": "final source mapping failed"})
                        continue
                else:
                    errors.append({"file": relative, "status": "NOT_COMPLETE", "reason": "final source mapping failed", "message": "final source mapping failed"})
                    continue
            answer = review_answers.answer_for(finding.logical_candidate or finding.text,
                                               finding.review_line or source_line(finding.context, finding.text),
                                               finding.review_key) if review_answers else None
            if answer == "not_person":
                occurrence_id = source_occurrence_id(finding, original.sha256 if original else source.sha256)
                decision = apply_name_policy(occurrence_id=occurrence_id, model_context=finding.context,
                                             judge_decision=None, code_sensitive_identifier=code,
                                             review_answer=answer)
                if decision.outcome == "leave_unchanged":
                    if decisions is not None:
                        decisions.append({**finding.to_dict(), "policy_outcome": decision.outcome, "policy_reading": decision.reading})
                    continue
            if code:
                code_sensitive.append(finding)
            else:
                replaceable.append(finding)
        if sample_count and len(errors) == error_before and len(replaceable) + len(code_sensitive) == unresolved_before:
            for line, text in enumerate(split_source_lines(source.text), 1):
                population += 1
                row = {"file": relative, "line": line, "text": text}
                if len(samples) < sample_count:
                    samples.append(row)
                else:
                    slot = randomizer.randrange(population)
                    if slot < sample_count:
                        samples[slot] = row
    return ResidualScan(tuple(replaceable), tuple(code_sensitive), tuple(errors), tuple(samples), population)


def _is_placeholder(value: str) -> bool:
    return bool(re.fullmatch(
        r"(?:Nome\d+|Cognome\d+|ANON_\d+)(?:\s+(?:Nome\d+|Cognome\d+|ANON_\d+))*|user\d+@example\.invalid",
        value.strip(), re.IGNORECASE))
