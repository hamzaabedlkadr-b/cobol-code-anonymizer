"""Final deterministic NAME scan for already-written anonymized output.

The pass is intentionally local and model-free. It looks for high-risk
watchlist spellings everywhere in the written text, skips occurrences that the
first pass already adjudicated, and separates identifier-shaped residuals from
ordinary replaceable text. Callers may safely mask ordinary residuals once and
must keep identifier residuals out of the shareable output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .evidence import assess_name_evidence
from .replacements import finding_key, group_findings, suggested_replacement
from .review_decisions import ReviewDecisions, source_line
from .scanner import (
    Finding,
    compile_name_regex,
    fold_watchlist_value,
    iter_all_files,
    is_text_candidate,
    relative_name,
    scan_case_shape_names,
    scan_folded_watchlist_names,
    scan_identifier_watchlist_names,
    scan_watchlist_names,
)
from .source_reader import SourceDecodingError, UnsupportedSourceEncodingError, read_source
from .cobol_layout import classify_source
from .text_matching import span_is_code, output_span_to_source


@dataclass(frozen=True)
class ResidualScan:
    """New residuals, split by whether automatic replacement is safe."""

    replaceable: tuple[Finding, ...]
    code_sensitive: tuple[Finding, ...]
    errors: tuple[dict[str, str], ...]


def scan_written_output(
    output_dir: Path,
    watchlist_values: list[str],
    adjudicated: list[Finding],
    review_answers: ReviewDecisions | None = None,
    replaced_spans: dict[str, list[tuple[int, int, int]]] | None = None,
) -> ResidualScan:
    """Scan all written text without calling a model or changing files."""

    if not watchlist_values:
        return ResidualScan((), (), ())
    name_regex = compile_name_regex(watchlist_values)
    known = {
        (finding.file, finding.start, finding.end, fold_watchlist_value(finding.text))
        for finding in adjudicated
        if finding.entity_type == "NAME"
    }
    replaceable: list[Finding] = []
    code_sensitive: list[Finding] = []
    errors: list[dict[str, str]] = []
    for path in iter_all_files(output_dir):
        if not is_text_candidate(path):
            continue
        relative = relative_name(path, output_dir)
        try:
            source = read_source(path)
            layout = classify_source(source, source_path=path)
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
        identifier_candidates = scan_identifier_watchlist_names(
            source.text,
            relative,
            watchlist_values,
        )
        candidates = [
            *scan_watchlist_names(source.text, relative, name_regex, "all"),
            *scan_folded_watchlist_names(source.text, relative, watchlist_values, "all"),
            *scan_case_shape_names(source.text, relative),
            *identifier_candidates,
        ]
        identifier_spans = [
            (finding.start, finding.end)
            for finding in identifier_candidates
        ]
        seen: set[tuple[int, int, str]] = set()
        for finding in candidates:
            folded = fold_watchlist_value(finding.text)
            key = (finding.start, finding.end, folded)
            if key in seen or _is_placeholder(finding.text):
                continue
            seen.add(key)
            mapped = output_span_to_source(finding.start, finding.end, (replaced_spans or {}).get(relative, []))
            if mapped is not None and (relative, *mapped, folded) in known:
                continue
            if (
                review_answers is not None
                and review_answers.answer_for(
                    finding.text,
                    source_line(finding.context, finding.text),
                )
                == "not_person"
            ):
                _, reasons = assess_name_evidence(
                    candidate=finding.text,
                    context=finding.context,
                )
                if "clue:direct_person_cue" not in reasons:
                    continue
            if span_is_code(layout, finding.start, finding.end) or any(finding.start < end and start < finding.end for start, end in identifier_spans):
                code_sensitive.append(finding)
            else:
                replaceable.append(finding)
    return ResidualScan(tuple(replaceable), tuple(code_sensitive), tuple(errors))


def residual_replacements(findings: tuple[Finding, ...]) -> dict[tuple[str, str], str]:
    """Use the ordinary pseudonym rule for the one repair pass."""
    return {group.key: suggested_replacement(group, index, "residual")
            for index, group in enumerate(group_findings(list(findings)), start=1)}


def _is_placeholder(value: str) -> bool:
    return bool(re.fullmatch(
        r"(?:Nome\d+|Cognome\d+|ANON_\d+)(?:\s+(?:Nome\d+|Cognome\d+|ANON_\d+))*|user\d+@example\.invalid",
        value.strip(), re.IGNORECASE))
