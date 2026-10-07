"""Write the small local queue used for human correction reviews.

Each block is grouped only by the folded candidate word. Context words are not
part of the key: in COBOL they commonly contain sequence-area fragments or
verbs such as ``MOVE`` and ``DISPLAY``, which made one review issue look like
many unrelated ones.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .review_decisions import folded_word, source_line

QUEUED_OUTCOMES = {"anonymize_and_review", "review_required"}


def write_review_queue_csv(
    path: Path,
    decisions: Iterable[Mapping[str, object]],
) -> int:
    """Write one complete, largest-first review block for each candidate word.

    ``line_texts`` contains every queued physical line, without fixed-format
    sequence/indicator columns or ``[[...]]`` markers. It is deliberately a
    newline-separated CSV cell so a reviewer can inspect all variants without
    a context-derived grouping rule hiding repetitions.
    """

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for decision in decisions:
        outcome = str(decision.get("policy_outcome") or decision.get("decision") or "")
        if outcome not in QUEUED_OUTCOMES:
            continue
        candidate = str(decision.get("text") or "").strip()
        context = str(decision.get("context") or "")
        if not candidate or not context:
            continue
        line = source_line(context, candidate)
        key = folded_word(candidate)
        groups[key].append(
            {
                "candidate": candidate,
                "outcome": outcome,
                "reason": str(decision.get("policy_reading") or ""),
                "occurrence_id": str(
                    (decision.get("policy_decision") or {}).get("occurrence_id", "")
                    if isinstance(decision.get("policy_decision"), Mapping)
                    else ""
                ),
                "location": _location(decision),
                "line_text": line,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_key",
        "review_kind",
        "occurrences",
        "candidate",
        "reasons",
        "locations",
        "occurrence_ids",
        "line_texts",
    ]
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, entries in ordered:
            row = {
                "group_key": key,
                "review_kind": _review_kind(entries),
                "occurrences": len(entries),
                "candidate": entries[0]["candidate"],
                "reasons": " | ".join(_unique(entry["reason"] for entry in entries)),
                "locations": " | ".join(_unique(entry["location"] for entry in entries)),
                "occurrence_ids": " | ".join(
                    _unique(entry["occurrence_id"] for entry in entries if entry["occurrence_id"])
                ),
                "line_texts": "\n".join(entry["line_text"] for entry in entries),
            }
            writer.writerow(row)
    return len(ordered)


def _location(decision: Mapping[str, object]) -> str:
    file = str(decision.get("file") or "")
    line = str(decision.get("line") or "")
    return f"{file}:{line}" if file and line else file or line


def _review_kind(entries: list[dict[str, str]]) -> str:
    outcomes = {entry["outcome"] for entry in entries}
    if outcomes == {"review_required"}:
        return "required"
    if outcomes == {"anonymize_and_review"}:
        return "correction"
    return "mixed"


def _unique(values: Iterable[str]) -> list[str]:
    """Preserve first occurrence order while omitting empty values."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
