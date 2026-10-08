"""Preserve overlap evidence and uncovered whole words of losing findings."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from bisect import bisect_left
from .text_matching import logical_bounds
from typing import Generic, Iterable, Protocol, TypeVar


class SpanFinding(Protocol):
    """The fields required to group and rank one scanner finding."""

    file: str
    entity_type: str
    start: int
    end: int


FindingT = TypeVar("FindingT", bound=SpanFinding)


ENTITY_PRIORITY = {
    "CODICE_FISCALE": 100,
    "IBAN": 95,
    "EMAIL": 90,
    "MATRICOLA": 80,
    "SUSPECTED_MATRICOLA": 79,
    "PHONE": 75,
    "NAME": 60,
}




@dataclass(frozen=True)
class OverlapResolution(Generic[FindingT]):
    """All preserved groups plus the temporary legacy-compatible selection."""

    selected: tuple[FindingT, ...]


def finding_order(item: SpanFinding) -> tuple[object, ...]:
    """Stable source order shared by grouping and compatibility selection."""

    return (
        item.file,
        item.start,
        -ENTITY_PRIORITY.get(item.entity_type, 0),
        -(item.end - item.start),
    )




def select_legacy_findings(findings: Iterable[FindingT]) -> tuple[FindingT, ...]:
    """Keep priority winners and every uncovered whole word of losing spans."""
    ordered = sorted(findings, key=lambda item: (
        -ENTITY_PRIORITY.get(item.entity_type, 0), -(item.end - item.start),
        item.file, item.start,
    ))
    kept: list[FindingT] = []
    by_file = {}
    for finding in ordered:
        starts, file_kept = by_file.setdefault(finding.file, ([], []))
        index = bisect_left(starts, finding.start)
        limit = bisect_left(starts, finding.end)
        overlaps = [item for item in file_kept[max(0, index - 1):limit]
                    if item.start < finding.end and finding.start < item.end]
        if not overlaps:
            kept.append(finding)
            starts.insert(index, finding.start)
            file_kept.insert(index, finding)
            continue
        for word in re.finditer(r"\w+(?:(?:''|'|’)[^\W_]+)*", finding.text):
            start, end = finding.start + word.start(), finding.start + word.end()
            if any(item.start < end and start < item.end for item in overlaps):
                continue
            before = finding.text[:word.start()]
            line = finding.line + before.count("\n")
            column = len(before.rsplit("\n", 1)[-1]) + 1 if "\n" in before else finding.column + word.start()
            marker = "[[" + finding.text + "]]"
            marked_word = finding.text[:word.start()] + "[[" + word.group() + "]]" + finding.text[word.end():]
            context = finding.context.replace(marker, marked_word, 1)
            fragment = replace(finding, text=word.group(), start=start, end=end,
                                line=line, column=column, context=context,
                                logical_start=finding.logical_start + word.start(),
                                logical_end=finding.logical_start + word.end(), logical_candidate=word.group())
            if finding.logical:
                local_start, local_end = logical_bounds(finding.logical, ((start, end),))
                logical = finding.logical.text
                fragment = replace(fragment, logical_start=local_start, logical_end=local_end,
                                   logical_context=logical[:local_start] + "[[" + logical[local_start:local_end] + "]]" + logical[local_end:])
            kept.append(fragment)
            index = bisect_left(starts, start)
            starts.insert(index, start)
            file_kept.insert(index, fragment)
    return tuple(sorted(kept, key=lambda item: (item.file, item.start, item.end)))


def resolve_overlaps(findings: Iterable[FindingT]) -> OverlapResolution[FindingT]:
    """Preserve evidence and select non-overlapping spans without losing words."""

    materialized = tuple(findings)
    return OverlapResolution(
        selected=select_legacy_findings(materialized),
    )
