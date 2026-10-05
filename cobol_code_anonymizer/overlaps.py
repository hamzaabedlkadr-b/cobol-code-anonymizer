"""Group overlapping findings without discarding detector evidence.

This module separates two concerns that the legacy scanner combined:

* ``build_overlap_groups`` records every directly or transitively overlapping
  finding, preserving detector provenance for later evidence and policy work.
* ``select_legacy_findings`` reproduces the scanner's current winner selection
  exactly. It is temporary compatibility behavior, not the final name policy.

Keeping the compatibility selector explicit lets cleanup paths migrate to
groups without changing current anonymization output in the same change set.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class OverlapGroup(Generic[FindingT]):
    """One same-file connected component of half-open finding spans."""

    file: str
    start: int
    end: int
    members: tuple[FindingT, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("overlap group must contain at least one finding")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("overlap group span must be non-empty and ordered")
        if any(member.file != self.file for member in self.members):
            raise ValueError("overlap group members must belong to one file")
        if min(member.start for member in self.members) != self.start:
            raise ValueError("overlap group start does not cover its members")
        if max(member.end for member in self.members) != self.end:
            raise ValueError("overlap group end does not cover its members")


@dataclass(frozen=True)
class OverlapResolution(Generic[FindingT]):
    """All preserved groups plus the temporary legacy-compatible selection."""

    groups: tuple[OverlapGroup[FindingT], ...]
    selected: tuple[FindingT, ...]


def finding_order(item: SpanFinding) -> tuple[object, ...]:
    """Stable source order shared by grouping and compatibility selection."""

    return (
        item.file,
        item.start,
        -ENTITY_PRIORITY.get(item.entity_type, 0),
        -(item.end - item.start),
    )


def build_overlap_groups(findings: Iterable[FindingT]) -> tuple[OverlapGroup[FindingT], ...]:
    """Return same-file transitive overlap groups without dropping members.

    Half-open spans overlap only when they share a source character. Adjacent
    spans therefore remain separate groups.
    """

    ordered = sorted(findings, key=finding_order)
    groups: list[OverlapGroup[FindingT]] = []
    members: list[FindingT] = []
    current_file = ""
    group_start = 0
    group_end = 0

    for finding in ordered:
        joins_group = (
            bool(members)
            and finding.file == current_file
            and finding.start < group_end
        )
        if not joins_group:
            if members:
                groups.append(
                    OverlapGroup(current_file, group_start, group_end, tuple(members))
                )
            members = [finding]
            current_file = finding.file
            group_start = finding.start
            group_end = finding.end
            continue
        members.append(finding)
        group_end = max(group_end, finding.end)

    if members:
        groups.append(OverlapGroup(current_file, group_start, group_end, tuple(members)))
    return tuple(groups)


def select_legacy_findings(findings: Iterable[FindingT]) -> tuple[FindingT, ...]:
    """Reproduce the old priority/length winner algorithm exactly."""

    ordered = sorted(findings, key=finding_order)
    kept: list[FindingT] = []
    for finding in ordered:
        overlaps = [
            existing
            for existing in kept
            if existing.file == finding.file
            and not (finding.end <= existing.start or finding.start >= existing.end)
        ]
        if not overlaps:
            kept.append(finding)
            continue
        best = max(
            overlaps,
            key=lambda item: (
                ENTITY_PRIORITY.get(item.entity_type, 0),
                item.end - item.start,
            ),
        )
        finding_rank = (
            ENTITY_PRIORITY.get(finding.entity_type, 0),
            finding.end - finding.start,
        )
        best_rank = (
            ENTITY_PRIORITY.get(best.entity_type, 0),
            best.end - best.start,
        )
        if finding_rank > best_rank:
            kept = [item for item in kept if item not in overlaps]
            kept.append(finding)
    return tuple(sorted(kept, key=lambda item: (item.file, item.start, item.end)))


def resolve_overlaps(findings: Iterable[FindingT]) -> OverlapResolution[FindingT]:
    """Preserve overlap groups while returning unchanged legacy output."""

    materialized = tuple(findings)
    return OverlapResolution(
        groups=build_overlap_groups(materialized),
        selected=select_legacy_findings(materialized),
    )
