"""Build model-facing free-text records without losing source locations.

COBOL fixed-format literals may continue on the next physical record.  Models
need the joined value to recognise a name, while the writer must replace the
original physical pieces without touching sequence areas, quotes, or line
breaks.  This module keeps those two views together in a small, explicit map.

The records here are only an input view for spaCy and the optional extractor.
They do not decide that anything is a name and never widen a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cobol_layout import COMMENT, LITERAL, TEXT, SourceLayout, SourceRegion


@dataclass(frozen=True)
class LogicalPiece:
    """One contiguous logical-text fragment and its exact source span."""

    logical_start: int
    logical_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class LogicalText:
    """Free text presented to a detector, with a reversible span mapping."""

    text: str
    pieces: tuple[LogicalPiece, ...]
    kind: str = TEXT

    def source_spans(self, logical_start: int, logical_end: int) -> tuple[tuple[int, int], ...]:
        """Map one logical half-open span to its physical source fragments.

        A continued literal may return two spans.  Each is a separate writer
        replacement, so no COBOL syntax between physical records is altered.
        """

        spans: list[tuple[int, int]] = []
        for piece in self.pieces:
            start = max(logical_start, piece.logical_start)
            end = min(logical_end, piece.logical_end)
            if start >= end:
                continue
            source_start = piece.source_start + start - piece.logical_start
            source_end = source_start + end - start
            spans.append((source_start, source_end))
        return tuple(spans)


def extraction_texts(layout: SourceLayout) -> tuple[LogicalText, ...]:
    """Return comments, literals, person paragraphs, and plain-text records.

    Plain COBOL/JCL code is intentionally absent.  Comments include inline
    ``*>`` text because layout classifies that suffix as ``comment``.  Literal
    payloads omit their delimiters so a copied name has an unambiguous source
    mapping.  A continuation is represented as one joined logical value.
    """

    records: list[LogicalText] = []
    records.extend(_single_region_text(layout, region) for region in layout.regions_of_kind(COMMENT, TEXT))
    records.extend(literal_texts(layout))
    return tuple(record for record in records if record.text.strip())


def continued_literal_texts(layout: SourceLayout) -> tuple[LogicalText, ...]:
    """Return only joined fixed-format literals for the supplemental spaCy pass."""

    return tuple(record for record in literal_texts(layout) if len(record.pieces) > 1)


def literal_texts(layout: SourceLayout) -> tuple[LogicalText, ...]:
    """Return every literal payload, joining adjacent fixed continuations."""

    literals = list(layout.regions_of_kind(LITERAL))
    records: list[LogicalText] = []
    index = 0
    while index < len(literals):
        regions = [literals[index]]
        index += 1
        while index < len(literals) and literals[index].continued:
            regions.append(literals[index])
            index += 1
        record = _literal_record(layout, regions)
        if record.text:
            records.append(record)
    return tuple(records)


def _single_region_text(layout: SourceLayout, region: SourceRegion) -> LogicalText:
    start = region.decoded_start
    text = layout.source.text[start:region.decoded_end]
    return LogicalText(
        text=text,
        pieces=(LogicalPiece(0, len(text), start, region.decoded_end),),
        kind=region.kind,
    )


def _literal_record(layout: SourceLayout, regions: list[SourceRegion]) -> LogicalText:
    source = layout.source.text
    fragments: list[str] = []
    pieces: list[LogicalPiece] = []
    logical_offset = 0
    for region in regions:
        start, end = _literal_payload_span(source, region)
        if end <= start:
            continue
        value = source[start:end]
        fragments.append(value)
        pieces.append(LogicalPiece(logical_offset, logical_offset + len(value), start, end))
        logical_offset += len(value)
    return LogicalText("".join(fragments), tuple(pieces), LITERAL)


def _literal_payload_span(source: str, region: SourceRegion) -> tuple[int, int]:
    """Drop only the opening/closing literal delimiters from one region."""

    start = region.decoded_start
    end = region.decoded_end
    if start < end and source[start] in {"'", '"'}:
        start += 1
    if start < end and source[end - 1] in {"'", '"'}:
        end -= 1
    return start, end
