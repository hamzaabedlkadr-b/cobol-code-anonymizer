"""Provide lightweight source-layout hints without limiting detector coverage.

This module never decides which text detectors may scan. Every detector must
receive the complete :attr:`DecodedSource.text`; regions only describe useful
context for later evidence and replacement decisions. Unknown or ambiguous
text therefore remains visible as ``code`` or ``text`` instead of being
discarded or causing the file to fail.

Only five region kinds exist: comments, quoted literals, code, plain text, and
line endings. The reader understands enough fixed-format COBOL to handle a
continued literal correctly, but it is intentionally not a COBOL/JCL parser.
"""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

from .source_reader import DecodedSource, split_source_lines


COMMENT = "comment"
LITERAL = "literal"
CODE = "code"
TEXT = "text"
LINE_ENDING = "line_ending"

FIXED_COBOL = "fixed_cobol"
FREE_COBOL = "free_cobol"
JCL = "jcl"
PLAIN_TEXT = "plain_text"

COBOL_SUFFIXES = frozenset({".cbl", ".cob", ".cobol", ".cpy"})
JCL_SUFFIXES = frozenset({".jcl", ".proc"})
PLAIN_TEXT_SUFFIXES = frozenset(
    {".txt", ".csv", ".sql", ".dat", ".ctl", ".md", ".xml", ".json", ".log"}
)

_FIXED_LINE_RE = re.compile(r"^[ 0-9A-Z]{6}[ */Dd-]", re.IGNORECASE)
_SOURCE_FORMAT_FREE_RE = re.compile(r"^\s*>>SOURCE\s+FORMAT\s+FREE\b", re.IGNORECASE)
_COBOL_EVIDENCE_RE = re.compile(
    r"\b(?:IDENTIFICATION|ID|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\b|"
    r"\b(?:PROGRAM-ID|PIC|PICTURE|WORKING-STORAGE|PROCEDURE)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceRegion:
    """One exact, non-empty source span carrying a contextual hint.

    Offsets are half-open. Decoded offsets index the complete source text;
    original offsets point to the corresponding bytes. ``continued`` is true
    only for the part of a fixed-format literal on a ``-`` continuation line.
    """

    kind: str
    decoded_start: int
    decoded_end: int
    original_start: int
    original_end: int
    line: int
    column: int
    continued: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {COMMENT, LITERAL, CODE, TEXT, LINE_ENDING}:
            raise ValueError(f"unknown source region kind: {self.kind}")
        if self.decoded_start < 0 or self.decoded_end <= self.decoded_start:
            raise ValueError("region decoded span must be non-empty and ordered")
        if self.original_start < 0 or self.original_end <= self.original_start:
            raise ValueError("region byte span must be non-empty and ordered")
        if self.line < 1 or self.column < 1:
            raise ValueError("region line and column must be one-based")
        if self.continued and self.kind != LITERAL:
            raise ValueError("only literal regions may be continued")


@dataclass(frozen=True)
class SourceLayout:
    """A complete hint partition for one decoded source.

    A complete partition makes it auditable that layout classification did not
    hide a character. Detection still scans ``source.text`` directly and must
    not use these regions as an allowlist.
    """

    source: DecodedSource
    format: str
    regions: tuple[SourceRegion, ...]

    def __post_init__(self) -> None:
        expected_start = 0
        for region in self.regions:
            if region.decoded_start != expected_start:
                raise ValueError("regions must form one complete decoded-text partition")
            if self.source.original_byte_span(
                region.decoded_start, region.decoded_end
            ) != (region.original_start, region.original_end):
                raise ValueError("region byte span does not match source byte mapping")
            expected_start = region.decoded_end
        if expected_start != len(self.source.text):
            raise ValueError("regions must cover the complete decoded source")

    @cached_property
    def region_starts(self) -> tuple[int, ...]:
        return tuple(region.decoded_start for region in self.regions)

    @property
    def detector_text(self) -> str:
        """Return the full text that every detector must scan."""

        return self.source.text

    def text_for(self, region: SourceRegion) -> str:
        return self.source.text[region.decoded_start : region.decoded_end]

    def regions_of_kind(self, *kinds: str) -> tuple[SourceRegion, ...]:
        """Return matching hints in source order; this is not a scan filter."""

        requested = frozenset(kinds)
        return tuple(region for region in self.regions if region.kind in requested)


@dataclass(frozen=True)
class _Line:
    start: int
    content_end: int
    end: int
    number: int


class _Builder:
    """Build ordered hints with exact decoded-to-byte mappings."""

    def __init__(self, source: DecodedSource) -> None:
        self.source = source
        self.regions: list[SourceRegion] = []

    def add(
        self,
        kind: str,
        start: int,
        end: int,
        line: _Line,
        *,
        continued: bool = False,
    ) -> None:
        if end <= start:
            return
        original_start, original_end = self.source.original_byte_span(start, end)
        self.regions.append(
            SourceRegion(
                kind=kind,
                decoded_start=start,
                decoded_end=end,
                original_start=original_start,
                original_end=original_end,
                line=line.number,
                column=start - line.start + 1,
                continued=continued,
            )
        )

    def line_ending(self, line: _Line) -> None:
        self.add(LINE_ENDING, line.content_end, line.end, line)


def classify_source(source: DecodedSource, *, source_path: str | Path) -> SourceLayout:
    """Return broad context hints while preserving full detector coverage."""

    path = _require_source_path(source_path)
    lines = tuple(_iter_lines(source.text))
    format_name = _detect_format(path, source.text, lines)

    if not lines:
        return SourceLayout(source, format_name, ())
    if format_name == FIXED_COBOL:
        regions = _classify_fixed(source, lines)
    elif format_name == FREE_COBOL:
        regions = _classify_free(source, lines)
    elif format_name == JCL:
        regions = _classify_jcl(source, lines)
    else:
        regions = _classify_text(source, lines)
    return SourceLayout(source, format_name, tuple(regions))


def _require_source_path(source_path: str | Path) -> Path:
    if source_path is None:
        raise TypeError("source_path is required and cannot be None")
    if not isinstance(source_path, (str, Path)):
        raise TypeError("source_path must be a string or pathlib.Path")
    return Path(source_path)


def _iter_lines(text: str) -> Iterator[_Line]:
    """Split only on CRLF, LF, or CR through the source-reader helper."""

    start = 0
    for number, value in enumerate(split_source_lines(text, keepends=True), start=1):
        end = start + len(value)
        content_end = end
        if value.endswith("\r\n"):
            content_end -= 2
        elif value.endswith(("\n", "\r")):
            content_end -= 1
        yield _Line(start, content_end, end, number)
        start = end


def _detect_format(path: Path, text: str, lines: tuple[_Line, ...]) -> str:
    """Use known extensions first and content only for extensionless files."""

    suffix = path.suffix.lower()
    contents = [text[line.start : line.content_end] for line in lines]
    meaningful = [value for value in contents if value.strip()]

    if suffix in JCL_SUFFIXES:
        return JCL
    if suffix in PLAIN_TEXT_SUFFIXES:
        return PLAIN_TEXT
    if suffix in COBOL_SUFFIXES:
        return _cobol_format(contents, meaningful)
    if suffix:
        return PLAIN_TEXT

    if any(value.startswith("//") for value in meaningful):
        return JCL
    if any(_COBOL_EVIDENCE_RE.search(value) for value in meaningful):
        return _cobol_format(contents, meaningful)
    return PLAIN_TEXT


def _cobol_format(contents: list[str], meaningful: list[str]) -> str:
    if any(_SOURCE_FORMAT_FREE_RE.match(value) for value in contents[:20]):
        return FREE_COBOL
    fixed_count = sum(bool(_FIXED_LINE_RE.match(value)) for value in meaningful)
    if fixed_count and (len(meaningful) == 1 or fixed_count * 2 >= len(meaningful)):
        return FIXED_COBOL
    return FREE_COBOL


def _classify_text(source: DecodedSource, lines: tuple[_Line, ...]) -> list[SourceRegion]:
    builder = _Builder(source)
    for line in lines:
        builder.add(TEXT, line.start, line.content_end, line)
        builder.line_ending(line)
    return builder.regions


def _identification_text(content: str, in_identification: bool, in_text: bool):
    """Locate comment-entry payloads, ending at the next header or division."""
    division = re.match(r"\s*([A-Z-]+)\s+DIVISION\b", content, re.IGNORECASE)
    if division:
        return division.group(1).upper() in {"ID", "IDENTIFICATION"}, False, None
    if not in_identification:
        return False, False, None
    header = re.match(
        r"\s*(AUTHOR|INSTALLATION|DATE-WRITTEN|DATE-COMPILED|SECURITY|REMARKS)\s*\.",
        content, re.IGNORECASE,
    )
    if header:
        return True, True, header.end()
    if re.match(r"\s*(?:PROGRAM-ID|FUNCTION-ID|CLASS-ID|METHOD-ID|INTERFACE-ID|FACTORY|OBJECT)\s*\.", content, re.IGNORECASE):
        return True, False, None
    return True, in_text, 0 if in_text else None


def _classify_fixed(source: DecodedSource, lines: tuple[_Line, ...]) -> list[SourceRegion]:
    """Classify fixed source while retaining all column areas as code."""

    builder = _Builder(source)
    open_quote: str | None = None
    in_identification, in_text = True, False

    for line in lines:
        content = source.text[line.start : line.content_end]
        indicator = content[6:7] if len(content) >= 7 else ""
        if indicator in {"*", "/"}:
            builder.add(CODE, line.start, line.start + 6, line)
            builder.add(COMMENT, line.start + 6, line.content_end, line)
            open_quote = None
            builder.line_ending(line)
            continue

        code_start = min(line.start + 7, line.content_end)
        code_end = min(line.start + 72, line.content_end)
        builder.add(CODE, line.start, code_start, line)
        in_identification, in_text, payload = _identification_text(
            source.text[code_start:code_end], in_identification, in_text)
        if payload is not None:
            builder.add(CODE, code_start, code_start + payload, line)
            builder.add(TEXT, code_start + payload, code_end, line)
            builder.add(CODE, code_end, line.content_end, line)
            builder.line_ending(line)
            open_quote = None
            continue
        carried_quote = open_quote if indicator == "-" else None
        open_quote = _add_quoted_segments(
            builder,
            line,
            code_start,
            code_end,
            carried_quote,
            fixed_continuation=indicator == "-",
            allow_inline_comment=True,
        )
        builder.add(CODE, code_end, line.content_end, line)
        builder.line_ending(line)
    return builder.regions


def _classify_free(source: DecodedSource, lines: tuple[_Line, ...]) -> list[SourceRegion]:
    builder = _Builder(source)
    in_identification, in_text = True, False
    for line in lines:
        content = source.text[line.start : line.content_end]
        if re.match(r"^\s*\*>", content):
            builder.add(COMMENT, line.start, line.content_end, line)
        else:
            in_identification, in_text, payload = _identification_text(
                content, in_identification, in_text)
            if payload is not None:
                builder.add(CODE, line.start, line.start + payload, line)
                builder.add(TEXT, line.start + payload, line.content_end, line)
            else:
                _add_quoted_segments(
                    builder, line, line.start, line.content_end, None,
                    fixed_continuation=False, allow_inline_comment=True)

        builder.line_ending(line)
    return builder.regions


def _classify_jcl(source: DecodedSource, lines: tuple[_Line, ...]) -> list[SourceRegion]:
    builder = _Builder(source)
    for line in lines:
        content = source.text[line.start : line.content_end]
        if content.startswith("//*"):
            builder.add(COMMENT, line.start, line.content_end, line)
        else:
            _add_quoted_segments(
                builder,
                line,
                line.start,
                line.content_end,
                None,
                fixed_continuation=False,
                allow_inline_comment=False,
            )
        builder.line_ending(line)
    return builder.regions


def _add_quoted_segments(
    builder: _Builder,
    line: _Line,
    start: int,
    end: int,
    open_quote: str | None,
    *,
    fixed_continuation: bool,
    allow_inline_comment: bool,
) -> str | None:
    """Add hints and return an unclosed quote, if any.

    In fixed format, the first matching quote after indentation on a ``-``
    line re-opens the carried literal. It is not its closing delimiter.
    """

    text = builder.source.text
    segment_start = start
    position = start
    quote = open_quote
    literal_start: int | None = start if quote is not None else None
    continued = fixed_continuation and quote is not None

    if continued:
        while position < end and text[position].isspace():
            position += 1
        builder.add(CODE, segment_start, position, line)
        if position < end and text[position] == quote:
            literal_start = position
            position += 1
        else:
            quote = None
            literal_start = None
            continued = False
            segment_start = position

    while position < end:
        character = text[position]
        if quote is None:
            if allow_inline_comment and text.startswith("*>", position):
                builder.add(CODE, segment_start, position, line)
                builder.add(COMMENT, position, end, line)
                return None
            if character in {"'", '"'}:
                builder.add(CODE, segment_start, position, line)
                literal_start = position
                quote = character
            position += 1
            continue

        if character == quote and position + 1 < end and text[position + 1] == quote:
            position += 2
            continue
        if character == quote:
            builder.add(
                LITERAL,
                literal_start if literal_start is not None else segment_start,
                position + 1,
                line,
                continued=continued,
            )
            segment_start = position + 1
            quote = None
            literal_start = None
            continued = False
        position += 1

    if quote is not None:
        builder.add(
            LITERAL,
            literal_start if literal_start is not None else segment_start,
            end,
            line,
            continued=continued,
        )
    else:
        builder.add(CODE, segment_start, end, line)
    return quote
