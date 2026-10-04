"""Decode source files without silently replacing or discarding bytes.

This module is the first, isolated part of the future COBOL/JCL region reader.
It records which decoder succeeded and which newline convention the decoded
file uses.  It does not classify source regions, change scanner behaviour, or
write output files.

The fallback order is intentional: strict UTF-8, strict Windows-1252, then
Latin-1.  Windows-1252 must come before Latin-1 so bytes used for punctuation,
such as a curly apostrophe, retain their intended character.  Many ordinary
Western European bytes mean the same thing in Windows-1252 and Latin-1, so
those encodings cannot always be distinguished from bytes alone; in that case
the first successful decoder is the stable result.

Fallback decoding is not yet approval to anonymize a file.  The next migration
step adds source plausibility scoring and EBCDIC rejection before these
single-byte fallbacks are connected to production scanning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


UTF8 = "utf-8"
WINDOWS_1252 = "cp1252"
LATIN1 = "latin-1"

NEWLINE_NONE = "none"
NEWLINE_LF = "lf"
NEWLINE_CRLF = "crlf"
NEWLINE_CR = "cr"
NEWLINE_MIXED = "mixed"

UTF8_BOM = b"\xef\xbb\xbf"
UTF16_LE_BOM = b"\xff\xfe"
UTF16_BE_BOM = b"\xfe\xff"

_SOURCE_LINE_BREAK_RE = re.compile(r"\r\n|\n|\r")


class UnsupportedSourceEncodingError(ValueError):
    """A declared source encoding cannot be preserved by the current pipeline."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class SourceDecodingError(ValueError):
    """A source file declares an encoding but is invalid in that encoding."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DecodedSource:
    """Exact decoded text together with the decoding facts needed later.

    ``encoding`` is the codec that successfully decoded the original bytes.
    ``newline_convention`` describes the line endings without normalizing
    them, so ``text`` remains an exact decoded representation of the input.
    ``has_bom`` records a UTF-8 BOM removed from ``text`` and restored by
    :meth:`encode`. ``embedded_utf8_multibyte_count`` counts valid UTF-8
    multibyte characters found inside a single-byte fallback. The later
    plausibility gate applies the threshold. ``sha256`` always identifies the
    complete original byte sequence, including any BOM.
    """

    text: str
    encoding: str
    newline_convention: str
    has_bom: bool
    embedded_utf8_multibyte_count: int
    sha256: str

    def encode(self) -> bytes:
        """Encode the current text and restore its original UTF-8 BOM.

        This recreates the original bytes while ``text`` is unchanged.  After
        an edit it can raise :class:`UnicodeEncodeError` when a new character
        is unavailable in the detected single-byte encoding.  The future
        writer must convert that failure to ``NOT_COMPLETE`` rather than emit
        a partially converted file.
        """

        encoded = self.text.encode(self.encoding, errors="strict")
        if self.has_bom:
            return UTF8_BOM + encoded
        return encoded


def decode_source_bytes(data: bytes) -> DecodedSource:
    """Decode source bytes using the pinned, lossless encoding order.

    Every attempt is strict.  In particular, this function never inserts the
    Unicode replacement character.  Latin-1 is the final decoding fallback
    because it maps every byte directly to one Unicode code point; a later
    validation step is responsible for rejecting binary or EBCDIC-looking
    input before production use.
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    if data.startswith(UTF16_LE_BOM):
        raise UnsupportedSourceEncodingError(
            "utf16_le_not_supported",
            "UTF-16 little-endian source is not supported; convert it to UTF-8",
        )
    if data.startswith(UTF16_BE_BOM):
        raise UnsupportedSourceEncodingError(
            "utf16_be_not_supported",
            "UTF-16 big-endian source is not supported; convert it to UTF-8",
        )

    has_bom = data.startswith(UTF8_BOM)
    content = data[len(UTF8_BOM) :] if has_bom else data
    digest = hashlib.sha256(data).hexdigest()

    if has_bom:
        try:
            text = content.decode(UTF8, errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceDecodingError(
                "invalid_utf8_bom",
                "source has a UTF-8 BOM but contains invalid UTF-8 bytes",
            ) from exc
        return DecodedSource(
            text=text,
            encoding=UTF8,
            newline_convention=detect_newline_convention(text),
            has_bom=True,
            embedded_utf8_multibyte_count=0,
            sha256=digest,
        )

    for encoding in (UTF8, WINDOWS_1252):
        try:
            text = content.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
        return DecodedSource(
            text=text,
            encoding=encoding,
            newline_convention=detect_newline_convention(text),
            has_bom=False,
            embedded_utf8_multibyte_count=(
                count_utf8_multibyte_sequences(content)
                if encoding == WINDOWS_1252
                else 0
            ),
            sha256=digest,
        )

    text = content.decode(LATIN1, errors="strict")
    return DecodedSource(
        text=text,
        encoding=LATIN1,
        newline_convention=detect_newline_convention(text),
        has_bom=False,
        embedded_utf8_multibyte_count=count_utf8_multibyte_sequences(content),
        sha256=digest,
    )


def read_source(path: Path) -> DecodedSource:
    """Read and decode one file without modifying it or normalizing newlines."""

    return decode_source_bytes(path.read_bytes())


def detect_newline_convention(text: str) -> str:
    """Return ``lf``, ``crlf``, ``cr``, ``mixed``, or ``none`` for text."""

    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    cr_count = text.count("\r") - crlf_count
    styles = sum(count > 0 for count in (lf_count, crlf_count, cr_count))

    if styles == 0:
        return NEWLINE_NONE
    if styles > 1:
        return NEWLINE_MIXED
    if crlf_count:
        return NEWLINE_CRLF
    if lf_count:
        return NEWLINE_LF
    return NEWLINE_CR


def split_source_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split only at CRLF, LF, or CR source line endings.

    Do not replace this helper with :meth:`str.splitlines`.  Python's generic
    method also treats form feed, vertical tab, NEL, and other Unicode control
    characters as line boundaries.  Those bytes can occur inside legacy source
    records and must not silently change line numbers or offsets.

    Like ``str.splitlines()``, this returns no extra empty item after a final
    line ending.  When ``keepends`` is true, the exact original delimiter stays
    attached to each terminated line.
    """

    lines: list[str] = []
    start = 0
    for match in _SOURCE_LINE_BREAK_RE.finditer(text):
        end = match.end() if keepends else match.start()
        lines.append(text[start:end])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


def count_utf8_multibyte_sequences(data: bytes) -> int:
    """Count independently valid UTF-8 multibyte characters in ``data``.

    This check runs only after whole-file UTF-8 decoding failed.  A valid
    multibyte character inside an otherwise single-byte file is evidence that
    separately encoded fragments may have been combined.  The count is
    deliberately diagnostic, not a decoder or an automatic rejection at this
    step.  Each non-overlapping valid sequence is counted once.
    """

    count = 0
    start = 0
    while start < len(data):
        first_byte = data[start]
        if first_byte < 0xC2 or first_byte > 0xF4:
            start += 1
            continue
        matched_width = None
        for width in (2, 3, 4):
            end = start + width
            if end > len(data):
                break
            try:
                character = data[start:end].decode(UTF8, errors="strict")
            except UnicodeDecodeError:
                continue
            if len(character) == 1 and ord(character) > 0x7F:
                matched_width = width
                break
        if matched_width is None:
            start += 1
        else:
            count += 1
            start += matched_width
    return count
