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

Fallback decoding is not by itself approval to anonymize a file.  Before a
single-byte fallback is accepted, this module checks its encoding integrity
and printable content.  Files with known COBOL/JCL extensions also receive
source-structure and EBCDIC checks.  The checks are deliberately conservative:
an unsupported file must be reported as ``NOT_COMPLETE`` when this reader is
later connected to production scanning.
"""

from __future__ import annotations

import hashlib
import re
import string
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
_DIVISION_RE = re.compile(
    r"\b(?:IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\b",
    re.IGNORECASE,
)
_COBOL_MARKER_RE = re.compile(
    r"\b(?:PROGRAM-ID|WORKING-STORAGE|FILE-CONTROL|PIC|PICTURE|VALUE|"
    r"MOVE|COPY|DISPLAY|PERFORM|PROCEDURE|SECTION|SELECT|ASSIGN|"
    r"STOP\s+RUN|GOBACK)\b",
    re.IGNORECASE,
)
_LEVEL_NUMBER_RE = re.compile(
    r"^\s*(?:0[1-9]|[1-4][0-9]|66|77|78|88)\s+[A-Z0-9]",
    re.IGNORECASE,
)
_FIXED_SOURCE_LINE_RE = re.compile(r"^[ 0-9A-Z]{6}[ */Dd-]", re.IGNORECASE)
_FREE_COMMENT_RE = re.compile(r"^\s*\*>")
_JCL_RECORD_RE = re.compile(r"^//(?:\*|[A-Z0-9#$@]{1,8}(?:\s|$))", re.IGNORECASE)

SOURCE_PLAUSIBILITY_SUFFIXES = frozenset(
    {".cbl", ".cob", ".cobol", ".cpy", ".jcl", ".proc"}
)

# These thresholds are named so the pilot can measure and review them.  They
# are safety policy, not guesses hidden inside individual conditionals.
MAX_EXCLUDED_C0_RATIO = 0.005
MIN_PRINTABLE_RATIO = 0.95
MIN_SOURCE_PLAUSIBILITY_SCORE = 2
MAX_EMBEDDED_UTF8_MULTIBYTE_COUNT = 0
MIN_EBCDIC_PLAUSIBILITY_SCORE = 4
EBCDIC_SCORE_MARGIN = 3
MIN_EBCDIC_AT_SIGNS = 3
MIN_EBCDIC_AT_SIGN_RATIO = 0.08
MAX_EBCDIC_ASCII_LETTER_RATIO = 0.20
MIN_EBCDIC_BYTE_SIGNATURE_LENGTH = 8
MIN_EBCDIC_SPACE_COUNT = 2
MIN_EBCDIC_SPACE_RATIO = 0.06
MIN_EBCDIC_ALNUM_RATIO = 0.60
MAX_EBCDIC_ASCII_RANGE_RATIO = 0.10


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
    :meth:`encode`. ``embedded_utf8_multibyte_count`` counts the narrow
    C2/C3 UTF-8 sequences that commonly reveal UTF-8 text embedded inside a
    single-byte fallback. The plausibility gate applies the threshold.
    ``sha256`` always identifies the complete original byte sequence,
    including any BOM.
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


def decode_source_bytes(
    data: bytes,
    *,
    source_path: str | Path,
) -> DecodedSource:
    """Decode source bytes using the pinned, lossless encoding order.

    Every attempt is strict.  In particular, this function never inserts the
    Unicode replacement character.  Latin-1 is the final decoding fallback
    because it maps every byte directly to one Unicode code point.  Universal
    byte checks reject NUL and control-heavy input.  Single-byte fallbacks are
    additionally required to look like COBOL/JCL rather than binary, mixed
    encoding, or EBCDIC source.  COBOL/JCL structural plausibility is required
    only when the required ``source_path`` has a known source extension.  The
    path is mandatory so callers cannot silently select weaker validation.
    Other scanned text formats are not expected to contain COBOL syntax, but
    every single-byte fallback still receives the encoding-independent EBCDIC
    byte-signature check.
    """

    if source_path is None:
        raise TypeError("source_path is required and cannot be None")
    if not isinstance(source_path, (str, Path)):
        raise TypeError("source_path must be a string or pathlib.Path")
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
    _validate_no_nul_bytes(content)
    require_source_plausibility = _requires_source_plausibility(source_path)
    if _has_ebcdic_byte_signature(content):
        _raise_suspected_ebcdic()

    if has_bom:
        try:
            text = content.decode(UTF8, errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceDecodingError(
                "invalid_utf8_bom",
                "source has a UTF-8 BOM but contains invalid UTF-8 bytes",
            ) from exc
        _validate_control_character_ratio(content)
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
        embedded_utf8_multibyte_count = (
            count_utf8_multibyte_sequences(content)
            if encoding == WINDOWS_1252
            else 0
        )
        if encoding == WINDOWS_1252:
            _validate_single_byte_fallback(
                content,
                text,
                embedded_utf8_multibyte_count,
                require_source_plausibility=require_source_plausibility,
            )
        else:
            _validate_control_character_ratio(content)
        return DecodedSource(
            text=text,
            encoding=encoding,
            newline_convention=detect_newline_convention(text),
            has_bom=False,
            embedded_utf8_multibyte_count=embedded_utf8_multibyte_count,
            sha256=digest,
        )

    text = content.decode(LATIN1, errors="strict")
    embedded_utf8_multibyte_count = count_utf8_multibyte_sequences(content)
    _validate_single_byte_fallback(
        content,
        text,
        embedded_utf8_multibyte_count,
        require_source_plausibility=require_source_plausibility,
    )
    return DecodedSource(
        text=text,
        encoding=LATIN1,
        newline_convention=detect_newline_convention(text),
        has_bom=False,
        embedded_utf8_multibyte_count=embedded_utf8_multibyte_count,
        sha256=digest,
    )


def read_source(path: Path) -> DecodedSource:
    """Read and decode one file without modifying it or normalizing newlines."""

    return decode_source_bytes(path.read_bytes(), source_path=path)


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
    """Count C2/C3 UTF-8 pairs that are strong mixed-encoding evidence.

    This check runs only after whole-file UTF-8 decoding failed.  UTF-8 pairs
    beginning with C2 or C3 appear as the characteristic ``Â``/``Ã`` mojibake
    when decoded as cp1252.  Other valid-looking pairs are ignored because
    ordinary cp1252 characters can accidentally form them; for example,
    ``C9 94`` and ``C8 A0`` are not reliable mixed-encoding evidence.
    """

    count = 0
    start = 0
    while start + 1 < len(data):
        if data[start] in (0xC2, 0xC3) and 0x80 <= data[start + 1] <= 0xBF:
            count += 1
            start += 2
        else:
            start += 1
    return count


def source_plausibility_score(text: str) -> int:
    """Return a small structural score for COBOL, copybook, or JCL text.

    This is an encoding guard, not a parser and not a name detector.  It uses
    several independent, easy-to-audit source markers so that one accidental
    keyword is not enough to approve arbitrary fallback-decoded data.  Fixed
    and free comments count because a source file may legitimately contain
    only comments, while copybook level numbers and JCL records let fragments
    pass without requiring a full COBOL division header.
    """

    score = min(len(_DIVISION_RE.findall(text)), 2) * 4
    score += min(len(_COBOL_MARKER_RE.findall(text)), 3)

    fixed_code_lines = 0
    fixed_comment_lines = 0
    free_comment_lines = 0
    level_number_lines = 0
    jcl_lines = 0
    for line in split_source_lines(text):
        if _FIXED_SOURCE_LINE_RE.match(line):
            if line[6] in "*/Dd":
                fixed_comment_lines += 1
            else:
                fixed_code_lines += 1
        if _FREE_COMMENT_RE.match(line):
            free_comment_lines += 1
        if _LEVEL_NUMBER_RE.match(line):
            level_number_lines += 1
        if _JCL_RECORD_RE.match(line):
            jcl_lines += 1

    score += min(fixed_code_lines, 2)
    score += min(fixed_comment_lines, 2) * 2
    score += min(free_comment_lines, 2) * 2
    score += min(level_number_lines, 2) * 3
    score += min(jcl_lines, 2) * 3
    return score


def _validate_no_nul_bytes(data: bytes) -> None:
    """Reject NUL bytes before any text can be accepted."""

    if b"\x00" in data:
        raise SourceDecodingError(
            "nul_byte",
            "source contains a NUL byte and may be binary or UTF-16 data",
        )


def _validate_control_character_ratio(data: bytes) -> None:
    """Reject excess C0 controls while allowing one final DOS EOF marker.

    A trailing ``0x1A`` is excluded only from this calculation.  It remains in
    the original bytes and decoded text, so :meth:`DecodedSource.encode`
    continues to reproduce the input exactly.
    """

    ratio_data = data[:-1] if data.endswith(b"\x1a") else data
    if not ratio_data:
        return
    excluded_controls = sum(_is_excluded_c0(byte) for byte in ratio_data)
    if excluded_controls / len(ratio_data) > MAX_EXCLUDED_C0_RATIO:
        raise SourceDecodingError(
            "excessive_control_characters",
            "source contains too many disallowed control characters",
        )


def _validate_single_byte_fallback(
    data: bytes,
    text: str,
    embedded_utf8_multibyte_count: int,
    *,
    require_source_plausibility: bool,
) -> None:
    """Approve cp1252/Latin-1 only when it safely resembles source text."""

    # This must run for every path before the Western control-byte check.
    # EBCDIC record separators such as 0x15 would otherwise hide the more
    # useful unsupported-encoding diagnosis.
    if _looks_like_ebcdic(
        data,
        text,
        use_source_structure=require_source_plausibility,
    ):
        _raise_suspected_ebcdic()

    _validate_control_character_ratio(data)

    if embedded_utf8_multibyte_count > MAX_EMBEDDED_UTF8_MULTIBYTE_COUNT:
        raise SourceDecodingError(
            "mixed_encoding_suspected",
            "single-byte source contains embedded UTF-8 multibyte sequences",
        )

    printable_ratio = _printable_ratio(text)
    if printable_ratio < MIN_PRINTABLE_RATIO:
        raise SourceDecodingError(
            "low_printable_ratio",
            "single-byte source contains too many non-printable characters",
        )

    if (
        require_source_plausibility
        and source_plausibility_score(text) < MIN_SOURCE_PLAUSIBILITY_SCORE
    ):
        raise SourceDecodingError(
            "implausible_source",
            "single-byte input does not contain enough COBOL/JCL structure",
        )


def _requires_source_plausibility(source_path: str | Path) -> bool:
    """Return whether a path is expected to contain COBOL/JCL structure."""

    return Path(source_path).suffix.lower() in SOURCE_PLAUSIBILITY_SUFFIXES


def _looks_like_ebcdic(
    data: bytes,
    western_text: str,
    *,
    use_source_structure: bool,
) -> bool:
    """Detect EBCDIC bytes, with optional COBOL/JCL structure comparison.

    The byte signature is format-independent and protects every scanned file.
    Trial-decoding structure scores are used only for extensions that promise
    COBOL or JCL content, so generic text is never required to contain source
    keywords.
    """

    if _has_ebcdic_byte_signature(data):
        return True
    if not use_source_structure:
        return False

    western_score = source_plausibility_score(western_text)
    ebcdic_score = max(
        source_plausibility_score(data.decode(codec, errors="strict"))
        for codec in ("cp037", "cp500")
    )
    materially_stronger = (
        ebcdic_score >= MIN_EBCDIC_PLAUSIBILITY_SCORE
        and ebcdic_score >= western_score + EBCDIC_SCORE_MARGIN
    )

    text_length = max(len(western_text), 1)
    at_sign_count = western_text.count("@")
    ascii_letter_count = sum(
        character in string.ascii_letters for character in western_text
    )
    western_signature = (
        at_sign_count >= MIN_EBCDIC_AT_SIGNS
        and at_sign_count / text_length >= MIN_EBCDIC_AT_SIGN_RATIO
        and ascii_letter_count / text_length <= MAX_EBCDIC_ASCII_LETTER_RATIO
        and ebcdic_score > western_score
    )
    return materially_stronger or western_signature


def _has_ebcdic_byte_signature(data: bytes) -> bool:
    """Recognize common EBCDIC space, letter, and digit ranges.

    EBCDIC text commonly uses ``0x40`` for spaces, ``0xC1`` through ``0xE9``
    for uppercase letter groups, ``0x81`` through ``0xA9`` for lowercase
    letter groups, and ``0xF0`` through ``0xF9`` for digits.  The same input
    normally contains very few bytes in the ASCII letter area. Requiring all
    signals avoids treating ordinary cp1252 ``@`` characters or email
    addresses as EBCDIC.
    """

    if len(data) < MIN_EBCDIC_BYTE_SIGNATURE_LENGTH:
        return False

    byte_count = len(data)
    space_count = data.count(0x40)
    non_space_count = byte_count - space_count
    if non_space_count == 0:
        return False
    ebcdic_alnum_count = sum(
        (
            0x81 <= byte <= 0x89
            or 0x91 <= byte <= 0x99
            or 0xA2 <= byte <= 0xA9
            or 0xC1 <= byte <= 0xE9
            or 0xF0 <= byte <= 0xF9
        )
        for byte in data
    )
    ascii_range_count = sum(0x41 <= byte <= 0x7A for byte in data)
    return (
        space_count >= MIN_EBCDIC_SPACE_COUNT
        and space_count / byte_count >= MIN_EBCDIC_SPACE_RATIO
        and ebcdic_alnum_count / non_space_count >= MIN_EBCDIC_ALNUM_RATIO
        and ascii_range_count / non_space_count <= MAX_EBCDIC_ASCII_RANGE_RATIO
    )


def _raise_suspected_ebcdic() -> None:
    """Raise the stable unsupported-encoding result used by later file status."""

    raise UnsupportedSourceEncodingError(
        "suspected_ebcdic",
        "source appears to be EBCDIC; convert it to UTF-8 before anonymizing",
    )


def _printable_ratio(text: str) -> float:
    """Return the share of printable characters or normal source whitespace."""

    if not text:
        return 1.0
    printable = sum(
        character.isprintable() or character in "\t\n\r\f" for character in text
    )
    return printable / len(text)


def _is_excluded_c0(byte: int) -> bool:
    """Match the C0 byte ranges excluded by the decoding policy."""

    return byte <= 0x08 or byte == 0x0B or 0x0E <= byte <= 0x1F
