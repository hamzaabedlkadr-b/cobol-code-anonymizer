"""Replacement generation and anonymization helpers."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import Finding, iter_all_files, is_text_candidate, relative_name, fold_watchlist_value
from .text_matching import physical_replacement
from .source_reader import (
    SourceDecodingError,
    UnsupportedSourceEncodingError,
    UTF8_BOM,
    DecodedSource,
    read_source,
    split_source_lines,
)

NON_LINKABLE_ENTITIES = {"MATRICOLA", "SUSPECTED_MATRICOLA"}


@dataclass
class ValueGroup:
    entity_type: str
    original: str
    key: tuple[str, str]
    findings: list[Finding] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.findings)

    @property
    def locations(self) -> str:
        sample = [f"{finding.file}:{finding.line}" for finding in self.findings[:8]]
        suffix = "" if len(self.findings) <= 8 else f" +{len(self.findings) - 8} more"
        return ", ".join(sample) + suffix


def normalize_value(value: str) -> str:
    return " ".join(value.strip().split()).upper()


def group_key(entity_type: str, value: str) -> tuple[str, str]:
    return entity_type, normalize_value(value)


def finding_key(finding: Finding) -> tuple[str, str]:
    if finding.entity_type == "NAME":
        return "NAME", fold_watchlist_value(finding.logical_candidate or finding.text)
    if finding.entity_type in NON_LINKABLE_ENTITIES:
        normalized = normalize_value(finding.text)
        return finding.entity_type, f"OCCURRENCE:{finding.file}:{finding.line}:{finding.column}:{normalized}"
    return group_key(finding.entity_type, finding.text)


def group_findings(findings: list[Finding]) -> list[ValueGroup]:
    groups: dict[tuple[str, str], ValueGroup] = {}
    for finding in findings:
        key = finding_key(finding)
        if key not in groups:
            groups[key] = ValueGroup(
                entity_type=finding.entity_type,
                original=" ".join((finding.logical_candidate or finding.text).split()),
                key=key,
            )
        groups[key].findings.append(finding)
    return sorted(
        groups.values(),
        key=lambda group: (entity_sort_order(group.entity_type), group.original.upper()),
    )


def entity_sort_order(entity_type: str) -> int:
    order = {
        "NAME": 10,
        "MATRICOLA": 20,
        "SUSPECTED_MATRICOLA": 21,
        "IBAN": 30,
        "CODICE_FISCALE": 40,
        "EMAIL": 50,
        "PHONE": 60,
    }
    return order.get(entity_type, 99)


def digest(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value.upper()}".encode("utf-8")).hexdigest()


def digits_from_hash(value: str, salt: str, length: int) -> str:
    raw = str(int(digest(value, salt), 16))
    while len(raw) < length:
        raw += raw
    return raw[:length]


def letters_from_hash(value: str, salt: str, length: int) -> str:
    number = int(digest(value, salt), 16)
    chars = []
    for _ in range(length):
        chars.append(chr(ord("A") + (number % 26)))
        number //= 26
    return "".join(chars)


def iban_mod97(value: str) -> int:
    converted = []
    for char in value:
        if char.isdigit():
            converted.append(char)
        elif "A" <= char <= "Z":
            converted.append(str(ord(char) - ord("A") + 10))
    remainder = 0
    for char in "".join(converted):
        remainder = (remainder * 10 + int(char)) % 97
    return remainder


def iban_check_digits(country: str, bban: str) -> str:
    check = 98 - iban_mod97(bban + country + "00")
    return f"{check:02d}"


def pseudonymize_iban(value: str, salt: str) -> str:
    compact = "".join(value.upper().split())
    if len(compact) < 5:
        return "IBAN_ANON"
    country = compact[:2]
    body_len = max(len(compact) - 4, 0)
    if country == "IT" and body_len == 23:
        bban = (
            letters_from_hash(compact, salt + ":iban-cin", 1)
            + digits_from_hash(compact, salt + ":iban-bank", 10)
            + digits_from_hash(compact, salt + ":iban-account", 12)
        )
    else:
        bban = letters_from_hash(compact, salt + ":iban", body_len)
    return country + iban_check_digits(country, bban) + bban


CF_ODD = {
    **{str(index): value for index, value in enumerate([1, 0, 5, 7, 9, 13, 15, 17, 19, 21])},
    **dict(
        zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23],
        )
    ),
}
CF_EVEN = {
    **{str(index): index for index in range(10)},
    **{char: index for index, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
}


def codice_fiscale_check_char(first_15: str) -> str:
    total = 0
    for index, char in enumerate(first_15):
        total += CF_ODD[char] if index % 2 == 0 else CF_EVEN[char]
    return chr(ord("A") + total % 26)


def pseudonymize_codice_fiscale(value: str, salt: str) -> str:
    compact = value.upper()
    first_15 = (
        letters_from_hash(compact, salt + ":cf1", 6)
        + digits_from_hash(compact, salt + ":cf2", 2)
        + letters_from_hash(compact, salt + ":cf3", 1)
        + digits_from_hash(compact, salt + ":cf4", 2)
        + letters_from_hash(compact, salt + ":cf5", 1)
        + digits_from_hash(compact, salt + ":cf6", 3)
    )
    return first_15 + codice_fiscale_check_char(first_15)


def suggested_replacement(group: ValueGroup, index: int, salt: str) -> str:
    original = group.original
    if group.entity_type == "NAME":
        parts = original.split()
        return f"Nome{index:03d} Cognome{index:03d}" if len(parts) > 1 else f"Nome{index:03d}"
    if group.entity_type == "IBAN":
        return pseudonymize_iban(original, salt)
    if group.entity_type == "CODICE_FISCALE":
        return pseudonymize_codice_fiscale(original, salt)
    if group.entity_type == "EMAIL":
        return f"user{index:03d}@example.invalid"
    if group.entity_type == "PHONE":
        digits = "".join(char for char in original if char.isdigit())
        return "0" * max(len(digits), 8)
    if group.entity_type in {"MATRICOLA", "SUSPECTED_MATRICOLA"}:
        compact = "".join(char for char in original if char.isdigit())
        seed = f"{original}:{group.key[1]}"
        if len(compact) == 7:
            prefix = "567"[int(digest(seed, salt + ":matricola-prefix"), 16) % 3]
            return prefix + digits_from_hash(seed, salt + ":matricola", 6)
        return digits_from_hash(seed, salt + ":matricola", len(compact))
    return f"ANON_{index:03d}"


def load_mapping(path: Path | None) -> dict[tuple[str, str], str]:
    """Load replacements for fields other than names."""

    if not path or not path.exists():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entity_type = (row.get("entity_type") or "").strip().upper()
            key = (row.get("key") or "").strip()
            original = (row.get("original") or "").strip()
            replacement = (row.get("replacement") or "").strip()
            if entity_type == "NAME" or not entity_type or not replacement:
                continue
            if key:
                mapping[(entity_type, key)] = replacement
            elif original and entity_type not in NON_LINKABLE_ENTITIES:
                mapping[group_key(entity_type, original)] = replacement
    return mapping


def write_mapping_template(
    path: Path,
    groups: list[ValueGroup],
    replacements: dict[tuple[str, str], str] | None,
    salt: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "entity_type",
            "key",
            "original",
            "suggested_replacement",
            "replacement",
            "hits",
            "locations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, group in enumerate(groups, start=1):
            writer.writerow(
                {
                    "entity_type": group.entity_type,
                    "key": group.key[1],
                    "original": group.original,
                    "suggested_replacement": (replacements or {}).get(group.key, suggested_replacement(group, index, salt)),
                    "replacement": (replacements or {}).get(group.key, ""),
                    "hits": group.count,
                    "locations": group.locations,
                }
            )


def apply_replacements(
    input_path: Path,
    output_dir: Path,
    findings: list[Finding],
    replacements: dict[tuple[str, str], str],
    not_complete_files: list[dict[str, str]] | None = None,
    skipped_files: list[dict[str, str]] | None = None,
    blocked_files: set[str] | None = None,
    written_files: list[str] | None = None,
    line_counts: dict[str, int] | None = None,
    skip_roots: list[Path] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Write safely encoded, byte-width-preserving replacements.

    The caller must make an explicit replacement choice for every remaining
    NAME finding.  This defensive check prevents a new UI or import path from
    recreating the old silent-skip leak. Each text source is decoded again
    through ``source_reader`` so the writer uses the same encoding, BOM, and
    byte boundaries as the scanner. A per-file write failure is recorded as
    ``NOT_COMPLETE`` and does not overwrite that file's output.
    """

    for finding in findings:
        if finding.entity_type != "NAME":
            continue
        replacement = replacements.get(finding_key(finding))
        if not replacement or not replacement.strip():
            raise ValueError(
                "every NAME finding requires a non-blank replacement: "
                f"{finding.file}:{finding.line}:{finding.column}"
            )

    incomplete = not_complete_files if not_complete_files is not None else []
    skipped = skipped_files if skipped_files is not None else []
    blocked = blocked_files or set()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        if finding_key(finding) in replacements:
            by_file.setdefault(finding.file, []).append(finding)

    changed_files = 0
    replacement_count = 0
    for source in iter_all_files(input_path, skip_root=[output_dir, *(skip_roots or [])]):
        rel = relative_name(source, input_path)
        target = output_dir / rel
        if rel in blocked:
            continue
        try:
            if not is_text_candidate(source):
                # Non-text files are deliberately not copied.  The shareable
                # output contains only files that passed the scanner's text
                # reader; callers receive an audit row for every omission.
                skipped.append(
                    {
                        "file": rel,
                        "reason": "unsupported_file_type",
                    }
                )
                continue

            decoded_source = read_source(source)
            if source_hashes is not None and decoded_source.sha256 != source_hashes.get(rel):
                raise ReplacementWriteError("source_changed", "source changed after scanning; withhold")
            file_findings = by_file.get(rel, [])
            output_text = apply_file_replacements(
                decoded_source,
                file_findings,
                replacements,
            )
            output_bytes = encode_output_text(decoded_source, output_text)
            validate_output_layout(decoded_source, output_text, output_bytes)
            write_output_bytes_atomically(source, target, output_bytes)
            if line_counts is not None:
                line_counts[rel] = lines_past_column_72(decoded_source.text, output_text)
            if written_files is not None:
                written_files.append(rel)
        except (
            SourceDecodingError,
            UnsupportedSourceEncodingError,
            ReplacementWriteError,
            UnicodeEncodeError,
            OSError,
        ) as exc:
            record_not_complete_file(incomplete, rel, exc)
            continue

        if file_findings:
            changed_files += 1
            replacement_count += len(file_findings)
    return changed_files, replacement_count


class ReplacementWriteError(ValueError):
    """One source file could not be modified without breaking its layout."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def apply_file_replacements(
    source: DecodedSource,
    findings: list[Finding],
    replacements: dict[tuple[str, str], str],
) -> str:
    """Apply one file's spans after validating their exact source boundaries."""

    ordered = sorted(findings, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end > current.start:
            raise ReplacementWriteError(
                "overlapping_replacements",
                "replacement spans overlap and cannot be applied safely",
            )

    output_text = source.text
    for finding in reversed(ordered):
        if not 0 <= finding.start <= finding.end <= len(source.text):
            raise ReplacementWriteError(
                "replacement_span_out_of_range",
                f"replacement span is outside the source: {finding.file}:{finding.line}",
            )
        if source.text[finding.start : finding.end] != finding.text:
            raise ReplacementWriteError(
                "replacement_text_mismatch",
                f"replacement text no longer matches the source: {finding.file}:{finding.line}",
            )

        replacement = physical_replacement(finding, replacements[finding_key(finding)])
        if "\r" in replacement or "\n" in replacement:
            raise ReplacementWriteError(
                "replacement_contains_line_break",
                f"replacement contains a line break: {finding.file}:{finding.line}",
            )
        output_text = output_text[:finding.start] + replacement + output_text[finding.end:]

    return output_text


def encode_output_text(source: DecodedSource, text: str) -> bytes:
    """Encode edited text in the source encoding and restore its UTF-8 BOM."""

    try:
        encoded = text.encode(source.encoding, errors="strict")
    except UnicodeEncodeError as exc:
        raise ReplacementWriteError("replacement_not_encodable", "replacement cannot use the source encoding") from exc
    return UTF8_BOM + encoded if source.has_bom else encoded


def validate_output_layout(
    source: DecodedSource,
    output_text: str,
    output_bytes: bytes,
) -> None:
    """Prove output kept source byte width and exact CRLF/LF/CR record shape."""

    original_lines = split_source_lines(source.text, keepends=True)
    output_lines = split_source_lines(output_text, keepends=True)
    if len(output_lines) != len(original_lines):
        raise ReplacementWriteError("line_count_changed", "edited source changed its record count")
    endings = lambda lines: [line[len(line.rstrip("\r\n")):] for line in lines]
    if endings(original_lines) != endings(output_lines):
        raise ReplacementWriteError("line_endings_changed", "edited source changed its line endings")


def write_output_bytes_atomically(source: Path, target: Path, data: bytes) -> None:
    """Replace one output file only after its complete byte sequence is ready."""

    temporary_path = _temporary_output_path(target)
    try:
        with temporary_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except OSError:
        _remove_temporary_file(temporary_path)
        raise


def copy_file_atomically(source: Path, target: Path) -> None:
    """Copy non-text input without exposing a partially written target file."""

    temporary_path = _temporary_output_path(target)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, target)
    except OSError:
        _remove_temporary_file(temporary_path)
        raise


def _temporary_output_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _remove_temporary_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def record_not_complete_file(
    not_complete_files: list[dict[str, str]],
    file_name: str,
    error: Exception,
) -> None:
    """Append the small audit record consumed by the current CLI bridge."""

    reason = getattr(error, "reason", "write_error")
    not_complete_files.append(
        {
            "file": file_name,
            "status": "NOT_COMPLETE",
            "reason": str(reason),
            "message": str(error),
        }
    )


def lines_past_column_72(original: str, output: str) -> int:
    """Count records newly extended past column 72, without truncation."""
    return sum(len(before) <= 72 < len(after)
               for before, after in zip(split_source_lines(original), split_source_lines(output)))


def replacement_spans(findings: list[Finding], replacements: dict[tuple[str, str], str]) -> dict[str, list[tuple[int, int, int]]]:
    """Record source spans and replacement character lengths for residual offsets."""
    spans: dict[str, list[tuple[int, int, int]]] = {}
    for finding in findings:
        if finding_key(finding) in replacements:
            spans.setdefault(finding.file, []).append((finding.start, finding.end, len(physical_replacement(finding, replacements[finding_key(finding)]))))
    return {file: sorted(items) for file, items in spans.items()}
