"""Replacement generation and anonymization helpers."""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import Finding, iter_all_files, is_text_candidate, read_text, relative_name


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


def group_findings(findings: list[Finding]) -> list[ValueGroup]:
    groups: dict[tuple[str, str], ValueGroup] = {}
    for finding in findings:
        key = group_key(finding.entity_type, finding.text)
        if key not in groups:
            groups[key] = ValueGroup(
                entity_type=finding.entity_type,
                original=" ".join(finding.text.split()),
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
    if group.entity_type == "MATRICOLA":
        if original.isdigit():
            width = len(original)
            number = int(digits_from_hash(original, salt + ":matricola", width))
            return f"{number:0{width}d}"[-width:]
        width = len(original)
        digits = digits_from_hash(original, salt + ":matricola", max(width - 1, 1))
        return ("M" + digits)[:width]
    return f"ANON_{index:03d}"


def load_mapping(path: Path | None) -> dict[tuple[str, str], str]:
    if not path or not path.exists():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entity_type = (row.get("entity_type") or "").strip()
            original = (row.get("original") or "").strip()
            replacement = (row.get("replacement") or "").strip()
            if not entity_type or not original or not replacement:
                continue
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
                    "original": group.original,
                    "suggested_replacement": suggested_replacement(group, index, salt),
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
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        if group_key(finding.entity_type, finding.text) in replacements:
            by_file.setdefault(finding.file, []).append(finding)

    changed_files = 0
    replacement_count = 0
    for source in iter_all_files(input_path, skip_root=output_dir):
        rel = relative_name(source, input_path)
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not is_text_candidate(source):
            shutil.copy2(source, target)
            continue
        text = read_text(source)
        file_findings = sorted(by_file.get(rel, []), key=lambda item: item.start, reverse=True)
        if not file_findings:
            target.write_text(text, encoding="utf-8", newline="")
            continue
        changed = False
        for finding in file_findings:
            replacement = replacements[group_key(finding.entity_type, finding.text)]
            if not replacement:
                continue
            text = text[: finding.start] + replacement + text[finding.end :]
            changed = True
            replacement_count += 1
        target.write_text(text, encoding="utf-8", newline="")
        if changed:
            changed_files += 1
    return changed_files, replacement_count
