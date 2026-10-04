"""Small, stable records used to describe detected source text.

This file does not scan files, call models, decide whether text is personal, or
replace anything.  It only records two facts:

* ``Occurrence`` says exactly where a piece of text exists in a source file.
* ``Detection`` says which detector reported that occurrence and how confident
  it was.

Keeping those facts separate allows several detectors to report the same text
without losing which detector supplied each piece of evidence.  File hashes
also make old decisions stale when the source file changes.

Offsets are half-open: ``start`` is included and ``end`` is excluded.
``decoded_*`` offsets refer to Python's decoded text.  ``original_*`` offsets
refer to bytes in the original file.  They remain ``None`` until the source
reader can provide a verified character-to-byte mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Protocol


RECORD_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Old Finding objects allow an empty source.  Detection requires a real detector
# name, so the adapter uses this reserved value and converts it back on return.
LEGACY_UNSPECIFIED_DETECTOR = "legacy_unspecified"


def _stable_hash(parts: list[object]) -> str:
    """Hash structured values without relying on ambiguous string joining."""
    encoded = json.dumps(
        parts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_sha256(value: str, field_name: str) -> str:
    normalized = value.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    record_name: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"Invalid {record_name} fields: {', '.join(details)}")


def _require_schema_version(payload: Mapping[str, object], record_name: str) -> None:
    if payload.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {record_name} schema version: "
            f"{payload.get('schema_version')!r}"
        )


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    # bool is an int subclass, but accepting true/false as an offset would hide
    # malformed audit data.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_integer(payload: Mapping[str, object], key: str) -> int | None:
    value = payload[key]
    if value is None:
        return None
    return _integer(payload, key)


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return _string(payload, key)


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


@dataclass(frozen=True)
class Occurrence:
    """The exact text location that later evidence and decisions refer to.

    ``file`` is the batch-relative path.  The file hash makes a location stale
    as soon as the source changes, preventing an old decision from silently
    applying to new code.
    """

    file: str
    file_sha256: str
    source_text: str
    decoded_start: int
    decoded_end: int
    line: int
    column: int
    region: str
    original_start: int | None = None
    original_end: int | None = None
    context: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.file, "file")
        _require_non_empty(self.source_text, "source_text")
        _require_non_empty(self.region, "region")
        object.__setattr__(
            self,
            "file_sha256",
            _normalize_sha256(self.file_sha256, "file_sha256"),
        )
        if (self.original_start is None) != (self.original_end is None):
            raise ValueError(
                "original_start and original_end must both be set or both be None"
            )
        if self.original_start is not None and self.original_end is not None:
            if self.original_start < 0 or self.original_end <= self.original_start:
                raise ValueError("original offsets must describe a non-empty span")
        if self.decoded_start < 0 or self.decoded_end <= self.decoded_start:
            raise ValueError("decoded offsets must describe a non-empty span")
        if self.decoded_end - self.decoded_start != len(self.source_text):
            raise ValueError("decoded offsets must exactly cover source_text")
        if self.line < 1 or self.column < 1:
            raise ValueError("line and column are one-based and must be positive")

    @property
    def occurrence_id(self) -> str:
        """Stable identity required by the implementation plan."""
        return _stable_hash(
            [
                self.file,
                self.file_sha256,
                self.decoded_start,
                self.decoded_end,
                self.source_text,
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "occurrence_id": self.occurrence_id,
            "file": self.file,
            "file_sha256": self.file_sha256,
            "source_text": self.source_text,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "decoded_start": self.decoded_start,
            "decoded_end": self.decoded_end,
            "line": self.line,
            "column": self.column,
            "region": self.region,
            "context": self.context,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Occurrence":
        expected = {
            "schema_version",
            "occurrence_id",
            "file",
            "file_sha256",
            "source_text",
            "original_start",
            "original_end",
            "decoded_start",
            "decoded_end",
            "line",
            "column",
            "region",
            "context",
        }
        _require_exact_keys(payload, expected, "Occurrence")
        _require_schema_version(payload, "Occurrence")
        occurrence = cls(
            file=_string(payload, "file"),
            file_sha256=_string(payload, "file_sha256"),
            source_text=_string(payload, "source_text"),
            decoded_start=_integer(payload, "decoded_start"),
            decoded_end=_integer(payload, "decoded_end"),
            line=_integer(payload, "line"),
            column=_integer(payload, "column"),
            region=_string(payload, "region"),
            original_start=_optional_integer(payload, "original_start"),
            original_end=_optional_integer(payload, "original_end"),
            context=_string(payload, "context"),
        )
        supplied_id = _string(payload, "occurrence_id")
        if supplied_id != occurrence.occurrence_id:
            raise ValueError("occurrence_id does not match the occurrence contents")
        return occurrence

    @classmethod
    def from_json(cls, value: str) -> "Occurrence":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Occurrence JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class Detection:
    """One detector's report about an occurrence.

    Several detections may point to the same occurrence.  Keeping the detector
    and its version explicit makes comparisons and audit reports reproducible
    when rules, models, or prompts change.  A Detection always covers its
    Occurrence exactly; a detector result with a different source span must use
    a different Occurrence.

    ``matched_value`` is always the exact source text.  A future watchlist
    detector can separately record its canonical entry in ``matched_entry``
    and the matching form in ``variant`` without changing the source value.
    """

    occurrence_id: str
    detector: str
    detector_version: str
    entity_type: str
    score: float
    matched_value: str
    matched_entry: str | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurrence_id",
            _normalize_sha256(self.occurrence_id, "occurrence_id"),
        )
        _require_non_empty(self.detector, "detector")
        _require_non_empty(self.detector_version, "detector_version")
        _require_non_empty(self.entity_type, "entity_type")
        _require_non_empty(self.matched_value, "matched_value")
        if (self.matched_entry is None) != (self.variant is None):
            raise ValueError(
                "matched_entry and variant must both be set or both be None"
            )
        if self.matched_entry is not None:
            _require_non_empty(self.matched_entry, "matched_entry")
        if self.variant is not None:
            _require_non_empty(self.variant, "variant")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a finite number between 0 and 1")
        object.__setattr__(self, "score", float(self.score))
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite number between 0 and 1")

    @property
    def detection_id(self) -> str:
        """Stable identity for this detector and version on the occurrence."""
        return _stable_hash(
            [
                self.occurrence_id,
                self.detector,
                self.detector_version,
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "detection_id": self.detection_id,
            "occurrence_id": self.occurrence_id,
            "detector": self.detector,
            "detector_version": self.detector_version,
            "entity_type": self.entity_type,
            "score": self.score,
            "matched_value": self.matched_value,
            "matched_entry": self.matched_entry,
            "variant": self.variant,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Detection":
        expected = {
            "schema_version",
            "detection_id",
            "occurrence_id",
            "detector",
            "detector_version",
            "entity_type",
            "score",
            "matched_value",
            "matched_entry",
            "variant",
        }
        _require_exact_keys(payload, expected, "Detection")
        _require_schema_version(payload, "Detection")
        detection = cls(
            occurrence_id=_string(payload, "occurrence_id"),
            detector=_string(payload, "detector"),
            detector_version=_string(payload, "detector_version"),
            entity_type=_string(payload, "entity_type"),
            score=_number(payload, "score"),
            matched_value=_string(payload, "matched_value"),
            matched_entry=_optional_string(payload, "matched_entry"),
            variant=_optional_string(payload, "variant"),
        )
        supplied_id = _string(payload, "detection_id")
        if supplied_id != detection.detection_id:
            raise ValueError("detection_id does not match the detection contents")
        return detection

    @classmethod
    def from_json(cls, value: str) -> "Detection":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Detection JSON must contain an object")
        return cls.from_dict(payload)


class FindingLike(Protocol):
    """The fields needed from scanner.Finding without importing the scanner."""

    file: str
    entity_type: str
    text: str
    start: int
    end: int
    line: int
    column: int
    confidence: float
    context: str
    source: str


def records_from_finding(
    finding: FindingLike,
    *,
    file_sha256: str,
    detector_version: str,
    region: str = "unknown",
    original_start: int | None = None,
    original_end: int | None = None,
    matched_entry: str | None = None,
    variant: str | None = None,
) -> tuple[Occurrence, Detection]:
    """Translate today's scanner result into the new records without changing it.

    Original byte offsets remain ``None`` because the current scanner only
    knows decoded character offsets.  A later source-reader step will supply a
    verified byte mapping.  The optional watchlist metadata is kept separate
    from the exact source text in ``matched_value``.
    """
    occurrence = Occurrence(
        file=finding.file,
        file_sha256=file_sha256,
        source_text=finding.text,
        decoded_start=finding.start,
        decoded_end=finding.end,
        line=finding.line,
        column=finding.column,
        region=region,
        original_start=original_start,
        original_end=original_end,
        context=finding.context,
    )
    detection = Detection(
        occurrence_id=occurrence.occurrence_id,
        detector=finding.source or LEGACY_UNSPECIFIED_DETECTOR,
        detector_version=detector_version,
        entity_type=finding.entity_type,
        score=float(finding.confidence),
        matched_value=finding.text,
        matched_entry=matched_entry,
        variant=variant,
    )
    return occurrence, detection


def finding_fields(
    occurrence: Occurrence,
    detection: Detection,
) -> dict[str, object]:
    """Return constructor fields for scanner.Finding during the migration."""
    if detection.occurrence_id != occurrence.occurrence_id:
        raise ValueError("Detection refers to a different occurrence")
    if detection.matched_value != occurrence.source_text:
        raise ValueError("Detection matched_value must equal the occurrence source_text")
    return {
        "file": occurrence.file,
        "entity_type": detection.entity_type,
        "text": occurrence.source_text,
        "start": occurrence.decoded_start,
        "end": occurrence.decoded_end,
        "line": occurrence.line,
        "column": occurrence.column,
        "confidence": detection.score,
        "context": occurrence.context,
        "source": (
            ""
            if detection.detector == LEGACY_UNSPECIFIED_DETECTOR
            else detection.detector
        ),
    }
