"""Audit records for evidence, decisions, and final file outcomes.

This module describes what the later evidence and policy stages need to record.
It does not collect evidence, call an LLM, decide whether text is personal, or
modify source files.  Keeping these records independent makes every future
decision serializable, reviewable, and tied to the exact source occurrence that
was evaluated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping


AUDIT_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DECISION_STAGES = {"judge", "verifier", "policy", "manual"}
JUDGE_OUTCOMES = {
    "anonymize_whole",
    "anonymize_part",
    "propose_unchanged",
    "uncertain",
    "error",
}
VERIFIER_OUTCOMES = {"possible", "not_possible", "unsure", "error"}
POLICY_OUTCOMES = {
    "anonymize_whole",
    "anonymize_part",
    "leave_unchanged",
    "review_required",
    "not_complete",
    "error",
}
MANUAL_OUTCOMES = {
    "person",
    "partial_person",
    "checked_non_person",
    "mixed",
    "defer",
    "error",
}
OUTCOMES_BY_STAGE = {
    "judge": JUDGE_OUTCOMES,
    "verifier": VERIFIER_OUTCOMES,
    "policy": POLICY_OUTCOMES,
    "manual": MANUAL_OUTCOMES,
}
MODEL_STAGES = {"judge", "verifier"}
PERSON_SCOPES = {"whole", "partial", "none", "unsure"}
NON_PERSON_CATEGORIES = {
    "none",
    "common_word",
    "date_or_month",
    "place",
    "organization",
    "code_or_identifier",
    "label_or_header",
    "abbreviation",
    "function_words",
    "number_or_symbol",
}
FILE_STATUSES = {
    "PASS",
    "PASS_WITH_OPTIONAL_REVIEW",
    "REVIEW_REQUIRED",
    "NOT_COMPLETE",
}


@dataclass(frozen=True)
class FieldRule:
    """One declarative requirement for a field in a Decision outcome."""

    kind: str
    value: object = None

    def __post_init__(self) -> None:
        if self.kind not in {"required", "forbidden", "must_equal", "optional"}:
            raise ValueError(f"unsupported decision field rule: {self.kind!r}")
        if self.kind == "must_equal" and self.value is None:
            raise ValueError("must_equal rules require a value")
        if self.kind != "must_equal" and self.value is not None:
            raise ValueError(f"{self.kind} rules cannot carry a value")


REQUIRED = FieldRule("required")
FORBIDDEN = FieldRule("forbidden")
OPTIONAL = FieldRule("optional")


def _must_equal(value: object) -> FieldRule:
    return FieldRule("must_equal", value)


DECISION_RULE_FIELDS = (
    "person_scope",
    "person_texts",
    "non_person_category",
    "evidence_quote",
    "reading",
)


def _field_rules(
    *,
    person_scope: FieldRule,
    person_texts: FieldRule,
    non_person_category: FieldRule,
    evidence_quote: FieldRule,
    reading: FieldRule,
) -> dict[str, FieldRule]:
    """Build a complete row so a new decision field cannot be overlooked."""

    return {
        "person_scope": person_scope,
        "person_texts": person_texts,
        "non_person_category": non_person_category,
        "evidence_quote": evidence_quote,
        "reading": reading,
    }


# This table is the single source of truth for the shape of every stage result.
# Empty tuples/strings, a missing scope, and the category ``none`` count as
# absent.  Metadata and error_message have separate provenance checks below.
OUTCOME_RULES: dict[tuple[str, str], dict[str, FieldRule]] = {
    ("judge", "anonymize_whole"): _field_rules(
        person_scope=_must_equal("whole"),
        person_texts=OPTIONAL,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("judge", "anonymize_part"): _field_rules(
        person_scope=_must_equal("partial"),
        person_texts=REQUIRED,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("judge", "propose_unchanged"): _field_rules(
        person_scope=_must_equal("none"),
        person_texts=FORBIDDEN,
        non_person_category=REQUIRED,
        evidence_quote=REQUIRED,
        reading=REQUIRED,
    ),
    ("judge", "uncertain"): _field_rules(
        person_scope=_must_equal("unsure"),
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("judge", "error"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=FORBIDDEN,
    ),
    ("verifier", "possible"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("verifier", "not_possible"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=REQUIRED,
        reading=REQUIRED,
    ),
    ("verifier", "unsure"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("verifier", "error"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=FORBIDDEN,
    ),
    ("policy", "anonymize_whole"): _field_rules(
        person_scope=_must_equal("whole"),
        person_texts=OPTIONAL,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("policy", "anonymize_part"): _field_rules(
        person_scope=_must_equal("partial"),
        person_texts=REQUIRED,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("policy", "leave_unchanged"): _field_rules(
        person_scope=_must_equal("none"),
        person_texts=FORBIDDEN,
        non_person_category=REQUIRED,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("policy", "review_required"): _field_rules(
        person_scope=REQUIRED,
        person_texts=OPTIONAL,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("policy", "not_complete"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=REQUIRED,
    ),
    ("policy", "error"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=FORBIDDEN,
    ),
    ("manual", "person"): _field_rules(
        person_scope=_must_equal("whole"),
        person_texts=OPTIONAL,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("manual", "partial_person"): _field_rules(
        person_scope=_must_equal("partial"),
        person_texts=REQUIRED,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("manual", "checked_non_person"): _field_rules(
        person_scope=_must_equal("none"),
        person_texts=FORBIDDEN,
        non_person_category=REQUIRED,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("manual", "mixed"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=REQUIRED,
    ),
    ("manual", "defer"): _field_rules(
        person_scope=_must_equal("unsure"),
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=REQUIRED,
    ),
    ("manual", "error"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=FORBIDDEN,
        reading=FORBIDDEN,
    ),
}


def _normalize_id(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(values)
    for value in result:
        _non_empty(value, field_name)
    return result


def _id_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of identifiers")
    return tuple(_normalize_id(value, field_name) for value in values)


def _require_choice(value: str, choices: set[str], field_name: str) -> str:
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}")
    return value


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


def _require_schema(payload: Mapping[str, object], record_name: str) -> None:
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {record_name} schema version: "
            f"{payload.get('schema_version')!r}"
        )


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return _string(payload, key)


def _validate_timestamp(value: str, field_name: str) -> str:
    _non_empty(value, field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _json_object(value: str, record_name: str) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{record_name} JSON must contain an object")
    return payload


def _decision_field_is_present(field_name: str, value: object) -> bool:
    """Return whether a decision field carries information rather than its sentinel."""

    if field_name == "person_scope":
        return value is not None
    if field_name == "non_person_category":
        return value != "none"
    return bool(value)


def _validate_outcome_fields(
    stage: str,
    outcome: str,
    values: Mapping[str, object],
) -> None:
    """Apply the one declarative rule row for a stage/outcome pair."""

    rules = OUTCOME_RULES[(stage, outcome)]
    for field_name in DECISION_RULE_FIELDS:
        rule = rules[field_name]
        value = values[field_name]
        present = _decision_field_is_present(field_name, value)
        if rule.kind == "required" and not present:
            raise ValueError(
                f"{stage}/{outcome} requires a non-empty {field_name}"
            )
        if rule.kind == "forbidden" and present:
            raise ValueError(f"{stage}/{outcome} forbids {field_name}")
        if rule.kind == "must_equal" and value != rule.value:
            raise ValueError(
                f"{stage}/{outcome} requires {field_name}={rule.value!r}"
            )


@dataclass(frozen=True)
class Evidence:
    """Deterministic facts collected for one source occurrence."""

    occurrence_id: str
    detection_ids: tuple[str, ...] = ()
    token_classifications: tuple[str, ...] = ()
    structural_patterns: tuple[str, ...] = ()
    person_cues: tuple[str, ...] = ()
    code_vocabulary_hits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurrence_id",
            _normalize_id(self.occurrence_id, "occurrence_id"),
        )
        object.__setattr__(
            self,
            "detection_ids",
            _id_tuple(self.detection_ids, "detection_ids"),
        )
        for field_name in (
            "token_classifications",
            "structural_patterns",
            "person_cues",
            "code_vocabulary_hits",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "occurrence_id": self.occurrence_id,
            "detection_ids": list(self.detection_ids),
            "token_classifications": list(self.token_classifications),
            "structural_patterns": list(self.structural_patterns),
            "person_cues": list(self.person_cues),
            "code_vocabulary_hits": list(self.code_vocabulary_hits),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Evidence":
        expected = {
            "schema_version",
            "occurrence_id",
            "detection_ids",
            "token_classifications",
            "structural_patterns",
            "person_cues",
            "code_vocabulary_hits",
        }
        _require_exact_keys(payload, expected, "Evidence")
        _require_schema(payload, "Evidence")
        return cls(
            occurrence_id=_string(payload, "occurrence_id"),
            detection_ids=tuple(
                _string_value(value, "detection_ids")
                for value in _list(payload, "detection_ids")
            ),
            token_classifications=tuple(
                _string_value(value, "token_classifications")
                for value in _list(payload, "token_classifications")
            ),
            structural_patterns=tuple(
                _string_value(value, "structural_patterns")
                for value in _list(payload, "structural_patterns")
            ),
            person_cues=tuple(
                _string_value(value, "person_cues")
                for value in _list(payload, "person_cues")
            ),
            code_vocabulary_hits=tuple(
                _string_value(value, "code_vocabulary_hits")
                for value in _list(payload, "code_vocabulary_hits")
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "Evidence":
        return cls.from_dict(_json_object(value, "Evidence"))


def _string_value(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must contain only strings")
    return value


@dataclass(frozen=True)
class Decision:
    """A validated outcome produced by one auditable processing stage.

    This is not the raw model response. Model calls keep their raw request and
    response in a separate audit; a judge or verifier Decision is created only
    after its answer passes schema and consistency validation.
    """

    occurrence_id: str
    stage: str
    outcome: str
    person_scope: str | None = None
    person_texts: tuple[str, ...] = ()
    non_person_category: str = "none"
    evidence_quote: str = ""
    reading: str = ""
    error_message: str | None = None
    model_digest: str | None = None
    prompt_version: str | None = None
    reviewer: str | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurrence_id",
            _normalize_id(self.occurrence_id, "occurrence_id"),
        )
        _require_choice(self.stage, DECISION_STAGES, "stage")
        _require_choice(self.outcome, OUTCOMES_BY_STAGE[self.stage], "outcome")
        if self.person_scope is not None:
            _require_choice(self.person_scope, PERSON_SCOPES, "person_scope")
        _require_choice(
            self.non_person_category,
            NON_PERSON_CATEGORIES,
            "non_person_category",
        )
        object.__setattr__(
            self,
            "person_texts",
            _string_tuple(self.person_texts, "person_texts"),
        )
        _string_value(self.evidence_quote, "evidence_quote")
        _string_value(self.reading, "reading")

        if self.stage in MODEL_STAGES:
            _non_empty(self.model_digest, "model_digest")
            _non_empty(self.prompt_version, "prompt_version")
        elif self.model_digest is not None or self.prompt_version is not None:
            raise ValueError("model metadata is only valid for model stages")
        if self.stage == "manual":
            _non_empty(self.reviewer, "reviewer")
            if self.timestamp is None:
                raise ValueError("manual decisions require a timestamp")
        elif self.reviewer is not None:
            raise ValueError("reviewer is only valid for manual decisions")
        if self.timestamp is not None:
            _validate_timestamp(self.timestamp, "timestamp")

        _validate_outcome_fields(
            self.stage,
            self.outcome,
            {field_name: getattr(self, field_name) for field_name in DECISION_RULE_FIELDS},
        )
        if self.outcome == "error":
            _non_empty(self.error_message, "error_message")
        elif self.error_message is not None:
            raise ValueError("error_message is only valid for an error decision")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "occurrence_id": self.occurrence_id,
            "stage": self.stage,
            "outcome": self.outcome,
            "person_scope": self.person_scope,
            "person_texts": list(self.person_texts),
            "non_person_category": self.non_person_category,
            "evidence_quote": self.evidence_quote,
            "reading": self.reading,
            "error_message": self.error_message,
            "model_digest": self.model_digest,
            "prompt_version": self.prompt_version,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Decision":
        expected = {
            "schema_version",
            "occurrence_id",
            "stage",
            "outcome",
            "person_scope",
            "person_texts",
            "non_person_category",
            "evidence_quote",
            "reading",
            "error_message",
            "model_digest",
            "prompt_version",
            "reviewer",
            "timestamp",
        }
        _require_exact_keys(payload, expected, "Decision")
        _require_schema(payload, "Decision")
        return cls(
            occurrence_id=_string(payload, "occurrence_id"),
            stage=_string(payload, "stage"),
            outcome=_string(payload, "outcome"),
            person_scope=_optional_string(payload, "person_scope"),
            person_texts=tuple(
                _string_value(value, "person_texts")
                for value in _list(payload, "person_texts")
            ),
            non_person_category=_string(payload, "non_person_category"),
            evidence_quote=_string(payload, "evidence_quote"),
            reading=_string(payload, "reading"),
            error_message=_optional_string(payload, "error_message"),
            model_digest=_optional_string(payload, "model_digest"),
            prompt_version=_optional_string(payload, "prompt_version"),
            reviewer=_optional_string(payload, "reviewer"),
            timestamp=_optional_string(payload, "timestamp"),
        )

    @classmethod
    def from_json(cls, value: str) -> "Decision":
        return cls.from_dict(_json_object(value, "Decision"))


@dataclass(frozen=True)
class ReplacementSpan:
    """One planned replacement tied to its source occurrence."""

    occurrence_id: str
    decoded_start: int
    decoded_end: int
    replacement: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurrence_id",
            _normalize_id(self.occurrence_id, "occurrence_id"),
        )
        if (
            isinstance(self.decoded_start, bool)
            or not isinstance(self.decoded_start, int)
            or isinstance(self.decoded_end, bool)
            or not isinstance(self.decoded_end, int)
        ):
            raise ValueError("replacement offsets must be integers")
        if self.decoded_start < 0 or self.decoded_end <= self.decoded_start:
            raise ValueError("replacement offsets must describe a non-empty span")
        _non_empty(self.replacement, "replacement")
        if len(self.replacement) != self.decoded_end - self.decoded_start:
            raise ValueError("replacement must have the same decoded width as its span")

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "decoded_start": self.decoded_start,
            "decoded_end": self.decoded_end,
            "replacement": self.replacement,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReplacementSpan":
        expected = {"occurrence_id", "decoded_start", "decoded_end", "replacement"}
        _require_exact_keys(payload, expected, "ReplacementSpan")
        return cls(
            occurrence_id=_string(payload, "occurrence_id"),
            decoded_start=_integer(payload, "decoded_start"),
            decoded_end=_integer(payload, "decoded_end"),
            replacement=_string(payload, "replacement"),
        )


@dataclass(frozen=True)
class ReviewItem:
    """One occurrence that may need required or optional human review."""

    occurrence_id: str
    reason: str
    required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurrence_id",
            _normalize_id(self.occurrence_id, "occurrence_id"),
        )
        _non_empty(self.reason, "reason")
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "reason": self.reason,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReviewItem":
        expected = {"occurrence_id", "reason", "required"}
        _require_exact_keys(payload, expected, "ReviewItem")
        return cls(
            occurrence_id=_string(payload, "occurrence_id"),
            reason=_string(payload, "reason"),
            required=_boolean(payload, "required"),
        )


@dataclass(frozen=True)
class FileResult:
    """The auditable final state of one processed source file.

    ``residual_occurrence_ids`` means possible person occurrences that remain
    readable in the output. Cleared residuals belong in the decision audit and
    are not stored in this final unresolved list.
    """

    file: str
    file_sha256: str
    status: str
    replacement_spans: tuple[ReplacementSpan, ...] = ()
    residual_occurrence_ids: tuple[str, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.file, "file")
        object.__setattr__(
            self,
            "file_sha256",
            _normalize_id(self.file_sha256, "file_sha256"),
        )
        _require_choice(self.status, FILE_STATUSES, "status")
        spans = tuple(self.replacement_spans)
        if not all(isinstance(span, ReplacementSpan) for span in spans):
            raise ValueError("replacement_spans must contain ReplacementSpan records")
        spans = tuple(sorted(spans, key=lambda span: (span.decoded_start, span.decoded_end)))
        for previous, current in zip(spans, spans[1:]):
            if previous.decoded_end > current.decoded_start:
                raise ValueError("replacement_spans must not overlap")
        object.__setattr__(self, "replacement_spans", spans)
        object.__setattr__(
            self,
            "residual_occurrence_ids",
            _id_tuple(self.residual_occurrence_ids, "residual_occurrence_ids"),
        )
        review_items = tuple(self.review_items)
        if not all(isinstance(item, ReviewItem) for item in review_items):
            raise ValueError("review_items must contain ReviewItem records")
        object.__setattr__(self, "review_items", review_items)
        object.__setattr__(self, "errors", _string_tuple(self.errors, "errors"))

        required_review = any(item.required for item in review_items)
        if self.status in {"PASS", "PASS_WITH_OPTIONAL_REVIEW"} and (
            self.residual_occurrence_ids
        ):
            raise ValueError(f"{self.status} cannot contain readable residuals")
        if self.status == "PASS" and (review_items or self.errors):
            raise ValueError("PASS cannot contain review items or errors")
        if self.status == "PASS_WITH_OPTIONAL_REVIEW" and (
            required_review or self.errors
        ):
            raise ValueError(
                "PASS_WITH_OPTIONAL_REVIEW allows only optional review items"
            )
        if self.status == "REVIEW_REQUIRED" and (
            not required_review or self.errors
        ):
            raise ValueError(
                "REVIEW_REQUIRED needs a required review item and no errors"
            )
        if self.status == "NOT_COMPLETE" and not self.errors:
            raise ValueError("NOT_COMPLETE requires at least one error")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "file": self.file,
            "file_sha256": self.file_sha256,
            "status": self.status,
            "replacement_spans": [span.to_dict() for span in self.replacement_spans],
            "residual_occurrence_ids": list(self.residual_occurrence_ids),
            "review_items": [item.to_dict() for item in self.review_items],
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FileResult":
        expected = {
            "schema_version",
            "file",
            "file_sha256",
            "status",
            "replacement_spans",
            "residual_occurrence_ids",
            "review_items",
            "errors",
        }
        _require_exact_keys(payload, expected, "FileResult")
        _require_schema(payload, "FileResult")
        replacement_payloads = _list(payload, "replacement_spans")
        review_payloads = _list(payload, "review_items")
        return cls(
            file=_string(payload, "file"),
            file_sha256=_string(payload, "file_sha256"),
            status=_string(payload, "status"),
            replacement_spans=tuple(
                ReplacementSpan.from_dict(_mapping(value, "replacement_spans"))
                for value in replacement_payloads
            ),
            residual_occurrence_ids=tuple(
                _string_value(value, "residual_occurrence_ids")
                for value in _list(payload, "residual_occurrence_ids")
            ),
            review_items=tuple(
                ReviewItem.from_dict(_mapping(value, "review_items"))
                for value in review_payloads
            ),
            errors=tuple(
                _string_value(value, "errors") for value in _list(payload, "errors")
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "FileResult":
        return cls.from_dict(_json_object(value, "FileResult"))


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must contain only objects")
    return value
