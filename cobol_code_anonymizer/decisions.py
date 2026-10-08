"""Validated judge, verifier, and policy decisions."""

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
VERIFIER_OUTCOMES = {"person", "not_person", "unsure", "error"}
POLICY_OUTCOMES = {
    "anonymize_whole",
    "anonymize_and_review",
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
        # The prompt requires complete person text. Keep this optional at the
        # schema boundary so an older/smaller local model with an empty safe
        # answer still hides the original candidate instead of aborting a file.
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
        evidence_quote=OPTIONAL,
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
    ("verifier", "person"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
        reading=OPTIONAL,
    ),
    ("verifier", "not_person"): _field_rules(
        person_scope=FORBIDDEN,
        person_texts=FORBIDDEN,
        non_person_category=FORBIDDEN,
        evidence_quote=OPTIONAL,
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
    ("policy", "anonymize_and_review"): _field_rules(
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
