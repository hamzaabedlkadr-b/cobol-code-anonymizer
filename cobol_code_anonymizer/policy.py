"""Deterministic privacy gates applied around model decisions.

The model is not allowed to override a gate in this module.  These checks are
kept outside ``judge.py`` so the judge only proposes a semantic reading while
policy decides whether source text is allowed to remain readable.

This module owns the single removal policy.  A non-person proposal can pass
only after all deterministic gates and a blinded verifier agree.
"""

from __future__ import annotations

import re

from .decisions import Decision


# These patterns identify source text that looks like an attempt to instruct
# the model.  Matching text is untrusted input, so the safe policy result is
# fixed: anonymize the candidate without asking the judge.
INSTRUCTION_PATTERNS = (
    re.compile(
        r"\b(?:IGNORE|DISREGARD)\b.{0,40}\b(?:INSTRUCTIONS?|PROMPT|SYSTEM)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bIGNOR(?:A|ARE)\b.{0,40}\b(?:ISTRUZIONI|PROMPT|SISTEMA)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ANSWER|RESPOND|RISPONDI)\b.{0,40}\b"
        r"(?:REJECT|KEEP|UNCERTAIN|ANONYMIZE_(?:WHOLE|PART)|PROPOSE_UNCHANGED)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:RETURN|RESTITUISCI)\b.{0,50}\b"
        r"(?:JSON|DECISION|REJECT|KEEP|ANONYMIZE_(?:WHOLE|PART)|PROPOSE_UNCHANGED)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bSYSTEM\s+PROMPT\b|\bPROMPT\s+DI\s+SISTEMA\b", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_FIXED_SEQUENCE_AND_INDICATOR_RE = re.compile(r"^[A-Z0-9 ]{6}[ */D-]", re.IGNORECASE)


def has_minimum_watchlist_context(context: str) -> bool:
    """Return whether a marked watchlist word has two other letter words.

    This is intentionally a small structural check, not vocabulary inference.
    The ``[[...]]`` candidate is removed before counting.  For conventional
    fixed-format lines, the sequence area and indicator are ignored; ordinary
    plain-text lines keep all of their words.
    """

    line = re.sub(r"\[\[.*?\]\]", "", context)
    if _FIXED_SEQUENCE_AND_INDICATOR_RE.match(line):
        line = line[7:]
    return len(_WORD_RE.findall(line)) >= 2


# One auditable table for ordinary judge/verifier outcomes. Deterministic
# gates above it always win. Every route other than independently verified
# non-person text keeps the candidate hidden.
NAME_POLICY_TABLE = {
    "code": ("review_required", "possible name in code; file needs review"),
    "watchlist_pair": ("anonymize_whole", "adjacent watchlist entries; hide"),
    "protected_identity": ("anonymize_whole", "protected identity; hide"),
    "person_overlap": ("anonymize_whole", "overlapping person span; hide"),
    "too_little_context": ("anonymize_and_review", "too little context"),
    "instruction_text": ("anonymize_and_review", "instruction text in line"),
    "extraction_unmatched": ("review_required", "extracted name not found in line"),
    "residual_unresolved": ("review_required", "name remains after repair; file needs review"),
    "no_judge": ("anonymize_whole", "judge was not called or returned no validated decision"),
    "judge_error": ("anonymize_and_review", "judge returned an invalid answer; candidate requires review"),
    "uncertain": ("anonymize_and_review", "judge uncertainty requires correction review"),
    "person": ("anonymize_whole", "judge identified a possible person"),
    "person_single_verified": ("anonymize_whole", "judge and verifier say person"),
    "person_single_review": ("anonymize_and_review", "single watchlist word needs review"),
    "non_person_no_verifier": ("anonymize_and_review", "non-person proposal requires independent verification"),
    "non_person_disagrees": ("anonymize_and_review", "judge and verifier did not agree on a non-person reading"),
    "non_person_verified": ("leave_unchanged", "judge proposal passed independent verification"),
}


def instruction_text_requires_anonymization(context: str) -> bool:
    """Return ``True`` when untrusted context must bypass the LLM judge.

    A positive result is an explicit policy decision to anonymize.  It is not
    evidence that the candidate is a person, and it is not a model rejection
    override.  Keeping that distinction makes the audit truthful and prevents
    prompt-like source text from influencing the judge at all.
    """

    return any(pattern.search(context) for pattern in INSTRUCTION_PATTERNS)


def _anonymize_whole(occurrence_id: str, reading: str) -> Decision:
    return Decision(
        occurrence_id=occurrence_id,
        stage="policy",
        outcome="anonymize_whole",
        person_scope="whole",
        reading=reading,
    )


def _anonymize_and_review(occurrence_id: str, reading: str) -> Decision:
    """Anonymize now, while recording that a later review may restore text."""

    return Decision(
        occurrence_id=occurrence_id,
        stage="policy",
        outcome="anonymize_and_review",
        person_scope="whole",
        reading=reading,
    )


def _review_identifier(occurrence_id: str, reading: str) -> Decision:
    return Decision(
        occurrence_id=occurrence_id,
        stage="policy",
        outcome="review_required",
        person_scope="unsure",
        reading=reading,
    )


def _table_decision(
    *,
    key: str,
    occurrence_id: str,
    judge_decision: Decision | None = None,
) -> Decision:
    """Build the final decision named by the single ordinary-outcome table."""

    outcome, reading = NAME_POLICY_TABLE[key]
    if outcome == "anonymize_whole":
        return _anonymize_whole(occurrence_id, reading)
    if outcome == "anonymize_and_review":
        return _anonymize_and_review(occurrence_id, reading)
    if outcome == "review_required":
        return _review_identifier(occurrence_id, reading)
    if outcome == "leave_unchanged" and judge_decision is not None:
        return Decision(
            occurrence_id=occurrence_id,
            stage="policy",
            outcome="leave_unchanged",
            person_scope="none",
            non_person_category=judge_decision.non_person_category,
            evidence_quote=judge_decision.evidence_quote,
            reading=reading,
        )
    raise ValueError(f"invalid policy-table route: {key}")


def apply_name_policy(
    *,
    occurrence_id: str,
    model_context: str,
    judge_decision: Decision | None,
    verifier_decision: Decision | None = None,
    protected_identity: bool = False,
    stronger_person_overlap: bool,
    code_sensitive_identifier: bool,
    watchlist_pair: bool = False,
    watchlist_single: bool = False,
    direct_person_cue: bool = False,
    unmatched_extraction: bool = False,
    residual_unresolved: bool = False,
) -> Decision:
    """Apply the only policy route that may eventually preserve NAME text.

    ``model_context`` must be exactly the untrusted context supplied to the
    judge.  The instruction gate therefore covers every source character the
    model can see.  If that context grows in the future, callers must pass the
    same expanded value to both this function and the model call.

    Only a valid verifier ``not_person`` decision can produce
    ``leave_unchanged``. An unclear non-person proposal produces
    ``anonymize_and_review``: the full candidate is still anonymized, but the
    pipeline may queue it for later correction review. Every other uncertainty
    and technical model outcome moves in the privacy-safe direction.
    """

    if unmatched_extraction:
        return _table_decision(key="extraction_unmatched", occurrence_id=occurrence_id)
    if residual_unresolved:
        return _table_decision(key="residual_unresolved", occurrence_id=occurrence_id)
    if judge_decision is not None:
        if judge_decision.stage != "judge":
            raise ValueError("judge_decision must have stage='judge'")
        if judge_decision.occurrence_id != occurrence_id:
            raise ValueError("judge_decision refers to a different occurrence")
    if verifier_decision is not None:
        if verifier_decision.stage != "verifier":
            raise ValueError("verifier_decision must have stage='verifier'")
        if verifier_decision.occurrence_id != occurrence_id:
            raise ValueError("verifier_decision refers to a different occurrence")

    if code_sensitive_identifier:
        return _table_decision(key="code", occurrence_id=occurrence_id)
    if watchlist_pair:
        return _table_decision(key="watchlist_pair", occurrence_id=occurrence_id)
    if instruction_text_requires_anonymization(model_context):
        return _table_decision(key="instruction_text", occurrence_id=occurrence_id)
    if protected_identity:
        return _table_decision(key="protected_identity", occurrence_id=occurrence_id)
    if stronger_person_overlap:
        return _table_decision(key="person_overlap", occurrence_id=occurrence_id)
    if judge_decision is None:
        return _table_decision(key="no_judge", occurrence_id=occurrence_id)
    if judge_decision.outcome == "error":
        return _table_decision(key="judge_error", occurrence_id=occurrence_id)
    if judge_decision.outcome == "uncertain":
        return _table_decision(key="uncertain", occurrence_id=occurrence_id)
    if judge_decision.outcome in {"anonymize_whole", "anonymize_part"}:
        if watchlist_single and not direct_person_cue:
            return _table_decision(
                key="person_single_verified" if verifier_decision is not None and verifier_decision.outcome == "person" else "person_single_review",
                occurrence_id=occurrence_id,
            )
        return _table_decision(key="person", occurrence_id=occurrence_id)
    if judge_decision.outcome != "propose_unchanged":
        return _table_decision(key="judge_error", occurrence_id=occurrence_id)

    # Evidence is retained for the model and audit but never clears a name by
    # itself. Every proposed readable candidate reaches the blinded verifier.
    if verifier_decision is None:
        return _table_decision(key="non_person_no_verifier", occurrence_id=occurrence_id)
    if verifier_decision.outcome != "not_person":
        return _table_decision(key="non_person_disagrees", occurrence_id=occurrence_id)
    if watchlist_single and not has_minimum_watchlist_context(model_context):
        return _table_decision(key="too_little_context", occurrence_id=occurrence_id)
    return _table_decision(
        key="non_person_verified",
        occurrence_id=occurrence_id,
        judge_decision=judge_decision,
    )
