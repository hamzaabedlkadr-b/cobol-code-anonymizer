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


def _review_identifier(occurrence_id: str, reading: str) -> Decision:
    return Decision(
        occurrence_id=occurrence_id,
        stage="policy",
        outcome="review_required",
        person_scope="unsure",
        reading=reading,
    )


def apply_name_policy(
    *,
    occurrence_id: str,
    model_context: str,
    judge_decision: Decision | None,
    verifier_decision: Decision | None = None,
    protected_identity: bool = False,
    stronger_person_overlap: bool,
    unresolved_evidence: bool,
    code_sensitive_identifier: bool,
) -> Decision:
    """Apply the only policy route that may eventually preserve NAME text.

    ``model_context`` must be exactly the untrusted context supplied to the
    judge.  The instruction gate therefore covers every source character the
    model can see.  If that context grows in the future, callers must pass the
    same expanded value to both this function and the model call.

    Only a valid verifier ``not_possible`` decision can produce
    ``leave_unchanged``. Every uncertainty and technical model outcome moves
    in the privacy-safe direction.
    """

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

    def protect(reading: str) -> Decision:
        if code_sensitive_identifier:
            return _review_identifier(occurrence_id, reading)
        return _anonymize_whole(occurrence_id, reading)

    if instruction_text_requires_anonymization(model_context):
        return protect("instruction-like model context requires anonymization")
    if protected_identity:
        return protect("protected identity overlap requires anonymization")
    if stronger_person_overlap:
        return protect("stronger overlapping person span requires anonymization")
    if judge_decision is None:
        return protect("judge was not called or returned no validated decision")
    if judge_decision.outcome == "error":
        return protect("judge failure requires anonymization")
    if judge_decision.outcome == "uncertain":
        return protect("judge uncertainty requires anonymization")
    if judge_decision.outcome == "anonymize_whole":
        return protect("judge identified a possible person in the whole candidate")
    if judge_decision.outcome == "anonymize_part":
        # The scanner cannot yet emit unresolved leftovers as fresh candidates.
        # Keeping the whole occurrence is the safe temporary implementation.
        return protect("judge identified a person part; whole-span fallback applied")
    if judge_decision.outcome != "propose_unchanged":
        return protect("unsupported judge outcome requires anonymization")
    if unresolved_evidence:
        return protect("unresolved deterministic evidence requires anonymization")
    if verifier_decision is None:
        return protect("independent verifier has not approved the proposal")
    if verifier_decision.outcome != "not_possible":
        return protect("verifier did not rule out a person reading")

    return Decision(
        occurrence_id=occurrence_id,
        stage="policy",
        outcome="leave_unchanged",
        person_scope="none",
        non_person_category=judge_decision.non_person_category,
        evidence_quote=judge_decision.evidence_quote,
        reading="judge proposal passed deterministic gates and independent verification",
    )
