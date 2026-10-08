"""One table for NAME privacy decisions."""

from __future__ import annotations

import re

from .decisions import Decision
from .text_matching import COBOL_KEYWORDS


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


def has_minimum_watchlist_context(context: str) -> bool:
    """Require two other letter words; sequence columns are removed by layout."""

    line = re.sub(r"\[\[.*?\]\]", "", context)
    return len(_WORD_RE.findall(line)) >= 2


NAME_POLICY_TABLE = {
    "spacy_structure": ("leave_unchanged", "spaCy span contains punctuation, digits or COBOL keyword; not a candidate"),
    "other_hide": ("anonymize_whole", "non-name detector; hide"),
    "other_show": ("leave_unchanged", "non-name replacement skipped; show"),
    "review_hide": ("anonymize_whole", "reviewer says name; hide"),
    "review_show": ("leave_unchanged", "reviewer says not a name; show"),
    "code_hide": ("anonymize_whole", "watchlist part in code or path; hide"),
    "code_show": ("leave_unchanged", "approved part in code or path; show"),
    "watchlist_pair": ("anonymize_whole", "adjacent watchlist words; hide"),
    "watchlist": ("anonymize_whole", "watchlist word not approved; hide"),
    "too_little_context": ("anonymize_whole", "approved word; too little context; hide"),
    "instruction_text": ("anonymize_whole", "instruction text in line"),
    "residual_unresolved": ("review_required", "readable name after final check"),
    "no_judge": ("anonymize_whole", "judge not available; hide"),
    "judge_error": ("anonymize_whole", "judge answer invalid; hide"),
    "uncertain": ("anonymize_whole", "judge unsure; hide"),
    "person": ("anonymize_whole", "judge says person; hide"),
    "non_person_disagrees": ("anonymize_whole", "verifier did not say not person; hide"),
    "non_person_verified": ("leave_unchanged", "judge and verifier say not person; show"),
    "non_person": ("leave_unchanged", "judge says not person; show"),
}


def instruction_text_requires_anonymization(context: str) -> bool:
    """Return ``True`` when untrusted context must bypass the LLM judge.

    A positive result is an explicit policy decision to anonymize.  It is not
    evidence that the candidate is a person, and it is not a model rejection
    override.  Keeping that distinction makes the audit truthful and prevents
    prompt-like source text from influencing the judge at all.
    """

    return any(pattern.search(context) for pattern in INSTRUCTION_PATTERNS)


def _table_decision(*, key: str, occurrence_id: str, judge_decision: Decision | None = None) -> Decision:
    outcome, reading = NAME_POLICY_TABLE[key]
    return Decision(
        occurrence_id=occurrence_id, stage="policy", outcome=outcome,
        person_scope={"anonymize_whole": "whole", "review_required": "unsure", "leave_unchanged": "none"}[outcome],
        non_person_category=judge_decision.non_person_category if outcome == "leave_unchanged" and judge_decision else "common_word" if outcome == "leave_unchanged" else "none",
        reading=reading,
    )


def apply_name_policy(
    *,
    occurrence_id: str,
    model_context: str,
    judge_decision: Decision | None,
    verifier_decision: Decision | None = None,
    code_sensitive_identifier: bool,
    watchlist_pair: bool = False,
    watchlist_single: bool = False,
    approved_word: bool = False,
    verifier_enabled: bool = True,
    residual_unresolved: bool = False,
    review_answer: str | None = None,
) -> Decision:
    """Choose hide, show, or required review from the single table."""

    if review_answer == "not_person":
        return _table_decision(key="review_show", occurrence_id=occurrence_id)
    if review_answer == "person":
        return _table_decision(key="review_hide", occurrence_id=occurrence_id)
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
        return _table_decision(key="code_show" if approved_word else "code_hide", occurrence_id=occurrence_id)
    if watchlist_pair:
        return _table_decision(key="watchlist_pair", occurrence_id=occurrence_id)
    if instruction_text_requires_anonymization(model_context):
        return _table_decision(key="instruction_text", occurrence_id=occurrence_id)
    if watchlist_single and not approved_word:
        return _table_decision(key="watchlist", occurrence_id=occurrence_id)
    if watchlist_single and not has_minimum_watchlist_context(model_context):
        return _table_decision(key="too_little_context", occurrence_id=occurrence_id)
    if judge_decision is None:
        return _table_decision(key="no_judge", occurrence_id=occurrence_id)
    if judge_decision.outcome == "error":
        return _table_decision(key="judge_error", occurrence_id=occurrence_id)
    if judge_decision.outcome == "uncertain":
        return _table_decision(key="uncertain", occurrence_id=occurrence_id)
    if judge_decision.outcome in {"anonymize_whole", "anonymize_part"}:
        return _table_decision(key="person", occurrence_id=occurrence_id)
    if judge_decision.outcome != "propose_unchanged":
        return _table_decision(key="judge_error", occurrence_id=occurrence_id)

    if not verifier_enabled:
        return _table_decision(key="non_person", occurrence_id=occurrence_id, judge_decision=judge_decision)
    if verifier_decision is None or verifier_decision.outcome != "not_person":
        return _table_decision(key="non_person_disagrees", occurrence_id=occurrence_id)
    return _table_decision(key="non_person_verified", occurrence_id=occurrence_id, judge_decision=judge_decision)


def spacy_candidate_reason(value: str) -> str:
    if re.search(r"[-=()./\d]", value) or any(word.upper() in COBOL_KEYWORDS for word in value.split()):
        return NAME_POLICY_TABLE["spacy_structure"][1]
    return ""
