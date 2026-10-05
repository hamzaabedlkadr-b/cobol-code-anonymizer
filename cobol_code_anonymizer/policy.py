"""Deterministic privacy gates applied around model decisions.

The model is not allowed to override a gate in this module.  These checks are
kept outside ``judge.py`` so the judge only proposes a semantic reading while
policy decides whether source text is allowed to remain readable.

Only the instruction-text gate is implemented here for now.  The remaining
single-removal policy is introduced in Step 27 of the implementation plan.
"""

from __future__ import annotations

import re


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
