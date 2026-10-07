"""Visible, non-decisive context clues for NAME candidates.

This module deliberately does not classify ordinary words, use an Italian
dictionary, or decide whether a candidate is a person. It records only three
things that a judge can see in the source line: a direct person cue, a place
cue, and a date containing a month. The return value is always unresolved so
that a clue can inform a later decision without clearing source text by itself.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


VOCABULARY_DIRECTORY = Path(__file__).with_name("data") / "vocab"
VOCABULARY_FILES = {
    "month": "months.txt",
    "place": "places.txt",
    "person_cue": "titles_person_cues.txt",
}

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
MARKED_CANDIDATE_RE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
DIRECT_SEPARATOR_RE = re.compile(r"[\s,.:;/-]*$")
DATE_BEFORE_RE = re.compile(r"\b\d{1,2}\s*$")
DATE_AFTER_RE = re.compile(r"^\s*(?:\d{2,4})\b")
PLACE_CUE_RE = re.compile(
    r"\b(?:REGIONE|PROVINCIA\s+DI|COMUNE\s+DI|SEDE\s+DI|PRESSO)\s*$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def builtin_vocabulary() -> dict[str, frozenset[str]]:
    """Load the three reviewed lists used only to recognize visible clues."""

    result: dict[str, frozenset[str]] = {}
    for category, filename in VOCABULARY_FILES.items():
        path = VOCABULARY_DIRECTORY / filename
        result[category] = frozenset(
            line.strip().casefold()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return result


def has_direct_person_cue(context: str) -> bool:
    """Return whether a title or person cue immediately introduces ``[[...]]``."""

    match = MARKED_CANDIDATE_RE.search(context)
    if match is None:
        return False
    before = DIRECT_SEPARATOR_RE.sub("", context[: match.start()])
    return any(
        re.search(r"(?:^|\b)" + re.escape(cue) + r"$", before, re.IGNORECASE)
        for cue in builtin_vocabulary()["person_cue"]
    )


def assess_name_evidence(*, candidate: str, context: str) -> tuple[bool, tuple[str, ...]]:
    """Return unresolved evidence and the visible source clues for a candidate.

    ``True`` is intentional: neither a date nor a place-shaped line is proof
    that a matching word is not a person. The caller forwards these audit
    reasons to the judge; policy remains responsible for every final outcome.
    """

    match = MARKED_CANDIDATE_RE.search(context)
    if match is None:
        return True, ("evidence:missing_candidate_marker",)

    left, right = context[: match.start()], context[match.end() :]
    words = {word.group(0).casefold() for word in WORD_RE.finditer(candidate)}
    vocab = builtin_vocabulary()
    reasons: list[str] = []

    if has_direct_person_cue(context):
        reasons.append("clue:direct_person_cue")
    if words & vocab["place"] and PLACE_CUE_RE.search(left):
        reasons.append("clue:place_cue")
    if words & vocab["month"] and (
        DATE_BEFORE_RE.search(left) or DATE_AFTER_RE.search(right)
    ):
        reasons.append("clue:date_with_month")

    if not reasons:
        reasons.append("evidence:no_visible_clue")
    return True, tuple(reasons)
