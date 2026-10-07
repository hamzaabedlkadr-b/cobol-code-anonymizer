"""Local, human-reviewed exceptions for the correction queue.

These answers are deliberately keyed by a folded candidate word and the
visible source line, not by an occurrence ID or file hash. A harmless copied
line should not require the same reviewer decision again in a later batch.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import fold_watchlist_value


FIELDNAMES = ("word", "line_text", "answer", "reviewer", "date")


def folded_word(value: str) -> str:
    """Normalize a queue word exactly as watchlist matching does."""

    return fold_watchlist_value(value.strip().replace("''", "'").replace("’", "'"))


def source_line(context: str, candidate: str) -> str:
    """Return the candidate line without markers or fixed-format columns."""

    marker = context.find("[[")
    if marker == -1:
        line = candidate
    else:
        start = context.rfind("\n", 0, marker) + 1
        end = context.find("\n", marker)
        line = context[start:] if end == -1 else context[start:end]
    line = line.replace("[[", "").replace("]]", "")
    if (
        len(line) >= 7
        and re.fullmatch(r"[A-Z0-9 ]{6}", line[:6], re.IGNORECASE)
        and line[6] in " */D-d"
    ):
        line = line[7:]
    return line.strip()


@dataclass(frozen=True)
class ReviewDecisions:
    """Validated human answers that can safely affect a later run."""

    person_all_lines: frozenset[str] = field(default_factory=frozenset)
    person_lines: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    not_person_lines: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def answer_for(self, word: str, line_text: str) -> str | None:
        """Return a matching answer, preferring the safe person decision."""

        key = (folded_word(word), line_text)
        if key[0] in self.person_all_lines or key in self.person_lines:
            return "person"
        if key in self.not_person_lines:
            return "not_person"
        return None


def ensure_review_decisions_csv(path: Path) -> None:
    """Create an empty local decisions file once; never overwrite answers."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()


def load_review_decisions(path: Path) -> ReviewDecisions:
    """Load only explicit, safe answers from ``review_decisions.csv``.

    Invalid rows are ignored rather than guessed. In particular, ``*`` can
    apply to every line only for a ``person`` answer; it can never clear text.
    """

    ensure_review_decisions_csv(path)
    person_all_lines: set[str] = set()
    person_lines: set[tuple[str, str]] = set()
    not_person_lines: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            word = folded_word(row.get("word", ""))
            line_text = str(row.get("line_text", "")).strip()
            answer = str(row.get("answer", "")).strip().casefold()
            if not word or answer not in {"person", "not_person"}:
                continue
            if answer == "person":
                if line_text == "*":
                    person_all_lines.add(word)
                elif line_text:
                    person_lines.add((word, line_text))
            elif line_text and line_text != "*":
                not_person_lines.add((word, line_text))
    return ReviewDecisions(
        person_all_lines=frozenset(person_all_lines),
        person_lines=frozenset(person_lines),
        not_person_lines=frozenset(not_person_lines),
    )
