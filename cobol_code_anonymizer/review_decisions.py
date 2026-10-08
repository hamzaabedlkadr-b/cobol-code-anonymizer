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
from datetime import date

from .scanner import fold_watchlist_value


FIELDNAMES = ("word", "line", "answer", "date", "key")


def folded_word(value: str) -> str:
    """Normalize a queue word exactly as watchlist matching does."""

    return fold_watchlist_value(value.strip().replace("''", "'").replace("’", "'"))


def source_line(context: str, candidate: str) -> str:
    """Remove display markers without guessing source format."""
    return (context or candidate).replace("[[", "").replace("]]", "").strip()


@dataclass(frozen=True)
class ReviewDecisions:
    """Validated human answers that can safely affect a later run."""

    person_lines: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    not_person_lines: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def answer_for(self, word: str, line_text: str, key: str = "") -> str | None:
        """Return a matching answer, preferring the safe person decision."""

        key = (folded_word(word), key or line_text)
        if key in self.person_lines:
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
    """Load exact word-and-line answers; the latest saved answer wins."""
    ensure_review_decisions_csv(path)
    answers = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            word = folded_word(row.get("word", ""))
            line = row.get("line", row.get("line_text", "")).strip()
            answer = row.get("answer", "").strip().casefold()
            answer = {"show": "not_person", "hide": "person"}.get(answer, answer)
            if word and line and line != "*" and answer in {"person", "not_person"}:
                answers[word, row.get("key") or line] = answer
    return ReviewDecisions(
        person_lines=frozenset(key for key, answer in answers.items() if answer == "person"),
        not_person_lines=frozenset(key for key, answer in answers.items() if answer == "not_person"),
    )


def import_review_answers(queue: Path, stored: Path) -> None:
    """Save explicit queue answers before replacing the previous run's queue."""
    if not queue.is_file():
        return
    ensure_review_decisions_csv(stored)
    answers = {}
    with stored.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            line = row.get("line", row.get("line_text", ""))
            answers[(folded_word(row.get("word", "")), row.get("key") or line)] = {
                "word": row.get("word", ""), "line": line,
                "answer": row.get("answer", ""), "date": row.get("date", ""), "key": row.get("key", "")}
    changed = False
    with queue.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            answer = row.get("answer", "").strip().casefold()
            word, line = row.get("name", row.get("word", "")).strip(), source_line(row.get("context", row.get("line", "")), row.get("name", row.get("word", "")))
            if answer in {"show", "hide"} and word and line:
                answers[folded_word(word), row.get("key") or line] = {"word": word, "line": line, "answer": answer, "date": date.today().isoformat(), "key": row.get("key", "")}
                changed = True
    if changed:
        temporary = stored.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(answers.values())
        temporary.replace(stored)
