"""Write required and optional reviews as distinct word-and-line decisions."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from .review_decisions import folded_word, source_line
from .text_matching import name_word_spans, IDENTIFIER_PART_RE


def write_review_queue_csv(path: Path, decisions: Iterable[Mapping[str, object]]) -> tuple[int, int]:
    """Write required items first; copied lines need only one answer."""
    groups = {}
    for decision in decisions:
        if decision.get("review_answer"):
            continue
        outcome = decision.get("policy_outcome")
        if decision.get("entity_type", "NAME") != "NAME" or outcome not in {"anonymize_whole", "review_required"}:
            continue
        candidate = str(decision.get("logical_candidate") or decision.get("text") or "")
        context = str(decision.get("context") or "")
        line = str(decision.get("review_line") or source_line(context, candidate))
        for start, end in name_word_spans(candidate):
            word = candidate[start:end]
            key = (folded_word(word), decision.get("review_key") or line)
            required = outcome == "review_required"
            display = line
            matches = [match for match in IDENTIFIER_PART_RE.finditer(line) if folded_word(match.group()) == folded_word(word)]
            for match in reversed(matches):
                display = display[:match.start()] + "[[" + display[match.start():match.end()] + "]]" + display[match.end():]
            if not matches and decision.get("logical_context"):
                joined = str(decision["logical_context"]).replace("[[", "").replace("]]", "")
                position = joined.find(word)
                if position >= 0:
                    display += "\nJoined: " + joined[:position] + "[[" + word + "]]" + joined[position + len(word):]
            row = {"priority": "REQUIRED" if required else "optional", "word": word,
                   "line": display.strip(), "where": f"{decision['file']}:{decision['line']}",
                   "action": ("not replaced (code)" if decision.get("code_sensitive_identifier") or decision.get("policy_reading") == "possible name in code" else "not replaced (text)") if required else
                             f"hidden as {decision['replacement']}" if decision.get("replacement") else "hidden",
                   "reason": decision.get("policy_reading", ""), "answer": "", "key": decision.get("review_key", "")}
            previous = groups.get(key)
            if previous is None or required and previous["priority"] != "REQUIRED":
                groups[key] = row
            elif row["where"] not in previous["where"].split("; "):
                previous["where"] += f"; {decision['file']}:{decision['line']}"
    rows = sorted(groups.values(), key=lambda row: (row["priority"] != "REQUIRED", folded_word(row["word"]), row["line"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("priority", "word", "line", "where", "action", "reason", "answer", "key"))
        writer.writeheader()
        writer.writerows(rows)
    required = sum(row["priority"] == "REQUIRED" for row in rows)
    return required, len(rows) - required
