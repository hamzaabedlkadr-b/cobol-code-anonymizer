"""Build the old review columns with exact word-and-line answer keys."""

import csv
from pathlib import Path
from .review_decisions import folded_word, source_line
from .text_matching import name_word_spans, IDENTIFIER_PART_RE
from .policy import NAME_POLICY_TABLE


def build_llm_name_review_rows(decisions, extractor=None):
    comparisons = {(row["file"], row["start"], row["end"]): row["status"]
                   for row in getattr(extractor, "comparisons", [])}
    groups = {}
    for decision in decisions:
        if decision.get("entity_type", "NAME") != "NAME":
            continue
        candidate = str(decision.get("logical_candidate") or decision["text"])
        outcome = decision["policy_outcome"]
        pending = not decision.get("review_answer") and outcome != "leave_unchanged"
        for start, end in name_word_spans(candidate) if pending else [(0, len(candidate))]:
            word = candidate[start:end]
            context = str(decision.get("review_line") or source_line(str(decision.get("context", "")), candidate))
            for match in reversed(list(IDENTIFIER_PART_RE.finditer(context))):
                if folded_word(match.group()) == folded_word(word):
                    context = context[:match.start()] + "[[" + match.group() + "]]" + context[match.end():]
            if decision.get("logical_context") and "[[" not in context:
                context += "\nJoined: " + str(decision["logical_context"])
            action = "review (required)" if outcome == "review_required" else "hide (optional review)" if pending else "show" if outcome == "leave_unchanged" else "hide"
            row = {"name": word, "file": decision["file"], "line": decision["line"], "column": decision.get("column", ""),
                   "extraction_status": comparisons.get((decision["file"], decision.get("start"), decision.get("end")), "detector_only"),
                   "judge_status": decision.get("judge_outcome", "not_called"), "final_action": action,
                   "reason": plain_reason(decision["policy_reading"]), "detector_sources": decision.get("source", ""),
                   "context": context, "verifier_status": decision.get("verifier_outcome", "not_called"),
                   "key": decision.get("review_key", ""), "answer": ""}
            key = (folded_word(word), row["key"] or source_line(context, word)) if pending else (row["file"], row["line"], row["column"], word)
            if key not in groups or action == "review (required)":
                groups[key] = row
    return sorted(groups.values(), key=lambda row: (row["final_action"] != "review (required)", folded_word(row["name"]), str(row["file"]), int(row["line"])))


def write_llm_name_review_csv(path: Path, rows):
    """Write the old review columns; answer is the last column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "file", "line", "column", "extraction_status", "judge_status", "final_action", "reason", "detector_sources", "context", "verifier_status", "key", "answer"))
        writer.writeheader()
        writer.writerows(rows)


def plain_reason(reason):
    labels = {
        "watchlist": "on the watchlist", "watchlist_pair": "two watchlist words together",
        "code_hide": "name inside code", "code_show": "approved word inside code",
        "person": "judge: person", "review_hide": "you answered hide", "review_show": "you answered show",
        "non_person_verified": "judge and verifier: not a person", "non_person": "judge: not a person",
        "spacy_structure": "looks like code (spaCy)", "uncertain": "judge: unsure",
        "judge_error": "judge answer invalid", "no_judge": "judge not available",
        "non_person_disagrees": "verifier did not confirm not a person",
        "too_little_context": "too little context", "instruction_text": "instruction text in line",
        "residual_unresolved": "still readable after the final check",
    }
    return next((label for key, label in labels.items() if NAME_POLICY_TABLE[key][1] == reason), reason.removesuffix("; hide").removesuffix("; show"))
