"""Unittest regression tests for overlap grouping and compatibility cleanup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.overlaps import (
    ENTITY_PRIORITY,
    resolve_overlaps,
)
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.scanner import Finding


def finding(
    start: int,
    end: int,
    *,
    file: str = "PROGRAM.CBL",
    entity_type: str = "NAME",
    source: str = "watchlist",
) -> Finding:
    return Finding(
        file=file,
        entity_type=entity_type,
        text=f"VALUE_{start}_{end}",
        start=start,
        end=end,
        line=1,
        column=start + 1,
        confidence=0.7,
        context="context",
        source=source,
    )


class OverlapTests(unittest.TestCase):



    def test_losing_span_keeps_uncovered_words_and_source_offsets(self):
        from dataclasses import replace
        text = "Mario Rossi Bianchi"
        short = replace(finding(0, 11), text="Mario Rossi", context="[[Mario Rossi]] Bianchi")
        winner = replace(finding(6, len(text)), text="Rossi Bianchi", context="Mario [[Rossi Bianchi]]")
        selected = resolve_overlaps([short, winner]).selected
        self.assertEqual([(item.text, item.start, item.end) for item in selected],
                         [("Mario", 0, 5), ("Rossi Bianchi", 6, 19)])
        self.assertEqual(selected[0].context, "[[Mario]] Rossi Bianchi")
        self.assertEqual(selected[0].column, 1)
        self.assertEqual(selected[0].source, "watchlist")

    def test_partial_words_are_never_created(self):
        from dataclasses import replace
        loser = replace(finding(0, 11), text="Mario Rossi")
        winner = replace(finding(2, 20, entity_type="EMAIL"), text="rio Rossi@example")
        selected = resolve_overlaps([loser, winner]).selected
        self.assertEqual(selected, (winner,))

    def test_uncovered_words_on_both_sides_remain_separate(self):
        from dataclasses import replace
        loser = replace(finding(0, 15), text="Anna Mario Nora", context="[[Anna Mario Nora]]")
        winner = replace(finding(5, 10, entity_type="EMAIL"), text="Mario", context="Anna [[Mario]] Nora")
        selected = resolve_overlaps([loser, winner]).selected
        self.assertEqual([item.text for item in selected], ["Anna", "Mario", "Nora"])
        self.assertTrue(all(a.end <= b.start for a, b in zip(selected, selected[1:])))


    def test_adjacent_watchlist_words_form_a_deterministic_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PROGRAM.CBL"
            source.write_text(
                "      * TOTALE: TOKENALPHA TOKENBETA\n",
                encoding="utf-8",
            )
            watchlist = root / "watchlist.txt"
            watchlist.write_text("TOKENALPHA\nTOKENBETA\n", encoding="utf-8")
            judge = NameJudge( model_digest="b" * 64)
            proposal = LlmJsonResult(
                parsed={
                    "decision": "propose_unchanged",
                    "person_scope": "none",
                    "person_texts": [],
                    "non_person_category": "common_word",
                    "evidence_quote": "TOTALE",
                    "reading": "The text is used as a non-person label.",
                },
                content="",
                latency_s=0.0,
                schema_ok=True,
            )
            with patch(
                "cobol_code_anonymizer.judge.call_ollama_json",
                return_value=proposal,
            ):
                findings = scan_path(
                    source,
                    entities={"NAME"},
                    extra_watchlists=[watchlist],
                    include_default_names=False,
                    use_presidio=False,
                    name_judge=judge,
                )

        self.assertEqual(
            [item.text for item in findings],
            ["TOKENALPHA TOKENBETA"],
        )
        self.assertEqual(
            [row["decision"] for row in judge.decisions],
            ["anonymize_whole"],
        )
        self.assertEqual(findings[0].source, "watchlist_pair")


if __name__ == "__main__":
    unittest.main()
