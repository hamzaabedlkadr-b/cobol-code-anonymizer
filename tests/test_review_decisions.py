"""Regression tests for local, line-scoped human review answers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.decisions import Decision
from cobol_code_anonymizer.pipeline import _apply_review_answers
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.review_decisions import (
    ReviewDecisions,
    ensure_review_decisions_csv,
    load_review_decisions,
)
from cobol_code_anonymizer.scanner import Finding


def finding(text: str, context: str) -> Finding:
    return Finding(
        file="PROGRAM.CBL",
        entity_type="NAME",
        text=text,
        start=0,
        end=len(text),
        line=1,
        column=1,
        confidence=0.7,
        context=context,
        source="watchlist",
    )


class ReviewDecisionTests(unittest.TestCase):


    def test_latest_queue_answer_replaces_old_answer(self):
        from cobol_code_anonymizer.review_decisions import import_review_answers
        with tempfile.TemporaryDirectory() as folder:
            stored, queue = Path(folder) / "answers.csv", Path(folder) / "queue.csv"
            stored.write_text("word,line_text,answer,reviewer,date\nSALA,NOTA SALA,person,,\n")
            queue.write_text('word,line,answer\nSALA,NOTA [[SALA]],show\n')
            import_review_answers(queue, stored)
            self.assertEqual(load_review_decisions(stored).answer_for("SALA", "NOTA SALA"), "not_person")

    def test_plain_line_prefix_is_not_a_sequence_area(self):
        from cobol_code_anonymizer.review_decisions import source_line
        self.assertEqual(source_line("PIANO  [[SALA]]", "SALA"), "PIANO  SALA")

    def test_template_has_the_documented_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review_decisions.csv"
            ensure_review_decisions_csv(path)
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    next(csv.reader(handle)),
                    ["word", "line", "answer", "date", "key"],
                )





if __name__ == "__main__":
    unittest.main()
