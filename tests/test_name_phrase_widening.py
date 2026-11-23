"""Regression tests for exact-span watchlist matching.

The filename is retained to keep the focused test location stable during the
migration. Adjacent-word widening was removed: phrase construction belongs to
the later judge-span step, not to deterministic watchlist detection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.pipeline import scan_path


class WatchlistExactSpanTests(unittest.TestCase):
    def scan(self, source_text: str, watchlist_text: str, *, name_scope: str = "context", reviews=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PROGRAM.CBL"
            source.write_text(source_text, encoding="utf-8")
            watchlist = root / "watchlist.txt"
            watchlist.write_text(watchlist_text, encoding="utf-8")
            findings = scan_path(
                source,
                entities={"NAME"},
                extra_watchlists=[watchlist],
                include_default_names=False,
                use_presidio=False,
                name_scope=name_scope,
                identifier_review_decisions=reviews,
            )
        return findings

    def test_single_word_match_does_not_absorb_neighbouring_words(self):
        self.assertEqual(
            [finding.text for finding in self.scan("      * MARIO ROSS DICE\n", "ROSS\n")],
            ["ROSS"],
        )

    def test_exact_phrase_watchlist_entry_remains_a_single_match(self):
        self.assertEqual(
            [
                finding.text
                for finding in self.scan("      * MARIO ROSS DICE\n", "MARIO ROSS\n")
            ],
            ["MARIO ROSS"],
        )

    def test_ordinary_and_technical_neighbours_do_not_change_the_span(self):
        with self.subTest("ordinary"):
            self.assertEqual(
                [
                    finding.text
                    for finding in self.scan("      * TOTALE CONTI\n", "CONTI\n")
                ],
                ["CONTI"],
            )
        with self.subTest("technical"):
            self.assertEqual(
                [
                    finding.text
                    for finding in self.scan("      * SALA SERVER\n", "SALA\n")
                ],
                ["SALA"],
            )

    def test_adjacent_entries_are_pair_findings(self):
        watchlist = "ROSSI\nMARIO\nGIAN\nMARIA\n"
        cases = (
            ("      * ROSSI, MARIO\n", "ROSSI, MARIO"),
            ("      * GIAN MARIA ROSSI\n", "GIAN MARIA ROSSI"),
            ("       DISPLAY 'ROSSI  MARIO'.\n", "ROSSI  MARIO"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                findings = self.scan(source, watchlist)
                self.assertEqual([finding.text for finding in findings], [expected])
                self.assertEqual(findings[0].source, "watchlist_pair")

    def test_short_words_and_initials_never_form_pairs(self):
        from cobol_code_anonymizer.scanner import find_watchlist_pair_spans
        for text in ("      * CASO DI", "      * DI TESTA", "      * M. ROSSI", "      * ROSSI M."):
            self.assertEqual(find_watchlist_pair_spans(text, ("CASO", "DI", "TESTA", "ROSSI", "M")), [])
        text = "      * CASO TESTA"
        self.assertEqual([text[a:b] for a, b in find_watchlist_pair_spans(text, ("CASO", "TESTA"))],
                         ["CASO TESTA"])

    def test_non_watchlist_neighbour_is_not_promoted_to_a_pair(self):
        findings = self.scan("      * SALA SERVER\n", "SALA\n")

        self.assertEqual([finding.text for finding in findings], ["SALA"])
        self.assertEqual(findings[0].source, "watchlist")



if __name__ == "__main__":
    unittest.main()
