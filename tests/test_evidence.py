"""Focused tests for the minimal deterministic NAME evidence layer."""

from __future__ import annotations

import unittest

from cobol_code_anonymizer.evidence import (
    VOCABULARY_DIRECTORY,
    VOCABULARY_FILES,
    assess_name_evidence,
    has_direct_person_cue,
)


class NameEvidenceTests(unittest.TestCase):
    def assess(
        self,
        candidate: str,
        context: str,
        watchlist_values: tuple[str, ...] = (),
    ):
        return assess_name_evidence(
            candidate=candidate,
            context=context,
            watchlist_values=watchlist_values,
        )

    def test_unknown_word_is_unresolved(self):
        unresolved, reasons = self.assess("Bianchi", "      * [[Bianchi]]")
        self.assertTrue(unresolved)
        self.assertIn("word:Bianchi:unknown", reasons)

    def test_watchlist_function_words_need_a_clean_neighbourhood(self):
        unresolved, reasons = self.assess(
            "A CUI", "      * [[A CUI]]", ("A", "CUI")
        )
        self.assertFalse(unresolved)
        self.assertIn("watchlist:resolved_by_vocabulary", reasons)
        self.assertIn("neighbourhood:clean", reasons)

    def test_watchlist_month_with_date_pattern_is_resolved(self):
        unresolved, reasons = self.assess(
            "APRILE", "      * 12 [[APRILE]] 2026", ("APRILE",)
        )
        self.assertFalse(unresolved)
        self.assertIn("pattern:date", reasons)

    def test_watchlist_place_requires_a_place_pattern(self):
        unresolved, reasons = self.assess(
            "CALABRIA", "      * REGIONE [[CALABRIA]]", ("CALABRIA",)
        )
        self.assertFalse(unresolved)
        self.assertIn("pattern:place", reasons)

    def test_direct_person_cue_makes_a_watchlist_occurrence_unresolved(self):
        context = "      * REFERENTE: [[APRILE]]"
        self.assertTrue(has_direct_person_cue(context))
        unresolved, reasons = self.assess("APRILE", context, ("APRILE",))
        self.assertTrue(unresolved)
        self.assertIn("neighbourhood:direct_person_cue", reasons)

    def test_requested_direct_person_cues_make_place_words_unresolved(self):
        cases = (
            ("ROMA", "AUTORE [[ROMA]]"),
            ("MILANO", "MODIFICATO DA [[MILANO]] IL 12/03/2004"),
            ("ROMA", "CHIEDERE A [[ROMA]]"),
        )
        for candidate, context in cases:
            with self.subTest(context=context):
                unresolved, reasons = self.assess(candidate, context, (candidate,))
                self.assertTrue(unresolved)
                self.assertIn("neighbourhood:direct_person_cue", reasons)

    def test_vocabularies_are_real_nonempty_reviewed_resources(self):
        minimum_entries = {
            "function_words.txt": 10,
            "months.txt": 12,
            "places.txt": 10,
            "cobol_keywords.txt": 10,
            "payroll_admin_terms.txt": 20,
            "titles_person_cues.txt": 20,
            "italian_common_words.txt": 100,
        }
        self.assertEqual(set(minimum_entries), set(VOCABULARY_FILES.values()))
        for filename, minimum in minimum_entries.items():
            with self.subTest(filename=filename):
                text = (VOCABULARY_DIRECTORY / filename).read_text(encoding="utf-8")
                entries = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                self.assertGreaterEqual(len(entries), minimum)
                self.assertNotIn("404", text.casefold())
                self.assertNotIn("not found", text.casefold())

    def test_distant_person_cue_does_not_protect_candidate(self):
        context = "      * REFERENTE PRATICA: [[ALLEGATO]]"
        self.assertFalse(has_direct_person_cue(context))
        unresolved, reasons = self.assess("ALLEGATO", context, ("ALLEGATO",))
        self.assertFalse(unresolved)

    def test_adjacent_unknown_word_makes_watchlist_occurrence_unresolved(self):
        unresolved, reasons = self.assess(
            "ALLEGATO", "      * Mario [[ALLEGATO]]", ("ALLEGATO", "Mario")
        )
        self.assertTrue(unresolved)
        self.assertIn("neighbourhood:name_like_before:Mario", reasons)


if __name__ == "__main__":
    unittest.main()
