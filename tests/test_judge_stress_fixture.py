import json
import re
import unittest
from collections import Counter
from pathlib import Path


class JudgeStressFixtureTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "samples" / "judge_stress"

    @staticmethod
    def logical_continued_literal(first: str, continuation: str) -> str:
        """Reconstruct a fixed-format literal carried to its continuation record."""
        if len(first) != 72:
            raise AssertionError("the first literal record must end in column 72")
        opening_quote = first.find("'")
        if opening_quote < 0 or "'" in first[opening_quote + 1:]:
            raise AssertionError("the first record must have one unclosed quote")
        if len(continuation) < 8 or continuation[6] != "-":
            raise AssertionError("the continuation indicator must be in column 7")
        reopen_at = 7
        while reopen_at < len(continuation) and continuation[reopen_at] == " ":
            reopen_at += 1
        if reopen_at >= len(continuation) or continuation[reopen_at] != "'":
            raise AssertionError("the literal must re-open after column-7 padding")
        closing_quote = continuation.find("'", reopen_at + 1)
        if closing_quote < 0 or closing_quote == len(continuation) - 1:
            raise AssertionError("the literal needs a closing quote and statement suffix")
        return first[opening_quote + 1:] + continuation[reopen_at + 1:closing_quote]

    def test_every_manifest_truth_span_matches_payroll_source_exactly(self):
        source = (self.root / "PAYROLL.CBL").read_text(encoding="utf-8")
        lines = source.splitlines()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(manifest), 300)
        self.assertEqual(
            Counter(case["expected_outcome"] for case in manifest),
            Counter(anonymize=150, leave=120, review_required=30),
        )
        self.assertNotRegex(source, r"anonymize|review_required|EXPECT-NAME|EXPECT-NO-NAME")

        case_ids = set()
        truth_count = 0
        for case in manifest:
            case_id = case["id"]
            self.assertRegex(case_id, r"^C\d{5}$")
            self.assertNotIn(case_id, case_ids)
            case_ids.add(case_id)
            line_number = case["line"]
            self.assertGreaterEqual(line_number, 1)
            self.assertLessEqual(len(lines[line_number - 1]), 72)
            self.assertEqual(lines[line_number - 1][:6], case_id)
            self.assertIn(lines[line_number - 1][6], " *-")

            for truth in case["truth_names"]:
                truth_count += 1
                self.assertIsInstance(truth["in_watchlist"], bool)
                actual = lines[truth["line"] - 1][truth["start"]:truth["end"]]
                self.assertEqual(actual, truth["text"], case_id)
                self.assertEqual(truth["column"], truth["start"] + 1)

        self.assertGreater(truth_count, 0)

    def test_every_manifest_text_appears_in_its_source_record(self):
        lines = (self.root / "PAYROLL.CBL").read_text(encoding="utf-8").splitlines()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

        for case in manifest:
            source = lines[case["line"] - 1]
            if case["line_type"] == "continued-literal":
                continuation = lines[case["line"]]
                self.assertEqual(
                    self.logical_continued_literal(source, continuation),
                    case["text"],
                    case["id"],
                )
            else:
                self.assertIn(case["text"], source, case["id"])

    def test_generator_balance_pairs_and_identifier_reviews(self):
        lines = (self.root / "PAYROLL.CBL").read_text(encoding="utf-8").splitlines()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        categories = Counter(case["category"] for case in manifest)
        hard_categories = {
            "no-cue-repeated-surname", "foreign-given-name", "prompt-injection",
            "continued-literal", "multi-person", "month-place-surname-person",
            "watchlist-word-person", "distant-cue", "name-inside-phrase",
            "apostrophe", "particles", "accents", "lowercase", "spaced-letters",
            "reversed-order", "paired-person", "paired-ordinary",
        }
        self.assertTrue(all(8 <= categories[category] <= 12 for category in hard_categories))
        self.assertLessEqual(categories["bulk-positive"], 40)

        positives = [case for case in manifest if case["expected_outcome"] == "anonymize"]
        directly_cued = sum(case.get("direct_person_cue", False) for case in positives)
        self.assertLessEqual(directly_cued * 3, len(positives))

        foreign_cases = [case for case in manifest if case["category"] == "foreign-given-name"]
        self.assertGreaterEqual(
            sum(not case["direct_person_cue"] for case in foreign_cases),
            5,
        )

        paired_people = [case for case in manifest if case["category"] == "paired-person"]
        self.assertGreaterEqual(
            sum(not case["direct_person_cue"] for case in paired_people),
            len(paired_people) // 2,
        )

        pairs = {}
        for case in manifest:
            if "pair_id" in case:
                pairs.setdefault(case["pair_id"], []).append(case)
        self.assertEqual(len(pairs), 8)
        for paired_cases in pairs.values():
            self.assertEqual(
                sorted(case["expected_outcome"] for case in paired_cases),
                ["anonymize", "leave"],
            )

        review_cases = [case for case in manifest if case["expected_outcome"] == "review_required"]
        self.assertEqual(len(review_cases), 30)
        self.assertTrue(all(case["category"].startswith("identifier-") for case in review_cases))
        self.assertTrue(all("'" not in lines[case["line"] - 1] for case in review_cases))

        continued_cases = [case for case in manifest if case["category"] == "continued-literal"]
        self.assertEqual(len(continued_cases), 8)
        self.assertGreaterEqual(
            sum(case.get("continuation_split") == "word" for case in continued_cases),
            4,
        )
        self.assertGreaterEqual(
            sum(case.get("continuation_split") == "between" for case in continued_cases),
            4,
        )

    def test_generated_source_has_fixed_format_case_records_and_continuation(self):
        lines = (self.root / "PAYROLL.CBL").read_text(encoding="utf-8").splitlines()
        case_lines = [line for line in lines if re.match(r"^C\d{5}[ *]", line)]
        self.assertEqual(len(case_lines), 300)
        self.assertTrue(any(line.startswith("      -") for line in lines))
        self.assertTrue(any(" *> " in line for line in case_lines))
        self.assertTrue(any(line[6] == "*" for line in case_lines))
        self.assertTrue(any(" DISPLAY '" in line for line in case_lines))
        self.assertTrue(any(" MOVE '" in line for line in case_lines))
        self.assertTrue(any(" VALUE '" in line for line in case_lines))

    def test_roster_contains_individual_name_entries_not_full_people(self):
        roster = (self.root / "roster.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(roster)
        self.assertNotIn("Ilaria D'Amico", roster)
        self.assertIn("Ilaria", roster)
        self.assertIn("D'Amico", roster)
        self.assertIn("sala", roster)
        self.assertIn("NOMINATIVO-INUTILIZZATO", roster)
