"""Exact-byte and output-scoring checks for the heavy noisy fixture."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "samples" / "judge_stress"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "noisy_stress_analysis", ROOT / "analyze_results.py"
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
stress_analysis = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(stress_analysis)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "noisy_stress_generator", ROOT / "generate_noisy.py"
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
noisy_generator = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(noisy_generator)


class NoisyJudgeStressFixtureTests(unittest.TestCase):
    def test_committed_development_fixture_is_frozen(self):
        expected = {
            "PAYROLL_NOISY.CBL": "4286f5a8384117b3a50a039b265808a80edfedbbe884381151180576907ffc25",
            "noisy_manifest.json": "daa23e3562aae33487871d6a996ea39bbee74bb1cade086790138a34e97349c3",
            "noisy_roster.txt": "d0a32eeafbb1f8db6421cadccfdd245f87d640de549ee1cf13f3767a9b23afa5",
        }
        for file_name, digest in expected.items():
            with self.subTest(file=file_name):
                actual = hashlib.sha256((ROOT / file_name).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)

    def test_traps_are_a_reviewed_phrase_list_not_a_cartesian_grid(self):
        self.assertGreaterEqual(len(noisy_generator.TRAPS), 40)
        self.assertEqual(len(noisy_generator.TRAPS), len(set(noisy_generator.TRAPS)))
        self.assertIn("SALA RIUNIONI", noisy_generator.TRAPS)
        self.assertIn("CONTI CORRENTI", noisy_generator.TRAPS)
        self.assertIn("ROSA DEI VENTI", noisy_generator.TRAPS)
        self.assertFalse(hasattr(noisy_generator, "_TRAP_WORDS"))
        self.assertFalse(hasattr(noisy_generator, "_TRAP_CONTEXTS"))

    def test_fiscal_code_uses_real_name_date_place_and_checksum_rules(self):
        value = noisy_generator.derive_fiscal_code(
            "MARIO", "ROSSI", date(1980, 1, 1), "M", "H501"
        )

        self.assertEqual(value, "RSSMRA80A01H501U")
        self.assertTrue(noisy_generator.is_valid_fiscal_code(value))

    def test_generated_fiscal_codes_are_valid_and_damaged_ones_are_not(self):
        _, manifest, _ = noisy_generator.generate(300, "messy", seed=20261006)
        valid = [case["text"] for case in manifest if case["category"] == "fiscal-code-valid"]
        damaged = [case["text"] for case in manifest if case["category"] == "fiscal-code-damaged"]

        self.assertTrue(valid)
        self.assertTrue(damaged)
        self.assertTrue(all(noisy_generator.is_valid_fiscal_code(value) for value in valid))
        self.assertTrue(all(not noisy_generator.is_valid_fiscal_code(value) for value in damaged))

    def test_sealed_seeds_are_checked_by_totals_only(self):
        for seed in (20261107, 20261208):
            with self.subTest(seed=seed):
                _, manifest, _ = noisy_generator.generate(300, "messy", seed=seed)
                outcomes = Counter(case["expected_outcome"] for case in manifest)
                self.assertEqual(
                    outcomes,
                    {"anonymize": 150, "leave": 120, "review_required": 30},
                )

    def test_seeded_generation_is_reproducible_and_seed_specific(self):
        first = noisy_generator.generate(150, "messy", seed=111)
        repeated = noisy_generator.generate(150, "messy", seed=111)
        different = noisy_generator.generate(150, "messy", seed=222)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0], different[0])

    def test_generator_records_isolated_fixture_file_names(self):
        _, manifest, _ = noisy_generator.generate(
            150,
            "messy",
            seed=333,
            source_name="SEALED.CBL",
            roster_name="SEALED_NAMES.txt",
        )

        self.assertTrue(all(case["source_file"] == "SEALED.CBL" for case in manifest))
        self.assertTrue(all(case["roster_file"] == "SEALED_NAMES.txt" for case in manifest))

    def test_stage_zero_runner_is_valid_shell_and_never_overwrites(self):
        runner = ROOT / "run_all.sh"
        result = subprocess.run(
            ["bash", "-n", str(runner)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        text = runner.read_text(encoding="utf-8")
        self.assertIn("Refusing to overwrite existing run", text)
        self.assertNotIn("rm -rf", text)

    def test_manifest_tracks_the_exact_written_noise_bytes(self):
        lines = (ROOT / "PAYROLL_NOISY.CBL").read_text(encoding="utf-8").splitlines()
        manifest = json.loads((ROOT / "noisy_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(manifest), 300)
        for case in manifest:
            source = lines[case["line"] - 1]
            self.assertIn(case["text"], source, case["id"])
            self.assertEqual(case["source_file"], "PAYROLL_NOISY.CBL")
            self.assertEqual(case["file_noise_level"], "messy")
            for truth in case["truth_names"]:
                self.assertEqual(
                    lines[truth["line"] - 1][truth["start"]:truth["end"]],
                    truth["text"],
                    case["id"],
                )

    def test_default_messy_file_has_every_noise_family_at_least_twenty_times(self):
        manifest = json.loads((ROOT / "noisy_manifest.json").read_text(encoding="utf-8"))
        counts = Counter(tag for case in manifest for tag in case["noise"])
        expected = {
            "spelling_damage", "messy_comment", "misleading_context",
            "name_like_not_name", "innocent_injection", "long_irrelevant_context",
        }
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(counts[tag] >= 20 for tag in expected))
        self.assertEqual(sum(bool(case["noise"]) for case in manifest), 120)

    def test_roster_is_flat_and_ordinary_roster_words_stay_leave(self):
        manifest = json.loads((ROOT / "noisy_manifest.json").read_text(encoding="utf-8"))
        roster = (ROOT / "noisy_roster.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(roster)
        self.assertTrue(all(" " not in entry for entry in roster))
        self.assertGreater(len(roster), len(set(roster)))
        self.assertIn("sala", roster)
        self.assertIn("Nicolo", roster)
        self.assertIn("NOME-INUTILIZZATO", roster)

        roster_words = {word.casefold() for entry in roster for word in re.findall(r"[^\W_]+", entry)}
        ordinary_traps = [
            case for case in manifest
            if case["expected_outcome"] == "leave"
            and ({word.casefold() for word in re.findall(r"[^\W_]+", case["text"])} & roster_words)
        ]
        self.assertGreaterEqual(len(ordinary_traps), 10)
        self.assertTrue(any(case["text"].startswith("SALA ") for case in ordinary_traps))
        self.assertTrue(any(case["text"].startswith("CONTI ") for case in ordinary_traps))

        review_cases = [case for case in manifest if case["expected_outcome"] == "review_required"]
        self.assertGreaterEqual(len(review_cases), 20)
        self.assertLessEqual(len(review_cases), 40)
        self.assertTrue(any(case["category"] == "fiscal-code-valid" for case in review_cases))
        self.assertTrue(any(case["category"] == "fiscal-code-damaged" for case in review_cases))
        self.assertTrue(any(case["category"] == "ambiguous-line" for case in review_cases))

    def test_noisy_fixture_limits_repeated_exact_text_and_titles_on_traps(self):
        manifest = json.loads((ROOT / "noisy_manifest.json").read_text(encoding="utf-8"))
        source = (ROOT / "PAYROLL_NOISY.CBL").read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(max(Counter(case["text"] for case in manifest).values()), 3)
        title = re.compile(r"\b(?:DOTT\.?|SIG\.?|ING\.?|AVV\.?|GEOM\.?)\b", re.IGNORECASE)
        for case in manifest:
            if case["expected_outcome"] != "anonymize":
                self.assertIsNone(title.search(source[case["line"] - 1]), case["id"])

    def test_output_span_scoring_counts_a_partial_mask_as_a_leak(self):
        manifest = [{
            "id": "N00001", "line": 1, "source_file": "PAYROLL_NOISY.CBL",
            "roster_file": "noisy_roster.txt", "file_noise_level": "messy",
            "noise": ["spelling_damage"], "category": "noisy-person",
            "line_type": "full-line-comment", "text": "Mario Rossi",
            "expected_outcome": "anonymize", "expected_route": "G3",
            "truth_names": [{
                "text": "Mario Rossi", "line": 1, "start": 8, "end": 19,
                "column": 9, "in_watchlist": False,
            }],
        }]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "decisions.json").write_text(json.dumps({"decisions": []}), encoding="utf-8")
            (root / "noisy_roster.txt").write_text("Pietro Verdi\n", encoding="utf-8")
            # Mario is hidden, but Rossi is still readable at its original span.
            (root / "PAYROLL_NOISY.CBL").write_text("      * XXXXX Rossi\n", encoding="utf-8")
            rows = stress_analysis.analyze(root / "manifest.json", root / "decisions.json", root)
            noise_rows = stress_analysis.noise_breakdown(
                root / "manifest.json", root / "decisions.json", root
            )

        self.assertEqual(rows[0]["name_leaks"], 1)
        self.assertEqual(noise_rows[0]["name_leaks"], 1)


if __name__ == "__main__":
    unittest.main()
