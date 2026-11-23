"""Regression tests for stress-fixture outcome accounting."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ANALYZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "judge_stress"
    / "analyze_results.py"
)
SPEC = importlib.util.spec_from_file_location("stress_analysis", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
stress_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stress_analysis)


class StressAnalysisTests(unittest.TestCase):
    def test_scoring_uses_entire_line_and_two_letter_truth_words(self):
        case = {"truth_names": [{"text": "Li Rossi", "line": 1, "start": 8, "end": 16}]}
        self.assertTrue(stress_analysis.span_contains_leak(case, ["LONG PSEUDONYM Li"]))
        self.assertFalse(stress_analysis.span_contains_leak(case, ["LONG PSEUDONYM Liliana Rossini"]))

    def test_release_and_withheld_are_scored_separately_and_review_is_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "out").mkdir()
            (root / "needs_review").mkdir()
            manifest = [{"id": "N1", "category": "person", "line_type": "comment", "line": 1,
                         "source_file": "A.CBL", "expected_outcome": "anonymize",
                         "truth_names": [{"text": "Mario Rossi", "line": 1}]},
                        {"id": "N2", "category": "person", "line_type": "comment", "line": 1,
                         "source_file": "B.CBL", "expected_outcome": "anonymize",
                         "truth_names": [{"text": "Mario Rossi", "line": 1}]}]
            (root / "manifest.json").write_text(json.dumps(manifest))
            decisions = root / "judge_decisions.json"
            decisions.write_text('{"decisions": []}')
            (root / "out" / "A.CBL").write_text("      * Nome001 Cognome001\n")
            (root / "needs_review" / "B.CBL").write_text("      * Mario Rossi\n")
            (root / "review_queue.csv").write_text('group_key,line_texts\nrossi,"NOTA ROSSI\nNOTA ROSSI\nVEDI ROSSI"\n')
            released = stress_analysis.analyze(root / "manifest.json", decisions, root)
            withheld = stress_analysis.analyze(root / "manifest.json", decisions, root, "needs_review")
            self.assertEqual(sum(row["cases"] for row in released), 1)
            self.assertEqual(sum(row["name_leaks"] for row in released), 0)
            self.assertEqual(sum(row["name_leaks"] for row in withheld), 1)
            self.assertEqual(stress_analysis.released_file_count(root), 1)
            self.assertEqual(stress_analysis.review_cost(root), (2, 2 / 3))

    def test_model_noise_never_exempts_a_new_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "N1", "category": "person", "line": 1,
                                            "source_file": "A.CBL", "expected_outcome": "anonymize",
                                            "truth_names": [{"text": "Mario Rossi", "line": 1}]}]))
            for name, text in (("old", "Nome001 Cognome001"), ("current", "Nome001 Rossi")):
                run = root / name
                run.mkdir()
                (run / "A.CBL").write_text(text)
                (run / "judge_decisions.json").write_text('{"decisions": []}')
            _, failed = stress_analysis.compare_runs(manifest, root / "current" / "judge_decisions.json",
                                                     root / "current", root / "old", model_noise_case_ids={"N1"})
            self.assertTrue(failed)

    def test_false_positives_split_by_reviewer_visibility(self):
        manifest = [
            {
                "category": "ordinary",
                "line_type": "comment",
                "line": 1,
                "truth_names": [],
                "expected_outcome": "leave",
            },
            {
                "category": "ordinary",
                "line_type": "comment",
                "line": 2,
                "truth_names": [],
                "expected_outcome": "leave",
            },
        ]
        decisions = {
            "decisions": [
                {"line": 1, "policy_outcome": "anonymize_and_review"},
                {"line": 2, "policy_outcome": "anonymize_whole"},
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "manifest.json"
            decisions_path = root / "decisions.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
            rows = stress_analysis.analyze(manifest_path, decisions_path)

        self.assertEqual(rows[0]["false_positives"], 2)
        self.assertEqual(rows[0]["fp_review"], 1)
        self.assertEqual(rows[0]["fp_hidden"], 1)

    def test_review_truth_uses_written_output_and_queue_not_decision_rows(self):
        manifest = [{
            "id": "R00001", "category": "identifier", "line_type": "code",
            "line": 1, "source_file": "PAYROLL.CBL", "text": "ABC123",
            "truth_names": [], "expected_outcome": "review_required",
        }]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "decisions.json").write_text(json.dumps({"decisions": []}), encoding="utf-8")
            (root / "PAYROLL.CBL").write_text("      * XXXXXX\n", encoding="utf-8")
            rows = stress_analysis.analyze(root / "manifest.json", root / "decisions.json", root)
            self.assertEqual(rows[0]["review_hidden"], 1)
            self.assertEqual(rows[0]["review_handled"], 0)

            (root / "review_queue.csv").write_text(
                "locations\nPAYROLL.CBL:1\n", encoding="utf-8"
            )
            rows = stress_analysis.analyze(root / "manifest.json", root / "decisions.json", root)
            self.assertEqual(rows[0]["review_handled"], 1)

            (root / "PAYROLL.CBL").write_text("      * ABC123\n", encoding="utf-8")
            rows = stress_analysis.analyze(root / "manifest.json", root / "decisions.json", root)
            self.assertEqual(rows[0]["review_leak"], 1)

    def test_current_csvs_score_by_file_and_read_each_output_once(self):
        from unittest.mock import patch
        manifest = [{"id": str(i), "category": "person", "line_type": "comment", "line": 1,
                     "source_file": file, "text": "TOKENALPHA", "truth_names": [{"text": "TOKENALPHA", "line": 1}],
                     "expected_outcome": "anonymize"} for i, file in enumerate(("FIRST.CBL", "SECOND.CBL", "FIRST.CBL"))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "out").mkdir()
            truth = root / "manifest.json"; truth.write_text(json.dumps(manifest))
            decisions = root / "decisions.csv"
            decisions.write_text("file,line,col,word,detector,judge,verifier,action,reason\nFIRST.CBL,1,1,TOKENALPHA,watchlist,not_called,not_called,hide,test\nSECOND.CBL,1,1,TOKENALPHA,watchlist,not_called,not_called,show,test\n")
            (root / "out" / "FIRST.CBL").write_text("* Nome001\n")
            (root / "out" / "SECOND.CBL").write_text("* TOKENALPHA\n")
            (root / "review_queue.csv").write_text("priority,word,line,where,action,reason,answer,key\noptional,TOKENALPHA,* TOKENALPHA,FIRST.CBL:1,hidden,test,,same\noptional,TOKENALPHA,* TOKENALPHA,SECOND.CBL:1,hidden,test,,same\n")
            with patch.object(stress_analysis, "load_output_lines", wraps=stress_analysis.load_output_lines) as reads:
                rows = stress_analysis.analyze(truth, decisions, root)
            self.assertEqual(reads.call_count, 2)
            self.assertEqual(sum(row["name_leaks"] for row in rows), 1)
            self.assertEqual(stress_analysis.review_cost(root), (1, 1 / 3))

    def test_compare_fails_on_new_hidden_false_positive(self):
        manifest = [{
            "id": "L00001", "category": "ordinary", "line_type": "comment",
            "line": 1, "source_file": "PAYROLL.CBL", "text": "ORDINARY",
            "truth_names": [], "expected_outcome": "leave",
        }]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "old"
            current = root / "current"
            old.mkdir()
            current.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (old / "judge_decisions.json").write_text(json.dumps({"decisions": []}), encoding="utf-8")
            (old / "PAYROLL.CBL").write_text("      * ORDINARY\n", encoding="utf-8")
            decisions_path = current / "judge_decisions.json"
            decisions_path.write_text(json.dumps({"decisions": [{
                "line": 1, "policy_outcome": "anonymize_whole",
            }]}), encoding="utf-8")
            (current / "PAYROLL.CBL").write_text("      * XXXXXX\n", encoding="utf-8")
            changes, failed = stress_analysis.compare_runs(
                manifest_path, decisions_path, current, old
            )

        self.assertEqual(changes, [("L00001", "handled", "fp_hidden")])
        self.assertTrue(failed)

    def test_same_code_class_change_is_model_noise_and_cannot_fail_compare(self):
        manifest = [{
            "id": "L00001", "category": "ordinary", "line_type": "comment",
            "line": 1, "source_file": "PAYROLL.CBL", "text": "ORDINARY",
            "truth_names": [], "expected_outcome": "leave",
        }]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "old"
            current = root / "current"
            repeat = root / "repeat"
            for run in (old, current, repeat):
                run.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            empty = json.dumps({"decisions": []})
            hidden = json.dumps({"decisions": [{
                "line": 1, "policy_outcome": "anonymize_whole",
            }]})
            (old / "judge_decisions.json").write_text(empty, encoding="utf-8")
            (old / "PAYROLL.CBL").write_text("      * ORDINARY\n", encoding="utf-8")
            (current / "judge_decisions.json").write_text(hidden, encoding="utf-8")
            (current / "PAYROLL.CBL").write_text("      * XXXXXX\n", encoding="utf-8")
            (repeat / "judge_decisions.json").write_text(empty, encoding="utf-8")
            (repeat / "PAYROLL.CBL").write_text("      * ORDINARY\n", encoding="utf-8")

            noise = stress_analysis.model_noise_changes(
                manifest_path, current, repeat
            )
            changes, failed = stress_analysis.compare_runs(
                manifest_path,
                current / "judge_decisions.json",
                current,
                old,
                model_noise_case_ids={case_id for case_id, _, _ in noise},
            )

        self.assertEqual(noise, [("L00001", "fp_hidden", "handled")])
        self.assertEqual(changes, [("L00001", "handled", "fp_hidden")])
        self.assertFalse(failed)

    def test_stable_same_code_case_is_not_model_noise(self):
        manifest = [{
            "id": "N00001", "category": "person", "line_type": "comment",
            "line": 1, "source_file": "PAYROLL.CBL", "text": "PERSON VALUE",
            "truth_names": [{"text": "PERSON VALUE", "line": 1, "start": 8, "end": 20}],
            "expected_outcome": "anonymize",
        }]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for name in ("first", "second"):
                run = root / name
                run.mkdir()
                (run / "judge_decisions.json").write_text(
                    json.dumps({"decisions": [{
                        "line": 1, "policy_outcome": "anonymize_whole",
                    }]}),
                    encoding="utf-8",
                )
                (run / "PAYROLL.CBL").write_text("      * XXXXXXXXXXXX\n", encoding="utf-8")

            noise = stress_analysis.model_noise_changes(
                manifest_path, root / "first", root / "second"
            )

        self.assertEqual(noise, [])

    def test_model_error_counts_keep_judge_and_verifier_separate(self):
        payload = {
            "decisions": [
                {"judge_outcome": "error", "verifier_outcome": "not_called"},
                {"judge_outcome": "propose_unchanged", "verifier_outcome": "error"},
                {"judge_outcome": "error", "verifier_outcome": "error"},
            ]
        }
        self.assertEqual(stress_analysis.model_error_counts(payload), (2, 2))

    def test_summary_helper_writes_required_stage_zero_columns(self):
        manifest = [{
            "id": "L00001", "category": "ordinary", "line_type": "comment",
            "line": 1, "source_file": "PAYROLL.CBL", "text": "ORDINARY",
            "truth_names": [], "expected_outcome": "leave",
        }]
        payload = {
            "llm_calls": 2,
            "decisions": [{
                "line": 1, "policy_outcome": "anonymize_and_review",
                "judge_outcome": "error", "verifier_outcome": "not_person",
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "judge_decisions.json").write_text(json.dumps(payload), encoding="utf-8")
            (root / "PAYROLL.CBL").write_text("      * XXXXXX\n", encoding="utf-8")
            (root / "review_queue.csv").write_text(
                "group_key,locations\nordinary,PAYROLL.CBL:1\n", encoding="utf-8"
            )
            (root / "runtime_seconds.txt").write_text("90\n", encoding="utf-8")
            result = subprocess.run(
                [
                    str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"),
                    str(ANALYZER_PATH.parent / "summarize_runs.py"),
                    "--run", "fixture", str(manifest_path), str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("leaks | fp_review | fp_hidden | review_groups", result.stdout)
        self.assertIn("fixture | 0 | 1 | 0 | 1 | 1 | 0 | 3 | 1.5", result.stdout)


if __name__ == "__main__":
    unittest.main()
