import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.export_candidates import export_candidates, main


class CandidateExportTests(unittest.TestCase):
    def test_export_writes_one_stable_row_per_occurrence_with_source_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input" / "PAYROLL.CBL"
            source.parent.mkdir()
            original = "      * REFERENTE: Alex Conti\n      * Alex Conti\n"
            source.write_text(original, encoding="utf-8")
            watchlist = root / "names.txt"
            watchlist.write_text("Alex Conti\n", encoding="utf-8")
            output = root / "candidates.jsonl"

            count, diagnostics = export_candidates(
                source.parent,
                output,
                watchlists=[watchlist],
                include_default_names=False,
                use_presidio=False,
            )

            self.assertEqual(diagnostics, [])
            self.assertEqual(count, 2)
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["start"] for row in rows], sorted(row["start"] for row in rows))
        self.assertEqual([row["text"] for row in rows], ["Alex Conti", "Alex Conti"])
        self.assertEqual([row["line"] for row in rows], [1, 2])
        self.assertTrue(all(row["export_stage"] == "pre_judge" for row in rows))
        self.assertTrue(all(row["source"] == "watchlist" for row in rows))
        self.assertTrue(all(row["entity_type"] == "NAME" for row in rows))
        self.assertTrue(all(original[row["start"]:row["end"]] == row["text"] for row in rows))
        self.assertTrue(
            all(
                row["file_sha256"] == hashlib.sha256(original.encode("utf-8")).hexdigest()
                for row in rows
            )
        )

    def test_export_is_byte_for_byte_stable_for_same_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.CBL"
            source.write_text("      * Mario Rossi\n      * Mario Rossi\n", encoding="utf-8")
            watchlist = root / "names.txt"
            watchlist.write_text("Mario Rossi\n", encoding="utf-8")
            first = root / "first.jsonl"
            second = root / "second.jsonl"

            options = {
                "watchlists": [watchlist],
                "include_default_names": False,
                "use_presidio": False,
            }
            export_candidates(source, first, **options)
            export_candidates(source, second, **options)

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_cli_rejects_an_output_inside_input_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.mkdir()
            (source / "PAYROLL.CBL").write_text("      * Mario Rossi\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main([str(source), "--output", str(source / "candidates.jsonl")])

        self.assertEqual(raised.exception.code, 2)
