"""Tests for the model-free final residual safety pass."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.residual import scan_written_output
from cobol_code_anonymizer.scanner import Finding
from cobol_code_anonymizer.text_matching import output_span_to_source
from cobol_code_anonymizer.residual import _is_placeholder


class ResidualTests(unittest.TestCase):
    def test_positions_after_long_alias_map_back_to_source(self):
        self.assertEqual(output_span_to_source(16, 20, [(0, 4, 10), (6, 8, 8)]), None)
        self.assertEqual(output_span_to_source(22, 26, [(0, 4, 10), (6, 8, 8)]), (10, 14))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "PROGRAM.CBL").write_text("      * Nome001 Cognome001 ROSSI\n")
            known = Finding("PROGRAM.CBL", "NAME", "ROSSI", 12, 17, 1, 13, 0.7,
                            "      * ALFA [[ROSSI]]", "watchlist")
            result = scan_written_output(output, ["ROSSI"], [known], replaced_spans={"PROGRAM.CBL": [(8, 11, 18)]})
        self.assertEqual(result.replaceable, ())

    def test_full_aliases_are_recognised(self):
        for value in ("Nome001", "Nome001 Cognome001", "ANON_123", "user001@example.invalid"):
            self.assertTrue(_is_placeholder(value))
        self.assertFalse(_is_placeholder("Nome001 ROSSI"))

    def test_code_reference_is_never_replaceable(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "PROGRAM.CBL").write_text("           MOVE SALA TO RESULT.\n")
            residual = scan_written_output(output, ["SALA"], [])
        self.assertEqual(residual.replaceable, ())
        self.assertEqual([item.text for item in residual.code_sensitive], ["SALA"])


    def test_already_adjudicated_occurrence_is_not_reopened(self):
        text = "      * ROSSI\n"
        finding = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text="ROSSI",
            start=text.index("ROSSI"),
            end=text.index("ROSSI") + len("ROSSI"),
            line=1,
            column=9,
            confidence=0.7,
            context="      * [[ROSSI]]",
            source="watchlist",
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "PROGRAM.CBL").write_text(text, encoding="utf-8")
            residual = scan_written_output(output, ["ROSSI"], [finding])

        self.assertEqual(residual.replaceable, ())

    def test_identifier_residual_requires_review_instead_of_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "PROGRAM.CBL").write_text(
                "       01 TAB-Bianca PIC X.\n",
                encoding="utf-8",
            )
            residual = scan_written_output(output, ["Bianca"], [])

        self.assertEqual(residual.replaceable, ())
        self.assertEqual([finding.text for finding in residual.code_sensitive], ["Bianca"])


if __name__ == "__main__":
    unittest.main()
