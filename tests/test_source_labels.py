import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.sample_source_lines import sample_source_lines
from experiments.validate_source_labels import validate_source_labels


class SourceLabelTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "batch"
        source.mkdir()
        (source / "PAYROLL.CBL").write_text(
            "       IDENTIFICATION DIVISION.\n"
            "       PROGRAM-ID. PAYROLL.\n"
            "      * REVIEWED BY MARIO\n"
            "       01 WS-CONTI PIC X(10).\n"
            "       MOVE 'Mario Rossi' TO WS-CONTI.\n",
            encoding="utf-8",
        )
        (source / "JOB.JCL").write_text("//JOB1 JOB USER=ABC\n", encoding="utf-8")
        return source

    def test_sampling_is_reproducible_and_includes_source_strata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_source(root)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self.assertEqual(sample_source_lines(source, first, per_stratum=1, seed=23), 5)
            self.assertEqual(sample_source_lines(source, second, per_stratum=1, seed=23), 5)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
            strata = {row["stratum"] for row in rows}
            self.assertEqual(
                strata,
                {"comment", "identification", "literal", "identifier_code", "jcl"},
            )
            self.assertTrue(all(row["labels"] == [] for row in rows))
            self.assertTrue(all("source" not in row and "finding" not in row for row in rows))

    def test_validator_accepts_exact_person_and_not_person_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_source(root)
            sample = root / "sample.jsonl"
            sample_source_lines(source, sample, per_stratum=1, seed=23)
            rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                start = row["line_start"]
                end = row["line_end"]
                row["labels"] = [{"label": "NOT_PERSON", "start": start, "end": end, "text": row["source_text"]}]
            target = root / "labeled.jsonl"
            target.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_source_labels(target, source), [])

    def test_validator_rejects_stale_hash_and_wrong_span(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_source(root)
            sample = root / "sample.jsonl"
            sample_source_lines(source, sample, per_stratum=1, seed=23)
            row = json.loads(sample.read_text(encoding="utf-8").splitlines()[0])
            row["file_sha256"] = hashlib.sha256(b"changed").hexdigest()
            row["labels"] = [{"label": "PERSON", "start": row["line_start"], "end": row["line_end"], "text": "wrong"}]
            labels = root / "bad.jsonl"
            labels.write_text(json.dumps(row) + "\n", encoding="utf-8")

            errors = validate_source_labels(labels, source)
            self.assertIn("source hash changed", errors[0])
