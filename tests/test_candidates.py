import hashlib
import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.candidates import Detection, Occurrence
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.scanner import Finding, write_json


FILE_HASH = "a" * 64


def make_occurrence(**changes):
    values = {
        "file": "PROGRAM.CBL",
        "file_sha256": FILE_HASH,
        "source_text": "TOKEN_A",
        "decoded_start": 10,
        "decoded_end": 17,
        "line": 2,
        "column": 5,
        "region": "comment",
        "context": "[[TOKEN_A]]",
    }
    values.update(changes)
    return Occurrence(**values)


def make_detection(occurrence, **changes):
    values = {
        "occurrence_id": occurrence.occurrence_id,
        "detector": "watchlist",
        "detector_version": "1",
        "entity_type": "NAME",
        "score": 0.7,
        "matched_value": occurrence.source_text,
        "matched_entry": None,
        "variant": None,
    }
    values.update(changes)
    return Detection(**values)


class CandidateRecordTests(unittest.TestCase):
    def test_same_occurrence_has_stable_id(self):
        first = make_occurrence()
        second = make_occurrence()

        self.assertEqual(first.occurrence_id, second.occurrence_id)

    def test_occurrence_id_changes_with_source_identity(self):
        base = make_occurrence()
        variants = [
            make_occurrence(file="OTHER.CBL"),
            make_occurrence(file_sha256="b" * 64),
            make_occurrence(source_text="TOKEN_B"),
            make_occurrence(
                decoded_start=20,
                decoded_end=27,
            ),
        ]

        self.assertTrue(
            all(item.occurrence_id != base.occurrence_id for item in variants)
        )
        self.assertEqual(len({item.occurrence_id for item in variants}), len(variants))

    def test_occurrence_id_does_not_depend_on_optional_byte_offsets(self):
        without_byte_map = make_occurrence()
        with_byte_map = make_occurrence(original_start=40, original_end=47)

        self.assertEqual(without_byte_map.occurrence_id, with_byte_map.occurrence_id)

    def test_detection_id_changes_with_detector_or_version(self):
        occurrence = make_occurrence()
        base = make_detection(occurrence)
        spacy = make_detection(occurrence, detector="presidio_spacy")
        newer = make_detection(occurrence, detector_version="2")

        self.assertNotEqual(base.detection_id, spacy.detection_id)
        self.assertNotEqual(base.detection_id, newer.detection_id)

    def test_json_round_trip_preserves_records(self):
        occurrence = make_occurrence()
        detection = make_detection(
            occurrence,
            matched_entry="CANONICAL_ENTRY",
            variant="exact",
        )

        self.assertEqual(Occurrence.from_json(occurrence.to_json()), occurrence)
        self.assertEqual(Detection.from_json(detection.to_json()), detection)

    def test_detection_must_match_the_occurrence_source_text(self):
        occurrence = make_occurrence()
        detection = make_detection(occurrence, matched_value="TOKEN_B")

        with self.assertRaisesRegex(ValueError, "matched_value"):
            Finding.from_candidate_records(occurrence, detection)

    def test_finding_round_trip_preserves_every_field(self):
        finding = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text="TOKEN_A",
            start=10,
            end=17,
            line=2,
            column=5,
            confidence=0.7,
            context="[[TOKEN_A]]",
            source="watchlist",
        )

        occurrence, detection = finding.to_candidate_records(
            file_sha256=FILE_HASH,
            detector_version="1",
            region="comment",
        )
        rebuilt = Finding.from_candidate_records(occurrence, detection)

        self.assertIsNone(occurrence.original_start)
        self.assertIsNone(occurrence.original_end)
        self.assertEqual(rebuilt, finding)
        self.assertEqual(rebuilt.to_dict(), finding.to_dict())

    def test_empty_legacy_source_is_preserved(self):
        finding = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text="TOKEN_A",
            start=10,
            end=17,
            line=2,
            column=5,
            confidence=0.7,
            context="[[TOKEN_A]]",
        )

        occurrence, detection = finding.to_candidate_records(
            file_sha256=FILE_HASH,
            detector_version="1",
        )

        self.assertEqual(Finding.from_candidate_records(occurrence, detection), finding)

    def test_scan_output_is_byte_for_byte_equal_after_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PROGRAM.CBL"
            source.write_text("      * TOKEN_A\n", encoding="utf-8")
            watchlist = root / "watchlist.txt"
            watchlist.write_text("TOKEN_A\n", encoding="utf-8")
            findings = scan_path(
                source,
                entities={"NAME"},
                extra_watchlists=[watchlist],
                include_default_names=False,
                use_presidio=False,
            )
            file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            rebuilt = []
            for finding in findings:
                occurrence, detection = finding.to_candidate_records(
                    file_sha256=file_hash,
                    detector_version="1",
                )
                rebuilt.append(Finding.from_candidate_records(occurrence, detection))

            before = root / "before.json"
            after = root / "after.json"
            write_json(before, findings)
            write_json(after, rebuilt)

            self.assertGreater(len(findings), 0)
            self.assertEqual(before.read_bytes(), after.read_bytes())
