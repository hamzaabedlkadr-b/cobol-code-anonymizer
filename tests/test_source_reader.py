import hashlib
import random
import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.source_reader import (
    DecodedByteOffsetMap,
    DecodedSource,
    LATIN1,
    SourceDecodingError,
    UTF8,
    UnsupportedSourceEncodingError,
    WINDOWS_1252,
    count_utf8_multibyte_sequences,
    decode_source_bytes,
    read_source,
    source_plausibility_score,
    split_source_lines,
)


class SourceReaderTests(unittest.TestCase):
    def assert_exact_byte_mapping(self, data: bytes, source) -> None:
        """Check every decoded boundary and character span against raw bytes."""

        prefix_length = 3 if source.has_bom else 0
        self.assertEqual(source.original_byte_offset(0), prefix_length)
        self.assertEqual(source.original_byte_offset(len(source.text)), len(data))
        for offset in range(len(source.text) + 1):
            expected = prefix_length + len(
                source.text[:offset].encode(source.encoding, errors="strict")
            )
            self.assertEqual(source.original_byte_offset(offset), expected)
        for start in range(len(source.text)):
            end = start + 1
            original_start, original_end = source.original_byte_span(start, end)
            self.assertEqual(
                data[original_start:original_end],
                source.text[start:end].encode(source.encoding, errors="strict"),
            )

    def test_strict_utf8_is_preferred_and_round_trips(self):
        original = "       IDENTIFICATION DIVISION.\n      * città e l’impiegato\n"
        data = original.encode("utf-8")

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.text, original)
        self.assertEqual(source.encoding, UTF8)
        self.assertFalse(source.has_bom)
        self.assertEqual(source.embedded_utf8_multibyte_count, 0)
        self.assertEqual(source.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(source.encode(), data)

    def test_windows_1252_preserves_curly_apostrophe(self):
        data = b"      * L\x92ARCHIVIO\r\n"

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.text, "      * L’ARCHIVIO\r\n")
        self.assertEqual(source.encoding, WINDOWS_1252)
        self.assertEqual(source.embedded_utf8_multibyte_count, 0)
        self.assertEqual(source.encode(), data)

    def test_latin1_is_used_when_strict_cp1252_cannot_decode(self):
        data = b"      * " + (b"A" * 80) + b" \x81\r"

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.text, data.decode("latin-1"))
        self.assertEqual(source.encoding, LATIN1)
        self.assertEqual(source.encode(), data)

    def test_utf8_bom_is_removed_from_text_and_restored_on_encode(self):
        data = b"\xef\xbb\xbf       IDENTIFICATION DIVISION.\r\n"

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.text, "       IDENTIFICATION DIVISION.\r\n")
        self.assertEqual(source.encoding, UTF8)
        self.assertTrue(source.has_bom)
        self.assertEqual(source.encode(), data)
        self.assertEqual(source.sha256, hashlib.sha256(data).hexdigest())

    def test_utf8_byte_mapping_tracks_multibyte_characters_sparsely(self):
        text = "Aà😀B"
        source = decode_source_bytes(text.encode("utf-8"), source_path="sample.txt")

        self.assertEqual(
            [source.original_byte_offset(offset) for offset in range(len(text) + 1)],
            [0, 1, 3, 7, 8],
        )
        self.assertEqual(source.original_byte_span(1, 3), (1, 7))
        self.assertEqual(source.original_byte_map.extra_byte_boundaries, (2, 3))
        self.assertEqual(source.original_byte_map.cumulative_extra_bytes, (1, 4))

        ascii_heavy = "A" * 10_000 + "à" + "B" * 10_000
        ascii_source = decode_source_bytes(
            ascii_heavy.encode("utf-8"),
            source_path="large.txt",
        )
        self.assertEqual(ascii_source.original_byte_map.extra_byte_boundaries, (10_001,))
        self.assertEqual(ascii_source.original_byte_map.cumulative_extra_bytes, (1,))

    def test_utf8_bom_byte_mapping_starts_after_the_bom(self):
        text = "AàB"
        data = b"\xef\xbb\xbf" + text.encode("utf-8")

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(
            [source.original_byte_offset(offset) for offset in range(len(text) + 1)],
            [3, 4, 6, 7],
        )
        self.assertEqual(source.original_byte_span(1, 2), (4, 6))
        self.assertEqual(data[4:6], "à".encode("utf-8"))

    def test_single_byte_byte_mapping_is_identity(self):
        cases = (
            (b"      * L\x92ARCHIVIO\r\n", "sample.cbl"),
            (b"      * " + (b"A" * 80) + b" \x81\r", "sample.cbl"),
        )

        for data, path in cases:
            with self.subTest(data=data):
                source = decode_source_bytes(data, source_path=path)
                self.assert_exact_byte_mapping(data, source)
                self.assertEqual(
                    source.original_byte_span(0, len(source.text)),
                    (0, len(data)),
                )

    def test_byte_mapping_preserves_newlines_and_trailing_dos_eof(self):
        text = "A\r\nà\rB\x1a"
        data = text.encode("utf-8")

        source = decode_source_bytes(data, source_path="sample.txt")

        self.assert_exact_byte_mapping(data, source)
        self.assertEqual(
            source.original_byte_span(1, 5),
            (1, 1 + len("\r\nà\r".encode("utf-8"))),
        )

    def test_byte_mapping_rejects_invalid_decoded_positions(self):
        source = decode_source_bytes(b"ABC", source_path="sample.txt")

        for offset in (-1, 4, True, "1"):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    source.original_byte_offset(offset)
        with self.assertRaises(ValueError):
            source.original_byte_span(2, 1)
        self.assertEqual(source.original_byte_span(1, 1), (1, 1))

    def test_decoded_source_rejects_a_map_that_does_not_fit_its_text(self):
        map_for_two_characters = DecodedByteOffsetMap(
            decoded_length=2,
            original_length=2,
            prefix_byte_count=0,
        )

        with self.assertRaisesRegex(ValueError, "decoded length"):
            DecodedSource(
                text="ABC",
                encoding=UTF8,
                has_bom=False,
                embedded_utf8_multibyte_count=0,
                sha256="0" * 64,
                original_byte_map=map_for_two_characters,
            )

    def test_invalid_bom_declared_utf8_does_not_fall_back(self):
        with self.assertRaisesRegex(SourceDecodingError, "UTF-8 BOM") as caught:
            decode_source_bytes(b"\xef\xbb\xbf\x92", source_path="sample.cbl")

        self.assertEqual(caught.exception.reason, "invalid_utf8_bom")

    def test_utf16_boms_are_rejected_before_single_byte_fallbacks(self):
        cases = (
            (b"\xff\xfeA\x00", "utf16_le_not_supported"),
            (b"\xfe\xff\x00A", "utf16_be_not_supported"),
        )
        for data, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(
                    UnsupportedSourceEncodingError,
                    "convert it to UTF-8",
                ) as caught:
                    decode_source_bytes(data, source_path="sample.cbl")
                self.assertEqual(caught.exception.reason, reason)

    def test_single_byte_fallback_rejects_c2_c3_utf8_sequences(self):
        cp1252_data = b"      * \x92PREFIX citt\xc3\xa0 SUFFIX\n"
        latin1_data = b"      * \x81PREFIX citt\xc3\xa0 SUFFIX\n"

        for data in (cp1252_data, latin1_data):
            with self.subTest(data=data):
                with self.assertRaises(SourceDecodingError) as caught:
                    decode_source_bytes(data, source_path="sample.cbl")
                self.assertEqual(caught.exception.reason, "mixed_encoding_suspected")

        self.assertEqual(count_utf8_multibyte_sequences(b"X\xc3\xa9Y"), 1)
        self.assertEqual(count_utf8_multibyte_sequences(b"X\x92Y"), 0)

    def test_mixed_encoding_check_ignores_accidental_non_mojibake_pairs(self):
        for accidental_pair in (b"\xc9\x94", b"\xc8\xa0"):
            with self.subTest(accidental_pair=accidental_pair):
                data = b"      * \x92PREFIX " + accidental_pair + b" SUFFIX\n"

                source = decode_source_bytes(data, source_path="sample.cbl")

                self.assertEqual(source.encoding, WINDOWS_1252)
                self.assertEqual(source.embedded_utf8_multibyte_count, 0)
                self.assertEqual(source.encode(), data)

    def test_embedded_utf8_count_is_limited_to_c2_and_c3_pairs(self):
        self.assertEqual(count_utf8_multibyte_sequences("é€😀".encode("utf-8")), 1)

    def test_nul_bytes_are_rejected_for_every_encoding(self):
        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(
                b"       MOVE X TO Y.\x00\n",
                source_path="sample.cbl",
            )

        self.assertEqual(caught.exception.reason, "nul_byte")

    def test_excessive_c0_control_characters_are_rejected(self):
        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(
                b"       MOVE X TO Y.\x01\x02\n",
                source_path="sample.cbl",
            )

        self.assertEqual(caught.exception.reason, "excessive_control_characters")

    def test_single_byte_fallback_requires_printable_text(self):
        data = b"      * VALID COMMENT " + (b"\x81" * 10) + b"\n"

        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(data, source_path="notes.txt")

        self.assertEqual(caught.exception.reason, "low_printable_ratio")

    def test_single_byte_fallback_requires_source_structure(self):
        data = b"Ordinary prose with Windows punctuation \x92 but no source structure."

        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(caught.exception.reason, "implausible_source")

    def test_indicator_column_space_alone_does_not_approve_prose(self):
        data = b"Hello, ordinary prose with Windows punctuation \x92 remains prose."

        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(caught.exception.reason, "implausible_source")

    def test_plausibility_score_covers_cobol_copybook_comments_and_jcl(self):
        examples = (
            "       IDENTIFICATION DIVISION.\n",
            "       01 RECORD-FIELD PIC X(10).\n",
            "      * fixed-format comment\n",
            "*> free-format comment\n",
            "//JOB1 JOB\n",
            "ABC123* alphanumeric sequence-area comment\n",
        )

        for text in examples:
            with self.subTest(text=text):
                self.assertGreaterEqual(source_plausibility_score(text), 2)

    def test_cp037_and_cp500_sources_are_rejected_as_ebcdic(self):
        text = (
            "       IDENTIFICATION DIVISION.\n"
            "       PROGRAM-ID. SAMPLE.\n"
            "       PROCEDURE DIVISION.\n"
            "           STOP RUN.\n"
        )

        for encoding in ("cp037", "cp500"):
            with self.subTest(encoding=encoding):
                with self.assertRaises(UnsupportedSourceEncodingError) as caught:
                    decode_source_bytes(
                        text.encode(encoding),
                        source_path="sample.cbl",
                    )
                self.assertEqual(caught.exception.reason, "suspected_ebcdic")

    def test_cp037_fixed_record_program_is_rejected_for_every_path_type(self):
        records = (
            "000100 IDENTIFICATION DIVISION.",
            "000200 PROGRAM-ID. PAYR010.",
            "000300 DATA DIVISION.",
            "000400 WORKING-STORAGE SECTION.",
            "000500 01 EMPLOYEE-NAME PIC X(30).",
            "000600 PROCEDURE DIVISION.",
            "000700     STOP RUN.",
        )
        data = "".join(record.ljust(80) for record in records).encode("cp037")

        for path in ("PAYR010", "PAYR010.txt", "PAYR010.dat"):
            with self.subTest(path=path):
                with self.assertRaises(UnsupportedSourceEncodingError) as caught:
                    decode_source_bytes(data, source_path=path)
                self.assertEqual(caught.exception.reason, "suspected_ebcdic")

    def test_cp037_name_data_without_source_structure_is_rejected(self):
        data = "MARIO ROSSI\nANNA BIANCHI\nLUCA FERRARI\n".encode("cp037")

        with self.assertRaises(UnsupportedSourceEncodingError) as caught:
            decode_source_bytes(data, source_path="EMP.dat")

        self.assertEqual(caught.exception.reason, "suspected_ebcdic")

    def test_lowercase_cp037_name_data_is_rejected_as_ebcdic(self):
        data = "mario rossi\nanna bianchi\nluca ferrari\n".encode("cp037")

        with self.assertRaises(UnsupportedSourceEncodingError) as caught:
            decode_source_bytes(data, source_path="EMP.dat")

        self.assertEqual(caught.exception.reason, "suspected_ebcdic")

    def test_ebcdic_record_separator_reports_ebcdic_before_controls(self):
        text = (
            "       IDENTIFICATION DIVISION.\n"
            "       PROGRAM-ID. SAMPLE.\n"
            "       PROCEDURE DIVISION.\n"
            "           STOP RUN.\n"
        )

        data = text.encode("cp037").replace(b"\x25", b"\x15")
        self.assertIn(b"\x15", data)
        with self.assertRaises(UnsupportedSourceEncodingError) as caught:
            decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(caught.exception.reason, "suspected_ebcdic")

    def test_non_cobol_text_extensions_do_not_require_source_structure(self):
        data = b"Ordinary prose with Windows punctuation \x92 and no COBOL syntax."
        names = (
            "notes.txt",
            "rows.csv",
            "query.sql",
            "records.dat",
            "loader.ctl",
            "README",
        )

        for name in names:
            with self.subTest(name=name):
                source = decode_source_bytes(data, source_path=name)
                self.assertEqual(source.encoding, WINDOWS_1252)
                self.assertEqual(source.encode(), data)

    def test_every_source_extension_requires_cobol_or_jcl_structure(self):
        data = b"Ordinary prose with Windows punctuation \x92 and no source syntax."

        for suffix in (".cbl", ".cob", ".cobol", ".cpy", ".jcl", ".proc"):
            with self.subTest(suffix=suffix):
                with self.assertRaises(SourceDecodingError) as caught:
                    decode_source_bytes(data, source_path=f"sample{suffix}")
                self.assertEqual(caught.exception.reason, "implausible_source")

    def test_one_trailing_dos_eof_is_ignored_but_preserved(self):
        data = b"       IDENTIFICATION DIVISION.\r\n\x1a"

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.text[-1], "\x1a")
        self.assertEqual(source.encode(), data)

        with self.assertRaises(SourceDecodingError) as caught:
            decode_source_bytes(data + b"\x1a", source_path="sample.cbl")
        self.assertEqual(caught.exception.reason, "excessive_control_characters")

    def test_at_signs_in_real_source_do_not_alone_trigger_ebcdic(self):
        data = b"      * CONTACT user@example.invalid \x92 OWNER\r\n"

        source = decode_source_bytes(data, source_path="sample.cbl")

        self.assertEqual(source.encoding, WINDOWS_1252)
        self.assertEqual(source.encode(), data)

    def test_cp1252_prose_with_at_signs_and_emails_is_not_ebcdic(self):
        data = (
            b"Contact alice@example.invalid and bob@example.invalid "
            b"or write @ support \x92 today."
        )

        source = decode_source_bytes(data, source_path="contacts.txt")

        self.assertEqual(source.encoding, WINDOWS_1252)
        self.assertEqual(source.encode(), data)

    def test_ascii_and_empty_input_have_stable_utf8_results(self):
        ascii_source = decode_source_bytes(
            b"       STOP RUN.",
            source_path="sample.cbl",
        )
        empty_source = decode_source_bytes(b"", source_path="empty.txt")

        self.assertEqual(ascii_source.encoding, UTF8)
        self.assertEqual(empty_source.encoding, UTF8)

    def test_mixed_newlines_are_preserved_without_normalizing(self):
        text = "FIRST\r\nSECOND\nTHIRD\rFOURTH"

        self.assertEqual(
            decode_source_bytes(text.encode("utf-8"), source_path="sample.txt").text,
            text,
        )

    def test_source_line_splitter_ignores_form_feed_and_other_controls(self):
        text = "FIRST\fMIDDLE\x85STILL-FIRST\r\nSECOND\nTHIRD\rFOURTH"

        self.assertEqual(
            split_source_lines(text),
            ["FIRST\fMIDDLE\x85STILL-FIRST", "SECOND", "THIRD", "FOURTH"],
        )
        self.assertEqual(
            split_source_lines(text, keepends=True),
            ["FIRST\fMIDDLE\x85STILL-FIRST\r\n", "SECOND\n", "THIRD\r", "FOURTH"],
        )

    def test_random_accepted_bytes_round_trip_exactly(self):
        generator = random.Random(20261005)
        checked = 0
        for _ in range(500):
            data = generator.randbytes(generator.randrange(0, 257))
            try:
                source = decode_source_bytes(data, source_path="random.bin")
            except (SourceDecodingError, UnsupportedSourceEncodingError):
                continue
            self.assertEqual(source.encode(), data)
            self.assertEqual(source.sha256, hashlib.sha256(data).hexdigest())
            self.assert_exact_byte_mapping(data, source)
            checked += 1

        self.assertGreater(checked, 0)

    def test_read_source_reads_bytes_without_changing_the_file(self):
        data = b"//JOB1 JOB\r\n//* COMMENT\r\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "JOB.JCL"
            path.write_bytes(data)

            source = read_source(path)

            self.assertEqual(source.text, data.decode("utf-8"))
            self.assertEqual(source.encode(), data)
            self.assertEqual(source.sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(path.read_bytes(), data)

    def test_read_source_uses_the_real_file_extension_for_plausibility(self):
        data = b"Ordinary prose with Windows punctuation \x92 and no source syntax."
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "notes.txt"
            source_path = Path(temp_dir) / "sample.CBL"
            text_path.write_bytes(data)
            source_path.write_bytes(data)

            self.assertEqual(read_source(text_path).encode(), data)
            with self.assertRaises(SourceDecodingError) as caught:
                read_source(source_path)
            self.assertEqual(caught.exception.reason, "implausible_source")

    def test_decode_requires_bytes_to_avoid_implicit_encoding_guesses(self):
        with self.assertRaisesRegex(TypeError, "data must be bytes"):
            decode_source_bytes("       STOP RUN.", source_path="sample.cbl")

    def test_decode_requires_an_explicit_source_path(self):
        with self.assertRaisesRegex(TypeError, "source_path"):
            decode_source_bytes(b"       STOP RUN.")

        with self.assertRaisesRegex(TypeError, "cannot be None"):
            decode_source_bytes(b"       STOP RUN.", source_path=None)


if __name__ == "__main__":
    unittest.main()
