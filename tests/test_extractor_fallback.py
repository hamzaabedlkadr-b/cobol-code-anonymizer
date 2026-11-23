import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.cli import main
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.policy import apply_name_policy


class ExtractorFallbackTests(unittest.TestCase):
    def test_policy_hides_unlocated_text_unless_explicitly_shown(self):
        for answer in (None, 'person', 'not_person'):
            decision = apply_name_policy(occurrence_id='1' * 64, model_context='TEXT',
                                         judge_decision=None, code_sensitive_identifier=False,
                                         extraction_unresolved=True, review_answer=answer)
            self.assertEqual(decision.outcome, 'leave_unchanged' if answer == 'not_person' else 'anonymize_whole')

    def test_unmatched_text_is_masked_and_released_with_cached_answers(self):
        for record_id in (2, 99):
            for ending in ('\n', '\r\n', '\r'):
                with self.subTest(record_id=record_id, ending=repr(ending)), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    source, watch, output = root / 'SYNTH.CBL', root / 'watch.txt', root / 'out'
                    lines = ['       PROCEDURE DIVISION.', '      * TOKENALPHA',
                             "           DISPLAY 'TOKENBETA 2024' 'OTHER TEXT'.",
                             '           MOVE 123 TO WS-RESULT.']
                    original = ending.join(lines) + ending
                    source.write_bytes(b'\xef\xbb\xbf' + original.encode())
                    watch.write_text('TOKENALPHA\n')
                    flags = [str(source), '--out-dir', str(output), '--watchlist', str(watch),
                             '--no-presidio', '--auto', '--llm', '--name-extract-model', 'fake',
                             '--sample-unchanged', '10']
                    primary = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENALPHA'},
                                                       {'record_id': record_id, 'text': 'TOKENOMEGA MISSING'}]}, '', .1, True)
                    retry = LlmJsonResult({'names': []}, '', .1, True)
                    reports = root / 'out_reports'
                    with redirect_stdout(io.StringIO()), \
                         patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                         patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                         patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                         patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, retry]) as call:
                        self.assertEqual(main(flags), 0)
                        result = (output / source.name).read_bytes()
                        expected = original.replace('TOKENALPHA', 'Nome001').replace('TOKENBETA', '*********')
                        if record_id == 99:
                            expected = expected.replace('OTHER TEXT', '***** ****')
                        self.assertEqual(result, b'\xef\xbb\xbf' + expected.encode())
                        self.assertFalse((reports / 'needs_review' / source.name).exists())
                        self.assertIn('Result: released 1, withheld 0, not complete 0', (reports / 'scan_summary.txt').read_text())
                        self.assertNotIn('[REVIEW REQUIRED]', (reports / 'llm_finding.txt').read_text())
                        self.assertIn('[released]', (reports / 'sample_check.txt').read_text())
                        with (reports / 'llm_name_review.csv').open() as handle:
                            rows = list(csv.DictReader(handle))
                        fallback = next(row for row in rows if row['extraction_status'] == 'unmatched')
                        self.assertEqual(fallback['final_action'], 'hide')
                        self.assertIn('text hidden automatically', fallback['reason'])
                        self.assertEqual(fallback['line'], '3' if record_id == 2 else '2')
                        self.assertEqual(main([*flags, '--overwrite']), 0)
                        self.assertEqual(call.call_count, 2)
                        for answer in ('show', 'hide'):
                            with (reports / 'llm_name_review.csv').open() as handle:
                                rows = list(csv.DictReader(handle))
                            next(row for row in rows if row['extraction_status'] == 'unmatched')['answer'] = answer
                            with (reports / 'llm_name_review.csv').open('w', newline='') as handle:
                                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                                writer.writeheader()
                                writer.writerows(rows)
                            self.assertEqual(main([*flags, '--overwrite']), 0)
                            self.assertEqual('TOKENBETA' in (output / source.name).read_text(), answer == 'show')
                            self.assertEqual(call.call_count, 2)

    def test_continued_literal_only_masks_payload_and_keeps_numbers(self):
        from cobol_code_anonymizer.extractor import NameExtractor
        from cobol_code_anonymizer.cobol_layout import classify_source
        from cobol_code_anonymizer.source_reader import decode_source_bytes
        from cobol_code_anonymizer.replacements import apply_file_replacements
        text = "       PROCEDURE DIVISION.\r\n           DISPLAY 'TOKEN 2024\r\n      -    'BETA'.\r\n"
        source = decode_source_bytes(text.encode(), source_path='SYNTH.CBL')
        extractor = NameExtractor()
        primary = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENOMEGA'}]}, '', .1, True)
        retry = LlmJsonResult({'names': []}, '', .1, True)
        with patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, retry]):
            extractor.extract(text, 'SYNTH.CBL', 'context', layout=classify_source(source, source_path='SYNTH.CBL'))
        result = apply_file_replacements(source, [], {}, redacted_spans=extractor.review_items[0]['spans'])
        self.assertEqual(result, text.replace('TOKEN', '*****').replace('BETA', '****'))

    def test_fallback_preserves_numeric_replacements(self):
        from cobol_code_anonymizer.scanner import Finding
        from cobol_code_anonymizer.source_reader import decode_source_bytes
        from cobol_code_anonymizer.replacements import apply_file_replacements, finding_key
        text = '      * TOKENBETA 2024 tel 123456789\n'
        start = text.index('123456789')
        number = Finding('SYNTH.CBL', 'PHONE', '123456789', start, start + 9, 1, start + 1, 1, '')
        source = decode_source_bytes(text.encode(), source_path='SYNTH.CBL')
        result = apply_file_replacements(source, [number], {finding_key(number): '000000000'},
                                         redacted_spans=[(8, len(text) - 1)])
        self.assertEqual(result, '      * ********* 2024 *** 000000000\n')

    def test_overlapping_chunks_on_one_line_mask_only_their_text_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text("       PROCEDURE DIVISION.\n           DISPLAY 'FIRST NOTE' 'SECOND NOTE'.\n           DISPLAY 'THIRD NOTE'.\n")
            primary = LlmJsonResult({'names': [{'record_id': 99, 'text': 'TOKENOMEGA'}]}, '', .1, True)
            retry = LlmJsonResult({'names': []}, '', .1, True)
            with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, retry, primary, retry]):
                self.assertEqual(main([str(source), '--out-dir', str(root / 'out'), '--no-presidio', '--auto',
                                       '--name-extract-model', 'fake', '--name-extract-chunk-lines', '2']), 0)
            self.assertEqual((root / 'out' / source.name).read_text(),
                             "       PROCEDURE DIVISION.\n           DISPLAY '***** ****' '****** ****'.\n           DISPLAY '***** ****'.\n")
            with (root / 'out_reports' / 'llm_name_review.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['final_action'] for row in rows}, {'hide'})
            self.assertEqual(len({row['key'] for row in rows}), 2)
