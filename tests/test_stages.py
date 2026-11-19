import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.checkpoints import finding_from_dict, load_checkpoint
from cobol_code_anonymizer.logical_text import LogicalPiece, LogicalText
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.scanner import Finding


class StageTests(unittest.TestCase):
    def test_logical_candidate_round_trip(self):
        logical = LogicalText('TOKENALPHA TOKENBETA',
                              (LogicalPiece(0, 11, 10, 21), LogicalPiece(11, 20, 40, 49)), 'literal')
        finding = Finding('SYNTH.CBL', 'NAME', 'TOKENALPHA ', 10, 21, 1, 11, .9,
                          '[[TOKENALPHA ]]', logical=logical, logical_start=0, logical_end=20,
                          logical_candidate=logical.text, review_key='literal:test')
        self.assertEqual(finding, finding_from_dict(json.loads(json.dumps(asdict(finding)))))

    def test_collect_candidates_does_not_make_policy_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * TOKENALPHA\n')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\n')
            with patch('cobol_code_anonymizer.pipeline.apply_name_policy', side_effect=AssertionError('no policy')):
                findings = scan_path(source, extra_watchlists=[watchlist], use_presidio=False, collect_only=True)
            self.assertEqual([hit.text for hit in findings], ['TOKENALPHA'])

    def test_extract_preserves_output_and_answers_and_skips_judge(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * TOKENALPHA\n')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\n')
            output = root / 'out'
            output.mkdir()
            (output / 'old.txt').write_text('existing')
            reports = root / 'reports'
            reports.mkdir()
            (reports / 'llm_finding.txt').write_text('existing')
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=AssertionError('no judge startup')):
                result = cli.main([str(source), '--step', 'extract', '--watchlist', str(watchlist),
                                   '--no-presidio', '--name-judge', '--out-dir', str(output), '--report-dir', str(reports)])
            self.assertEqual(result, 0)
            self.assertEqual((output / 'old.txt').read_text(), 'existing')
            self.assertEqual((reports / 'llm_finding.txt').read_text(), 'existing')
            self.assertIn('Step: extract', (reports / 'scan_summary.txt').read_text())
            self.assertIn('TOKENALPHA', (reports / 'scan_summary.txt').read_text())
            data = json.loads((reports / 'candidates.json').read_text())
            self.assertEqual(data['findings'][0]['text'], 'TOKENALPHA')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_checkpoint(reports / 'candidates.json', {})

    def test_decide_reuses_candidates_and_calls_only_judge_and_verifier(self):
        from cobol_code_anonymizer.decisions import Decision

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * VALORE TOKENALPHA PREVISTO\n')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\n')
            args = [str(source), '--watchlist', str(watchlist), '--no-presidio',
                    '--name-judge-model', 'fake', '--name-verifier-model', 'fake',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            def judged(candidate, context, occurrence_id):
                return Decision(occurrence_id=occurrence_id, stage='judge', outcome='propose_unchanged',
                                person_scope='none', non_person_category='common_word',
                                reading='ordinary word here', model_digest='1' * 64,
                                prompt_version='fake'), False
            def verified(**kwargs):
                return Decision(occurrence_id=kwargs['occurrence_id'], stage='verifier', outcome='not_person',
                                reading='ordinary word here', model_digest='1' * 64, prompt_version='fake')
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                with patch('cobol_code_anonymizer.scanner.scan_text', side_effect=AssertionError('no extraction')), \
                     patch('cobol_code_anonymizer.scanner.build_presidio_analyzer', side_effect=AssertionError('no spacy')), \
                     patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                     patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                     patch('cobol_code_anonymizer.judge.NameJudge.decide', side_effect=judged) as judge, \
                     patch('cobol_code_anonymizer.verifier.NameVerifier.verify', side_effect=verified) as verifier:
                    self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                    self.assertEqual(judge.call_count, 1)
                    self.assertEqual(verifier.call_count, 1)
            data = json.loads((root / 'reports' / 'decisions.json').read_text())
            self.assertEqual(data['findings'], [])
            self.assertEqual(data['rows'][0]['policy_outcome'], 'leave_unchanged')
            self.assertIn('[REJECTED - left as is]', (root / 'reports' / 'llm_finding.txt').read_text())
            self.assertFalse((root / 'out').exists())

    def test_changed_input_or_watchlist_rejected_before_model_startup(self):
        for change in ('source', 'watchlist'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'SYNTH.CBL'
                source.write_text('      * TOKENALPHA\n')
                watchlist = root / 'words.txt'
                watchlist.write_text('TOKENALPHA\n')
                args = [str(source), '--watchlist', str(watchlist), '--no-presidio',
                        '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                    (source if change == 'source' else watchlist).write_text('CHANGED\n')
                    with patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=AssertionError('no startup')), \
                         self.assertRaises(SystemExit):
                        cli.main(args + ['--step', 'decide', '--name-judge'])
                self.assertFalse((root / 'reports' / 'decisions.json').exists())

    def test_three_steps_match_full_run_including_numbers_paths_and_encoding(self):
        from cobol_code_anonymizer.decisions import Decision

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            batch = root / 'input'
            batch.mkdir()
            original = ("       IDENTIFICATION DIVISION.\r\n"
                        "       PROGRAM-ID. SYNTH.\r\n"
                        "       AUTHOR. TOKENALPHA.\r\n"
                        "       PROCEDURE DIVISION.\r\n"
                        "      * TOKENALPHA TOKENBETA\r\n"
                        "      * VALORE TOKENCHARLIE PREVISTO è\r\n"
                        "           DISPLAY 'TOKENDELTA \r\n"
                        "      -    'TOKENOMEGA'.\r\n"
                        "           MOVE WS-TOKENGAMMA1 TO RESULT\r\n"
                        "      * MATRICOLA: 123456\r\n"
                        "      * EMAIL: a@example.invalid\r\n")
            (batch / 'TOKENBETA.CBL').write_bytes(original.encode('cp1252'))
            (batch / 'EMPTY.CPY').write_bytes(b'      * PLAIN TEXT\r\n')
            (batch / 'BAD.CBL').write_bytes(b'\x00binary')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\nTOKENBETA\nTOKENCHARLIE\nTOKENDELTA\nTOKENOMEGA\nTOKENGAMMA\n123456\n')
            base = [str(batch), '--watchlist', str(watchlist), '--no-presidio', '--auto',
                    '--name-judge-model', 'fake', '--name-verifier-model', 'fake']
            def judged(candidate, context, occurrence_id):
                return Decision(occurrence_id=occurrence_id, stage='judge', outcome='propose_unchanged',
                                person_scope='none', non_person_category='common_word', reading='ordinary word here',
                                model_digest='1' * 64, prompt_version='fake'), False
            def verified(**kwargs):
                return Decision(occurrence_id=kwargs['occurrence_id'], stage='verifier', outcome='not_person',
                                reading='ordinary word here', model_digest='1' * 64, prompt_version='fake')
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.judge.NameJudge.decide', side_effect=judged), \
                 patch('cobol_code_anonymizer.verifier.NameVerifier.verify', side_effect=verified):
                full = base + ['--out-dir', str(root / 'full'), '--report-dir', str(root / 'full_reports')]
                split = base + ['--out-dir', str(root / 'split'), '--report-dir', str(root / 'split_reports')]
                self.assertEqual(cli.main(full), 1)  # The binary source is not complete.
                self.assertEqual(cli.main(split + ['--step', 'extract']), 1)
                self.assertEqual(cli.main(split + ['--step', 'decide']), 1)
                with patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=AssertionError('no startup')), \
                     patch('cobol_code_anonymizer.cli.scan_path', side_effect=AssertionError('no rescan')):
                    self.assertEqual(cli.main(split + ['--step', 'write']), 1)
            def files(path):
                return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob('*') if file.is_file()}
            self.assertEqual(files(root / 'full'), files(root / 'split'))
            self.assertEqual(set(files(root / 'split')), {'Cognome001.CBL', 'EMPTY.CPY'})
            written = (root / 'split' / 'Cognome001.CBL').read_bytes()
            self.assertIn('è'.encode('cp1252'), written)
            self.assertIn(b'VALORE TOKENCHARLIE PREVISTO', written)
            self.assertNotIn(b'TOKENDELTA', written)
            self.assertNotIn(b'TOKENGAMMA1', written)
            for name in ('replacement_map.csv', 'replacement_map.txt', 'llm_finding.txt', 'llm_name_review.csv', 'scan_summary.txt'):
                self.assertEqual((root / 'full_reports' / name).read_bytes(), (root / 'split_reports' / name).read_bytes(), name)

    def test_saved_extractor_fallback_is_written_without_model_calls(self):
        from cobol_code_anonymizer.llm import LlmJsonResult

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_bytes(b'\xef\xbb\xbf      * UNKNOWN TEXT 123\r\n')
            args = [str(source), '--no-presidio', '--auto', '--name-extract-model', 'fake',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            response = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENUNKNOWN'}]}, '', .1, True)
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', return_value=response):
                self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                with patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=AssertionError('no startup')), \
                     patch('cobol_code_anonymizer.extractor.NameExtractor.extract', side_effect=AssertionError('no extraction')):
                    self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                    self.assertEqual(cli.main(args + ['--step', 'write']), 0)
            self.assertEqual((root / 'out' / source.name).read_bytes(), b'\xef\xbb\xbf      * ******* **** 123\r\n')
            stats = (root / 'reports' / 'run_stats.txt').read_text()
            self.assertIn('Extractor: fake', stats)
            self.assertIn('Model requests (excluding startup): 2', stats)

    def test_write_rejects_changed_answers_until_decide_runs_again(self):
        import csv

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * TOKENALPHA\n')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\n')
            args = [str(source), '--watchlist', str(watchlist), '--no-presidio', '--auto',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                path = root / 'reports' / 'llm_name_review.csv'
                with path.open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    fields, rows = reader.fieldnames, list(reader)
                rows[0]['answer'] = 'show'
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(SystemExit):
                    cli.main(args + ['--step', 'write'])
                self.assertFalse((root / 'out').exists())
                self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                self.assertEqual(cli.main(args + ['--step', 'write']), 0)
            self.assertEqual((root / 'out' / source.name).read_text(), source.read_text())

    def test_write_still_withholds_final_check_matches(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * TOKENALPHA\n')
            args = [str(source), '--no-presidio', '--auto', '--out-dir', str(root / 'out'),
                    '--report-dir', str(root / 'reports')]
            hit = Finding(source.name, 'NAME', 'TOKENALPHA', 8, 18, 1, 9, 1,
                          '      * [[TOKENALPHA]]', source='watchlist')
            residual = SimpleNamespace(replaceable=[hit], code_sensitive=[], errors=[], samples=[])
            with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                with patch('cobol_code_anonymizer.cli.scan_written_output', return_value=residual) as final_check:
                    self.assertEqual(cli.main(args + ['--step', 'write']), 1)
                    final_check.assert_called_once()
            self.assertFalse((root / 'out' / source.name).exists())
            self.assertEqual((root / 'reports' / 'needs_review' / source.name).read_text(), source.read_text())
            self.assertIn('[REVIEW REQUIRED]', (root / 'reports' / 'llm_finding.txt').read_text())

    def test_failed_stage_rerun_cannot_reuse_old_stage_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * PLAIN TEXT\n')
            reports = root / 'reports'
            args = [str(source), '--no-presidio', '--auto', '--out-dir', str(root / 'out'), '--report-dir', str(reports)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                with patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=OSError('offline')):
                    self.assertEqual(cli.main(args + ['--step', 'decide', '--name-judge']), 1)
                    self.assertFalse((reports / 'decisions.json').exists())
                    with self.assertRaises(SystemExit):
                        cli.main(args + ['--step', 'write'])
                    self.assertEqual(cli.main(args + ['--step', 'extract', '--name-extract']), 1)
                    self.assertFalse((reports / 'candidates.json').exists())
            self.assertFalse((root / 'out').exists())

    def test_changed_source_before_write_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * PLAIN TEXT\n')
            args = [str(source), '--no-presidio', '--auto', '--out-dir', str(root / 'out'),
                    '--report-dir', str(root / 'reports')]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for step in ('extract', 'decide', 'write'):
                    self.assertEqual(cli.main(args + ['--step', step]), 0)
                before = (root / 'out' / source.name).read_bytes()
                source.write_text('      * CHANGED TEXT\n')
                with self.assertRaises(SystemExit):
                    cli.main(args + ['--step', 'write', '--overwrite'])
            self.assertEqual((root / 'out' / source.name).read_bytes(), before)
