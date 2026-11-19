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

    def test_extract_leaves_existing_output_and_reports_and_skips_judge(self):
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
            (reports / 'scan_summary.txt').write_text('existing')
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=AssertionError('no judge startup')):
                result = cli.main([str(source), '--step', 'extract', '--watchlist', str(watchlist),
                                   '--no-presidio', '--name-judge', '--out-dir', str(output), '--report-dir', str(reports)])
            self.assertEqual(result, 0)
            self.assertEqual((output / 'old.txt').read_text(), 'existing')
            self.assertEqual((reports / 'scan_summary.txt').read_text(), 'existing')
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
