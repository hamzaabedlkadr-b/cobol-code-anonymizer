import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.cli import main
from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult, call_ollama_json
from cobol_code_anonymizer.scanner import Finding
from cobol_code_anonymizer.verifier import NameVerifier


class RunStatsTests(unittest.TestCase):
    def test_step_time_is_separate_from_saved_pipeline_time(self):
        from cobol_code_anonymizer.cli import write_run_stats

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stats.txt'
            with patch('cobol_code_anonymizer.cli.time.perf_counter', return_value=150), redirect_stdout(io.StringIO()) as output:
                write_run_stats(path, 50, [], 'Step: decide', step_started=140)
            self.assertIn('100.0 s; cumulative', path.read_text())
            self.assertIn('Step duration: 0.00 h (10.0 s)', path.read_text())
            self.assertIn('Step time: 10.0 s   Cumulative time: 100.0 s', output.getvalue())

    def test_judge_counts_startup_retry_and_cache_tokens(self):
        context = '* NOTE [[TOKENALPHA]]'
        finding = Finding('SYNTH.CBL', 'NAME', 'TOKENALPHA', 7, 17, 1, 8, 1, context)
        person = {'decision': 'anonymize_whole', 'person_scope': 'whole',
                  'person_texts': ['TOKENALPHA'], 'non_person_category': 'none',
                  'evidence_quote': 'TOKENALPHA', 'reading': 'person'}
        judge = NameJudge(model='fake', model_digest='1' * 64)
        replies = [LlmJsonResult(person, '', .1, True, prompt_tokens=5, completion_tokens=2),
                   LlmJsonResult({**person, 'person_texts': ['TOKENOMEGA']}, '', .2, True,
                                 prompt_tokens=20, completion_tokens=4),
                   LlmJsonResult(person, '', .3, True, prompt_tokens=10, completion_tokens=3)]
        with patch('cobol_code_anonymizer.judge.call_ollama_json', side_effect=replies) as call:
            self.assertEqual(judge._ask(context).kind, 'valid')
            decision, cached = judge.decide(finding, context, '2' * 64)
            self.assertEqual(decision.outcome, 'anonymize_whole')
            self.assertFalse(cached)
            self.assertTrue(judge.decide(finding, context, '2' * 64)[1])
        self.assertEqual(call.call_count, 3)
        self.assertEqual((judge.prompt_tokens, judge.completion_tokens), (35, 9))
        self.assertEqual((judge.canary_calls, judge.calls, judge.cache_hits, judge.http_attempts), (1, 2, 1, 3))

    def test_verifier_counts_reported_tokens_once_for_cached_answers(self):
        verifier = NameVerifier(model='fake', model_digest='1' * 64)
        response = LlmJsonResult({'decision': 'not_person', 'reading': 'ordinary word'}, '', .2, True,
                                 prompt_tokens=50, completion_tokens=7)
        with patch('cobol_code_anonymizer.verifier.call_ollama_json', return_value=response) as call:
            for _ in range(2):
                self.assertEqual(verifier.verify(occurrence_id='2' * 64, candidate='TOKENALPHA',
                                                 context='* NOTE [[TOKENALPHA]]').outcome, 'not_person')
        self.assertEqual(call.call_count, 1)
        self.assertEqual((verifier.prompt_tokens, verifier.completion_tokens, verifier.cache_hits), (50, 7, 1))

    def test_batch_writes_duration_results_tokens_and_zero_new_tokens_on_cached_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * PLAIN TEXT\n')
            output, reports = root / 'out', root / 'out_reports'
            flags = [str(source), '--out-dir', str(output), '--no-presidio', '--auto', '--name-extract-model', 'fake']
            response = LlmJsonResult({'names': []}, '', .5, True, prompt_tokens=200, completion_tokens=30)
            with redirect_stdout(io.StringIO()) as stdout, \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', return_value=response) as call:
                self.assertEqual(main(flags), 0)
                report = (reports / 'run_stats.txt').read_text()
                self.assertIn('Duration:', report)
                self.assertIn('Finished (UTC):', report)
                self.assertIn('Result: released 1, withheld 0, not complete 0', report)
                self.assertIn('Extractor: fake', report)
                self.assertIn('Total input tokens (reported): 200', report)
                self.assertIn('Total output tokens (reported): 30', report)
                self.assertNotIn('PLAIN TEXT', report)
                self.assertIn(str(reports / 'run_stats.txt'), stdout.getvalue())
                self.assertEqual(main([*flags, '--overwrite']), 0)
                report = (reports / 'run_stats.txt').read_text()
                self.assertEqual(call.call_count, 1)
                self.assertIn('Cache hits: 1', report)
                self.assertIn('Total input tokens (reported): 0', report)
                self.assertIn('Total output tokens (reported): 0', report)
                source.write_text('      * CHANGED TEXT\n')
                call.side_effect = call_ollama_json
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                           return_value=LlmJsonResult(None, '', .3, False, error='fake outage')) as http:
                    self.assertEqual(main([*flags, '--overwrite']), 1)
                    self.assertEqual(http.call_count, 2)
                report = (reports / 'run_stats.txt').read_text()
                self.assertIn('Result: released 0, withheld 0, not complete 1', report)
                self.assertIn('Model requests (excluding startup): 1', report)
                self.assertIn('HTTP attempts (chat, including startup/retries): 2', report)
                self.assertIn('Model request time: 0.6 s', report)
                self.assertIn('Failures: 1', report)

    def test_failed_rerun_removes_previous_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, reports = root / 'SYNTH.CBL', root / 'out_reports'
            source.write_text('      * PLAIN TEXT\n')
            reports.mkdir()
            (reports / 'run_stats.txt').write_text('old successful run')
            with redirect_stdout(io.StringIO()), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', side_effect=OSError('fake outage')):
                self.assertEqual(main([str(source), '--out-dir', str(root / 'out'), '--no-presidio',
                                       '--name-extract-model', 'fake', '--auto']), 1)
            self.assertFalse((reports / 'run_stats.txt').exists())
