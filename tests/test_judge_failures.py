"""Transport isolation, safe release, and durable per-answer caches."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.extractor import NameExtractor
from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult, call_ollama_json
from cobol_code_anonymizer.llm_cache import PersistentResponseCache
from cobol_code_anonymizer.scanner import Finding
from cobol_code_anonymizer.verifier import NameVerifier


CONTEXT = '* VALORE [[TOKENALPHA]] PREVISTO'
CANDIDATE = Finding('SYNTH.CBL', 'NAME', 'TOKENALPHA', 9, 19, 1, 10, 1, CONTEXT)
FAILURE = LlmJsonResult(None, '', .1, False, error='synthetic timeout')


def reply(role):
    payload = {'decision': 'not_person', 'reading': 'ordinary word here'}
    if role == 'judge':
        payload.update(decision='propose_unchanged', person_scope='none', person_texts=[],
                       non_person_category='common_word', evidence_quote='')
    elif role == 'extractor':
        payload = {'names': []}
    return LlmJsonResult(payload, '', .1, True)


class JudgeFailureTests(unittest.TestCase):
    def test_transport_and_schema_failures_share_one_retry_budget(self):
        invalid = LlmJsonResult(None, '', .2, False)
        for responses in ([FAILURE, FAILURE], [FAILURE, invalid], [invalid, FAILURE], [FAILURE, reply('judge')]):
            with self.subTest(responses=responses), \
                 patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=responses) as call:
                result = call_ollama_json('http://unused', 'fake', [], {})
                self.assertEqual(call.call_count, 2)
                self.assertTrue(result.retried)
                self.assertAlmostEqual(result.latency_s, sum(answer.latency_s for answer in responses))
                self.assertEqual(result.error, responses[-1].error)

    def test_judge_continues_after_exhausted_failure(self):
        judge = NameJudge(model_digest='1' * 64)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                   side_effect=[FAILURE, FAILURE, reply('judge')]) as call:
            first, _ = judge.decide(CANDIDATE, CONTEXT, 'a' * 64)
            second, _ = judge.decide(CANDIDATE, CONTEXT, 'b' * 64)
        self.assertEqual(first.outcome, 'error')
        self.assertEqual(second.outcome, 'propose_unchanged')
        self.assertEqual(call.call_count, 3)
        self.assertEqual((judge.calls, judge.http_attempts, judge.errors), (2, 3, 1))

    def test_failed_person_text_retry_does_not_disable_later_requests(self):
        judge = NameJudge(model_digest='1' * 64)
        person = {'decision': 'anonymize_whole', 'person_scope': 'whole', 'person_texts': ['TOKENOMEGA'],
                  'non_person_category': 'none', 'evidence_quote': '', 'reading': 'person'}
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                   side_effect=[LlmJsonResult(person, '', .1, True), FAILURE, reply('judge')]) as call:
            first, _ = judge.decide(CANDIDATE, CONTEXT, 'a' * 64)
            second, _ = judge.decide(CANDIDATE, CONTEXT, 'b' * 64)
        self.assertEqual(first.outcome, 'error')
        self.assertIn('a' * 64, judge.transport_failures)
        self.assertEqual(second.outcome, 'propose_unchanged')
        self.assertEqual((call.call_count, judge.http_attempts, judge.person_text_retries), (3, 3, 1))

    def test_transient_failure_recovers_with_one_retry(self):
        judge = NameJudge(model_digest='1' * 64)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=[FAILURE, reply('judge')]) as call:
            decision, _ = judge.decide(CANDIDATE, CONTEXT, 'a' * 64)
        self.assertEqual(decision.outcome, 'propose_unchanged')
        self.assertEqual((call.call_count, judge.http_attempts, judge.errors), (2, 2, 0))

    def test_verifier_continues_after_exhausted_failure(self):
        verifier = NameVerifier(model_digest='1' * 64)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                   side_effect=[FAILURE, FAILURE, reply('verifier')]) as call:
            first = verifier.verify(candidate=CANDIDATE.text, context=CONTEXT, occurrence_id='a' * 64)
            second = verifier.verify(candidate=CANDIDATE.text, context=CONTEXT, occurrence_id='b' * 64)
        self.assertEqual(first.outcome, 'error')
        self.assertEqual(second.outcome, 'not_person')
        self.assertEqual(call.call_count, 3)

    def test_malformed_http_reply_is_not_a_transport_failure(self):
        for body in (b'not-json', b'[]'):
            with self.subTest(body=body), patch('urllib.request.urlopen') as call:
                call.return_value.__enter__.return_value.read.return_value = body
                result = call_ollama_json('http://unused', 'fake', [], {})
            self.assertEqual(call.call_count, 2)
            self.assertFalse(result.schema_ok)
            self.assertEqual(result.error, '')

    def test_each_role_saves_answer_before_file_end(self):
        for role in ('judge', 'verifier', 'extractor'):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                def create():
                    cls = {'judge': NameJudge, 'verifier': NameVerifier, 'extractor': NameExtractor}[role]
                    return cls(model_digest='1' * 64, cache_dir=root)
                def run(model):
                    if role == 'judge':
                        model.decide(CANDIDATE, CONTEXT, 'a' * 64)
                    elif role == 'verifier':
                        model.verify(candidate=CANDIDATE.text, context=CONTEXT, occurrence_id='a' * 64)
                    else:
                        model.extract('      * PLAIN TEXT\n', 'SYNTH.CBL', 'context')
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=reply(role)) as call:
                    run(create())
                    run(create())  # Fresh process state, with no file-end flush.
                self.assertEqual(call.call_count, 1)

    def test_crash_during_next_answer_keeps_previous_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            judge = NameJudge(model_digest='1' * 64, cache_dir=root)
            with patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=[reply('judge'), SystemExit('crash')]):
                judge.decide(CANDIDATE, CONTEXT, 'a' * 64)
                with self.assertRaises(SystemExit):
                    judge.decide(CANDIDATE, CONTEXT + ' EXTRA', 'b' * 64)
            reloaded = NameJudge(model_digest='1' * 64, cache_dir=root)
            with patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                decision, cached = reloaded.decide(CANDIDATE, CONTEXT, 'c' * 64)
            self.assertTrue(cached)
            self.assertEqual(decision.outcome, 'propose_unchanged')

    def test_journal_recovers_legacy_cache_and_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = PersistentResponseCache(report_dir=root, stage='judge')
            cache.path.parent.mkdir(parents=True)
            cache.path.write_text(json.dumps({'stage': 'judge', 'entries': {'old': {'answer': 1}}}))
            cache = PersistentResponseCache(report_dir=root, stage='judge')
            cache.put('first', {'answer': 2})
            journal = cache.path.with_suffix('.jsonl')
            with journal.open('ab') as handle:
                handle.write(b'{"key":"partial","value":"\xc3')
            recovered = PersistentResponseCache(report_dir=root, stage='judge')
            self.assertEqual(recovered.entries, {'old': {'answer': 1}, 'first': {'answer': 2}})
            recovered.put('second', {'answer': 3})
            recovered = PersistentResponseCache(report_dir=root, stage='judge')
            self.assertEqual(recovered.get('second'), {'answer': 3})
            recovered.flush()
            self.assertFalse(journal.exists())
            self.assertEqual(PersistentResponseCache(report_dir=root, stage='judge').entries, recovered.entries)

    def test_failed_compaction_does_not_disable_later_answer_saves(self):
        from cobol_code_anonymizer.pipeline import scan_path

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input'
            source.mkdir()
            for name in ('A', 'B'):
                (source / (name + '.CBL')).write_text(f'      * VALORE TOKENALPHA PREVISTO {name}\n')
            words = root / 'words.txt'
            words.write_text('TOKENALPHA\n')
            judge = NameJudge(model_digest='1' * 64, cache_dir=root / 'reports')
            with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=reply('judge')), \
                 patch.object(PersistentResponseCache, 'flush', side_effect=OSError('synthetic compaction failure')):
                scan_path(source, extra_watchlists=[words], use_presidio=False, name_judge=judge)
            reloaded = NameJudge(model_digest='1' * 64, cache_dir=root / 'reports')
            self.assertEqual(len(reloaded._response_cache.entries), 2)

    def test_failed_transport_blocks_only_affected_file_in_split_and_full_runs(self):
        for failed_role in ('judge', 'verifier'):
            for split in (False, True):
                with self.subTest(role=failed_role, split=split):
                    self.run_batch(failed_role, split, transport=True)

    def test_malformed_answer_hides_but_does_not_block_file(self):
        for role in ('judge', 'verifier'):
            with self.subTest(role=role):
                self.run_batch(role, True, transport=False)

    def run_batch(self, failed_role, split, transport):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input'
            source.mkdir()
            for file, word in (('A.CBL', 'TOKENALPHA'), ('B.CBL', 'TOKENBETA')):
                (source / file).write_text(f'      * VALORE {word} PREVISTO\n')
            words = root / 'words.txt'
            words.write_text('TOKENALPHA\nTOKENBETA\n')
            args = [str(source), '--watchlist', str(words), '--no-presidio', '--auto',
                    '--name-judge-model', 'fake-judge', '--name-verifier-model', 'fake-verifier',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            def answer(host, model, messages, *rest):
                role = model.removeprefix('fake-')
                if role == failed_role and 'TOKENALPHA' in messages[-1]['content']:
                    return FAILURE if transport else LlmJsonResult(None, '', .1, False)
                return reply(role)
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake-judge': '1' * 64, 'fake-verifier': '2' * 64}), \
                 patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=answer):
                if split:
                    self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                    self.assertEqual(cli.main(args + ['--step', 'decide']), int(transport))
                    result = cli.main(args + ['--step', 'write'])
                else:
                    result = cli.main(args)
            self.assertEqual(result, int(transport))
            self.assertTrue((root / 'out' / 'B.CBL').exists())
            self.assertFalse((root / 'reports' / 'needs_review' / 'A.CBL').exists())
            summary = (root / 'reports' / 'scan_summary.txt').read_text()
            if transport:
                self.assertFalse((root / 'out' / 'A.CBL').exists())
                self.assertIn('Result: released 1, withheld 0, not complete 1', summary)
            else:
                self.assertNotIn('TOKENALPHA', (root / 'out' / 'A.CBL').read_text())
                self.assertIn('Result: released 2, withheld 0, not complete 0', summary)
