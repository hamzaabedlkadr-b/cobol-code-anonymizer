"""Bound transport waits per model without releasing unchecked files."""

import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult, call_ollama_json
from cobol_code_anonymizer.scanner import Finding
from cobol_code_anonymizer.verifier import NameVerifier


FAILURE = LlmJsonResult(None, '', .1, False, error='synthetic timeout')


def reply(role):
    payload = {'decision': 'not_person', 'reading': 'ordinary word'}
    if role == 'judge':
        payload.update(decision='propose_unchanged', person_scope='none', person_texts=[],
                       non_person_category='common_word', evidence_quote='')
    return LlmJsonResult(payload, '', .1, True)


class TransportCutoffTests(unittest.TestCase):
    def test_three_exhausted_requests_stop_further_network_attempts(self):
        state = {}
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=FAILURE) as call:
            for _ in range(3):
                result = call_ollama_json('http://unused', 'fake', [], {}, transport_state=state)
                self.assertTrue(result.retried)
            stopped = call_ollama_json('http://unused', 'fake', [], {}, transport_state=state)
        self.assertEqual(call.call_count, 6)
        self.assertFalse(stopped.attempted)
        self.assertEqual(stopped.latency_s, 0)
        self.assertIn('3 consecutive transport failures', stopped.error)

    def test_response_resets_streak_even_when_answer_is_malformed(self):
        for response in (reply('judge'), LlmJsonResult(None, '', .1, False)):
            with self.subTest(valid=response.schema_ok):
                state = {}
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=FAILURE):
                    for _ in range(2):
                        call_ollama_json('http://unused', 'fake', [], {}, transport_state=state)
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=response):
                    call_ollama_json('http://unused', 'fake', [], {}, transport_state=state)
                self.assertEqual(state['failures'], 0)
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=FAILURE) as call:
                    for _ in range(2):
                        result = call_ollama_json('http://unused', 'fake', [], {}, transport_state=state)
                self.assertEqual(call.call_count, 4)
                self.assertTrue(result.attempted)

    def test_roles_share_model_cutoff_and_do_not_count_skipped_requests(self):
        state = {}
        judge = NameJudge(model='fake', model_digest='1' * 64, transport_state=state)
        verifier = NameVerifier(model='fake', model_digest='1' * 64, transport_state=state)
        context = '* VALORE [[TOKENALPHA]] PREVISTO'
        finding = Finding('SYNTH.CBL', 'NAME', 'TOKENALPHA', 9, 19, 1, 10, 1, context)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=FAILURE) as call:
            judge.decide(finding, context, 'a' * 64)
            verifier.verify(candidate=finding.text, context=context, occurrence_id='b' * 64)
            judge.decide(finding, context, 'c' * 64)
            verifier.verify(candidate=finding.text, context=context, occurrence_id='d' * 64)
        self.assertEqual(call.call_count, 6)
        self.assertEqual((judge.calls, judge.http_attempts), (2, 4))
        self.assertEqual((verifier.calls, verifier.http_attempts), (1, 2))
        other = NameJudge(model='other', model_digest='2' * 64)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=reply('judge')) as call:
            self.assertEqual(other.decide(finding, context, 'e' * 64)[0].outcome, 'propose_unchanged')
        self.assertEqual(call.call_count, 1)

    def test_cached_answer_can_still_be_used_after_cutoff(self):
        state = {}
        judge = NameJudge(model='fake', model_digest='1' * 64, transport_state=state)
        context = '* VALORE [[TOKENALPHA]] PREVISTO'
        finding = Finding('SYNTH.CBL', 'NAME', 'TOKENALPHA', 9, 19, 1, 10, 1, context)
        with patch('cobol_code_anonymizer.llm._call_ollama_json_once', return_value=reply('judge')):
            judge.decide(finding, context, 'a' * 64)
        state['failures'] = 3
        with patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
            decision, cached = judge.decide(finding, context, 'b' * 64)
        self.assertTrue(cached)
        self.assertEqual(decision.outcome, 'propose_unchanged')

    def test_cli_shares_cutoff_for_different_tags_with_same_digest(self):
        self.run_batch('verifier', True, shared_model=True)

    def test_cutoff_blocks_remaining_affected_files_in_full_and_split_runs(self):
        for role in ('judge', 'verifier'):
            for split in (False, True):
                with self.subTest(role=role, split=split):
                    self.run_batch(role, split)

    def run_batch(self, failed_role, split, shared_model=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input'
            source.mkdir()
            for name in ('A', 'B', 'C', 'D', 'E'):
                (source / (name + '.CBL')).write_text(f'      * VALORE TOKENALPHA PREVISTO {name}\n')
            (source / 'CLEAR.CBL').write_text('      * PLAIN TEXT\n')
            watchlist = root / 'words.txt'
            watchlist.write_text('TOKENALPHA\n')
            args = [str(source), '--watchlist', str(watchlist), '--no-presidio', '--auto',
                    '--name-judge-model', 'fake-judge', '--name-verifier-model', 'fake-verifier',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            calls = Counter()
            def answer(host, model, *rest):
                role = model.removeprefix('fake-')
                calls[role] += 1
                if shared_model:
                    return reply(role) if sum(calls.values()) == 1 else FAILURE
                return FAILURE if role == failed_role else reply(role)
            with redirect_stdout(io.StringIO()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake-judge': '1' * 64, 'fake-verifier': ('1' if shared_model else '2') * 64}), \
                 patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=answer):
                if split:
                    self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                    self.assertEqual(cli.main(args + ['--step', 'decide']), 1)
                    self.assertEqual(cli.main(args + ['--step', 'write']), 1)
                else:
                    self.assertEqual(cli.main(args), 1)
                if shared_model:
                    self.assertEqual(sum(calls.values()), 7)  # One reply, then three failed requests across roles.
                else:
                    self.assertEqual(calls[failed_role], 6)
                self.assertEqual([path.name for path in (root / 'out').iterdir()], ['CLEAR.CBL'])
                summary = (root / 'reports' / 'scan_summary.txt').read_text()
                self.assertIn('Result: released 1, withheld 0, not complete 5', summary)
                self.assertFalse((root / 'reports' / 'needs_review').exists())
                # A new invocation can recover, with finished answers reused from cache.
                with patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                           side_effect=lambda host, model, *rest: reply(model.removeprefix('fake-'))):
                    if split:
                        self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                        self.assertEqual(cli.main(args + ['--step', 'write', '--overwrite']), 0)
                    else:
                        self.assertEqual(cli.main(args + ['--overwrite']), 0)
                self.assertEqual(len(list((root / 'out').iterdir())), 6)
