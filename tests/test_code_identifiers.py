"""Exact CODE approvals never approve a word in other identifiers or TEXT."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.verifier import NameVerifier
from cobol_code_anonymizer.review_decisions import load_identifier_answers
from cobol_code_anonymizer.text_matching import code_identifier_span
from cobol_code_anonymizer.cobol_layout import classify_source
from cobol_code_anonymizer.source_reader import read_source


def ordinary(role):
    payload = {'decision': 'not_person', 'reading': 'ordinary identifier in all examples'}
    if role == 'judge':
        payload.update(decision='propose_unchanged', person_scope='none', person_texts=[],
                       non_person_category='code_or_identifier', evidence_quote='')
    return LlmJsonResult(payload, '', .01, True)


class CodeIdentifierTests(unittest.TestCase):
    def test_exact_code_approval_reused_but_text_and_other_identifier_stay_hidden(self):
        for split in (False, True):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                batch = root / 'input'
                batch.mkdir()
                for name in ('A', 'B', 'C', 'D'):
                    (batch / (name + '.CBL')).write_text(
                        "       PROCEDURE DIVISION.\n"
                        "           MOVE WS-RIGA4-TESTA TO RESULT\n"
                        "           MOVE WS-RIGA5-TESTA TO RESULT\n"
                        "           DISPLAY 'RIGA'\n"
                        "      * RIGA\n")
                words = root / 'words.txt'
                words.write_text('RIGA\nTESTA\n')
                reports = root / 'reports'
                args = [str(batch), '--watchlist', str(words), '--no-presidio', '--auto',
                        '--name-judge-model', 'fake-judge', '--name-verifier-model', 'fake-verifier',
                        '--out-dir', str(root / 'out'), '--report-dir', str(reports)]
                contexts = []
                def answer(host, model, messages, *args):
                    role = model.removeprefix('fake-')
                    contexts.append((role, messages[-1]['content']))
                    if role == 'verifier' and 'WS-RIGA5-TESTA' in messages[-1]['content']:
                        return LlmJsonResult({'decision': 'unsure', 'reading': 'ambiguous identifier'}, '', .01, True)
                    return ordinary(role)
                with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                     patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake-judge': '1' * 64, 'fake-verifier': '2' * 64}), \
                     patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                     patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=answer) as call:
                    if split:
                        self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                        data = json.loads((reports / 'candidates.json').read_text())
                        self.assertEqual(set(data['code_identifiers']), {'WS-RIGA4-TESTA', 'WS-RIGA5-TESTA'})
                        self.assertTrue(all(len(entries) == 3 for entries in data['code_identifiers'].values()))
                        self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                        self.assertEqual(cli.main(args + ['--step', 'write']), 0)
                    else:
                        self.assertEqual(cli.main(args), 0)
                    self.assertEqual(call.call_count, 4)
                    self.assertTrue(all('.CBL' not in context for _, context in contexts))
                    self.assertTrue(all(context.count('[[') == 1 for _, context in contexts))
                    for path in (root / 'out').glob('*.CBL'):
                        text = path.read_text()
                        self.assertIn('MOVE WS-RIGA4-TESTA TO RESULT', text)
                        self.assertNotIn('WS-RIGA5-TESTA', text)
                        self.assertNotIn("DISPLAY 'RIGA'", text)
                        self.assertNotIn('* RIGA\n', text)
                    with (reports / 'code_identifiers.csv').open(newline='') as handle:
                        reader = csv.DictReader(handle)
                        fields, rows = reader.fieldnames, list(reader)
                    rows[0]['answer'] = 'hide'
                    with (reports / 'code_identifiers.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        cli.main(args + ['--step', 'write', '--overwrite'])
                    self.assertEqual(cli.main(args + ['--step', 'decide']), 0)
                    self.assertEqual(cli.main(args + ['--step', 'write', '--overwrite']), 0)
                    self.assertEqual(call.call_count, 4)  # Other identifier answers came from cache.
                    self.assertNotIn('WS-RIGA4-TESTA', (root / 'out' / 'A.CBL').read_text())
                    self.assertEqual(load_identifier_answers(reports / 'code_identifiers.csv'), {'WS-RIGA4-TESTA': 'hide'})

    def test_code_transport_failure_blocks_every_affected_file_and_no_other_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            batch = root / 'input'
            batch.mkdir()
            for name in ('A', 'B'):
                (batch / (name + '.CBL')).write_text('       MOVE WS-RIGA4 TO RESULT\n')
            (batch / 'C.CBL').write_text('      * NOTHING PERSONAL\n')
            words = root / 'words.txt'
            words.write_text('RIGA\n')
            args = [str(batch), '--watchlist', str(words), '--no-presidio', '--auto',
                    '--name-judge-model', 'fake', '--name-verifier-model', 'fake',
                    '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
            with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                       return_value=LlmJsonResult(None, '', .01, False, error='synthetic outage')) as call:
                self.assertEqual(cli.main(args), 1)
                self.assertEqual(call.call_count, 2)
            self.assertEqual([path.name for path in (root / 'out').iterdir()], ['C.CBL'])
            self.assertIn('not complete 2', (root / 'reports' / 'scan_summary.txt').read_text())

    def test_metadata_comments_and_literals_cannot_be_approved_as_code_identifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'SYNTH.CBL'
            text = "23E31I* RIGA\n       MOVE WS-RIGA4 TO RESULT\n           DISPLAY 'RIGA'\n"
            path.write_text(text)
            layout = classify_source(read_source(path), source_path=path)
            for word in ('23E31I', "* RIGA", "'RIGA'"):
                start = text.index(word)
                self.assertIsNone(code_identifier_span(layout, start, start + len(word)))
            start = text.index('RIGA4')
            span = code_identifier_span(layout, start, start + 4)
            self.assertEqual(text[slice(*span)], 'WS-RIGA4')

    def test_code_exception_blocks_affected_files_and_continues_other_identifiers(self):
        for stage in ('judge', 'verifier'):
            for split in (False, True):
                with self.subTest(stage=stage, split=split), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    batch, reports = root / 'input', root / 'reports'
                    batch.mkdir()
                    for name in ('A', 'B', 'C', 'D'):
                        (batch / f'{name}.CBL').write_text('       MOVE WS-TOKENALPHA1 TO RESULT\n')
                    (batch / 'E.CBL').write_text('       MOVE WS-TOKENALPHA2 TO RESULT\n')
                    words = root / 'words.txt'
                    words.write_text('TOKENALPHA\n')
                    args = [str(batch), '--watchlist', str(words), '--no-presidio', '--auto',
                            '--name-judge-model', 'fake-judge', '--name-verifier-model', 'fake-verifier',
                            '--out-dir', str(root / 'out'), '--report-dir', str(reports)]
                    role, method = (NameJudge, 'decide') if stage == 'judge' else (NameVerifier, 'verify')
                    original = getattr(role, method)
                    def fail_one(model, *args, **kwargs):
                        candidate = args[0].text if stage == 'judge' else kwargs['candidate']
                        if candidate == 'WS-TOKENALPHA1':
                            raise ValueError('synthetic model exception')
                        return original(model, *args, **kwargs)
                    with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                         patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake-judge': '1' * 64, 'fake-verifier': '2' * 64}), \
                         patch.object(NameJudge, 'canary_ok', return_value=(True, '')), \
                         patch.object(role, method, autospec=True, side_effect=fail_one), \
                         patch('cobol_code_anonymizer.llm._call_ollama_json_once',
                               side_effect=lambda host, model, *args: ordinary(model.removeprefix('fake-'))) as call:
                        if split:
                            self.assertEqual(cli.main(args + ['--step', 'extract']), 0)
                            self.assertEqual(cli.main(args + ['--step', 'decide']), 1)
                            data = json.loads((reports / 'decisions.json').read_text())
                            self.assertEqual({row['file'] for row in data['not_complete_files']},
                                             {'A.CBL', 'B.CBL', 'C.CBL', 'D.CBL'})
                            requests = call.call_count
                            self.assertEqual(cli.main(args + ['--step', 'write']), 1)
                            self.assertEqual(call.call_count, requests)
                        else:
                            self.assertEqual(cli.main(args), 1)
                    self.assertEqual([path.name for path in (root / 'out').iterdir()], ['E.CBL'])
                    self.assertIn('WS-TOKENALPHA2', (root / 'out' / 'E.CBL').read_text())
                    self.assertIn('not complete 4', (reports / 'scan_summary.txt').read_text())
                    with (reports / 'code_identifiers.csv').open() as handle:
                        decisions = {row['identifier']: row['decision'] for row in csv.DictReader(handle)}
                    self.assertEqual(decisions, {'WS-TOKENALPHA1': 'not_complete', 'WS-TOKENALPHA2': 'leave_unchanged'})
