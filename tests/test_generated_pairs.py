"""Generated pairs need models; complete company entries remain direct hides."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.preflight import run_preflight


class GeneratedPairTests(unittest.TestCase):
    def test_generated_pairs_use_both_models_but_full_entries_and_baseline_do_not(self):
        for full_entry, models in ((False, True), (True, True), (False, False)):
            with self.subTest(full_entry=full_entry, models=models), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'SYNTH.CBL'
                source.write_text('      * VALORI REALI\n')
                words = root / 'words.txt'
                words.write_text('VALORI\nREALI\n' + ('VALORI REALI\n' if full_entry else ''))
                args = [str(source), '--watchlist', str(words), '--no-presidio', '--auto',
                        '--out-dir', str(root / 'out'), '--report-dir', str(root / 'reports')]
                if models:
                    args += ['--name-judge-model', 'fake-judge', '--name-verifier-model', 'fake-verifier']
                def answer(host, model, *args):
                    payload = {'decision': 'not_person', 'reading': 'ordinary phrase'}
                    if model == 'fake-judge':
                        payload.update(decision='propose_unchanged', person_scope='none', person_texts=[],
                                       non_person_category='common_word', evidence_quote='')
                    return LlmJsonResult(payload, '', .01, True)
                with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                     patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake-judge': '1' * 64, 'fake-verifier': '2' * 64}), \
                     patch('cobol_code_anonymizer.judge.NameJudge.canary_ok', return_value=(True, '')), \
                     patch('cobol_code_anonymizer.llm._call_ollama_json_once', side_effect=answer) as call:
                    self.assertEqual(cli.main(args), 0)
                text = (root / 'out' / source.name).read_text()
                shown = models and not full_entry
                self.assertEqual(text == source.read_text(), shown)
                self.assertEqual(call.call_count, 2 if shown else 0)
                rows = json.loads((root / 'reports' / 'decisions.json').read_text())['rows']
                self.assertTrue(rows)
                if shown:
                    self.assertTrue(all(row['policy_outcome'] == 'leave_unchanged' for row in rows))
                if not full_entry:
                    with redirect_stdout(io.StringIO()):
                        counts = run_preflight(source, watchlist_paths=[words],
                                               include_default_names=False, name_scope='context',
                                               use_presidio=False, presidio_model='unused')
                    self.assertEqual(counts.judge_calls, 1)
