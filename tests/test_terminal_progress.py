import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import cli
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.pipeline import scan_path


class TerminalProgressTests(unittest.TestCase):
    def test_scan_reports_selected_spans_and_file_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'SYNTH.CBL').write_text('      * a@example.invalid\n')
            messages = []
            scan_path(root, entities={'EMAIL'}, use_presidio=False, progress=messages.append)
        self.assertTrue(any(message.startswith('Scanned SYNTH.CBL: selected spans 1, shown 0, automatic text fallbacks 0;')
                            and 'model calls 0, cache hits 0;' in message and message.endswith(' s') for message in messages))

    def test_live_line_includes_stage_counts(self):
        stream = io.StringIO()
        with redirect_stdout(stream), patch.object(stream, 'isatty', return_value=True):
            cli.print_progress('Analyzing file 1/2: SYNTH.CBL')
            cli.print_progress('[LLM extraction] SYNTH.CBL: extraction chunk 2/10 (lines 1-25)')
            cli.print_progress('[LLM judge] candidate 3/8: judging SYNTH.CBL:1')
            cli.print_progress('[LLM verifier] reviewing proposed non-person candidate')
            cli.end_progress()
        self.assertIn('finding names 2/10 (20%)', stream.getvalue())
        self.assertIn('checking names 3/8 (37%)', stream.getvalue())
        self.assertIn('verifying a name', stream.getvalue())

    def test_cli_connects_progress_and_labels_approved_words(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'SYNTH.CBL'
            source.write_text('      * PLAIN TEXT\n')
            primary = LlmJsonResult({'names': []}, '', .1, True)
            with redirect_stdout(io.StringIO()) as stream, \
                 patch('urllib.request.urlopen', side_effect=AssertionError('no network')), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', return_value=primary):
                self.assertEqual(cli.main([str(source), '--out-dir', str(root / 'out'), '--no-presidio',
                                           '--name-extract-model', 'fake', '--auto']), 0)
            text = stream.getvalue()
            self.assertIn('Approved CODE/path words: 0 (none supplied)', text)
            self.assertIn('finding names 1/1 (100%)', text)
            self.assertIn('Writing anonymized copies', text)
            self.assertIn('Final check: 1 files', text)
            self.assertIn('model calls 1, cache hits 0;', text)
