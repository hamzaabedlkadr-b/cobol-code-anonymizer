import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer import review


class ReviewTests(unittest.TestCase):
    def test_names_group_only_same_line_with_explicit_connectors(self):
        pattern = review.name_pattern(['Mario', 'Rossi', 'Sala'])
        for value in ['MARIO ROSSI', 'Mario D. Rossi', "Mario D’Rossi", 'Mario De Rossi']:
            spans = review.name_spans(value, pattern)
            self.assertEqual(len(spans), 1)
            self.assertEqual(value[spans[0][0]:spans[0][1]], value)
            self.assertEqual(len(spans[0][2]), 2)
        self.assertEqual(len(review.name_spans('Mario, Rossi', pattern)), 2)
        self.assertEqual(len(review.name_spans('Mario\nRossi', pattern)), 2)
        self.assertEqual(review.name_spans('mario.rossi@example.com', pattern), [])
        self.assertEqual(len(review.name_spans('WS-ROSSI-COD', pattern)), 1)
        self.assertEqual(review.name_spans('ROSSINI', pattern), [])

    def test_structured_values_preserve_positions_and_leading_zeros(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'demo.cbl'
            text = 'prima\nMario D. Rossi 00058 IT60X0542811101000000123456 mario@example.com\ndopo\n'
            source.write_text(text)
            rows = review.extract_file(source, root, review.name_pattern(['Mario', 'Rossi']), {'00058'})
            self.assertEqual({row['entity_type'] for row in rows}, {'NAME', 'MATRICOLA', 'IBAN', 'EMAIL'})
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row['line'], 2)
                self.assertEqual(row['previous_line'], 'prima')
                self.assertEqual(row['next_line'], 'dopo')
                self.assertEqual(row['current_line'][row['column']-1:row['end_column']], row['value'])
            matricola = next(row for row in rows if row['entity_type'] == 'MATRICOLA')
            self.assertEqual(matricola['value'], '00058')
            self.assertTrue(matricola['validation']['in_numeric_watchlist'])
            self.assertTrue(next(row for row in rows if row['entity_type'] == 'IBAN')['validation']['mod97_valid'])
            self.assertFalse(review.validate_iban('IT61X0542811101000000123456')['mod97_valid'])

    def test_invalid_model_answers_are_uncertain_and_payload_has_only_three_lines(self):
        candidate = dict(candidate_id='id', entity_type='NAME', value='Sala', column=1, end_column=4,
                         validation={}, matched_watchlist=['Sala'], inferred_connectors=[],
                         previous_line='before', current_line='Sala server', next_line='after', file='secret.cbl')
        payloads = []
        def invalid(host, endpoint, payload, timeout):
            payloads.append(payload)
            return {'message': {'content': json.dumps(dict(candidate_id='wrong', candidate='Sala', decision='REJECTED', reason='x'))}}
        with patch.object(review, 'request_json', side_effect=invalid):
            result = review.classify(candidate, review.DEFAULT_HOST, 'test', 1)
        self.assertEqual(result['decision'], 'UNCERTAIN')
        self.assertTrue(result['error'])
        self.assertEqual(len(payloads), 2)
        evidence = json.loads(payloads[0]['messages'][1]['content'])
        self.assertNotIn('file', evidence)
        self.assertEqual(evidence['current_line'], '[[Sala]] server')
        self.assertEqual((evidence['previous_line'], evidence['next_line']), ('before', 'after'))

    def test_full_command_cache_and_changed_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src = root / 'source'
            src.mkdir()
            code = src / 'test.cbl'
            code.write_text('* Sala server\n* Mario D. Rossi\nMOVE 00058 TO COD-DIP\n')
            original = code.read_bytes()
            watchlist = root / 'names.txt'
            watchlist.write_text('Sala\nMario\nRossi\n00058\n')
            out = root / 'reports'
            args = [str(src), '--watchlist', str(watchlist), '--report-dir', str(out)]
            calls = []
            def local_model(host, endpoint, payload=None, timeout=120):
                if endpoint == '/api/tags':
                    return {'models': [{'name': review.DEFAULT_MODEL, 'digest': 'test-digest'}]}
                evidence = json.loads(payload['messages'][1]['content'])
                calls.append(evidence)
                answer = {key: evidence[key] for key in ['candidate_id', 'candidate']}
                answer.update(decision='REJECTED' if evidence['candidate'] == 'Sala' else 'CONFIRMED', reason='Contesto verificato')
                return {'message': {'content': json.dumps(answer)}}
            with patch.object(review, 'request_json', side_effect=local_model), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review.main(args), 0)
                self.assertEqual(len(calls), 3)
                self.assertEqual(code.read_bytes(), original)
                self.assertEqual(review.main(args), 0)
                self.assertEqual(len(calls), 3)
                code.write_text(code.read_text() + '* changed\n')
                self.assertEqual(review.main(args), 0)
                self.assertEqual(len(calls), 6)
            self.assertEqual(json.loads((out/'run.json').read_text())['decisions'], {'REJECTED': 1, 'CONFIRMED': 2})
            self.assertIn('Mario D. Rossi', (out/'report.txt').read_text())
            self.assertTrue((out/'candidates.csv').exists())
            self.assertEqual(original, b'* Sala server\n* Mario D. Rossi\nMOVE 00058 TO COD-DIP\n')

    def test_startup_failure_and_bad_source_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src = root/'source'
            src.mkdir()
            (src/'bad.cbl').write_bytes(b'\x00binary')
            code = src/'good.cbl'
            code.write_text('* Mario\n')
            original = code.read_bytes()
            watchlist = root/'names.txt'
            watchlist.write_text('Mario\n')
            out = root/'reports'
            with patch.object(review, 'request_json', side_effect=OSError('server unavailable')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review.main([str(src), '--watchlist', str(watchlist), '--report-dir', str(out)]), 1)
            run = json.loads((out/'run.json').read_text())
            self.assertFalse(run['complete'])
            self.assertEqual(len(run['errors']), 2)
            self.assertEqual(json.loads((out/'review.json').read_text())[0]['decision'], 'UNCERTAIN')
            self.assertEqual(code.read_bytes(), original)

    def test_reports_cannot_be_inside_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            watchlist = root/'names.txt'
            watchlist.write_text('Mario\n')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                review.main([str(root), '--watchlist', str(watchlist), '--report-dir', str(root/'reports')])


if __name__ == '__main__':
    unittest.main()
