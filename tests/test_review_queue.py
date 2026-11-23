import csv
import tempfile
import unittest
from pathlib import Path
from cobol_code_anonymizer.review_queue import build_llm_name_review_rows, write_llm_name_review_csv
from cobol_code_anonymizer.review_decisions import import_review_answers, load_review_decisions

class ReviewReportTests(unittest.TestCase):
    def test_accepted_full_name_stays_whole_and_answer_import_still_works(self):
        from cobol_code_anonymizer.pipeline import scan_path
        name = 'DE TOKENALPHA TOKENBETA'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch = root / 'SYNTH.CBL', root / 'watch.txt'
            source.write_text(f'      * REFERENTE {name} APPROVA.\n')
            watch.write_text(name + '\n')
            decisions = []
            hits = scan_path(source, entities={'NAME'}, extra_watchlists=[watch],
                             include_default_names=False, use_presidio=False,
                             identifier_review_decisions=decisions)
            self.assertEqual([hit.text for hit in hits], [name])
            rows = build_llm_name_review_rows(decisions)
            self.assertEqual([row['name'] for row in rows], [name])
            self.assertIn(f'[[{name}]]', rows[0]['context'])
            rows[0]['answer'] = 'show'
            write_llm_name_review_csv(root / 'review.csv', rows)
            import_review_answers(root / 'review.csv', root / 'answers.csv')
            answers = load_review_decisions(root / 'answers.csv')
            self.assertEqual(scan_path(source, entities={'NAME'}, extra_watchlists=[watch],
                                       include_default_names=False, use_presidio=False,
                                       review_answers=answers), [])

    def test_full_name_group_combines_reasons_and_keeps_other_occurrences_separate(self):
        from cobol_code_anonymizer.cli import write_llm_name_review_text
        name = 'DE TOKENALPHA TOKENBETA'
        decisions = [{'file': 'SYNTH.CBL', 'line': line, 'column': 9, 'text': word,
                      'context': f'* [[{word}]]', 'review_key': f'line-{line}',
                      'policy_outcome': outcome, 'policy_reading': reason}
                     for line, word, outcome, reason in (
                         (20, name, 'anonymize_whole', 'multi-word watchlist entry; hide'),
                         (10, name, 'anonymize_whole', 'judge says person; hide'),
                         (30, name, 'anonymize_whole', 'judge says person; hide'),
                         (40, 'DE', 'anonymize_whole', 'judge says person; hide'),
                         (41, 'DE', 'anonymize_whole', 'judge says person; hide'),
                         (1, name, 'leave_unchanged', 'judge and verifier say not person; show'),
                     )]
        rows = build_llm_name_review_rows(decisions)
        self.assertEqual(len(rows), 6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'llm_finding.txt'
            write_llm_name_review_text(path, 'union-judge', rows)
            report = path.read_text()
        hidden = report.split('[ACCEPTED AS NAME - hidden]')[1].split('[REJECTED - left as is]')[0]
        self.assertEqual(hidden.count(name + '  hits='), 1)
        self.assertIn(name + '  hits=3  locations=SYNTH.CBL:10, SYNTH.CBL:20, SYNTH.CBL:30', hidden)
        self.assertIn('reason=judge: person; on the watchlist', hidden)
        self.assertIn('DE  hits=2  locations=SYNTH.CBL:40, SYNTH.CBL:41', hidden)
        self.assertEqual(report.count(name + '  hits='), 2)

    def test_findings_follow_source_order_with_required_review_first(self):
        from cobol_code_anonymizer.cli import write_llm_name_review_text
        decisions = [
            {'file': file, 'line': line, 'column': column, 'text': word,
             'context': f'* [[{word}]]', 'review_key': f'{file}:{line}:{column}',
             'policy_outcome': outcome, 'policy_reading': 'test reason'}
            for file, line, column, word, outcome in (
                ('A.CBL', 100, 9, 'TOKENALPHA', 'anonymize_whole'),
                ('B.CBL', 1, 9, 'TOKENBETA', 'anonymize_whole'),
                ('A.CBL', 10, 9, 'TOKENZETA', 'anonymize_whole'),
                ('A.CBL', 2, 9, 'TOKENZETA', 'anonymize_whole'),
                ('A.CBL', 2, 20, 'TOKENCHARLIE', 'anonymize_whole'),
                ('A.CBL', 999, '', 'TOKENDELTA', 'review_required'),
                ('A.CBL', 1, 9, 'TOKENSHOWN', 'leave_unchanged'),
            )]
        rows = build_llm_name_review_rows(decisions)
        self.assertEqual([row['name'] for row in rows],
                         ['TOKENDELTA', 'TOKENSHOWN', 'TOKENZETA', 'TOKENCHARLIE',
                          'TOKENZETA', 'TOKENALPHA', 'TOKENBETA'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'llm_finding.txt'
            write_llm_name_review_text(path, 'union-judge', list(reversed(rows)))
            report = path.read_text()
        self.assertLess(report.index('[REVIEW REQUIRED]'), report.index('[ACCEPTED AS NAME - hidden]'))
        self.assertLess(report.index('[ACCEPTED AS NAME - hidden]'), report.index('[REJECTED - left as is]'))
        hidden = report.split('[ACCEPTED AS NAME - hidden]')[1].split('[REJECTED - left as is]')[0]
        self.assertLess(hidden.index('TOKENZETA'), hidden.index('TOKENCHARLIE'))
        self.assertLess(hidden.index('TOKENCHARLIE'), hidden.index('TOKENALPHA'))
        self.assertLess(hidden.index('TOKENALPHA'), hidden.index('TOKENBETA'))
        self.assertEqual(hidden.count('TOKENZETA  hits='), 1)
        self.assertIn('TOKENZETA  hits=2  locations=A.CBL:2, A.CBL:10', hidden)

    def test_hidden_words_and_pairs_first_with_one_context_per_group(self):
        from cobol_code_anonymizer.cli import write_llm_name_review_text
        rows = [dict(name=name, file='SYNTH.CBL', line=line, column=9,
                     final_action=action, reason='test reason', context=context)
                for line, name, action, context in (
                    (1, 'TOKENALPHA TOKENBETA TOKENZETA', 'hide', '* [[TOKENALPHA TOKENBETA TOKENZETA]]'),
                    (2, 'RIGA LETTA', 'hide', '* [[RIGA LETTA]] DA ARCHIVIO'),
                    (3, 'VALORI', 'hide', '* [[VALORI]] PREVISTI'),
                    (4, 'VALORI', 'hide', '* ALTRI [[VALORI]]'),
                    (5, 'TOKENREVIEW', 'review (required)', '* [[TOKENREVIEW]]'),
                    (6, 'TOKENSHOWN', 'show', '* [[TOKENSHOWN]]'),
                )]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'llm_finding.txt'
            write_llm_name_review_text(path, 'union-judge', list(reversed(rows)))
            report = path.read_text()
        hidden = report.split('[ACCEPTED AS NAME - hidden]')[1].split('[REJECTED - left as is]')[0]
        self.assertLess(report.index('[REVIEW REQUIRED]'), report.index('[ACCEPTED AS NAME - hidden]'))
        self.assertLess(hidden.index('RIGA LETTA  hits='), hidden.index('VALORI  hits='))
        self.assertLess(hidden.index('VALORI  hits='), hidden.index('TOKENALPHA TOKENBETA TOKENZETA  hits='))
        self.assertIn('VALORI  hits=2  locations=SYNTH.CBL:3, SYNTH.CBL:4  reason=test reason', hidden)
        self.assertIn('* [[VALORI]] PREVISTI', hidden)
        self.assertNotIn('* ALTRI [[VALORI]]', hidden)

    def test_required_first_distinct_word_keys_and_answer_round_trip(self):
        decisions = [{"file": "SYNTH.CBL", "line": 1, "text": "TOKENALPHA TOKENBETA", "context": "* [[TOKENALPHA TOKENBETA]]", "review_key": "source-key", "policy_outcome": outcome, "policy_reading": "test reason"} for outcome in ("anonymize_whole", "review_required")]
        rows = build_llm_name_review_rows(decisions)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["final_action"] == "review (required)" for row in rows[:2]))
        self.assertEqual(rows[2]["name"], "TOKENALPHA TOKENBETA")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows[0]["answer"] = "show"
            write_llm_name_review_csv(root / "llm_name_review.csv", rows)
            with (root / "llm_name_review.csv").open() as handle:
                self.assertEqual(next(csv.reader(handle))[-1], "answer")
            import_review_answers(root / "llm_name_review.csv", root / "review_decisions.csv")
            self.assertEqual(load_review_decisions(root / "review_decisions.csv").answer_for(rows[0]["name"], "unused", "source-key"), "not_person")

    def test_explicit_answer_can_change_a_shown_decision(self):
        row = {"file": "SYNTH.CBL", "line": 1, "text": "TOKENALPHA", "context": "* [[TOKENALPHA]]", "review_key": "source-key", "policy_outcome": "leave_unchanged", "policy_reading": "judge and verifier say not person; show"}
        rows = build_llm_name_review_rows([row])
        rows[0]["answer"] = "hide"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_llm_name_review_csv(root / "llm_name_review.csv", rows)
            import_review_answers(root / "llm_name_review.csv", root / "review_decisions.csv")
            self.assertEqual(load_review_decisions(root / "review_decisions.csv").answer_for("TOKENALPHA", "unused", "source-key"), "person")
