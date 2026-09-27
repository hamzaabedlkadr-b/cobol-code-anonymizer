import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.cli import choose_replacements
from cobol_code_anonymizer.replacements import group_findings, suggested_replacement
from cobol_code_anonymizer.scanner import scan_path


class AnonymizationTests(unittest.TestCase):
    def scan(self, option):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'sample.cbl'
            source.write_text("      * Mario Rossi 00058 00273 123456 5123456\n")
            roster = root / 'workers.txt'
            roster.write_text('Mario Rossi\n00058\n00273\n123456\n5123456\n')
            return scan_path(source, entities={'NAME', 'MATRICOLA'},
                             include_default_names=False, use_presidio=False,
                             **{option: [roster]})

    def test_numeric_entries_are_ids_with_preserved_width(self):
        for option in ('extra_watchlists', 'employee_rosters'):
            with self.subTest(option=option):
                findings = self.scan(option)
                ids = [f for f in findings if f.entity_type == 'MATRICOLA']
                self.assertEqual({f.text for f in ids}, {'00058', '00273', '123456', '5123456'})
                self.assertTrue(all(not f.text.isdecimal() for f in findings if f.entity_type == 'NAME'))
                for i, group in enumerate(group_findings(ids), 1):
                    replacement = suggested_replacement(group, i, 'test')
                    self.assertTrue(replacement.isdecimal())
                    self.assertEqual(len(replacement), len(group.original))

    def test_accept_all_prompts_once_and_matches_auto(self):
        groups = group_findings(self.scan('extra_watchlists'))
        with patch('builtins.input', return_value='all') as prompt:
            actual = choose_replacements(groups, {}, 'test', False)
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(actual, choose_replacements(groups, {}, 'test', True))

    def test_skip_all_preserves_previous_choices(self):
        groups = group_findings(self.scan('extra_watchlists'))
        with patch('builtins.input', side_effect=['custom', 'skip-all']) as prompt:
            actual = choose_replacements(groups, {}, 'test', False)
        self.assertEqual(prompt.call_count, 2)
        self.assertEqual(actual, {groups[0].key: 'custom'})


if __name__ == '__main__':
    unittest.main()
