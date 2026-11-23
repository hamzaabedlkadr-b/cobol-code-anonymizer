import csv
import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.replacements import group_findings, load_mapping, suggested_replacement, write_mapping_template
from cobol_code_anonymizer.scanner import Finding


class MappingReportTests(unittest.TestCase):
    def test_text_map_has_only_unique_names_and_applied_replacements(self):
        findings = [Finding('SYNTH.CBL', entity, word, start, start + len(word), line, 8, 1, '')
                    for entity, word, start, line in [('NAME', 'TOKENZETA TOKENBETA', 8, 1),
                                                     ('PHONE', '123456789', 40, 2),
                                                     ('NAME', 'TOKENALPHA', 70, 3),
                                                     ('NAME', 'TOKENALPHA', 100, 4)]]
        groups = group_findings(findings)
        replacements = {group.key: suggested_replacement(group, index, 'test')
                        for index, group in enumerate(groups, 1)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'replacement_map.csv'
            write_mapping_template(path, groups, replacements, 'test')
            text_path = path.with_suffix('.txt')
            self.assertEqual(text_path.read_text(), 'TOKENZETA TOKENBETA -> Nome002 Cognome002\nTOKENALPHA -> Nome001\n')
            write_mapping_template(path, groups, {}, 'test')
            self.assertEqual(text_path.read_text(), '')
            custom_path = Path(folder) / 'custom_map.txt'
            write_mapping_template(custom_path, groups, replacements, 'test')
            with custom_path.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_map_shows_original_and_replacement_first_in_source_order(self):
        findings = [Finding('SYNTH.CBL', 'NAME', word, start, start + len(word), line, 8, 1, '')
                    for word, start, line in [('TOKENZETA', 8, 1), ('TOKENALPHA', 30, 2)]]
        groups = group_findings(findings)
        replacements = {group.key: suggested_replacement(group, index, 'test')
                        for index, group in enumerate(groups, 1)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.csv'
            write_mapping_template(path, groups, replacements, 'test')
            with path.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, ['original', 'replacement', 'hits', 'locations', 'entity_type', 'key'])
            self.assertEqual([(row['original'], row['replacement'], row['locations']) for row in rows],
                             [('TOKENZETA', 'Nome002', 'SYNTH.CBL:1'), ('TOKENALPHA', 'Nome001', 'SYNTH.CBL:2')])
            self.assertEqual(load_mapping(path), {})

    def test_template_keeps_suggestions_and_numeric_mapping_import(self):
        finding = Finding('SYNTH.CBL', 'PHONE', '123456789', 8, 17, 1, 8, 1, '')
        groups = group_findings([finding])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.csv'
            write_mapping_template(path, groups, None, 'test')
            with path.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['replacement'], '000000000')
            self.assertNotIn('suggested_replacement', row)
            self.assertEqual(load_mapping(path), {groups[0].key: '000000000'})
            write_mapping_template(path, groups, {}, 'test')
            with path.open() as handle:
                self.assertEqual(next(csv.DictReader(handle))['replacement'], '')

    def test_csv_groups_names_first_without_changing_replacements_or_import(self):
        findings = [Finding('SYNTH.CBL', entity, word, line * 20, line * 20 + len(word), line, 8, 1, '')
                    for line, entity, word in ((1, 'PHONE', '123456789'),
                                              (2, 'MATRICOLA', '654321'),
                                              (3, 'NAME', 'TOKENZETA'),
                                              (4, 'NAME', 'TOKENALPHA'),
                                              (5, 'PHONE', '987654321'))]
        groups = group_findings(findings)
        replacements = {group.key: suggested_replacement(group, index, 'test')
                        for index, group in enumerate(groups, 1)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.csv'
            write_mapping_template(path, groups, replacements, 'test')
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['entity_type'] for row in rows], ['NAME', 'NAME', 'MATRICOLA', 'PHONE', 'PHONE'])
            self.assertEqual([row['original'] for row in rows[:2]], ['TOKENZETA', 'TOKENALPHA'])
            for row in rows:
                self.assertEqual(row['replacement'], replacements[(row['entity_type'], row['key'])])
            self.assertEqual(load_mapping(path), {key: value for key, value in replacements.items() if key[0] != 'NAME'})
            self.assertEqual(path.with_suffix('.txt').read_text(),
                             f"TOKENZETA -> {replacements[groups[1].key]}\nTOKENALPHA -> {replacements[groups[0].key]}\n")
