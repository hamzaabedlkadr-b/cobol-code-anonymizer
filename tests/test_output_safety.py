import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.cli import main


class OutputSafetyTests(unittest.TestCase):
    def test_watchlist_word_in_comment_sequence_area_is_hidden_without_models(self):
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("TOKENX* ordinary comment\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENX\n")
        output = root / "out"
        with redirect_stdout(io.StringIO()), patch("urllib.request.urlopen", side_effect=AssertionError("no models")):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        self.assertEqual((output / source.name).read_text(), "Nome001* ordinary comment\n")

    def test_failed_overwrite_does_not_leave_previous_shared_output(self):
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("      * TOKENALPHA\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        output = root / "out"
        output.mkdir()
        (output / "previous.txt").write_text("old output")
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.load_model_digests", side_effect=OSError("fake outage")):
            self.assertEqual(self.run_safe(source, watch, output, "--overwrite", "--name-extract-model", "fake"), 1)
        self.assertFalse(output.exists())

    def test_name_alias_is_generated_even_with_a_custom_map(self):
        from cobol_code_anonymizer.cli import choose_replacements
        from cobol_code_anonymizer.replacements import group_findings, load_mapping
        from cobol_code_anonymizer.scanner import Finding
        root = Path(tempfile.mkdtemp())
        mapping = root / "map.csv"
        mapping.write_text("entity_type,key,original,replacement\nNAME,TOKENALPHA,TOKENALPHA,TOKENBETA\n")
        self.assertEqual(load_mapping(mapping), {})
        group = group_findings([Finding("SYNTH.CBL", "NAME", "TOKENALPHA", 0, 10, 1, 1, 1, "[[TOKENALPHA]]")])[0]
        with patch("builtins.input", side_effect=AssertionError("name prompt")), redirect_stdout(io.StringIO()):
            values = choose_replacements([group], {group.key: "TOKENBETA"}, "test", False)
        self.assertEqual(values[group.key], "Nome001")

    def test_source_metadata_is_not_exported(self):
        import os
        if not hasattr(os, "setxattr"):
            self.skipTest("extended attributes unavailable")
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("      * TOKENALPHA\n")
        try:
            os.setxattr(source, "user.private", b"TOKENALPHA")
        except OSError:
            self.skipTest("filesystem does not support extended attributes")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        output = root / "out"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        self.assertNotIn("user.private", os.listxattr(output / source.name))

    def test_map_input_is_excluded_from_run_and_preflight(self):
        root = Path(tempfile.mkdtemp())
        source = root / "input"
        source.mkdir()
        (source / "SYNTH.CBL").write_text("      * TOKENALPHA\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        mapping = source / "private.csv"
        mapping.write_text("entity_type,key,original,replacement\n")
        output = root / "out"
        args = [str(source), "--out-dir", str(output), "--watchlist", str(watch),
                "--map-file", str(mapping), "--no-presidio", "--auto"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
            self.assertEqual(main([*args, "--preflight"]), 0)
        self.assertFalse((output / mapping.name).exists())
        reports = root / "out_reports"
        self.assertIn("files_scanned=1", (reports / "preflight.txt").read_text())

    def test_missing_watchlist_preserves_previous_output(self):
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("      * TOKENALPHA\n")
        output = root / "out"
        output.mkdir()
        (output / "previous.txt").write_text("previous")
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            main([str(source), "--out-dir", str(output), "--overwrite", "--watchlist", str(root / "missing.txt")])
        self.assertEqual((output / "previous.txt").read_text(), "previous")

    def test_empty_watchlist_still_checks_source_hash(self):
        from cobol_code_anonymizer.residual import scan_written_output
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("      * SOURCE\n")
        output = root / "out"
        output.mkdir()
        (output / source.name).write_text(source.read_text())
        result = scan_written_output(output, [], [], input_path=source, source_hashes={source.name: "wrong"})
        self.assertEqual(len(result.errors), 1)

    def test_output_and_report_ancestors_are_rejected_before_deletion(self):
        root = Path(tempfile.mkdtemp())
        source = root / "input"
        source.mkdir()
        original = source / "SYNTH.CBL"
        original.write_text("      * TOKENALPHA\n")
        for extra in (("--out-dir", str(root)), ("--out-dir", str(root / "out"), "--report-dir", str(root))):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                main([str(source), "--overwrite", *extra])
            self.assertEqual(original.read_text(), "      * TOKENALPHA\n")

    def test_readable_text_after_final_check_withholds_without_repair(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        original = (source_dir / "sample.cbl").read_text()
        from cobol_code_anonymizer.pipeline import scan_path
        def missed(*args, **kwargs):
            scan_path(*args, **kwargs)
            return []
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_path", side_effect=missed):
            self.assertEqual(self.run_safe(source_dir, watchlist, output), 1)
        self.assertFalse((output / "sample.cbl").exists())
        self.assertEqual((root / "share_reports" / "needs_review" / "sample.cbl").read_text(), original)

    def test_plain_text_name_is_hidden_on_first_pass(self):
        root, source_dir, watchlist = self.make_case()
        (source_dir / "note.txt").write_text("ROSSI\n")
        output = root / "share"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source_dir, watchlist, output), 0)
        self.assertNotIn("ROSSI", (output / "note.txt").read_text())




    def test_negative_sample_size_is_rejected(self):
        root, source_dir, watchlist = self.make_case()
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.run_safe(source_dir, watchlist, root / "share", "--sample-unchanged", "-1")

    def test_nested_reports_never_reach_release_or_manifest(self):
        root, source_dir, watchlist = self.make_case()
        reports = source_dir / "private_reports"
        reports.mkdir()
        (reports / "secret.json").write_text('{"original": "Mario Rossi"}')
        output = root / "share"
        from cobol_code_anonymizer.residual import scan_written_output
        def checked(*args, **kwargs):
            self.assertEqual(args[0], reports / "staging")
            self.assertFalse((output / "sample.cbl").exists())
            return scan_written_output(*args, **kwargs)
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_written_output", side_effect=checked):
            self.assertEqual(self.run_safe(source_dir, watchlist, output, "--report-dir", str(reports)), 0)
        self.assertEqual([p.name for p in output.rglob("*") if p.is_file()], ["sample.cbl"])


    def test_changed_source_is_not_released(self):
        root, source_dir, watchlist = self.make_case()
        from cobol_code_anonymizer.replacements import apply_replacements
        failures = []
        result = apply_replacements(source_dir, root / "stage", [], {},
                                    source_hashes={"sample.cbl": "wrong"}, not_complete_files=failures)
        self.assertEqual(result, (0, 0))
        self.assertEqual(failures[0]["reason"], "source_changed")


    def test_author_name_is_replaced_and_released(self):
        root, source_dir, watchlist = self.make_case()
        (source_dir / "sample.cbl").write_text("       IDENTIFICATION DIVISION.\n       AUTHOR. ROSSI.\n       PROCEDURE DIVISION.\n           STOP RUN.\n")
        output = root / "share"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source_dir, watchlist, output), 0)
        result = (output / "sample.cbl").read_text()
        self.assertNotIn("ROSSI", result)
        self.assertIn("AUTHOR. Nome", result)
        self.assertIn("STOP RUN.", result)

    def test_model_tags_are_trimmed_and_missing_tags_fail_before_canaries(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        for available in ({"test:9b": "a" * 64}, {}):
            with self.subTest(available=available), redirect_stdout(io.StringIO()) as stdout, \
                    patch("cobol_code_anonymizer.cli.load_model_digests", return_value=available) as tags, \
                    patch("cobol_code_anonymizer.extractor.NameExtractor") as extractor:
                extractor.return_value.canary_ok.return_value = (False, "fake startup failure")
                status = self.run_safe(source_dir, watchlist, output, "--name-extract-model", " test:9b ")
                self.assertEqual(status, 1)
                tags.assert_called_once()
                if available:
                    self.assertEqual(extractor.call_args.args[1], "test:9b")
                else:
                    extractor.assert_not_called()
                    self.assertIn("available models: none available", stdout.getvalue())

    def test_missing_requested_spacy_stops_normal_run_and_preflight(self):
        root, source_dir, watchlist = self.make_case()
        for extra in ([], ["--preflight"]):
            with self.subTest(extra=extra), patch.dict("sys.modules", {"presidio_analyzer": None}), redirect_stdout(io.StringIO()) as stdout:
                status = main([str(source_dir), "--watchlist", str(watchlist), "--auto",
                               "--out-dir", str(root / "share"), *extra])
                self.assertEqual(status, 1)
                self.assertIn("--no-presidio", stdout.getvalue())
                self.assertFalse((root / "share").exists())
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(self.run_safe(source_dir, watchlist, root / "share"), 0)
        self.assertIn("Active detectors:", stdout.getvalue())
        self.assertLess(stdout.getvalue().index("Active detectors:"), stdout.getvalue().index("Analyzing file"))

    def test_auto_never_prompts_or_prints_interactive_instructions(self):
        root, source_dir, watchlist = self.make_case()
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")) as prompt, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(self.run_safe(source_dir, watchlist, root / "share"), 0)
        prompt.assert_not_called()
        self.assertNotIn("Choose replacements", stdout.getvalue())
        self.assertNotIn("Press Enter", stdout.getvalue())
        self.assertNotIn("skip-all", stdout.getvalue())

    def test_rerun_clears_old_outputs_but_preserves_cache_answers_and_inputs(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        reports = root / "share_reports"
        reports.mkdir()
        cache = reports / "llm_cache"
        cache.mkdir()
        (cache / "judge.json").write_text("cache must survive")
        answers = "word,line_text,answer,reviewer,date\n"
        (reports / "review_decisions.csv").write_text(answers)
        (reports / "notes.txt").write_text("user notes")
        (reports / "not_complete_files.json").write_text("old startup failure")
        (reports / "sample_check.csv").write_text("old sample")
        for name in ("staging", "needs_review"):
            (reports / name).mkdir()
            (reports / name / "old.cbl").write_text("old private source")
        supplied = reports / "names_findings.csv"
        supplied.write_text("entity_type,original,replacement\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source_dir, watchlist, output, "--map-file", str(supplied)), 0)
        self.assertFalse((reports / "not_complete_files.json").exists())
        self.assertFalse((reports / "sample_check.csv").exists())
        self.assertFalse((reports / "needs_review").exists())
        self.assertFalse((reports / "staging" / "old.cbl").exists())
        self.assertEqual((cache / "judge.json").read_text(), "cache must survive")
        self.assertEqual((reports / "review_decisions.csv").read_text(), answers)
        self.assertEqual((reports / "notes.txt").read_text(), "user notes")
        self.assertEqual(supplied.read_text(), "entity_type,original,replacement\n")

    def make_case(self):
        root = Path(tempfile.mkdtemp())
        source_dir = root / "input"
        source_dir.mkdir()
        watchlist = root / "watchlist.txt"
        watchlist.write_text("ROSSI\n", encoding="utf-8")
        (source_dir / "sample.cbl").write_text(
            "      * REFERENTE ROSSI\n", encoding="utf-8"
        )
        return root, source_dir, watchlist

    def run_safe(self, source_dir, watchlist, output, *extra):
        return main([
            str(source_dir), "--out-dir", str(output), "--watchlist", str(watchlist),
            "--no-default-name-watchlist", "--no-presidio", "--auto", *extra,
        ])

    def test_report_is_sibling_and_nested_report_is_rejected(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source_dir, watchlist, output), 0)
            with self.assertRaises(SystemExit):
                main([
                    str(source_dir), "--out-dir", str(output),
                    "--report-dir", str(output / "reports"),
                ])
        self.assertTrue((root / "share_reports").is_dir())

    def test_all_local_reports_and_map_stay_outside_shareable_folder(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                self.run_safe(source_dir, watchlist, output, "--create-map", "map.csv"),
                0,
            )
        self.assertFalse((output / "map.csv").exists())
        self.assertTrue((root / "share_reports" / "map.csv").exists())

    def test_unsupported_files_are_not_copied_and_are_audited(self):
        root, source_dir, watchlist = self.make_case()
        (source_dir / "binary.bin").write_bytes(b"not source")
        output = root / "share"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source_dir, watchlist, output), 0)
        self.assertFalse((output / "binary.bin").exists())


    def test_stale_output_requires_overwrite(self):
        root, source_dir, watchlist = self.make_case()
        output = root / "share"
        output.mkdir()
        (output / "old.txt").write_text("old", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_safe(source_dir, watchlist, output)
            self.assertEqual(self.run_safe(source_dir, watchlist, output, "--overwrite"), 0)
        self.assertFalse((output / "old.txt").exists())






if __name__ == "__main__":
    unittest.main()


class HybridReportTests(unittest.TestCase):
    def test_extractor_retry_transport_failure_writes_no_copy(self):
        from cobol_code_anonymizer.llm import LlmJsonResult
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch, output = root / 'SYNTH.CBL', root / 'watch.txt', root / 'out'
            source.write_text('      * TOKENALPHA\n')
            watch.write_text('TOKENALPHA\n')
            primary = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENOMEGA'}]}, '', .1, True)
            error = LlmJsonResult(None, '', .1, False, error='transport failed')
            with patch('urllib.request.urlopen', side_effect=AssertionError('network')), redirect_stdout(io.StringIO()), \
                 patch('cobol_code_anonymizer.cli.load_model_digests', return_value={'fake': '1' * 64}), \
                 patch('cobol_code_anonymizer.extractor.NameExtractor.canary_ok', return_value=(True, '')), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, error, error]):
                self.assertEqual(self.run_safe(source, watch, output, '--llm', '--name-extract-model', 'fake'), 1)
            self.assertFalse((output / source.name).exists())
            reports = root / 'out_reports'
            self.assertFalse((reports / 'needs_review' / source.name).exists())
            self.assertIn('Result: released 0, withheld 0, not complete 1', (reports / 'scan_summary.txt').read_text())

    make_case = OutputSafetyTests.make_case
    run_safe = OutputSafetyTests.run_safe
    def test_changed_source_after_staging_is_not_released(self):
        from cobol_code_anonymizer.residual import scan_written_output
        root, source, watchlist = self.make_case()
        output = root / "share"
        def changed(*args, **kwargs):
            (source / "sample.cbl").write_text("      * MODIFICATO\n")
            return scan_written_output(*args, **kwargs)
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_written_output", side_effect=changed):
            self.assertEqual(self.run_safe(source, watchlist, output), 1)
        self.assertFalse((output / "sample.cbl").exists())
        self.assertIn("source changed", (root / "share_reports" / "scan_summary.txt").read_text())







def load_answer(reports):
    from cobol_code_anonymizer.review_decisions import load_review_decisions
    return load_review_decisions(reports / "review_decisions.csv").answer_for("ROSSI", "MOVE ROSSI TO RESULT.")


class CodeAliasTests(unittest.TestCase):
    run_safe = OutputSafetyTests.run_safe



    def test_sample_uses_withheld_copy_when_nothing_is_released(self):
        from cobol_code_anonymizer.pipeline import scan_path
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "SYNTH.CBL", root / "watch.txt", root / "out"
        source.write_text("      * TOKENALPHA / TOKENBETA\n")
        watch.write_text("TOKENALPHA\nTOKENBETA\n")
        def missed(*args, **kwargs):
            return [hit for hit in scan_path(*args, **kwargs) if hit.text != "TOKENBETA"]
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_path", side_effect=missed):
            self.assertEqual(self.run_safe(source, watch, output, "--sample-unchanged", "300"), 1)
        reports = root / "out_reports"
        self.assertIn("Sample: 1 lines in sample_check.txt (withheld copy)", (reports / "scan_summary.txt").read_text())
        self.assertIn("[withheld copy]", (reports / "sample_check.txt").read_text())
        self.assertIn("Nome001 / TOKENBETA", (reports / "sample_check.txt").read_text())
        self.assertFalse((output / source.name).exists())
        self.assertIn("[REVIEW REQUIRED]", (reports / "llm_finding.txt").read_text())

    def test_scan_only_uses_the_same_old_report_formats(self):
        root = Path(tempfile.mkdtemp())
        source, watch = root / "SYNTH.CBL", root / "watch.txt"
        source.write_text("      * TOKENALPHA\n")
        watch.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, root / "out", "--scan-only"), 0)
        reports = root / "out_reports"
        self.assertIn("[NAME]", (reports / "scan_summary.txt").read_text())
        self.assertTrue((reports / "replacement_map.csv").is_file())
        self.assertFalse((root / "out").exists())

    def test_old_report_files_and_formats_are_restored(self):
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "SYNTH.CBL", root / "watch.txt", root / "out"
        source.write_text("      * TOKENALPHA TOKENBETA\n")
        watch.write_text("TOKENALPHA\nTOKENBETA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        reports = root / "out_reports"
        summary = (reports / "scan_summary.txt").read_text()
        self.assertTrue(summary.startswith("Result: released 1, withheld 0, not complete 0"))
        self.assertIn("Mode: baseline", summary)
        self.assertIn("[NAME]", summary)
        self.assertIn("hits=", summary)
        self.assertIn("locations=", summary)
        review = (reports / "llm_finding.txt").read_text()
        self.assertIn("[ACCEPTED AS NAME - hidden]", review)
        self.assertIn("reason=two watchlist words together", review)
        self.assertNotIn("action=", review)
        self.assertNotIn("optional review", review)
        self.assertNotIn("; hide", review)
        with (reports / "llm_name_review.csv").open() as handle:
            fields = next(csv.reader(handle))
        self.assertEqual(fields[:10], ["name", "file", "line", "column", "extraction_status", "judge_status", "final_action", "reason", "detector_sources", "context"])
        self.assertEqual(fields[-3:], ["verifier_status", "key", "answer"])
        self.assertTrue((reports / "replacement_map.csv").is_file())
        for removed in ("decisions.csv", "review_queue.csv", "out_manifest.csv", "changes.txt"):
            self.assertFalse((reports / removed).exists())

    def test_digit_boundaries_in_code_and_paths_share_alias_without_models(self):
        root = Path(tempfile.mkdtemp())
        source = root / "input"
        folder = source / "TOKENALPHA01"
        folder.mkdir(parents=True)
        (folder / "WS-TOKENALPHA1.cbl").write_text(
            "           MOVE WS-TOKENALPHA1 TO 02TOKENALPHA_RESULT.\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        output = root / "out"
        with redirect_stdout(io.StringIO()), patch("urllib.request.urlopen", side_effect=AssertionError("no models")):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        self.assertEqual((output / "Nome00101" / "WS-Nome0011.cbl").read_text(),
                         "           MOVE WS-Nome0011 TO 02Nome001_RESULT.\n")

    def test_final_check_blocks_missed_code_parts_at_digit_boundaries(self):
        from cobol_code_anonymizer.pipeline import scan_path
        for identifier in ("TOKENALPHA01", "WS-TOKENALPHA1", "02TOKENALPHA"):
            with self.subTest(identifier=identifier):
                root = Path(tempfile.mkdtemp())
                source = root / "SYNTH.CBL"
                source.write_text(f"           MOVE {identifier} TO RESULT.\n")
                watch = root / "watch.txt"
                watch.write_text("TOKENALPHA\n")
                def missed(*args, **kwargs):
                    scan_path(*args, **kwargs)
                    return []
                with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_path", side_effect=missed):
                    self.assertEqual(self.run_safe(source, watch, root / "out"), 1)
                self.assertFalse((root / "out" / source.name).exists())

    def test_code_paths_and_text_share_one_alias(self):
        root = Path(tempfile.mkdtemp())
        source = root / "input"
        folder = source / "TOKENALPHA"
        folder.mkdir(parents=True)
        text = "           MOVE WS-TOKENALPHA TO TOKENALPHA_RESULT.\n      * TOKENALPHA TOKENBETA\n"
        (folder / "TOKENBETA.cbl").write_text(text)
        (source / "OTHER.cbl").write_text("           COPY TOKENALPHA.\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\nTOKENBETA\n")
        output = root / "out"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        renamed = output / "Nome001" / "Cognome001.cbl"
        result = renamed.read_text()
        self.assertIn("WS-Nome001", result)
        self.assertIn("Nome001_RESULT", result)
        self.assertIn("Nome001 Cognome001", result)
        self.assertIn("COPY Nome001", (output / "OTHER.cbl").read_text())

    def test_approved_code_stays_and_glued_parts_are_not_split(self):
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        text = "           MOVE TOKENALPHA TO TOKENALPHARESULT.\n"
        source.write_text(text)
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        approved = root / "approved.txt"
        approved.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, root / "out", "--approved-words", str(approved)), 0)
        self.assertEqual((root / "out" / source.name).read_text(), text)

    def test_missed_code_word_blocks_at_final_check(self):
        from cobol_code_anonymizer.pipeline import scan_path
        root = Path(tempfile.mkdtemp())
        source = root / "SYNTH.CBL"
        source.write_text("           MOVE TOKENALPHA TO RESULT.\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        def missed(*args, **kwargs):
            scan_path(*args, **kwargs)
            return []
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_path", side_effect=missed):
            self.assertEqual(self.run_safe(source, watch, root / "out"), 1)
        self.assertFalse((root / "out" / source.name).exists())

    def test_cli_uses_real_role_classes_with_fake_responses_and_saved_cache(self):
        from cobol_code_anonymizer.llm import LlmJsonResult
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "SYNTH.CBL", root / "watch.txt", root / "out"
        source.write_text("      * TOKENALPHA\n")
        watch.write_text("TOKENALPHA\n")
        flags = ("--name-extract-model", "fake", "--name-judge-model", "fake", "--name-verifier-model", "fake")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")), redirect_stdout(io.StringIO()), \
             patch("cobol_code_anonymizer.cli.load_model_digests", return_value={"fake": "1" * 64}), \
             patch("cobol_code_anonymizer.extractor.NameExtractor.canary_ok", return_value=(True, "")), \
             patch("cobol_code_anonymizer.judge.NameJudge.canary_ok", return_value=(True, "")), \
             patch("cobol_code_anonymizer.extractor.call_ollama_json", return_value=LlmJsonResult({"names": []}, "", 0, True)) as call:
            self.assertEqual(self.run_safe(source, watch, output, *flags), 0)
            self.assertEqual(self.run_safe(source, watch, output, *flags, "--overwrite"), 0)
            self.assertEqual(call.call_count, 1)
        self.assertTrue((output / source.name).exists())
        self.assertTrue((root / "out_reports" / "llm_cache" / "extractor.json").exists())

    def test_sample_includes_changed_code_and_text_with_no_extra_read(self):
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "TOKENALPHA.CBL", root / "watch.txt", root / "out"
        source.write_text("       AUTHOR. TOKENALPHA.\n       PROCEDURE DIVISION.\n           MOVE TOKENALPHA TO RESULT.\n      * TOKENALPHA\n           DISPLAY 'TOKENALPHA'.\n")
        watch.write_text("TOKENALPHA\n")
        from cobol_code_anonymizer.source_reader import read_source
        with patch("cobol_code_anonymizer.residual.read_source", wraps=read_source) as reads, redirect_stdout(io.StringIO()) as terminal:
            self.assertEqual(self.run_safe(source, watch, output, "--sample-unchanged", "300", "--sample-seed", "42"), 0)
        self.assertEqual(reads.call_count, 2)
        summary = (root / "out_reports" / "scan_summary.txt").read_text()
        self.assertIn("Sample: 5 lines in sample_check.txt", summary)
        sample = (root / "out_reports" / "sample_check.txt").read_text()
        self.assertIn("Nome001.CBL:3", sample)
        self.assertIn("MOVE Nome001 TO RESULT", sample)
        self.assertIn("AUTHOR. Nome001", sample)
        self.assertIn("Check each line", sample)
        self.assertNotIn("Nome001.CBL:3", terminal.getvalue())
        self.assertFalse((root / "out_reports" / "sample_check.csv").exists())

    def test_sample_excludes_files_blocked_by_the_final_check(self):
        from cobol_code_anonymizer.residual import scan_written_output
        root = Path(tempfile.mkdtemp())
        (root / "CLEAN.CBL").write_text("           STOP RUN.\n")
        (root / "BLOCKED.CBL").write_text("      * TOKENALPHA\n")
        result = scan_written_output(root, ["TOKENALPHA"], [], sample_count=300, sample_seed=42)
        self.assertEqual(result.sample_population, 1)
        self.assertEqual([row["file"] for row in result.samples], ["CLEAN.CBL"])

    def test_final_review_answer_uses_source_key_after_length_change(self):
        from cobol_code_anonymizer.pipeline import scan_path
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "SYNTH.CBL", root / "watch.txt", root / "out"
        source.write_text("      * TOKENALPHA / TOKENBETA\n")
        watch.write_text("TOKENALPHA\nTOKENBETA\n")
        def missed(*args, **kwargs):
            return [hit for hit in scan_path(*args, **kwargs) if hit.text != "TOKENBETA"]
        with redirect_stdout(io.StringIO()), patch("cobol_code_anonymizer.cli.scan_path", side_effect=missed):
            self.assertEqual(self.run_safe(source, watch, output), 1)
        queue = root / "out_reports" / "llm_name_review.csv"
        with queue.open() as handle:
            rows = list(csv.DictReader(handle))
        row = next(row for row in rows if row["final_action"] == "review (required)")
        self.assertTrue(row["key"])
        self.assertIn("TOKENALPHA / [[TOKENBETA]]", row["context"])
        row["answer"] = "show"
        with queue.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output, "--overwrite"), 0)
        self.assertIn(" / TOKENBETA", (output / source.name).read_text())
        self.assertNotIn("TOKENALPHA", (output / source.name).read_text())

    def test_approved_path_stays_even_when_text_uses_an_alias(self):
        root = Path(tempfile.mkdtemp())
        source, watch, approved, output = root / "TOKENALPHA.CBL", root / "watch.txt", root / "approved.txt", root / "out"
        source.write_text("      * TOKENALPHA\n           MOVE TOKENALPHA TO RESULT.\n")
        watch.write_text("TOKENALPHA\n"); approved.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output, "--approved-words", str(approved)), 0)
        self.assertTrue((output / source.name).exists())
        text = (output / source.name).read_text()
        self.assertIn("* Nome001", text)
        self.assertIn("MOVE TOKENALPHA", text)
        report = (root / "out_reports" / "llm_finding.txt").read_text()
        self.assertIn("[REJECTED - left as is]", report)

    def test_path_correction_answer_works_on_the_next_run(self):
        root = Path(tempfile.mkdtemp())
        source, watch, output = root / "TOKENALPHA.CBL", root / "watch.txt", root / "out"
        source.write_text("           STOP RUN.\n"); watch.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output), 0)
        queue = root / "out_reports" / "llm_name_review.csv"
        with queue.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["key"])
        rows[0]["answer"] = "show"
        with queue.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, output, "--overwrite"), 0)
        self.assertTrue((output / source.name).exists())
        with queue.open() as handle:
            self.assertTrue(all("review" not in row["final_action"] for row in csv.DictReader(handle)))

    def test_case_insensitive_renamed_path_collision_stops_before_writing(self):
        root = Path(tempfile.mkdtemp()); source = root / "input"; source.mkdir()
        (source / "TOKENALPHA.CBL").write_text("           STOP RUN.\n")
        (source / "Nome001.cbl").write_text("           STOP RUN.\n")
        watch = root / "watch.txt"; watch.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.run_safe(source, watch, root / "out")
        self.assertFalse((root / "out").exists())

    def test_name_only_in_a_path_is_renamed(self):
        root = Path(tempfile.mkdtemp())
        source = root / "TOKENALPHA.cbl"
        source.write_text("           STOP RUN.\n")
        watch = root / "watch.txt"
        watch.write_text("TOKENALPHA\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.run_safe(source, watch, root / "out"), 0)
        self.assertEqual([p.name for p in (root / "out").iterdir()], ["Nome001.cbl"])
