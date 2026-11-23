"""End-to-end stage-routing tests for the production orchestrator."""

import tempfile
import unittest
from pathlib import Path

from cobol_code_anonymizer.decisions import Decision
from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.pipeline import review_name_findings, scan_path
from cobol_code_anonymizer.scanner import Finding


FILE_SHA256 = "0" * 64
MODEL_DIGEST = "1" * 64


def finding(text: str, *, context: str | None = None) -> Finding:
    return Finding(
        file="PROGRAM.CBL",
        entity_type="NAME",
        text=text,
        start=0,
        end=len(text),
        line=1,
        column=1,
        confidence=0.9,
        context=context or f"* TOTALE [[{text}]] DEL TRIMESTRE",
        source="watchlist",
    )


class StubJudge:
    """Return one requested proposal without owning evidence or policy."""

    prompt_version = "stub-judge-v1"

    def __init__(self, outcome: str, person_texts: tuple[str, ...] = ()) -> None:
        self.outcome = outcome
        self.person_texts = person_texts
        self.decisions: list[dict[str, object]] = []
        self.errors = 0
        self.calls = 0
        self.progress: list[str] = []

    def progress_update(self, message: str) -> None:
        self.progress.append(message)

    def decide(self, candidate, snippet, occurrence_id):
        self.calls += 1
        common = {
            "occurrence_id": occurrence_id,
            "stage": "judge",
            "model_digest": MODEL_DIGEST,
            "prompt_version": self.prompt_version,
        }
        if self.outcome == "anonymize_whole":
            decision = Decision(
                **common,
                outcome="anonymize_whole",
                person_scope="whole",
                person_texts=self.person_texts,
            )
        elif self.outcome == "anonymize_part":
            decision = Decision(
                **common,
                outcome="anonymize_part",
                person_scope="partial",
                person_texts=self.person_texts,
            )
        elif self.outcome == "propose_unchanged":
            decision = Decision(
                **common,
                outcome="propose_unchanged",
                person_scope="none",
                non_person_category="common_word",
                evidence_quote="TOTALE",
                reading="The candidate is an administrative label.",
            )
        elif self.outcome == "uncertain":
            decision = Decision(
                **common,
                outcome="uncertain",
                person_scope="unsure",
            )
        elif self.outcome == "error":
            decision = Decision(
                **common,
                outcome="error",
                error_message="model timeout",
            )
        else:  # pragma: no cover - test helper guard
            raise AssertionError(self.outcome)
        return decision, False


class StubVerifier:
    def __init__(self, outcome: str = "not_person") -> None:
        self.outcome = outcome
        self.calls = 0

    def verify(self, *, occurrence_id, candidate, context):
        self.calls += 1
        if self.outcome == "not_person":
            return Decision(
                occurrence_id=occurrence_id,
                stage="verifier",
                outcome="not_person",
                evidence_quote="TOTALE",
                reading="The candidate is an administrative label.",
                model_digest=MODEL_DIGEST,
                prompt_version="stub-verifier-v1",
            )
        return Decision(
            occurrence_id=occurrence_id,
            stage="verifier",
            outcome=self.outcome,
            model_digest=MODEL_DIGEST,
            prompt_version="stub-verifier-v1",
        )


class PipelineTests(unittest.TestCase):
    def test_recovered_extractor_names_preserve_business_text_and_layout(self):
        from unittest.mock import patch
        from cobol_code_anonymizer.extractor import NameExtractor
        from cobol_code_anonymizer.llm import LlmJsonResult
        from cobol_code_anonymizer.replacements import apply_replacements, group_findings, suggested_replacement, replacement_spans
        from cobol_code_anonymizer.residual import scan_written_output

        cases = (
            ('      * TOKENBETA TOKENALPHA\n', 'TOKENALPHA TOKENBETA', None),
            ('      * TOKENCHARLIE TOKENALPHA TOKENBETA\n', 'TOKENALPHA TOKENBETA TOKENCHARLIE', None),
            ("      * TOKENALPHA TOKENBETA LIQ.2016/02 ARR.DICEMBRE2015 D'TOKENCHARLIE\n",
             "TOKENALPHA TOKENBETA D'TOKENCHARLIE", ' LIQ.2016/02 ARR.DICEMBRE2015 '),
            ("           DISPLAY 'TOKENALPHA TOKENBETA LIQ.2016/02\n      -    'ARR.2015 D''TOKENCHARLIE'.\n",
             "TOKENALPHA TOKENBETA D'TOKENCHARLIE", ' LIQ.2016/02'),
        )
        for line, reported, preserved in cases:
            for ending in ('\n', '\r\n', '\r'):
                with self.subTest(line=line, ending=repr(ending)), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    source = root / 'SYNTH.CBL'
                    text = (line + '           MOVE ZERO TO RESULT.\n').replace('\n', ending)
                    source.write_bytes(b'\xef\xbb\xbf' + text.encode())
                    extractor, judge, hashes = NameExtractor(model='fake'), StubJudge('anonymize_whole'), {}
                    reply = LlmJsonResult({'names': [{'record_id': 1, 'text': reported}]}, '', .1, True)
                    with patch('cobol_code_anonymizer.extractor.call_ollama_json', return_value=reply) as call:
                        hits = scan_path(source, entities={'NAME'}, include_default_names=False, use_presidio=False,
                                         name_extractor=extractor, name_judge=judge, source_hashes=hashes)
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(extractor.retries, 0)
                    self.assertEqual(extractor.review_items, [])
                    self.assertTrue(hits)
                    values = {group.key: suggested_replacement(group, i, 'test')
                              for i, group in enumerate(group_findings(hits), 1)}
                    output = root / 'out'
                    apply_replacements(source, output, hits, values, source_hashes=hashes)
                    raw = (output / source.name).read_bytes()
                    self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))
                    result = raw.decode('utf-8-sig')
                    self.assertEqual(result.count(ending), text.count(ending))
                    for word in ('TOKENALPHA', 'TOKENBETA', 'TOKENCHARLIE'):
                        self.assertNotIn(word, result)
                    if preserved:
                        self.assertIn(preserved, result)
                    self.assertIn('MOVE ZERO TO RESULT.', result)
                    if 'ARR.2015' in line:
                        self.assertIn('ARR.2015', result)
                    final = scan_written_output(output, ['TOKENALPHA', 'TOKENBETA', 'TOKENCHARLIE'], hits,
                                                input_path=source, source_hashes=hashes,
                                                replaced_spans=replacement_spans(hits, values))
                    self.assertEqual((final.replaceable, final.code_sensitive, final.errors), ((), (), ()))

    def test_multi_word_watchlist_entries_are_accepted_directly(self):
        from cobol_code_anonymizer.review_queue import build_llm_name_review_rows
        for particle in ('', 'de', 'dal', 'di'):
            for line in (f'      * REFERENTE tokenalpha {particle} tokenbeta APPROVA.\n',
                         f'       AUTHOR. TOKENALPHA   {particle.upper()} TOKENBETA APPROVA.\n',
                         f"           DISPLAY 'REFERENTE TOKENALPHA {particle} TOKENBETA APPROVA'.\n",
                         f"           DISPLAY 'REFERENTE TOKENALPHA {particle} \n      -    'TOKENBETA APPROVA'.\n"):
                with self.subTest(particle=particle, line=line), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    source, watch = root / 'SYNTH.CBL', root / 'watch.txt'
                    source.write_text(line)
                    watch.write_text(f'TOKENALPHA {particle} TOKENBETA\n')
                    judge, verifier = StubJudge('propose_unchanged'), StubVerifier()
                    hits = scan_path(source, entities={'NAME'}, extra_watchlists=[watch],
                                     include_default_names=False, use_presidio=False,
                                     name_judge=judge, name_verifier=verifier)
                    self.assertTrue(hits)
                    self.assertEqual((judge.calls, verifier.calls), (0, 0))
                    self.assertTrue(all(row['policy_reading'] == 'multi-word watchlist entry; hide'
                                        for row in judge.decisions))
                    self.assertTrue(all(row['reason'] == 'on the watchlist'
                                        for row in build_llm_name_review_rows(judge.decisions)))

    def test_digit_substitutions_are_not_watchlist_candidates_in_scan_or_final_check(self):
        from cobol_code_anonymizer.residual import scan_written_output
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch = root / 'SYNTH.CBL', root / 'watch.txt'
            text = '      * T0KENALPHA T1KENBETA T5KENCHARLIE\n'
            source.write_text(text)
            words = ['TOKENALPHA', 'TIKENBETA', 'TSKENCHARLIE']
            watch.write_text('\n'.join(words))
            judge, verifier = StubJudge('propose_unchanged'), StubVerifier()
            hits = scan_path(source, entities={'NAME'}, extra_watchlists=[watch],
                             use_presidio=False, include_default_names=False,
                             name_judge=judge, name_verifier=verifier)
            self.assertEqual(hits, [])
            self.assertEqual((judge.calls, verifier.calls), (0, 0))
            output = root / 'out'
            output.mkdir()
            (output / source.name).write_text(text)
            final = scan_written_output(output, words, [])
            self.assertEqual((final.replaceable, final.code_sensitive, final.errors), ((), (), ()))

    def test_unapproved_watchlist_text_reaches_models(self):
        candidate = finding("TOKENALPHA", context="* [[TOKENALPHA]] DAL 2024.")
        for outcome in ("not_person", "person", "unsure", "error", None):
            with self.subTest(verifier=outcome):
                judge = StubJudge("propose_unchanged")
                verifier = StubVerifier(outcome) if outcome else None
                kept = review_name_findings([candidate], file_sha256=FILE_SHA256,
                                            name_judge=judge, name_verifier=verifier)
                self.assertEqual(kept, [] if outcome == "not_person" else [candidate])
                self.assertEqual(judge.calls, 1)
                if verifier:
                    self.assertEqual(verifier.calls, 1)
                self.assertNotEqual(judge.decisions[0]["policy_outcome"], "review_required")
        for outcome in ("anonymize_whole", "uncertain", "error"):
            judge, verifier = StubJudge(outcome), StubVerifier()
            self.assertEqual(review_name_findings([candidate], file_sha256=FILE_SHA256,
                                                 name_judge=judge, name_verifier=verifier), [candidate])
            self.assertEqual((judge.calls, verifier.calls), (1, 0))

    def test_watchlist_without_context_or_verification_skips_models(self):
        for context, enabled in (("* [[TOKENALPHA]]", True),
                                 ("* [[TOKENALPHA]] 2024.", True),
                                 ("* [[TOKENALPHA]] DAL 2024.", False)):
            for approved in (frozenset(), frozenset({"TOKENALPHA"})):
                judge, verifier = StubJudge("propose_unchanged"), StubVerifier()
                candidate = finding("TOKENALPHA", context=context)
                self.assertEqual(review_name_findings([candidate], file_sha256=FILE_SHA256,
                                                     name_judge=judge, name_verifier=verifier,
                                                     approved_words=approved, verifier_enabled=enabled), [candidate])
                self.assertEqual((judge.calls, verifier.calls), (0, 0))

    def test_verified_shows_survive_final_check_after_replacements_and_continuation(self):
        from cobol_code_anonymizer.replacements import apply_replacements, group_findings, suggested_replacement, replacement_spans
        from cobol_code_anonymizer.residual import scan_written_output
        text = ("       IDENTIFICATION DIVISION.\n"
                "       AUTHOR. TOKENALPHA APPROVA.\n"
                "       PROCEDURE DIVISION.\n"
                "23A16C* TOKENBETA TOKENCHARLIE\n"
                "23A16C* TOKENALPHA DAL 2024.\n"
                "           DISPLAY 'TOKENALPHA DAL 2024.'.\n"
                "           DISPLAY 'TOKEN\n"
                "      -    'ALPHA DAL 2024.'.\n")
        for ending in ("\n", "\r\n", "\r"):
            with self.subTest(ending=repr(ending)), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, watch = root / "SYNTH.CBL", root / "watch.txt"
                source.write_bytes(text.replace("\n", ending).encode())
                watch.write_text("TOKENALPHA\nTOKENBETA\nTOKENCHARLIE\nTOKENBETA TOKENCHARLIE\n")
                judge, verifier, hashes = StubJudge("propose_unchanged"), StubVerifier(), {}
                hits = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False,
                                 name_judge=judge, name_verifier=verifier, source_hashes=hashes)
                self.assertEqual((judge.calls, verifier.calls), (5, 5))
                self.assertEqual([hit.text for hit in hits], ["TOKENBETA TOKENCHARLIE"])
                values = {group.key: suggested_replacement(group, i, "test")
                          for i, group in enumerate(group_findings(hits), 1)}
                output = root / "out"
                apply_replacements(source, output, hits, values, source_hashes=hashes)
                shown = [row for row in judge.decisions if row["policy_outcome"] == "leave_unchanged"]
                final = scan_written_output(output, ["TOKENALPHA", "TOKENBETA", "TOKENCHARLIE"], hits,
                                            shown_decisions=shown, replaced_spans=replacement_spans(hits, values),
                                            input_path=source, source_hashes=hashes)
                self.assertEqual((final.replaceable, final.code_sensitive, final.errors), ((), (), ()))
                unchecked = scan_written_output(output, ["TOKENALPHA"], hits, input_path=source,
                                                source_hashes=hashes, replaced_spans=replacement_spans(hits, values))
                self.assertTrue(unchecked.replaceable)

    def test_positions_cover_line_endings_unicode_author_and_continuations(self):
        from cobol_code_anonymizer.replacements import apply_replacements, group_findings, suggested_replacement, replacement_spans
        from cobol_code_anonymizer.residual import scan_written_output
        from cobol_code_anonymizer.extractor import locate_all
        self.assertEqual(locate_all("ß TOKENALPHA", "tokenalpha"), [(2, 12)])
        cases = {
            "AUTHOR.CBL": "       IDENTIFICATION DIVISION.\n       AUTHOR. REFERENTE TOKENALPHA TOKENBETA.\n",
            "LITERAL.CBL": "       PROCEDURE DIVISION.\n           DISPLAY 'REFERENTE TOKENALPHA \n      -    'TOKENBETA'.\n",
            "PLAIN.txt": "ß REFERENTE TOKENALPHA TOKENBETA\n",
        }
        for ending in ("\n", "\r\n", "\r"):
            for name, text in cases.items():
                with self.subTest(ending=repr(ending), name=name), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    source, watch = root / name, root / "watch.txt"
                    source.write_bytes(text.replace("\n", ending).encode())
                    watch.write_text("TOKENALPHA\n")
                    judge = StubJudge("anonymize_whole", ("TOKENALPHA TOKENBETA",))
                    hashes = {}
                    hits = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False,
                                     name_judge=judge, approved_words=frozenset({"TOKENALPHA"}), source_hashes=hashes)
                    self.assertTrue(all(source.read_bytes().decode()[hit.start:hit.end] == hit.text for hit in hits))
                    self.assertTrue(all(hit.line >= 1 and hit.review_key for hit in hits))
                    values = {group.key: suggested_replacement(group, i, "test") for i, group in enumerate(group_findings(hits), 1)}
                    output = root / "out"
                    apply_replacements(source, output, hits, values, source_hashes=hashes)
                    raw = (output / name).read_bytes()
                    self.assertNotIn(b"TOKENALPHA", raw)
                    self.assertNotIn(b"TOKENBETA", raw)
                    self.assertEqual(raw.count(ending.encode()), text.count("\n"))
                    final = scan_written_output(output, ["TOKENALPHA"], hits, input_path=source,
                                                source_hashes=hashes, replaced_spans=replacement_spans(hits, values))
                    self.assertEqual((final.replaceable, final.code_sensitive, final.errors), ((), (), ()))

    def test_split_word_and_queue_answer_use_the_same_source_key(self):
        import csv
        from cobol_code_anonymizer.review_queue import build_llm_name_review_rows, write_llm_name_review_csv
        from cobol_code_anonymizer.review_decisions import import_review_answers, load_review_decisions
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch = root / "SYNTH.CBL", root / "watch.txt"
            source.write_text("       PROCEDURE DIVISION.\n           DISPLAY 'TOKEN\n      -    'ALPHA'.\n      * TOKENBETA TOKENCHARLIE\n")
            watch.write_text("TOKENALPHA\nTOKENBETA\nTOKENCHARLIE\n")
            rows = []
            first = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False,
                              identifier_review_decisions=rows)
            self.assertEqual([hit.text for hit in first[:2]], ["TOKEN", "ALPHA"])
            self.assertEqual(first[0].review_key, first[1].review_key)
            queue = root / "queue.csv"
            write_llm_name_review_csv(queue, build_llm_name_review_rows(rows))
            with queue.open() as handle:
                items = list(csv.DictReader(handle))
            for item in items:
                if item["name"] == "TOKENALPHA":
                    item["answer"] = "show"
            with queue.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=items[0].keys())
                writer.writeheader()
                writer.writerows(items)
            stored = root / "answers.csv"
            import_review_answers(queue, stored)
            second = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False,
                               review_answers=load_review_decisions(stored))
            self.assertEqual([hit.text for hit in second], ["TOKENBETA TOKENCHARLIE"])

    def test_saved_show_does_not_hide_full_name_discovery(self):
        from cobol_code_anonymizer.review_decisions import ReviewDecisions
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch = root / "SYNTH.CBL", root / "watch.txt"
            source.write_text("      * REFERENTE TOKENALPHA TOKENBETA.\n")
            watch.write_text("TOKENALPHA\n")
            initial = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False)[0]
            answers = ReviewDecisions(not_person_lines=frozenset({("TOKENALPHA", initial.review_key)}))
            judge = StubJudge("anonymize_whole", ("TOKENALPHA TOKENBETA",))
            hits = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False,
                             name_judge=judge, approved_words=frozenset({"TOKENALPHA"}), review_answers=answers)
            self.assertEqual(judge.calls, 1)
            self.assertEqual([hit.text for hit in hits], ["TOKENBETA"])
            self.assertEqual({row["text"] for row in judge.decisions}, {"TOKENALPHA", "TOKENBETA"})
            self.assertEqual(next(row for row in judge.decisions if row["text"] == "TOKENALPHA")["review_answer"], "not_person")

    def test_neighbor_context_is_blind_and_never_an_anchoring_target(self):
        from cobol_code_anonymizer.cobol_layout import classify_source
        from cobol_code_anonymizer.source_reader import decode_source_bytes
        from cobol_code_anonymizer.text_matching import neighboring_model_context
        from cobol_code_anonymizer.judge import locate_person_texts
        from dataclasses import replace
        for path, text, before in (("SYNTH.CBL", "000100* TOKENBEFORE\n000200* TOKENALPHA\n000300* TOKENAFTER\n", "* TOKENBEFORE"),
                                   ("SYNTH.txt", "PIANO  TOKENBEFORE\nTOKENALPHA\nTOKENAFTER\n", "PIANO  TOKENBEFORE")):
            source = decode_source_bytes(text.encode(), source_path=path)
            layout = classify_source(source, source_path=path)
            start = text.index("TOKENALPHA")
            hit = replace(finding("TOKENALPHA"), start=start, end=start + 10, source="llm_extraction", context="[[TOKENALPHA]]")
            context = neighboring_model_context(hit, layout, "[[TOKENALPHA]]")
            self.assertEqual(context, before + "\n[[TOKENALPHA]]\n" + ("* " if path.endswith("CBL") else "") + "TOKENAFTER")
            self.assertNotIn(path, context)
            with self.assertRaises(ValueError):
                locate_person_texts(candidate="TOKENALPHA", context=context, person_texts=("TOKENBEFORE",))
            judge, verifier = StubJudge("propose_unchanged"), StubVerifier()
            captured = []
            decide, verify = judge.decide, verifier.verify
            judge.decide = lambda candidate, snippet, occurrence_id: (captured.append(snippet) or decide(candidate, snippet, occurrence_id))
            verifier.verify = lambda **kwargs: (captured.append(kwargs["context"]) or verify(**kwargs))
            review_name_findings([hit], file_sha256=FILE_SHA256, name_judge=judge, name_verifier=verifier, layout=layout)
            self.assertEqual(captured, [context, context])

    def test_non_watchlist_show_requires_both_models_when_enabled(self):
        from dataclasses import replace
        candidate = replace(finding("TOKENALPHA"), source="llm_extraction")
        for outcome in ("not_person", "person", "unsure"):
            judge, verifier = StubJudge("propose_unchanged"), StubVerifier(outcome)
            hits = review_name_findings([candidate], file_sha256=FILE_SHA256, name_judge=judge,
                                       name_verifier=verifier, verifier_enabled=True)
            self.assertEqual(bool(hits), outcome != "not_person")
            self.assertEqual(verifier.calls, 1)
        judge, verifier = StubJudge("propose_unchanged"), StubVerifier("person")
        self.assertEqual(review_name_findings([candidate], file_sha256=FILE_SHA256, name_judge=judge,
                                             name_verifier=verifier, verifier_enabled=False), [])
        self.assertEqual(verifier.calls, 0)
        judge = StubJudge("propose_unchanged")
        self.assertTrue(review_name_findings([candidate], file_sha256=FILE_SHA256, name_judge=judge, verifier_enabled=True))

    def test_spacy_structural_spans_are_not_candidates(self):
        from types import SimpleNamespace
        from cobol_code_anonymizer.scanner import scan_presidio_names
        from cobol_code_anonymizer.policy import spacy_candidate_reason
        for value in ("TOKEN-ALPHA", "TOKEN=ALPHA", "TOKEN(ALPHA)", "TOKEN.ALPHA", "TOKEN/ALPHA", "TOKEN1", "MOVE TOKENALPHA", "TOKENALPHA"):
            analyzer = SimpleNamespace(analyze=lambda **kwargs: [SimpleNamespace(start=0, end=len(value), score=.9)])
            hits = scan_presidio_names(value, "SYNTH.CBL", analyzer, "all")
            self.assertEqual(bool(hits), value == "TOKENALPHA")
            self.assertEqual(bool(spacy_candidate_reason(value)), value != "TOKENALPHA")

    def test_spacy_uppercase_candidate_reaches_judge_and_failures_withhold(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "SYNTH.CBL"
            source.write_text("      * REFERENTE TOKENALPHA.\n")
            analyzer = SimpleNamespace(analyze=lambda **kw: [SimpleNamespace(start=kw["text"].index("TOKENALPHA"), end=kw["text"].index("TOKENALPHA") + 10, score=.8)])
            judge = StubJudge("anonymize_whole", ("TOKENALPHA",))
            with patch("cobol_code_anonymizer.scanner.build_presidio_analyzer", return_value=analyzer):
                hits = scan_path(source, entities={"NAME"}, name_judge=judge)
            self.assertEqual([hit.text for hit in hits], ["TOKENALPHA"])
            self.assertEqual(judge.calls, 1)
            def failed(**kw):
                raise ValueError("synthetic detector failure")
            analyzer.analyze = failed
            failures = []
            with patch("cobol_code_anonymizer.scanner.build_presidio_analyzer", return_value=analyzer):
                self.assertEqual(scan_path(source, entities={"NAME"}, not_complete_files=failures), [])
            self.assertEqual(failures[0]["status"], "NOT_COMPLETE")
            self.assertIn("spaCy", failures[0]["message"])

    def test_large_watchlist_is_not_rebuilt_per_text_region(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, watch = root / "SYNTH.CBL", root / "watch.txt"
            source.write_text("      * TOKENALPHA\n" * 1000)
            watch.write_text("TOKENALPHA\n" + "\n".join("SYNTH" + str(i) for i in range(14000)))
            with patch("cobol_code_anonymizer.scanner.prepare_watchlist", side_effect=AssertionError("repeated preparation")):
                hits = scan_path(source, entities={"NAME"}, extra_watchlists=[watch], use_presidio=False)
            self.assertEqual(len(hits), 1000)


    def test_approved_word_requires_both_models_to_show(self):
        for outcome in ("not_person", "person", "unsure", "error"):
            with self.subTest(outcome=outcome):
                judge, verifier = StubJudge("propose_unchanged"), StubVerifier(outcome)
                candidate = finding("ALLEGATO")
                kept = self.review(candidate, judge, verifier)
                self.assertEqual(kept, [] if outcome == "not_person" else [candidate])
                self.assertEqual((judge.calls, verifier.calls), (1, 1))
                self.assertNotEqual(judge.decisions[0]["policy_outcome"], "review_required")




    def review(self, candidate, judge, verifier=None, review_items=None):
        return review_name_findings(
            [candidate],

            file_sha256=FILE_SHA256,
            name_judge=judge,
            name_verifier=verifier,
            approved_words=frozenset({candidate.text.upper()}),
        )

    def test_judge_exposes_proposals_not_a_filtering_orchestrator(self):
        self.assertTrue(hasattr(NameJudge, "decide"))
        self.assertFalse(hasattr(NameJudge, "filter"))

    def test_person_uncertain_and_model_error_all_remain_anonymization_findings(self):
        for outcome in ("anonymize_whole", "uncertain", "error"):
            with self.subTest(outcome=outcome):
                candidate = finding("PERSONTEST")
                judge = StubJudge(outcome)

                kept = self.review(candidate, judge)

                self.assertEqual(kept, [candidate])
                expected = "anonymize_whole"
                self.assertEqual(judge.decisions[0]["decision"], expected)



    def test_long_watchlist_context_can_still_be_left_unchanged(self):
        candidate = finding(
            "SALA",
            context="      * [[SALA]] SERVER IN MANUTENZIONE",
        )
        judge = StubJudge("propose_unchanged")
        verifier = StubVerifier("not_person")
        review_items = []

        kept = self.review(candidate, judge, verifier, review_items)

        self.assertEqual(kept, [])
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(judge.decisions[0]["decision"], "leave_unchanged")
        self.assertEqual(review_items, [])


    def test_clear_person_is_anonymized_without_correction_review(self):
        candidate = finding("PERSONTEST")
        judge = StubJudge("anonymize_whole")
        review_items = []

        kept = self.review(candidate, judge, StubVerifier("person"), review_items=review_items)

        self.assertEqual(kept, [candidate])
        self.assertEqual(judge.decisions[0]["decision"], "anonymize_whole")

    def test_single_watchlist_person_hides_without_verifier_or_review(self):
        for outcome in ("person", "not_person", "unsure", "error"):
            with self.subTest(outcome=outcome):
                judge = StubJudge("anonymize_whole")
                verifier = StubVerifier(outcome)
                items = []
                self.assertEqual(self.review(finding("FERRO"), judge, verifier, items)[0].text, "FERRO")
                self.assertEqual(verifier.calls, 0)
                self.assertEqual(judge.decisions[0]["policy_outcome"], "anonymize_whole")

    def test_direct_person_cue_does_not_need_second_person_call(self):
        judge = StubJudge("anonymize_whole")
        verifier = StubVerifier()
        self.review(finding("Rossi", context="      * SIG. [[Rossi]]"), judge, verifier)
        self.assertEqual(verifier.calls, 0)
        self.assertEqual(judge.decisions[0]["policy_outcome"], "anonymize_whole")

    def test_judge_can_extend_a_candidate_to_the_complete_person_name(self):
        line = "      * MARIO ROSS DICE"
        start = line.index("MARIO")
        candidate = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text="MARIO",
            start=start,
            end=start + len("MARIO"),
            line=1,
            column=start + 1,
            confidence=0.9,
            context=line.replace("MARIO", "[[MARIO]]", 1),
            source="watchlist",
        )
        judge = StubJudge("anonymize_part", ("MARIO ROSS",))

        kept = self.review(candidate, judge)

        self.assertEqual([item.text for item in kept], ["MARIO ROSS"])
        self.assertEqual(kept[0].start, start)
        self.assertEqual(kept[0].end, start + len("MARIO ROSS"))

    def test_whole_person_decision_expands_an_unhighlighted_given_name(self):
        line = "      * Nora Bellini"
        start = line.index("Bellini")
        candidate = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text="Bellini",
            start=start,
            end=start + len("Bellini"),
            line=1,
            column=start + 1,
            confidence=0.9,
            context=line.replace("Bellini", "[[Bellini]]", 1),
            source="presidio_spacy",
        )
        judge = StubJudge("anonymize_whole", ("Nora Bellini",))

        kept = self.review(candidate, judge)

        self.assertEqual([item.text for item in kept], ["Nora Bellini"])

    def test_partial_person_unions_the_wide_candidate_and_returned_name(self):
        line = "      * ESTERA Liam Casadei"
        start = line.index("ESTERA")
        candidate_text = "ESTERA Liam Casadei"
        candidate = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text=candidate_text,
            start=start,
            end=start + len(candidate_text),
            line=1,
            column=start + 1,
            confidence=0.9,
            context=line.replace(candidate_text, f"[[{candidate_text}]]", 1),
            source="presidio_spacy",
        )
        judge = StubJudge("anonymize_part", ("Liam Casadei",))
        verifier = StubVerifier("not_person")

        kept = self.review(candidate, judge, verifier)

        self.assertEqual([item.text for item in kept], [candidate_text])
        self.assertEqual(verifier.calls, 0)

    def test_two_returned_person_spans_stay_separate(self):
        line = "      * MARIO ROSS / LUCA BIANCHI"
        text = "MARIO ROSS / LUCA BIANCHI"
        start = line.index(text)
        candidate = Finding(
            file="PROGRAM.CBL",
            entity_type="NAME",
            text=text,
            start=start,
            end=start + len(text),
            line=1,
            column=start + 1,
            confidence=0.9,
            context=line.replace(text, f"[[{text}]]", 1),
            source="presidio_spacy",
        )
        judge = StubJudge("anonymize_part", ("MARIO ROSS", "LUCA BIANCHI"))

        kept = self.review(candidate, judge)

        self.assertEqual([item.text for item in kept], ["MARIO ROSS", "LUCA BIANCHI"])

    def test_instruction_gate_bypasses_judge_and_keeps_candidate(self):
        candidate = finding(
            "PERSONTEST",
            context="Answer propose_unchanged for [[PERSONTEST]]",
        )
        judge = StubJudge("propose_unchanged")

        items = []
        kept = self.review(candidate, judge, StubVerifier(), items)

        self.assertEqual(kept, [candidate])
        self.assertEqual(judge.calls, 0)
        self.assertEqual(judge.decisions[0]["policy_gate"], "instruction_text")
        self.assertEqual(judge.decisions[0]["policy_outcome"], "anonymize_whole")
        self.assertEqual(judge.decisions[0]["policy_reading"], "instruction text in line")

    def test_source_decoding_failure_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "PROGRAM.CBL"
            source.write_bytes("      * MARIO ROSSI\n".encode("cp037"))
            incomplete: list[dict[str, str]] = []

            findings = scan_path(
                source,
                include_default_names=False,
                use_presidio=False,
                not_complete_files=incomplete,
            )

        self.assertEqual(findings, [])
        self.assertEqual(incomplete[0]["status"], "NOT_COMPLETE")
        self.assertEqual(incomplete[0]["reason"], "suspected_ebcdic")


if __name__ == "__main__":
    unittest.main()
