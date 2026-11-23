"""Regression tests for configuration-scoped judge and verifier caches."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.llm import load_model_digests
from cobol_code_anonymizer.llm_cache import response_cache_key
from cobol_code_anonymizer.scanner import Finding
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.verifier import NameVerifier


CONTEXT = "000100 * TOTALE [[ALLEGATO]] DEL TRIMESTRE"
OCCURRENCE_ID = "a" * 64


def finding() -> Finding:
    return Finding(
        file="PROGRAM.CBL",
        entity_type="NAME",
        text="ALLEGATO",
        start=16,
        end=24,
        line=1,
        column=17,
        confidence=0.9,
        context=CONTEXT,
        source="watchlist",
    )


def judge_reply() -> LlmJsonResult:
    return LlmJsonResult(
        parsed={
            "decision": "propose_unchanged",
            "person_scope": "none",
            "person_texts": [],
            "non_person_category": "common_word",
            "evidence_quote": "TOTALE",
            "reading": "Administrative label.",
        },
        content="",
        latency_s=0.01,
        schema_ok=True,
    )


def verifier_reply() -> LlmJsonResult:
    return LlmJsonResult(
        parsed={
            "decision": "not_person",
            "evidence_quote": "TOTALE",
            "reading": "Administrative label.",
        },
        content="",
        latency_s=0.01,
        schema_ok=True,
    )


class LlmCacheTests(unittest.TestCase):
    def test_models_share_plain_input_and_cache_across_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "input"
            inputs.mkdir()
            for number in (100, 200):
                (inputs / f"sample{number}.cbl").write_text(
                    f"{number:06d}* TOTALE ALLEGATO DEL TRIMESTRE\n", encoding="utf-8"
                )
            watchlist = root / "watchlist.txt"
            watchlist.write_text("ALLEGATO\n", encoding="utf-8")
            judge = NameJudge(model="fake", model_digest="b" * 64)
            verifier = NameVerifier(model="fake", model_digest="b" * 64)
            with patch("cobol_code_anonymizer.judge.call_ollama_json", return_value=judge_reply()) as jc, \
                    patch("cobol_code_anonymizer.verifier.call_ollama_json", return_value=verifier_reply()) as vc:
                result = scan_path(inputs, entities={"NAME"}, extra_watchlists=[watchlist],
                                   include_default_names=False, use_presidio=False,
                                   name_judge=judge, name_verifier=verifier)
            self.assertEqual(result, [])
            self.assertEqual(jc.call_count, 1)
            self.assertEqual(vc.call_count, 1)
            self.assertEqual(judge.cache_hits, 1)
            self.assertEqual(verifier.cache_hits, 1)
            judge_input = jc.call_args.args[2][1]
            self.assertEqual(judge_input, vc.call_args.args[2][1])
            self.assertIn("* TOTALE [[ALLEGATO]] DEL TRIMESTRE", judge_input["content"])
            for metadata in ("000100", "000200", "sample", "Detected by", "clue:"):
                self.assertNotIn(metadata, judge_input["content"])

    def test_plain_text_prefixes_and_different_detectors(self):
        from dataclasses import replace
        from cobol_code_anonymizer.judge import build_messages as judge_messages
        from cobol_code_anonymizer.verifier import build_messages as verifier_messages
        for context in ("SIG.R [[SALA]]", "PIANO [[SALA]]"):
            self.assertEqual(judge_messages(context, "SALA")[1],
                             verifier_messages(context=context, candidate="SALA")[1])
            self.assertIn(context, judge_messages(context, "SALA")[1]["content"])
        judge = NameJudge(model="fake", model_digest="b" * 64)
        with patch("cobol_code_anonymizer.judge.call_ollama_json", return_value=judge_reply()) as call:
            judge.decide(finding(), CONTEXT, OCCURRENCE_ID)
            _, cached = judge.decide(replace(finding(), source="presidio_spacy", file="other.cbl"),
                                     CONTEXT, "b" * 64)
        self.assertTrue(cached)
        self.assertEqual(call.call_count, 1)

    def test_exact_request_keys_and_model_digest(self):
        request = dict(stage="judge", messages=[{"role": "user", "content": "SIG.R [[SALA]]"}],
                       schema={"type": "object"}, options={"temperature": 0}, model_digest="a" * 64)
        key = response_cache_key(**request)
        self.assertEqual(key, response_cache_key(**request))
        for field, value in (
            ("messages", [{"role": "user", "content": "PIANO [[SALA]]"}]),
            ("stage", "verifier"), ("schema", {}), ("options", {"temperature": 1}),
            ("model_digest", "b" * 64),
        ):
            self.assertNotEqual(key, response_cache_key(**{**request, field: value}))

    def test_requests_disable_thinking_and_cache_keys_include_it(self):
        import hashlib
        import json
        import socket
        from unittest.mock import MagicMock
        from cobol_code_anonymizer.llm import call_ollama_json
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"message":{"content":"{}"}}'
        with patch("urllib.request.urlopen", return_value=response) as call:
            call_ollama_json("http://fake", "model", [], {"type": "object"})
        payload = json.loads(call.call_args.args[0].data)
        self.assertIs(payload["think"], False)
        request = dict(stage="judge", messages=[], schema={}, options={}, model_digest="a" * 64)
        old_key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertNotEqual(response_cache_key(**request), old_key)
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            result = call_ollama_json("http://fake", "model", [], {})
        self.assertIn("timed out", result.error)

    def test_model_digests_come_from_one_tags_request(self):
        import json
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "models": [{"name": "model:latest", "digest": "a" * 64},
                       {"name": "other:3b", "digest": "b" * 64}]
        }).encode()
        with patch("urllib.request.urlopen", return_value=response) as call:
            digests = load_model_digests("http://fake")
        self.assertEqual(digests["model"], "a" * 64)
        self.assertEqual(digests["other:3b"], "b" * 64)
        call.assert_called_once()

    def test_request_cost_includes_schema_retries_but_excludes_cache_hits(self):
        from dataclasses import replace
        judge = NameJudge(model="fake", model_digest="b" * 64)
        response = replace(judge_reply(), latency_s=2.5, retried=True)
        with patch("cobol_code_anonymizer.judge.call_ollama_json", return_value=response):
            judge.decide(finding(), CONTEXT, OCCURRENCE_ID)
            judge.decide(finding(), CONTEXT, "b" * 64)
        self.assertEqual(judge.calls, 1)
        self.assertEqual(judge.http_attempts, 2)
        self.assertEqual(judge.latency_s, 2.5)
        self.assertEqual(judge.cache_hits, 1)

    def test_many_answers_write_once_and_clean_flush_does_not_write(self):
        import json
        import os
        from cobol_code_anonymizer.llm_cache import PersistentResponseCache
        with tempfile.TemporaryDirectory() as folder:
            cache = PersistentResponseCache(report_dir=Path(folder), stage="judge")
            with patch("cobol_code_anonymizer.llm_cache.os.replace", wraps=os.replace) as write:
                for index in range(100):
                    cache.put(str(index), {"answer": index})
                self.assertFalse(cache.path.exists())
                cache.flush()
                cache.flush()
                self.assertEqual(write.call_count, 1)
            saved = json.loads(cache.path.read_text())
            self.assertEqual(len(saved["entries"]), 100)
            reloaded = PersistentResponseCache(report_dir=Path(folder), stage="judge")
            self.assertEqual(reloaded.get("42"), {"answer": 42})

    def test_pipeline_flushes_each_file_and_reuses_saved_answers(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "input"
            inputs.mkdir()
            for name in ("A", "B"):
                (inputs / f"{name}.cbl").write_text(f"      * TOTALE ALLEGATO DEL TRIMESTRE {name}\n")
            watchlist = root / "words.txt"
            watchlist.write_text("ALLEGATO\n")
            judge = NameJudge(model="fake", model_digest="b" * 64, cache_dir=root / "reports")
            with patch("cobol_code_anonymizer.llm_cache.os.replace", wraps=os.replace) as write, \
                 patch("cobol_code_anonymizer.judge.call_ollama_json", return_value=judge_reply()) as call:
                scan_path(inputs, entities={"NAME"}, extra_watchlists=[watchlist], use_presidio=False,
                          name_judge=judge)
                self.assertEqual(write.call_count, 2)
                self.assertEqual(call.call_count, 2)
                reloaded = NameJudge(model="fake", model_digest="b" * 64, cache_dir=root / "reports")
                scan_path(inputs, entities={"NAME"}, extra_watchlists=[watchlist], use_presidio=False,
                          name_judge=reloaded)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(write.call_count, 2)
                self.assertEqual(reloaded.cache_hits, 2)

    def test_same_judge_line_twice_uses_one_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            judge = NameJudge(model="model-a", cache_dir=Path(folder), model_digest="b" * 64)
            with patch(
                "cobol_code_anonymizer.judge.call_ollama_json",
                return_value=judge_reply(),
            ) as call:
                first, first_cached = judge.decide(finding(), CONTEXT, OCCURRENCE_ID)
                second, second_cached = judge.decide(finding(), CONTEXT, "b" * 64)

            self.assertEqual(first.outcome, "propose_unchanged")
            self.assertEqual(second.outcome, "propose_unchanged")
            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(judge.cache_hits, 1)
            judge._response_cache.flush()
            self.assertTrue(list((Path(folder) / "llm_cache").rglob("judge.json")))

    def test_configuration_change_does_not_reuse_judge_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_dir = Path(folder)
            first = NameJudge(model="model-a", cache_dir=cache_dir, model_digest="b" * 64)
            second = NameJudge(model="model-b", cache_dir=cache_dir, model_digest="c" * 64)
            with patch(
                "cobol_code_anonymizer.judge.call_ollama_json",
                return_value=judge_reply(),
            ) as call:
                first.decide(finding(), CONTEXT, OCCURRENCE_ID)
                second.decide(finding(), CONTEXT, "b" * 64)

            self.assertEqual(call.call_count, 2)
            self.assertEqual(second.cache_hits, 0)

    def test_verifier_has_a_separate_configuration_scoped_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_dir = Path(folder)
            verifier = NameVerifier(model="model-a", cache_dir=cache_dir, model_digest="b" * 64)
            with patch(
                "cobol_code_anonymizer.verifier.call_ollama_json",
                return_value=verifier_reply(),
            ) as call:
                first = verifier.verify(
                    occurrence_id=OCCURRENCE_ID,
                    candidate="ALLEGATO",
                    context=CONTEXT,
                )
                second = verifier.verify(
                    occurrence_id="b" * 64,
                    candidate="ALLEGATO",
                    context=CONTEXT,
                )

            self.assertEqual(first.outcome, "not_person")
            self.assertEqual(second.outcome, "not_person")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(verifier.cache_hits, 1)
            verifier._response_cache.flush()
            cache_root = cache_dir / "llm_cache"
            self.assertTrue(list(cache_root.rglob("verifier.json")))
            self.assertFalse(list(cache_root.rglob("judge.json")))


if __name__ == "__main__":
    unittest.main()


class OutageTests(unittest.TestCase):
    def test_non_object_cache_is_ignored(self):
        import tempfile
        from pathlib import Path
        from cobol_code_anonymizer.llm_cache import PersistentResponseCache
        root = Path(tempfile.mkdtemp())
        (root / "llm_cache").mkdir()
        (root / "llm_cache" / "judge.json").write_text("[]")
        self.assertEqual(PersistentResponseCache(report_dir=root, stage="judge").entries, {})


class ExtractorCacheTests(unittest.TestCase):
    def test_old_cached_unmatched_answer_recovers_without_another_model_call(self):
        from cobol_code_anonymizer.extractor import NameExtractor
        text = '      * TOKENBETA TOKENALPHA\n'
        primary = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENALPHA TOKENBETA'}]}, '', .1, True)
        retry = LlmJsonResult({'names': []}, '', .1, True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = NameExtractor(model_digest='1' * 64, cache_dir=root)
            with patch('cobol_code_anonymizer.extractor.exact_name_word_spans', return_value=[]), \
                 patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, retry]):
                self.assertEqual(first.extract(text, 'SYNTH.CBL', 'context'), [])
                self.assertEqual(len(first.review_items), 1)
            first._response_cache.flush()
            second = NameExtractor(model_digest='1' * 64, cache_dir=root)
            with patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=AssertionError('network')):
                hits = second.extract(text, 'SYNTH.CBL', 'context')
            self.assertEqual([hit.text for hit in hits], ['TOKENBETA TOKENALPHA'])
            self.assertEqual(second.cache_hits, 1)
            self.assertEqual(second.review_items, [])
            self.assertEqual(second.retries, 0)

    def test_unmatched_and_retry_answers_are_cached_without_losing_review(self):
        from cobol_code_anonymizer.extractor import NameExtractor
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            primary = LlmJsonResult({'names': [{'record_id': 1, 'text': 'TOKENOMEGA'}]}, '', .1, True)
            retry = LlmJsonResult({'names': []}, '', .1, True)
            text = '      * TOKENALPHA\n'
            first = NameExtractor(model_digest='1' * 64, cache_dir=root)
            with patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=[primary, retry]) as call:
                first.extract(text, 'FIRST.CBL', 'context')
                self.assertEqual(call.call_count, 2)
            first._response_cache.flush()
            second = NameExtractor(model_digest='1' * 64, cache_dir=root)
            with patch('cobol_code_anonymizer.extractor.call_ollama_json', side_effect=AssertionError('network')):
                second.extract(text, 'SECOND.CBL', 'context')
            self.assertEqual(second.cache_hits, 2)
            self.assertTrue(second.complete)
            self.assertEqual(second.review_items[0]['review_key'], first.review_items[0]['review_key'])
            self.assertEqual(second.unlocatable, 1)
            self.assertEqual(len(second._response_cache.entries), 2)

    def test_invalid_extractor_answers_are_not_cached(self):
        from cobol_code_anonymizer.extractor import NameExtractor
        extractor = NameExtractor(model_digest='1' * 64)
        with patch('cobol_code_anonymizer.extractor.call_ollama_json',
                   return_value=LlmJsonResult(None, '', .1, False, retried=True)):
            extractor.extract('      * TOKENALPHA\n', 'SYNTH.CBL', 'context')
        self.assertEqual(extractor._response_cache.entries, {})
        self.assertFalse(extractor.complete)

    def test_reuse_across_files_positions_and_runs_but_not_model_digests(self):
        from cobol_code_anonymizer.extractor import NameExtractor
        from cobol_code_anonymizer.cobol_layout import classify_source
        from cobol_code_anonymizer.source_reader import decode_source_bytes
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            extractor = NameExtractor(model_digest="1" * 64, cache_dir=root)
            response = LlmJsonResult({"names": [{"record_id": 1, "text": "TOKENALPHA"}]}, "", .1, True)
            def extract(model, text, file):
                layout = classify_source(decode_source_bytes(text.encode(), source_path=file), source_path=file)
                return model.extract(text, file, "context", layout=layout)
            with patch("cobol_code_anonymizer.extractor.call_ollama_json", return_value=response) as call:
                first = extract(extractor, "000100* TOKENALPHA\n", "FIRST.CBL")
                extractor._response_cache.flush()
                reloaded = NameExtractor(model_digest="1" * 64, cache_dir=root)
                second = extract(reloaded, "000100     STOP RUN.\n000200* TOKENALPHA\n", "SECOND.CBL")
                self.assertEqual(call.call_count, 1)
                self.assertEqual(first[0].line, 1)
                self.assertEqual(second[0].line, 2)
                self.assertEqual(reloaded.cache_hits, 1)
                changed = NameExtractor(model_digest="2" * 64, cache_dir=root)
                extract(changed, "000100* TOKENALPHA\n", "FIRST.CBL")
                self.assertEqual(call.call_count, 2)
