"""Tests for the blinded independent verifier contract."""

from __future__ import annotations

import unittest

from cobol_code_anonymizer.verifier import (
    VERIFIER_RESPONSE_SCHEMA,
    build_messages,
    validate_verifier_response,
)


OCCURRENCE_ID = "a" * 64
MODEL_DIGEST = "b" * 64
CONTEXT = "      * TOTALE [[ALLEGATO]] DEL TRIMESTRE"


class NameVerifierTests(unittest.TestCase):
    def test_schema_has_only_fail_safe_semantic_outcomes(self):
        self.assertEqual(
            set(VERIFIER_RESPONSE_SCHEMA["properties"]["decision"]["enum"]),
            {"person", "not_person", "unsure"},
        )

    def test_not_person_needs_anchored_evidence_and_reading(self):
        decision = validate_verifier_response(
            {
                "decision": "not_person",
                "evidence_quote": "TOTALE",
                "reading": "It is an accounting label.",
            },
            occurrence_id=OCCURRENCE_ID,
            context=CONTEXT,
            model_digest=MODEL_DIGEST,
        )
        self.assertEqual(decision.outcome, "not_person")

        dropped = validate_verifier_response(
            {
                "decision": "not_person",
                "evidence_quote": "missing",
                "reading": "It is an accounting label.",
            },
            occurrence_id=OCCURRENCE_ID,
            context=CONTEXT,
            model_digest=MODEL_DIGEST,
        )
        self.assertEqual(dropped.evidence_quote, "")

    def test_evidence_quote_ignores_markers_and_whitespace(self):
        decision = validate_verifier_response(
            {
                "decision": "not_person",
                "evidence_quote": "TOTALE ALLEGATO DEL",
                "reading": "It is an accounting label.",
            },
            occurrence_id=OCCURRENCE_ID,
            context="      * TOTALE   [[ALLEGATO]]\tDEL TRIMESTRE",
            model_digest=MODEL_DIGEST,
        )

        self.assertEqual(decision.outcome, "not_person")

    def test_evidence_quote_tolerates_sequence_area_and_small_copy_errors(self):
        decision = validate_verifier_response(
            {
                "decision": "not_person",
                "evidence_quote": "TOTALE ALLEGAT DEL",
                "reading": "It is an accounting label.",
            },
            occurrence_id=OCCURRENCE_ID,
            context="000100 TOTALE [[ALLEGATO]] DEL TRIMESTRE",
            model_digest=MODEL_DIGEST,
        )
        self.assertEqual(decision.evidence_quote, "TOTALE ALLEGAT DEL")

    def test_quote_is_optional_for_all_outcomes(self):
        from cobol_code_anonymizer.llm import validate_json_content
        import json
        for outcome in ("person", "not_person", "unsure"):
            payload = {"decision": outcome, "reading": "Use in this line."}
            self.assertTrue(validate_json_content(json.dumps(payload), VERIFIER_RESPONSE_SCHEMA)[1])
            result = validate_verifier_response(payload, occurrence_id=OCCURRENCE_ID,
                                              context=CONTEXT, model_digest=MODEL_DIGEST)
            self.assertEqual(result.outcome, outcome)
            self.assertEqual(result.evidence_quote, "")

    def test_prompt_is_blind_to_the_judge_answer(self):
        messages = build_messages(
            context=CONTEXT,
            candidate="ALLEGATO",
        )
        combined = "\n".join(message["content"] for message in messages)
        self.assertNotIn("Deterministic evidence", combined)
        self.assertNotIn("clue:", combined)
        self.assertIn(CONTEXT, messages[1]["content"])
        self.assertNotIn("propose_unchanged", combined)
        self.assertNotIn("judge outcome", combined.lower())
        self.assertIn("IN THIS LINE", combined)
        self.assertNotIn("impossible", combined.lower())


if __name__ == "__main__":
    unittest.main()
