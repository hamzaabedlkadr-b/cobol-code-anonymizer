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
            {"possible", "not_possible", "unsure"},
        )

    def test_not_possible_needs_anchored_evidence_and_reading(self):
        decision = validate_verifier_response(
            {
                "decision": "not_possible",
                "evidence_quote": "TOTALE",
                "reading": "It is an accounting label.",
            },
            occurrence_id=OCCURRENCE_ID,
            context=CONTEXT,
            model_digest=MODEL_DIGEST,
        )
        self.assertEqual(decision.outcome, "not_possible")

        with self.assertRaisesRegex(ValueError, "copied exactly"):
            validate_verifier_response(
                {
                    "decision": "not_possible",
                    "evidence_quote": "missing",
                    "reading": "It is an accounting label.",
                },
                occurrence_id=OCCURRENCE_ID,
                context=CONTEXT,
                model_digest=MODEL_DIGEST,
            )

    def test_prompt_is_blind_to_the_judge_answer(self):
        messages = build_messages(
            context=CONTEXT,
            candidate="ALLEGATO",
            evidence_reasons=("word:ALLEGATO:payroll_admin",),
        )
        combined = "\n".join(message["content"] for message in messages)
        self.assertIn("word:ALLEGATO:payroll_admin", combined)
        self.assertNotIn("propose_unchanged", combined)
        self.assertNotIn("judge outcome", combined.lower())


if __name__ == "__main__":
    unittest.main()
