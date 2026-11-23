"""Unittest coverage for the one privacy policy introduced in Step 27."""

from __future__ import annotations

import unittest

from cobol_code_anonymizer.decisions import Decision
from cobol_code_anonymizer.policy import apply_name_policy


OCCURRENCE_ID = "1" * 64
OTHER_OCCURRENCE_ID = "2" * 64
MODEL_DIGEST = "model-digest-for-tests"
PROMPT_VERSION = "test-v1"


def judge_decision(outcome: str) -> Decision:
    common = {
        "occurrence_id": OCCURRENCE_ID,
        "stage": "judge",
        "outcome": outcome,
        "model_digest": MODEL_DIGEST,
        "prompt_version": PROMPT_VERSION,
    }
    if outcome == "anonymize_whole":
        return Decision(**common, person_scope="whole")
    if outcome == "anonymize_part":
        return Decision(
            **common,
            person_scope="partial",
            person_texts=("Alpha",),
        )
    if outcome == "propose_unchanged":
        return Decision(
            **common,
            person_scope="none",
            non_person_category="common_word",
            evidence_quote="TOTALE",
            reading="Used as an accounting label.",
        )
    if outcome == "uncertain":
        return Decision(**common, person_scope="unsure")
    if outcome == "error":
        return Decision(**common, error_message="model failure")
    raise AssertionError(f"unsupported test outcome: {outcome}")


def verifier_decision(outcome: str, occurrence_id: str = OCCURRENCE_ID) -> Decision:
    common = {
        "occurrence_id": occurrence_id,
        "stage": "verifier",
        "outcome": outcome,
        "model_digest": "independent-verifier-digest",
        "prompt_version": "verifier-test-v1",
    }
    if outcome == "not_person":
        return Decision(
            **common,
            evidence_quote="TOTALE",
            reading="The line uses the span as an administrative label.",
        )
    if outcome in {"person", "unsure"}:
        return Decision(**common)
    if outcome == "error":
        return Decision(**common, error_message="verifier failure")
    raise AssertionError(f"unsupported verifier outcome: {outcome}")


class NamePolicyTests(unittest.TestCase):
    def test_full_watchlist_entry_hides_without_model_verification(self):
        for judge in (None, judge_decision('propose_unchanged')):
            result = self.apply(judge, watchlist_full_name=True,
                                verifier_decision=verifier_decision('not_person'))
            self.assertEqual((result.outcome, result.reading),
                             ('anonymize_whole', 'multi-word watchlist entry; hide'))
        self.assertEqual(self.apply(None, watchlist_full_name=True, review_answer='not_person').outcome,
                         'leave_unchanged')


    def test_watchlist_text_requires_both_models_to_show(self):
        judge = judge_decision("propose_unchanged")
        for verified in (None, "not_person", "person", "unsure", "error"):
            with self.subTest(verifier=verified):
                result = self.apply(judge, watchlist_single=True,
                                    verifier_decision=verifier_decision(verified) if verified else None)
                self.assertEqual(result.outcome, "leave_unchanged" if verified == "not_person" else "anonymize_whole")
        for outcome in ("anonymize_whole", "anonymize_part", "uncertain", "error"):
            with self.subTest(judge=outcome):
                self.assertEqual(self.apply(judge_decision(outcome), watchlist_single=True).outcome,
                                 "anonymize_whole")

    def test_watchlist_context_needs_one_other_letter_word(self):
        for context, allowed in (("* [[TOKENALPHA]] DAL 2024.", True),
                                 ("* [[TOKENALPHA]] a 2024.", True),
                                 ("* [[TOKENALPHA]]", False),
                                 ("* [[TOKENALPHA]] 2024.", False)):
            with self.subTest(context=context):
                result = self.apply(judge_decision("propose_unchanged"), watchlist_single=True,
                                    model_context=context, verifier_decision=verifier_decision("not_person"))
                self.assertEqual(result.outcome, "leave_unchanged" if allowed else "anonymize_whole")

    def test_watchlist_verifier_disabled_hides_even_approved_words(self):
        for approved in (False, True):
            result = self.apply(judge_decision("propose_unchanged"), watchlist_single=True,
                                approved_word=approved, verifier_enabled=False)
            self.assertEqual((result.outcome, result.reading),
                             ("anonymize_whole", "verification disabled; hide"))
        self.assertEqual(self.apply(judge_decision("propose_unchanged"), verifier_enabled=False).outcome,
                         "leave_unchanged")

    def apply(self, judge: Decision | None, **changes) -> Decision:
        arguments = {
            "occurrence_id": OCCURRENCE_ID,
            "model_context": "      * TOTALE [[Alpha]]",
            "judge_decision": judge,
            # Production callers must state these gates explicitly.  The test
            # helper supplies the ordinary, non-blocking case by default.
            "code_sensitive_identifier": False,
        }
        arguments.update(changes)
        return apply_name_policy(**arguments)

    def test_safe_judge_outcomes_all_anonymize(self):
        for outcome in ("anonymize_whole", "uncertain"):
            with self.subTest(outcome=outcome):
                result = self.apply(judge_decision(outcome))
                self.assertEqual(result.stage, "policy")
                expected = "anonymize_whole"
                self.assertEqual(result.outcome, expected)

        partial = self.apply(judge_decision("anonymize_part"))
        self.assertEqual(partial.outcome, "anonymize_whole")

        invalid = self.apply(judge_decision("error"))
        self.assertEqual(invalid.outcome, "anonymize_whole")

    def test_evidence_and_overlap_gates_must_be_stated_by_the_caller(self):
        with self.assertRaises(TypeError):
            apply_name_policy(
                occurrence_id=OCCURRENCE_ID,
                model_context="[[Alpha]]",
                judge_decision=judge_decision("uncertain"),
            )




    def test_clear_person_is_anonymized_without_correction_outcome(self):
        result = self.apply(judge_decision("anonymize_whole"))

        self.assertEqual(result.outcome, "anonymize_whole")


    def test_stage_and_occurrence_mismatches_are_rejected(self):
        wrong_stage = Decision(
            occurrence_id=OCCURRENCE_ID,
            stage="policy",
            outcome="anonymize_whole",
            person_scope="whole",
            reading="test",
        )
        with self.assertRaisesRegex(ValueError, "stage='judge'"):
            self.apply(wrong_stage)

        with self.assertRaisesRegex(ValueError, "different occurrence"):
            self.apply(
                judge_decision("propose_unchanged"),
                verifier_decision=verifier_decision(
                    "not_person",
                    occurrence_id=OTHER_OCCURRENCE_ID,
                ),
            )


if __name__ == "__main__":
    unittest.main()
