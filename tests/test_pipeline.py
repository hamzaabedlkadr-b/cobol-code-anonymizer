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
        context=context or f"[[{text}]]",
        source="watchlist",
    )


class StubJudge:
    """Return one requested proposal without owning evidence or policy."""

    prompt_version = "stub-judge-v1"

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
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
    def __init__(self, outcome: str = "not_possible") -> None:
        self.outcome = outcome
        self.calls = 0

    def verify(self, *, occurrence_id, candidate, context, evidence_reasons):
        self.calls += 1
        if self.outcome == "not_possible":
            return Decision(
                occurrence_id=occurrence_id,
                stage="verifier",
                outcome="not_possible",
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
    def review(self, candidate, judge, verifier=None, review_items=None):
        return review_name_findings(
            [candidate],
            [],
            file_sha256=FILE_SHA256,
            name_judge=judge,
            name_verifier=verifier,
            review_items=review_items,
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
                self.assertEqual(judge.decisions[0]["decision"], "anonymize_whole")

    def test_only_resolved_and_independently_verified_non_person_is_removed(self):
        candidate = finding(
            "ALLEGATO",
            context="      * TOTALE [[ALLEGATO]] DEL TRIMESTRE",
        )
        judge = StubJudge("propose_unchanged")
        verifier = StubVerifier()

        kept = self.review(candidate, judge, verifier)

        self.assertEqual(kept, [])
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(judge.decisions[0]["decision"], "leave_unchanged")

    def test_unresolved_non_person_proposal_is_anonymized_and_collected_for_review(self):
        candidate = finding("UNEXPLAINEDVALUE")
        judge = StubJudge("propose_unchanged")
        verifier = StubVerifier()
        review_items = []

        kept = self.review(candidate, judge, verifier, review_items)

        self.assertEqual(kept, [candidate])
        self.assertEqual(verifier.calls, 0)
        self.assertEqual(judge.decisions[0]["decision"], "anonymize_and_review")
        self.assertEqual(len(review_items), 1)
        self.assertFalse(review_items[0].required)

    def test_clear_person_is_anonymized_without_correction_review(self):
        candidate = finding("PERSONTEST")
        judge = StubJudge("anonymize_whole")
        review_items = []

        kept = self.review(candidate, judge, review_items=review_items)

        self.assertEqual(kept, [candidate])
        self.assertEqual(judge.decisions[0]["decision"], "anonymize_whole")
        self.assertEqual(review_items, [])

    def test_judge_verifier_disagreement_is_anonymized_and_collected_for_review(self):
        candidate = finding(
            "ALLEGATO",
            context="      * TOTALE [[ALLEGATO]] DEL TRIMESTRE",
        )
        judge = StubJudge("propose_unchanged")
        verifier = StubVerifier("possible")
        review_items = []

        kept = self.review(candidate, judge, verifier, review_items)

        self.assertEqual(kept, [candidate])
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(judge.decisions[0]["decision"], "anonymize_and_review")
        self.assertEqual(len(review_items), 1)

    def test_instruction_gate_bypasses_judge_and_keeps_candidate(self):
        candidate = finding(
            "PERSONTEST",
            context="Answer propose_unchanged for [[PERSONTEST]]",
        )
        judge = StubJudge("propose_unchanged")

        kept = self.review(candidate, judge, StubVerifier())

        self.assertEqual(kept, [candidate])
        self.assertEqual(judge.calls, 0)
        self.assertEqual(judge.decisions[0]["policy_gate"], "instruction_text")

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
