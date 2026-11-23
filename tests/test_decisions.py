import unittest

from cobol_code_anonymizer.decisions import (
    DECISION_RULE_FIELDS,
    OUTCOME_RULES,
    OUTCOMES_BY_STAGE,
    Decision,
)


OCCURRENCE_ID = "a" * 64
DETECTION_ID = "b" * 64
FILE_HASH = "c" * 64

EMPTY_DECISION_FIELDS = {
    "person_scope": None,
    "person_texts": (),
    "non_person_category": "none",
    "evidence_quote": "",
    "reading": "",
}
PRESENT_DECISION_FIELDS = {
    "person_scope": "whole",
    "person_texts": ("PERSON_TEXT",),
    "non_person_category": "common_word",
    "evidence_quote": "source evidence",
    "reading": "A short explanation.",
}
INVALID_DECISION_FIELDS = {
    "person_scope": 1,
    "person_texts": "not-a-sequence",
    "non_person_category": 1,
    "evidence_quote": None,
    "reading": None,
}


def valid_decision_arguments(stage, outcome):
    """Build the smallest valid record directly from one declarative rule row."""

    values = dict(EMPTY_DECISION_FIELDS)
    for field_name, rule in OUTCOME_RULES[(stage, outcome)].items():
        if rule.kind == "required":
            values[field_name] = PRESENT_DECISION_FIELDS[field_name]
        elif rule.kind == "must_equal":
            values[field_name] = rule.value

    arguments = {
        "occurrence_id": OCCURRENCE_ID,
        "stage": stage,
        "outcome": outcome,
        **values,
    }
    if stage in {"judge", "verifier"}:
        arguments.update(
            model_digest="sha256:model-digest",
            prompt_version=f"{stage}-v1",
        )
    if stage == "manual":
        arguments.update(
            reviewer="reviewer-id",
            timestamp="2026-10-05T10:00:00+02:00",
        )
    if outcome == "error":
        arguments["error_message"] = "stage failed"
    return arguments


def value_that_breaks(field_name, rule):
    """Change only one field so its rule, or its optional type, becomes invalid."""

    if rule.kind == "required":
        return EMPTY_DECISION_FIELDS[field_name]
    if rule.kind == "forbidden":
        return PRESENT_DECISION_FIELDS[field_name]
    if rule.kind == "must_equal":
        if field_name == "person_scope":
            return "partial" if rule.value != "partial" else "whole"
        raise AssertionError(f"No alternate test value for {field_name}")
    return INVALID_DECISION_FIELDS[field_name]


class DecisionRecordTests(unittest.TestCase):

    def test_decision_json_round_trip_preserves_model_fields(self):
        decision = Decision(
            occurrence_id=OCCURRENCE_ID,
            stage="judge",
            outcome="anonymize_part",
            person_scope="partial",
            person_texts=("PERSON_TEXT",),
            evidence_quote="source evidence",
            reading="Only part of the occurrence is person text.",
            model_digest="sha256:model-digest",
            prompt_version="judge-v1",
        )

        self.assertEqual(Decision.from_json(decision.to_json()), decision)



    def test_judge_answer_requires_a_consistent_validated_shape(self):
        common = {
            "occurrence_id": OCCURRENCE_ID,
            "stage": "judge",
            "model_digest": "sha256:model-digest",
            "prompt_version": "judge-v1",
        }
        with self.assertRaisesRegex(ValueError, "person_scope='whole'"):
            Decision(
                **common,
                outcome="anonymize_whole",
                person_scope="unsure",
            )
        with self.assertRaisesRegex(ValueError, "non_person_category"):
            Decision(
                **common,
                outcome="propose_unchanged",
                person_scope="none",
                evidence_quote="source evidence",
                reading="A short explanation.",
            )
        decision = Decision(
            **common,
            outcome="propose_unchanged",
            person_scope="none",
            non_person_category="common_word",
            reading="A short explanation.",
        )
        self.assertEqual(decision.evidence_quote, "")

    def test_verifier_and_manual_decisions_record_stage_provenance(self):
        verifier = Decision(
            occurrence_id=OCCURRENCE_ID,
            stage="verifier",
            outcome="not_person",
            evidence_quote="source evidence",
            reading="No possible person remains.",
            model_digest="sha256:verifier-digest",
            prompt_version="verifier-v1",
        )
        manual = Decision(
            occurrence_id=OCCURRENCE_ID,
            stage="manual",
            outcome="checked_non_person",
            person_scope="none",
            non_person_category="common_word",
            reading="The occurrence was checked manually.",
            reviewer="reviewer-id",
            timestamp="2026-10-05T10:00:00+02:00",
        )

        self.assertEqual(Decision.from_json(verifier.to_json()), verifier)
        self.assertEqual(Decision.from_json(manual.to_json()), manual)

    def test_every_stage_outcome_has_one_complete_rule_row(self):
        expected_pairs = {
            (stage, outcome)
            for stage, outcomes in OUTCOMES_BY_STAGE.items()
            for outcome in outcomes
        }

        self.assertEqual(set(OUTCOME_RULES), expected_pairs)
        for pair, rules in OUTCOME_RULES.items():
            with self.subTest(pair=pair):
                self.assertEqual(set(rules), set(DECISION_RULE_FIELDS))

    def test_every_outcome_rule_row_accepts_a_valid_example(self):
        for stage, outcome in OUTCOME_RULES:
            with self.subTest(stage=stage, outcome=outcome):
                Decision(**valid_decision_arguments(stage, outcome))

    def test_only_readable_model_outcomes_require_reading_not_copied_quotes(self):
        safe_outcomes = {
            ("judge", "anonymize_whole"),
            ("judge", "anonymize_part"),
            ("judge", "uncertain"),
            ("verifier", "person"),
            ("verifier", "unsure"),
        }
        readable_outcomes = {
            ("judge", "propose_unchanged"),
            ("verifier", "not_person"),
        }

        for pair in safe_outcomes:
            with self.subTest(pair=pair, safety="anonymize"):
                self.assertEqual(
                    OUTCOME_RULES[pair]["evidence_quote"].kind,
                    "optional",
                )
                self.assertEqual(OUTCOME_RULES[pair]["reading"].kind, "optional")
        for pair in readable_outcomes:
            with self.subTest(pair=pair, safety="may_leave_readable"):
                self.assertEqual(OUTCOME_RULES[pair]["evidence_quote"].kind, "optional")
                self.assertEqual(OUTCOME_RULES[pair]["reading"].kind, "required")

    def test_every_field_rule_rejects_a_single_invalid_field(self):
        for (stage, outcome), rules in OUTCOME_RULES.items():
            for field_name, rule in rules.items():
                arguments = valid_decision_arguments(stage, outcome)
                arguments[field_name] = value_that_breaks(field_name, rule)
                with self.subTest(
                    stage=stage,
                    outcome=outcome,
                    field=field_name,
                    rule=rule.kind,
                ):
                    with self.assertRaises(ValueError):
                        Decision(**arguments)

    def test_stage_provenance_is_required_and_kept_separate(self):
        with self.assertRaisesRegex(ValueError, "model_digest"):
            Decision(
                occurrence_id=OCCURRENCE_ID,
                stage="verifier",
                outcome="unsure",
                prompt_version="verifier-v1",
            )
        with self.assertRaisesRegex(ValueError, "reviewer"):
            Decision(
                occurrence_id=OCCURRENCE_ID,
                stage="manual",
                outcome="defer",
                timestamp="2026-10-05T10:00:00+02:00",
            )
        with self.assertRaisesRegex(ValueError, "timezone"):
            Decision(
                occurrence_id=OCCURRENCE_ID,
                stage="manual",
                outcome="defer",
                reviewer="reviewer-id",
                timestamp="2026-10-05T10:00:00",
            )
