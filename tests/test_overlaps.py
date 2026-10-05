"""Unittest regression tests for overlap grouping and compatibility cleanup."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_code_anonymizer.judge import NameJudge
from cobol_code_anonymizer.llm import LlmJsonResult
from cobol_code_anonymizer.overlaps import (
    ENTITY_PRIORITY,
    build_overlap_groups,
    resolve_overlaps,
)
from cobol_code_anonymizer.pipeline import scan_path
from cobol_code_anonymizer.scanner import Finding


def finding(
    start: int,
    end: int,
    *,
    file: str = "PROGRAM.CBL",
    entity_type: str = "NAME",
    source: str = "watchlist",
) -> Finding:
    return Finding(
        file=file,
        entity_type=entity_type,
        text=f"VALUE_{start}_{end}",
        start=start,
        end=end,
        line=1,
        column=start + 1,
        confidence=0.7,
        context="context",
        source=source,
    )


def legacy_reference(findings: list[Finding]) -> list[Finding]:
    """Frozen copy of the pre-group scanner selection for equivalence tests."""

    ordered = sorted(
        findings,
        key=lambda item: (
            item.file,
            item.start,
            -ENTITY_PRIORITY.get(item.entity_type, 0),
            -(item.end - item.start),
        ),
    )
    kept: list[Finding] = []
    for item in ordered:
        overlaps = [
            existing
            for existing in kept
            if existing.file == item.file
            and not (item.end <= existing.start or item.start >= existing.end)
        ]
        if not overlaps:
            kept.append(item)
            continue
        best = max(
            overlaps,
            key=lambda existing: (
                ENTITY_PRIORITY.get(existing.entity_type, 0),
                existing.end - existing.start,
            ),
        )
        item_rank = (ENTITY_PRIORITY.get(item.entity_type, 0), item.end - item.start)
        best_rank = (ENTITY_PRIORITY.get(best.entity_type, 0), best.end - best.start)
        if item_rank > best_rank:
            kept = [existing for existing in kept if existing not in overlaps]
            kept.append(item)
    return sorted(kept, key=lambda item: (item.file, item.start, item.end))


class OverlapTests(unittest.TestCase):
    def test_groups_preserve_transitive_overlaps_and_detector_provenance(self):
        first = finding(0, 5, source="watchlist")
        bridge = finding(4, 9, source="presidio_spacy")
        last = finding(8, 12, source="llm_extraction")

        groups = build_overlap_groups([last, first, bridge])

        self.assertEqual(len(groups), 1)
        self.assertEqual((groups[0].start, groups[0].end), (0, 12))
        self.assertEqual(groups[0].members, (first, bridge, last))
        self.assertEqual(
            {item.source for item in groups[0].members},
            {"watchlist", "presidio_spacy", "llm_extraction"},
        )

    def test_adjacent_and_different_file_spans_are_separate_groups(self):
        groups = build_overlap_groups(
            [finding(0, 5), finding(5, 10), finding(0, 5, file="OTHER.CBL")]
        )

        self.assertEqual(
            [(group.file, group.start, group.end) for group in groups],
            [
                ("OTHER.CBL", 0, 5),
                ("PROGRAM.CBL", 0, 5),
                ("PROGRAM.CBL", 5, 10),
            ],
        )

    def test_exact_duplicate_spans_remain_separate_group_members(self):
        watchlist = finding(3, 10, source="watchlist")
        spacy = finding(3, 10, source="presidio_spacy")

        groups = build_overlap_groups([watchlist, spacy])

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].members, (watchlist, spacy))

    def test_compatibility_selection_matches_old_algorithm_for_random_spans(self):
        randomizer = random.Random(41)
        entity_types = tuple(ENTITY_PRIORITY)
        for iteration in range(300):
            with self.subTest(iteration=iteration):
                generated = []
                for index in range(randomizer.randint(0, 25)):
                    start = randomizer.randint(0, 80)
                    end = start + randomizer.randint(1, 20)
                    generated.append(
                        finding(
                            start,
                            end,
                            file=f"P{randomizer.randint(1, 3)}.CBL",
                            entity_type=randomizer.choice(entity_types),
                            source=f"detector-{index}",
                        )
                    )

                self.assertEqual(
                    list(resolve_overlaps(generated).selected),
                    legacy_reference(generated),
                )

    def test_two_token_watchlist_rejection_remains_anonymized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PROGRAM.CBL"
            text = "      * TOTALE: TOKENALPHA TOKENBETA\n"
            source.write_text(text, encoding="utf-8")
            watchlist = root / "watchlist.txt"
            watchlist.write_text("TOKENALPHA TOKENBETA\n", encoding="utf-8")

            judge = NameJudge()
            rejected_as_common_word = LlmJsonResult(
                parsed={
                    "decision": "propose_unchanged",
                    "person_scope": "none",
                    "person_texts": [],
                    "non_person_category": "common_word",
                    "evidence_quote": "TOTALE",
                    "reading": "The text is used as a non-person label.",
                },
                content="",
                latency_s=0.0,
                schema_ok=True,
            )
            with patch(
                "cobol_code_anonymizer.judge.call_ollama_json",
                return_value=rejected_as_common_word,
            ):
                findings = scan_path(
                    source,
                    entities={"NAME"},
                    extra_watchlists=[watchlist],
                    include_default_names=False,
                    use_presidio=False,
                    name_judge=judge,
                )

        self.assertEqual([item.text for item in findings], ["TOKENALPHA TOKENBETA"])
        self.assertEqual(findings[0].source, "watchlist")
        self.assertEqual(judge.decisions[0]["decision"], "anonymize_and_review")
        self.assertEqual(
            judge.decisions[0]["judge_outcome"], "propose_unchanged"
        )
        self.assertEqual(
            (findings[0].start, findings[0].end),
            (text.index("TOKENALPHA"), text.index("TOKENBETA") + len("TOKENBETA")),
        )

    def test_adjacent_single_token_findings_stay_separate_and_anonymized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "PROGRAM.CBL"
            source.write_text(
                "      * TOTALE: TOKENALPHA TOKENBETA\n",
                encoding="utf-8",
            )
            watchlist = root / "watchlist.txt"
            watchlist.write_text("TOKENALPHA\nTOKENBETA\n", encoding="utf-8")
            judge = NameJudge()
            proposal = LlmJsonResult(
                parsed={
                    "decision": "propose_unchanged",
                    "person_scope": "none",
                    "person_texts": [],
                    "non_person_category": "common_word",
                    "evidence_quote": "TOTALE",
                    "reading": "The text is used as a non-person label.",
                },
                content="",
                latency_s=0.0,
                schema_ok=True,
            )
            with patch(
                "cobol_code_anonymizer.judge.call_ollama_json",
                return_value=proposal,
            ):
                findings = scan_path(
                    source,
                    entities={"NAME"},
                    extra_watchlists=[watchlist],
                    include_default_names=False,
                    use_presidio=False,
                    name_judge=judge,
                )

        self.assertEqual(
            [item.text for item in findings],
            ["TOKENALPHA", "TOKENBETA"],
        )
        self.assertEqual(
            [row["decision"] for row in judge.decisions],
            ["anonymize_and_review", "anonymize_and_review"],
        )


if __name__ == "__main__":
    unittest.main()
