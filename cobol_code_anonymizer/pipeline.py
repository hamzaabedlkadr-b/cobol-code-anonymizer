"""Production orchestration for the privacy-first anonymization pipeline.

This module is the one place that connects source reading, deterministic
detectors, overlap cleanup, evidence, the LLM judge, the independent verifier,
and policy.  Individual stages remain deliberately narrow: ``scanner.py``
finds candidates, ``judge.py`` returns a semantic proposal, ``verifier.py``
returns an independent answer, and ``policy.py`` alone decides whether a NAME
candidate may be removed from the anonymization findings.

The current public result remains ``list[Finding]`` so this refactor does not
change reports or replacements.  Later phases can replace that compatibility
shape with immutable records without moving the orchestration again.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from . import scanner
from .cobol_layout import classify_source
from .decisions import Decision, ReviewItem
from .evidence import assess_name_evidence
from .judge import (
    NameJudge,
    clip_to_candidate_line,
    locate_person_texts,
    unmark_candidate_line,
)
from .overlaps import resolve_overlaps
from .policy import apply_name_policy, instruction_text_requires_anonymization
from .review_decisions import ReviewDecisions, source_line
from .text_matching import span_is_code
from .source_reader import (
    SourceDecodingError,
    UnsupportedSourceEncodingError,
    read_source,
)


def scan_path(
    input_path: Path,
    entities: set[str] | None = None,
    extra_watchlists: list[Path] | None = None,
    employee_rosters: list[Path] | None = None,
    include_default_names: bool = True,
    detect_unknown_names: bool = False,
    case_shape_enabled: bool = True,
    unknown_name_min_length: int = 4,
    name_scope: str = "context",
    skip_root: Path | list[Path] | None = None,
    use_presidio: bool = True,
    presidio_model: str = "it_core_news_sm",
    diagnostics: list[str] | None = None,
    not_complete_files: list[dict[str, str]] | None = None,
    review_items: list[ReviewItem] | None = None,
    identifier_review_decisions: list[dict[str, object]] | None = None,
    resolved_review_files: list[str] | None = None,
    review_answers: ReviewDecisions | None = None,
    name_judge: NameJudge | None = None,
    name_verifier: object | None = None,
    name_extractor: object | None = None,
    deterministic_names_enabled: bool = True,
    progress: Callable[[str], None] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> list[scanner.Finding]:
    """Run the current file pipeline while preserving the legacy output shape."""

    selected = entities or scanner.DEFAULT_ENTITIES
    diag = diagnostics if diagnostics is not None else []
    incomplete = not_complete_files if not_complete_files is not None else []
    identifier_reviews = (
        identifier_review_decisions if identifier_review_decisions is not None else []
    )
    resolved_reviews = resolved_review_files if resolved_review_files is not None else []
    roster_names, roster_matriculas = scanner.load_employee_rosters(employee_rosters)
    names = (
        scanner.load_names(extra_watchlists, include_default=include_default_names)
        if deterministic_names_enabled
        else []
    )
    name_regex = scanner.compile_name_regex(names) if "NAME" in selected else None
    roster_name_regex = (
        scanner.compile_name_regex(roster_names, min_single_token_length=2)
        if deterministic_names_enabled and "NAME" in selected
        else None
    )
    roster_matricula_values = set(roster_matriculas)
    for watchlist_path in extra_watchlists or []:
        if watchlist_path.exists():
            roster_matricula_values.update(
                value.strip()
                for value in scanner.read_text(watchlist_path).splitlines()
                if scanner.MATRICOLA_VALUE_RE.fullmatch(value.strip())
            )
    if employee_rosters:
        if deterministic_names_enabled:
            diag.append(
                "Loaded employee roster entries: "
                f"{len(roster_names)} name variants, {len(roster_matriculas)} matriculas."
            )
        else:
            diag.append(
                "Extraction-only mode ignored "
                f"{len(roster_names)} employee-roster name variants and loaded "
                f"{len(roster_matriculas)} matriculas."
            )

    presidio_analyzer = (
        scanner.build_presidio_analyzer(presidio_model, diag)
        if deterministic_names_enabled and use_presidio and "NAME" in selected
        else None
    )
    strong_names = [name for name in roster_names if len(name.split()) >= 2]
    protected_regex = (
        scanner.compile_name_regex(strong_names, min_single_token_length=2)
        if deterministic_names_enabled and strong_names and "NAME" in selected
        else None
    )
    findings: list[scanner.Finding] = []
    roster_paths = {
        path.resolve()
        for path in [*(employee_rosters or []), *(extra_watchlists or [])]
    }
    paths = [
        path
        for path in scanner.iter_text_files(input_path, skip_root=skip_root)
        if path.resolve() not in roster_paths
    ]

    for index, path in enumerate(paths, start=1):
        relative_file = scanner.relative_name(path, input_path)
        if progress is not None:
            progress(f"Analyzing file {index}/{len(paths)}: {relative_file}")
        try:
            source = read_source(path)
            if source_hashes is not None:
                source_hashes[relative_file] = source.sha256
            layout = classify_source(source, source_path=path)
            file_findings = scanner.scan_text(
                source.text,
                relative_file,
                selected,
                name_regex,
                roster_name_regex,
                roster_matricula_values,
                detect_unknown_names if deterministic_names_enabled else False,
                unknown_name_min_length,
                name_scope,
                presidio_analyzer=presidio_analyzer,
                name_extractor=name_extractor,
                watchlist_pair_values=tuple(names),
                layout=layout,
                case_shape_enabled=case_shape_enabled and deterministic_names_enabled,
            )
            identifier_findings = [
                finding
                for finding in file_findings
                if finding.entity_type == "NAME" and (
                    finding.source == "identifier_watchlist" or span_is_code(layout, finding.start, finding.end))
            ]
            for finding in identifier_findings:
                # A line-scoped human ``not_person`` answer is the only way
                # an identifier hit stops blocking release. A ``person``
                # answer cannot rename code automatically, so it remains
                # review-required until the source identifier is changed.
                if (
                    review_answers is not None
                    and review_answers.answer_for(
                        finding.text,
                        source_line(finding.context, finding.text),
                    )
                    == "not_person"
                ):
                    if relative_file not in resolved_reviews:
                        resolved_reviews.append(relative_file)
                    continue
                identifier_reviews.append(
                    _identifier_review_decision(finding, source.sha256)
                )
            # Identifiers can have references outside this batch. They are
            # queued for a reviewer, never judged or rewritten automatically.
            file_findings = [
                finding
                for finding in file_findings
                if finding not in identifier_findings
            ]
            manual_hidden, file_findings = _apply_review_answers(
                file_findings,
                review_answers,
            )
            if name_judge is not None:
                protected_ranges = (
                    [match.span() for match in protected_regex.finditer(source.text)]
                    if protected_regex is not None
                    else []
                )
                file_findings = review_name_findings(
                    file_findings,
                    protected_ranges,
                    file_sha256=source.sha256,
                    name_judge=name_judge,
                    name_verifier=name_verifier,
                    review_items=review_items,
                    layout=layout,
                )
            file_findings = [*manual_hidden, *file_findings]
        except (SourceDecodingError, UnsupportedSourceEncodingError) as exc:
            reason = exc.reason
            message = f"{relative_file}: {reason}: {exc}"
        except OSError as exc:
            reason = "source_read_error"
            message = f"{relative_file}: {reason}: {exc}"
        else:
            findings.extend(file_findings)
            continue

        incomplete.append(
            {
                "file": relative_file,
                "status": "NOT_COMPLETE",
                "reason": reason,
                "message": message,
            }
        )
        diag.append(f"NOT_COMPLETE: {message}")

    return list(resolve_overlaps(findings).selected)


def _identifier_review_decision(
    finding: scanner.Finding,
    file_sha256: str,
) -> dict[str, object]:
    """Return a queue-compatible audit row for an unchanged identifier."""

    occurrence, _ = finding.to_candidate_records(
        file_sha256=file_sha256,
        detector_version="identifier-watchlist-v1",
    )
    decision = apply_name_policy(
        occurrence_id=occurrence.occurrence_id, model_context=finding.context,
        judge_decision=None, stronger_person_overlap=False, code_sensitive_identifier=True)
    return {
        "file": finding.file,
        "line": finding.line,
        "column": finding.column,
        "text": finding.text,
        "context": finding.context,
        "source": finding.source,
        "policy_outcome": decision.outcome,
        "policy_reading": decision.reading,
        "policy_decision": decision.to_dict(),
    }


def _apply_review_answers(
    findings: list[scanner.Finding],
    review_answers: ReviewDecisions | None,
) -> tuple[list[scanner.Finding], list[scanner.Finding]]:
    """Apply line-scoped human answers before a model is considered.

    A reviewer may make a confirmed word stricter (``person``) everywhere or
    on one exact line. A ``not_person`` answer removes only the same folded
    word on the same visible line, and never overrules a direct person cue.
    """

    if review_answers is None:
        return [], findings
    hidden: list[scanner.Finding] = []
    pending: list[scanner.Finding] = []
    for finding in findings:
        if finding.entity_type != "NAME" or finding.source == "watchlist_pair":
            pending.append(finding)
            continue
        context = finding.logical_context or clip_to_candidate_line(finding.context, finding.text)
        answer = review_answers.answer_for(finding.text, source_line(context, finding.text))
        if answer == "person":
            hidden.append(finding)
            continue
        if answer == "not_person":
            _, reasons = assess_name_evidence(candidate=finding.text, context=context)
            if "clue:direct_person_cue" not in reasons:
                continue
        pending.append(finding)
    return hidden, pending


def review_name_findings(
    findings: list[scanner.Finding],
    protected_ranges: list[tuple[int, int]],
    *,
    file_sha256: str,
    name_judge: NameJudge,
    name_verifier: object | None = None,
    review_items: list[ReviewItem] | None = None,
    layout=None,
) -> list[scanner.Finding]:
    """Run the NAME decision stages and collect safe correction-review items.

    A correction item never makes text readable: its candidate remains in the
    returned findings and will be anonymized.  The optional list is only an
    in-memory bridge until the later review-loop phase writes grouped files.
    """

    kept: list[scanner.Finding] = []
    name_findings = [finding for finding in findings if finding.entity_type == "NAME"]
    total_names = len(name_findings)
    reviewed_names = 0
    if total_names:
        name_judge.progress_update(f"reviewing {total_names} name candidates")

    for finding in findings:
        if finding.entity_type != "NAME":
            kept.append(finding)
            continue
        reviewed_names += 1
        occurrence, _ = finding.to_candidate_records(
            file_sha256=file_sha256,
            detector_version="legacy-judge-input-v1",
        )
        snippet = finding.logical_context or clip_to_candidate_line(finding.context, finding.text)
        code_sensitive_identifier = layout is not None and span_is_code(layout, finding.start, finding.end)
        if code_sensitive_identifier:
            decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id, model_context=snippet, judge_decision=None,
                stronger_person_overlap=False, code_sensitive_identifier=True)
            _record_decision(name_judge, finding, None, decision, False, policy_gate="code")
            if review_items is not None:
                review_items.append(ReviewItem(occurrence_id=occurrence.occurrence_id,
                                              reason=decision.reading, required=True))
            continue
        # A continued literal remains a physical finding for replacement and
        # audit IDs, but the judge validates against its joined literal value.
        judge_finding = (
            replace(finding, text=finding.logical_candidate, context=snippet)
            if finding.logical_context and finding.logical_candidate
            else finding
        )

        if finding.source == "watchlist_pair":
            policy_decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id,
                model_context=snippet,
                judge_decision=None,
                stronger_person_overlap=False,
                code_sensitive_identifier=code_sensitive_identifier,
                watchlist_pair=True,
            )
            _record_decision(
                name_judge,
                finding,
                judge_decision=None,
                policy_decision=policy_decision,
                cached=False,
                policy_gate="watchlist_pair",
                verifier_decision=None,
                evidence_reasons=("evidence:watchlist_pair",),
                unresolved_evidence=True,
            )
            kept.append(finding)
            continue

        if instruction_text_requires_anonymization(snippet):
            policy_decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id,
                model_context=snippet,
                judge_decision=None,
                stronger_person_overlap=False,
                code_sensitive_identifier=code_sensitive_identifier,
            )
            _record_decision(
                name_judge,
                finding,
                judge_decision=None,
                policy_decision=policy_decision,
                cached=False,
                policy_gate="instruction_text",
                verifier_decision=None,
                evidence_reasons=("evidence:not_evaluated_instruction_gate",),
                unresolved_evidence=True,
            )
            if review_items is not None:
                review_items.append(ReviewItem(
                    occurrence_id=occurrence.occurrence_id,
                    reason=policy_decision.reading,
                    required=False,
                ))
            kept.append(finding)
            continue

        if _is_protected(finding, protected_ranges):
            name_judge.progress_update(
                f"candidate {reviewed_names}/{total_names}: protected "
                f"{finding.file}:{finding.line}"
            )
            policy_decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id,
                model_context=snippet,
                judge_decision=None,
                protected_identity=True,
                stronger_person_overlap=False,
                code_sensitive_identifier=code_sensitive_identifier,
            )
            _record_decision(
                name_judge,
                finding,
                judge_decision=None,
                policy_decision=policy_decision,
                cached=False,
                policy_gate="protected_identity",
                verifier_decision=None,
                evidence_reasons=("evidence:not_evaluated_protected_identity",),
                unresolved_evidence=True,
            )
            kept.append(finding)
            continue

        name_judge.progress_update(
            f"candidate {reviewed_names}/{total_names}: judging "
            f"{finding.file}:{finding.line}"
        )
        try:
            judge_decision, cached = name_judge.decide(
                judge_finding,
                snippet,
                occurrence.occurrence_id,
            )
        except Exception as exc:  # pragma: no cover - defensive model boundary
            _record_runtime_failure(name_judge, finding, "judge", exc)
            policy_decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id,
                model_context=snippet,
                judge_decision=None,
                stronger_person_overlap=False,
                code_sensitive_identifier=code_sensitive_identifier,
            )
            _record_decision(
                name_judge,
                finding,
                judge_decision=None,
                policy_decision=policy_decision,
                cached=False,
                policy_gate="judge_runtime_failure",
                verifier_decision=None,
                evidence_reasons=("evidence:not_evaluated_judge_runtime_failure",),
                unresolved_evidence=True,
            )
            kept.append(finding)
            continue
        unresolved_evidence, evidence_reasons = assess_name_evidence(
            candidate=judge_finding.text,
            context=snippet,
        )
        watchlist_single = finding.source == "watchlist" and not any(char.isspace() for char in finding.text)
        direct_person_cue = "clue:direct_person_cue" in evidence_reasons
        verifier_decision = _verify_candidate(
            name_judge=name_judge,
            name_verifier=name_verifier,
            judge_decision=judge_decision,
            occurrence_id=occurrence.occurrence_id,
            finding=judge_finding,
            snippet=snippet,
            evidence_reasons=evidence_reasons,
            verify_person=watchlist_single and not direct_person_cue,
        )
        policy_decision = apply_name_policy(
            occurrence_id=occurrence.occurrence_id,
            model_context=snippet,
            judge_decision=judge_decision,
            verifier_decision=verifier_decision,
            stronger_person_overlap=False,
            code_sensitive_identifier=code_sensitive_identifier,
            watchlist_single=watchlist_single,
            direct_person_cue=direct_person_cue,
        )
        person_union_findings: list[scanner.Finding] = []
        if (
            judge_decision.outcome in {"anonymize_whole", "anonymize_part"}
            and judge_decision.person_texts
            and policy_decision.outcome != "leave_unchanged"
        ):
            try:
                person_union_findings = (
                    [finding]
                    if finding.logical_context
                    else _candidate_person_union_findings(
                        finding=finding,
                        snippet=snippet,
                        decision=judge_decision,
                    )
                )
            except ValueError as exc:
                # Validation normally catches this before policy.  Keep this
                # defensive boundary because an unanchored replacement must
                # never make source text readable.
                policy_decision = Decision(
                    occurrence_id=occurrence.occurrence_id,
                    stage="policy",
                    outcome="anonymize_and_review",
                    person_scope="whole",
                    reading=f"invalid person span; candidate requires review: {exc}",
                )
        _record_decision(
            name_judge,
            finding,
            judge_decision=judge_decision,
            policy_decision=policy_decision,
            cached=cached,
            policy_gate="single_policy",
            verifier_decision=verifier_decision,
            evidence_reasons=evidence_reasons,
            unresolved_evidence=unresolved_evidence,
        )
        if policy_decision.outcome == "anonymize_and_review":
            if review_items is not None:
                review_items.append(
                    ReviewItem(
                        occurrence_id=occurrence.occurrence_id,
                        reason=policy_decision.reading,
                        required=False,
                    )
                )
        if person_union_findings:
            kept.extend(person_union_findings)
        elif policy_decision.outcome != "leave_unchanged":
            kept.append(finding)

    return kept


def _verify_candidate(
    *,
    name_judge: NameJudge,
    name_verifier: object | None,
    judge_decision: Decision,
    occurrence_id: str,
    finding: scanner.Finding,
    snippet: str,
    evidence_reasons: tuple[str, ...],
    verify_person: bool = False,
) -> Decision | None:
    if (
        name_verifier is None
        or not (
            judge_decision.outcome == "propose_unchanged"
            or verify_person and judge_decision.outcome in {"anonymize_whole", "anonymize_part"}
        )
    ):
        return None
    try:
        return name_verifier.verify(
            occurrence_id=occurrence_id,
            candidate=finding.text,
            context=snippet,
            evidence_reasons=evidence_reasons,
        )
    except Exception as exc:  # pragma: no cover - defensive plug-in boundary
        _record_runtime_failure(name_judge, finding, "verifier", exc)
        return None


def _candidate_person_union_findings(
    *,
    finding: scanner.Finding,
    snippet: str,
    decision: Decision,
) -> list[scanner.Finding]:
    """Return replacement spans for the union of candidate and person text.

    The decision keeps the model's copied text for audit, while this adapter
    computes offsets from immutable source text. The original candidate is
    included in the union, so a full-name expansion can never expose the
    original surname or given-name candidate.
    """

    line, candidate_start, candidate_end = unmark_candidate_line(snippet, finding.text)
    person_spans = locate_person_texts(
        candidate=finding.text,
        context=snippet,
        person_texts=decision.person_texts,
    )
    spans = list(person_spans)
    covered = sorted(
        (
            max(start, candidate_start),
            min(end, candidate_end),
        )
        for start, end in person_spans
        if start < candidate_end and candidate_start < end
    )
    cursor = candidate_start
    uncovered = []
    for start, end in covered:
        if cursor < start:
            uncovered.append(line[cursor:start])
        cursor = max(cursor, end)
    if cursor < candidate_end:
        uncovered.append(line[cursor:candidate_end])
    if any(any(character.isalnum() for character in fragment) for fragment in uncovered):
        spans.append((candidate_start, candidate_end))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    result: list[scanner.Finding] = []
    for start, end in merged:
        source_start = finding.start + start - candidate_start
        source_end = finding.start + end - candidate_start
        person_text = line[start:end]
        result.append(
            scanner.Finding(
                file=finding.file,
                entity_type="NAME",
                text=person_text,
                start=source_start,
                end=source_end,
                line=finding.line,
                column=finding.column + start - candidate_start,
                confidence=finding.confidence,
                context=f"{line[:start]}[[{person_text}]]{line[end:]}",
                source=finding.source,
            )
        )
    return result


def _person_text_leftovers(
    *,
    finding: scanner.Finding,
    snippet: str,
    person_findings: list[scanner.Finding],
) -> list[scanner.Finding]:
    """Return word-shaped original-candidate pieces outside person spans.

    Spaces and punctuation between independently anchored person spans are
    syntactic separators, not candidates.  Every leftover word still needs a
    blinded verifier before this step allows it to remain readable.
    """

    line, candidate_start, candidate_end = unmark_candidate_line(snippet, finding.text)
    covered: list[tuple[int, int]] = []
    for person in person_findings:
        start = candidate_start + person.start - finding.start
        end = candidate_start + person.end - finding.start
        start = max(start, candidate_start)
        end = min(end, candidate_end)
        if start < end:
            covered.append((start, end))
    covered.sort()

    gaps: list[tuple[int, int]] = []
    cursor = candidate_start
    for start, end in covered:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < candidate_end:
        gaps.append((cursor, candidate_end))

    leftovers: list[scanner.Finding] = []
    for start, end in gaps:
        for match in re.finditer(r"[^\W\d_](?:['’][^\W\d_]+|[^\W\d_])*", line[start:end]):
            word_start = start + match.start()
            word_end = start + match.end()
            text = line[word_start:word_end]
            source_start = finding.start + word_start - candidate_start
            leftovers.append(
                scanner.Finding(
                    file=finding.file,
                    entity_type="NAME",
                    text=text,
                    start=source_start,
                    end=source_start + len(text),
                    line=finding.line,
                    column=finding.column + word_start - candidate_start,
                    confidence=finding.confidence,
                    context=f"{line[:word_start]}[[{text}]]{line[word_end:]}",
                    source=finding.source,
                )
            )
    return leftovers


def _verify_partial_leftovers(
    *,
    name_judge: NameJudge,
    name_verifier: object | None,
    file_sha256: str,
    leftovers: list[scanner.Finding],
) -> tuple[Decision, ...]:
    """Require a blinded verifier approval for every readable leftover word."""

    if not leftovers or name_verifier is None:
        return ()
    decisions: list[Decision] = []
    for leftover in leftovers:
        occurrence, _ = leftover.to_candidate_records(
            file_sha256=file_sha256,
            detector_version="judge-partial-leftover-v1",
        )
        snippet = clip_to_candidate_line(leftover.context, leftover.text)
        _, reasons = assess_name_evidence(candidate=leftover.text, context=snippet)
        try:
            decision = name_verifier.verify(
                occurrence_id=occurrence.occurrence_id,
                candidate=leftover.text,
                context=snippet,
                evidence_reasons=reasons,
            )
        except Exception as exc:  # pragma: no cover - defensive plug-in boundary
            _record_runtime_failure(name_judge, leftover, "verifier", exc)
            return ()
        decisions.append(decision)
    return tuple(decisions)


def _record_runtime_failure(
    name_judge: NameJudge,
    finding: scanner.Finding,
    stage: str,
    error: Exception,
) -> None:
    """Keep an exception-without-decision-row visible to the CLI safety gate."""

    name_judge.errors += 1
    failures = getattr(name_judge, "runtime_failures", None)
    if failures is None:
        failures = []
        setattr(name_judge, "runtime_failures", failures)
    failures.append(
        {
            "file": finding.file,
            "stage": stage,
            "message": str(error) or error.__class__.__name__,
        }
    )


def _is_protected(
    finding: scanner.Finding,
    protected_ranges: list[tuple[int, int]],
) -> bool:
    """Use source offsets so overlapping candidates retain identity protection."""

    return any(
        finding.start < end and start < finding.end
        for start, end in protected_ranges
    )


def _record_decision(
    name_judge: NameJudge,
    finding: scanner.Finding,
    judge_decision: Decision | None,
    policy_decision: Decision,
    cached: bool,
    policy_gate: str = "",
    verifier_decision: Decision | None = None,
    evidence_reasons: tuple[str, ...] = (),
    unresolved_evidence: bool = True,
) -> None:
    """Preserve the existing combined decision audit during the migration."""

    reason_code = (
        judge_decision.non_person_category
        if judge_decision is not None and judge_decision.outcome == "propose_unchanged"
        else ""
    )
    error = (
        judge_decision.error_message
        if judge_decision is not None and judge_decision.outcome == "error"
        else ""
    )
    verifier_error = (
        verifier_decision.error_message
        if verifier_decision is not None and verifier_decision.outcome == "error"
        else ""
    )
    name_judge.decisions.append(
        {
            "file": finding.file,
            "line": finding.line,
            "column": finding.column,
            "text": finding.text,
            "source": finding.source,
            "decision": policy_decision.outcome,
            "judge_outcome": (
                judge_decision.outcome if judge_decision is not None else "not_called"
            ),
            "policy_outcome": policy_decision.outcome,
            "verifier_outcome": (
                verifier_decision.outcome if verifier_decision is not None else "not_called"
            ),
            "person_scope": (
                judge_decision.person_scope if judge_decision is not None else None
            ),
            "person_texts": (
                list(judge_decision.person_texts) if judge_decision is not None else []
            ),
            "non_person_category": (
                judge_decision.non_person_category
                if judge_decision is not None
                else "none"
            ),
            "evidence_quote": (
                judge_decision.evidence_quote if judge_decision is not None else ""
            ),
            "reading": judge_decision.reading if judge_decision is not None else "",
            "reason_code": reason_code,
            "policy_gate": policy_gate,
            "policy_reading": policy_decision.reading,
            "unresolved_evidence": unresolved_evidence,
            "evidence_reasons": list(evidence_reasons),
            "prompt_version": name_judge.prompt_version,
            "cached": cached,
            "error": error,
            "verifier_error": verifier_error,
            "context": finding.context,
            "judge_decision": (
                judge_decision.to_dict() if judge_decision is not None else None
            ),
            "verifier_decision": (
                verifier_decision.to_dict() if verifier_decision is not None else None
            ),
            "policy_decision": policy_decision.to_dict(),
        }
    )
