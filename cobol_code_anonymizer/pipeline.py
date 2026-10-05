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

from pathlib import Path
from typing import Callable

from . import scanner
from .decisions import Decision, ReviewItem
from .evidence import assess_name_evidence
from .judge import NameJudge, clip_to_candidate_line
from .overlaps import resolve_overlaps
from .policy import apply_name_policy, instruction_text_requires_anonymization
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
    unknown_name_min_length: int = 4,
    name_scope: str = "context",
    skip_root: Path | list[Path] | None = None,
    use_presidio: bool = True,
    presidio_model: str = "it_core_news_sm",
    diagnostics: list[str] | None = None,
    not_complete_files: list[dict[str, str]] | None = None,
    review_items: list[ReviewItem] | None = None,
    name_judge: NameJudge | None = None,
    name_verifier: object | None = None,
    name_extractor: object | None = None,
    deterministic_names_enabled: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[scanner.Finding]:
    """Run the current file pipeline while preserving the legacy output shape."""

    selected = entities or scanner.DEFAULT_ENTITIES
    diag = diagnostics if diagnostics is not None else []
    incomplete = not_complete_files if not_complete_files is not None else []
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
    watchlist_values = frozenset(name.casefold() for name in names)
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
                    watchlist_values=watchlist_values,
                    review_items=review_items,
                )
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


def review_name_findings(
    findings: list[scanner.Finding],
    protected_ranges: list[tuple[int, int]],
    *,
    file_sha256: str,
    name_judge: NameJudge,
    name_verifier: object | None = None,
    watchlist_values: frozenset[str] = frozenset(),
    review_items: list[ReviewItem] | None = None,
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
        snippet = clip_to_candidate_line(finding.context, finding.text)

        if instruction_text_requires_anonymization(snippet):
            policy_decision = apply_name_policy(
                occurrence_id=occurrence.occurrence_id,
                model_context=snippet,
                judge_decision=None,
                stronger_person_overlap=False,
                unresolved_evidence=True,
                code_sensitive_identifier=False,
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
                unresolved_evidence=True,
                code_sensitive_identifier=False,
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
        judge_decision, cached = name_judge.decide(
            finding,
            snippet,
            occurrence.occurrence_id,
        )
        unresolved_evidence, evidence_reasons = assess_name_evidence(
            candidate=finding.text,
            context=snippet,
            watchlist_values=watchlist_values,
        )
        verifier_decision = _verify_non_person_proposal(
            name_judge=name_judge,
            name_verifier=name_verifier,
            judge_decision=judge_decision,
            occurrence_id=occurrence.occurrence_id,
            finding=finding,
            snippet=snippet,
            unresolved_evidence=unresolved_evidence,
            evidence_reasons=evidence_reasons,
        )
        policy_decision = apply_name_policy(
            occurrence_id=occurrence.occurrence_id,
            model_context=snippet,
            judge_decision=judge_decision,
            verifier_decision=verifier_decision,
            stronger_person_overlap=False,
            unresolved_evidence=unresolved_evidence,
            code_sensitive_identifier=False,
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
        if policy_decision.outcome != "leave_unchanged":
            kept.append(finding)

    return kept


def _verify_non_person_proposal(
    *,
    name_judge: NameJudge,
    name_verifier: object | None,
    judge_decision: Decision,
    occurrence_id: str,
    finding: scanner.Finding,
    snippet: str,
    unresolved_evidence: bool,
    evidence_reasons: tuple[str, ...],
) -> Decision | None:
    if (
        name_verifier is None
        or judge_decision.outcome != "propose_unchanged"
        or unresolved_evidence
    ):
        return None
    try:
        return name_verifier.verify(
            occurrence_id=occurrence_id,
            candidate=finding.text,
            context=snippet,
            evidence_reasons=evidence_reasons,
        )
    except Exception:  # pragma: no cover - defensive plug-in boundary
        name_judge.errors += 1
        return None


def _is_protected(
    finding: scanner.Finding,
    protected_ranges: list[tuple[int, int]],
) -> bool:
    """Use source offsets so widened candidates cannot lose identity protection."""

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
