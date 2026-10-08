"""Read, detect, decide, and collect source findings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

from . import scanner
from .cobol_layout import FIXED_COBOL, PLAIN_TEXT, classify_source
from .decisions import Decision
from .judge import (
    NameJudge,
    clip_to_candidate_line,
    locate_person_texts,
    unmark_candidate_line,
)
from .overlaps import resolve_overlaps
from .policy import NAME_POLICY_TABLE, apply_name_policy, has_minimum_watchlist_context, instruction_text_requires_anonymization
from .review_decisions import ReviewDecisions, source_line
from .text_matching import prepare_watchlist, span_is_code, name_word_spans, marked_source_line, source_findings, source_occurrence_id
from .llm import NAME_VERIFIER_ENABLED
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
    include_default_names: bool = False,
    name_scope: str = "context",
    skip_root: Path | list[Path] | None = None,
    use_presidio: bool = True,
    presidio_model: str = "it_core_news_sm",
    diagnostics: list[str] | None = None,
    not_complete_files: list[dict[str, str]] | None = None,
    identifier_review_decisions: list[dict[str, object]] | None = None,
    resolved_review_files: list[str] | None = None,
    review_answers: ReviewDecisions | None = None,
    name_judge: NameJudge | None = None,
    name_verifier: object | None = None,
    approved_words: frozenset[str] = frozenset(),
    verifier_enabled: bool = NAME_VERIFIER_ENABLED,
    name_extractor: object | None = None,
    deterministic_names_enabled: bool = True,
    progress: Callable[[str], None] | None = None,
    source_hashes: dict[str, str] | None = None,
    file_times: dict[str, float] | None = None,
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
    entries, pair_words = prepare_watchlist(names)
    watchlist_words = frozenset(scanner.fold_watchlist_value(word) for word in names)
    name_regex = scanner.compile_name_regex([word for word in names if scanner.fold_watchlist_value(word) not in entries]) if "NAME" in selected else None
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
    if deterministic_names_enabled and use_presidio and "NAME" in selected and presidio_analyzer is None:
        raise RuntimeError("spaCy/Presidio is unavailable; use --no-presidio to disable it")
    if progress is not None:
        detectors = sorted(selected - {"NAME"})
        if deterministic_names_enabled and "NAME" in selected:
            detectors.extend(["watchlist", "watchlist pairs", "folded watchlist", "identifier watchlist"])
            if presidio_analyzer is not None:
                detectors.append(f"spaCy ({presidio_model})")
        if name_extractor is not None and "NAME" in selected:
            detectors.append("LLM extractor")
        progress("Active detectors: " + (", ".join(detectors) or "none"))
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
        file_started = time.perf_counter()
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
                "all" if layout.format == PLAIN_TEXT else name_scope,
                presidio_analyzer=presidio_analyzer,
                name_extractor=name_extractor,
                watchlist_pair_values=entries, pair_words=pair_words,
                layout=layout,
            )
            file_findings = [replace(finding, context=marked_source_line(source.text, finding.start, finding.end))
                             if finding.entity_type == "NAME" else finding for finding in file_findings]
            identifier_findings = [
                finding
                for finding in file_findings
                if finding.entity_type == "NAME" and (
                    finding.source == "identifier_watchlist" or span_is_code(layout, finding.start, finding.end))
            ]
            for finding in identifier_findings:
                answer = review_answers.answer_for(finding.logical_candidate or finding.text, finding.review_line or source_line(finding.context, finding.text), finding.review_key) if review_answers else None
                row = policy_row(finding, source.sha256, code=True, answer=answer,
                                       approved=scanner.fold_watchlist_value(finding.text) in approved_words)
                identifier_reviews.append(row)
                if row["policy_outcome"] == "leave_unchanged" and relative_file not in resolved_reviews:
                    resolved_reviews.append(relative_file)
            code_spans = {(finding.start, finding.end) for finding in identifier_findings}
            file_findings = [finding for finding in file_findings
                             if (finding.start, finding.end) not in code_spans or finding.entity_type != "NAME"]
            code_hidden = [finding for finding, row in zip(identifier_findings, identifier_reviews[-len(identifier_findings):])
                           if row["policy_outcome"] == "anonymize_whole"] if identifier_findings else []
            file_findings = review_name_findings(
                file_findings, file_sha256=source.sha256, name_judge=name_judge,
                name_verifier=name_verifier, layout=layout, approved_words=approved_words,
                verifier_enabled=verifier_enabled,
                watchlist_words=watchlist_words,
                review_answers=review_answers, decisions=identifier_reviews,
            )
            file_findings = [*code_hidden, *file_findings]
        except (SourceDecodingError, UnsupportedSourceEncodingError) as exc:
            reason = exc.reason
            message = f"{relative_file}: {reason}: {exc}"
        except RuntimeError as exc:
            reason = "detector failed"
            message = f"{relative_file}: {exc}"
        except OSError as exc:
            reason = "source_read_error"
            message = f"{relative_file}: {reason}: {exc}"
        else:
            findings.extend(file_findings)
            if file_times is not None:
                file_times[relative_file] = time.perf_counter() - file_started
            continue

        finally:
            for model in (name_extractor, name_judge, name_verifier):
                cache = getattr(model, "_response_cache", None)
                if cache is not None:
                    try:
                        cache.flush()
                    except OSError as exc:
                        diag.append(f"Cache not saved; answers remain in memory: {exc}")
                        cache.path = None

        incomplete.append(
            {
                "file": relative_file,
                "status": "NOT_COMPLETE",
                "reason": reason,
                "message": message,
            }
        )
        diag.append(f"NOT_COMPLETE: {message}")
        if file_times is not None:
            file_times[relative_file] = time.perf_counter() - file_started

    return list(resolve_overlaps(findings).selected)


def policy_row(
    finding: scanner.Finding,
    file_sha256: str,
    *, code: bool = False, answer: str | None = None, approved: bool = False,
) -> dict[str, object]:
    """Record a human answer or a required code review through policy."""

    occurrence_id = source_occurrence_id(finding, file_sha256)
    decision = apply_name_policy(
        occurrence_id=occurrence_id, model_context=finding.context,
        judge_decision=None, code_sensitive_identifier=code,
        review_answer=answer, watchlist_pair=finding.source == "watchlist_pair",
        watchlist_single=finding.source == "watchlist", approved_word=approved)
    return {
        "file": finding.file,
        "start": finding.start, "end": finding.end, "review_answer": answer,
        "line": finding.line,
        "column": finding.column,
        "text": finding.text,
        "context": finding.context,
        "logical_candidate": finding.logical_candidate, "logical_context": finding.logical_context,
            "review_line": finding.review_line, "review_key": finding.review_key,
        "source": finding.source, "approved": approved, "watchlist": finding.source in {"watchlist", "path_watchlist"},
        "code_sensitive_identifier": code,
        "policy_outcome": decision.outcome,
        "policy_reading": decision.reading,
        "policy_decision": decision.to_dict(),
    }


def _apply_review_answers(
    findings: list[scanner.Finding],
    review_answers: ReviewDecisions | None,
    *, decisions: list[dict[str, object]] | None = None, file_sha256: str = "0" * 64, layout=None,
) -> tuple[list[scanner.Finding], list[scanner.Finding]]:
    """Apply exact word-and-line answers to discovered source spans."""
    hidden, pending = [], []
    for finding in findings:
        if finding.entity_type != "NAME" or review_answers is None:
            pending.append(finding)
            continue
        context = finding.logical_context or clip_to_candidate_line(finding.context, finding.text)
        line = finding.review_line or source_line(context, finding.text)
        word = finding.logical_candidate or finding.text
        answer = review_answers.answer_for(word, line, finding.review_key)
        parts = [finding]
        spans = name_word_spans(word)
        if answer is None and len(spans) > 1 and any(
            review_answers.answer_for(word[start:end], line, finding.review_key) for start, end in spans
        ):
            if finding.logical and layout is not None:
                parts = [hit for start, end in spans for hit in source_findings(
                    replace(finding, start=finding.logical_start + start, end=finding.logical_start + end), finding.logical, layout)]
            else:
                parts = [replace(finding, text=word[start:end], start=finding.start + start,
                                 end=finding.start + end, column=finding.column + start,
                                 context=context.replace(f"[[{word}]]", f"{word[:start]}[[{word[start:end]}]]{word[end:]}", 1))
                         for start, end in spans]
        for part in parts:
            answer = review_answers.answer_for(part.logical_candidate or part.text, line, part.review_key)
            if answer is None:
                pending.append(part)
                continue
            row = policy_row(part, file_sha256, answer=answer)
            if decisions is not None:
                decisions.append(row)
            if row["policy_outcome"] == "anonymize_whole":
                hidden.append(part)
    return hidden, pending


def review_name_findings(
    findings: list[scanner.Finding], *, file_sha256: str,
    name_judge: NameJudge | None, name_verifier: object | None = None,
    approved_words: frozenset[str] = frozenset(),
    watchlist_words: frozenset[str] = frozenset(),
    verifier_enabled: bool = NAME_VERIFIER_ENABLED, layout=None,
    review_answers: ReviewDecisions | None = None,
    decisions: list[dict[str, object]] | None = None,
) -> list[scanner.Finding]:
    """Discover full names, then apply exact word-and-line corrections."""
    kept = []
    audit = name_judge.decisions if name_judge else decisions if decisions is not None else []
    total = sum(hit.entity_type == "NAME" for hit in findings)
    if name_judge:
        name_judge.progress_update(f"reviewing {total} name candidates")
    count = 0
    for finding in findings:
        if finding.entity_type != "NAME":
            kept.append(finding)
            continue
        occurrence_id = source_occurrence_id(finding, file_sha256)
        context = finding.logical_context or clip_to_candidate_line(finding.context, finding.text)
        candidate = replace(finding, text=finding.logical_candidate or finding.text, context=context)
        word = scanner.fold_watchlist_value(candidate.text)
        code = layout is not None and span_is_code(layout, finding.start, finding.end)
        pair = finding.source == "watchlist_pair"
        watchlist = finding.source == "watchlist" or word in watchlist_words
        approved = word in approved_words
        gates = dict(model_context=context, code_sensitive_identifier=code, watchlist_pair=pair,
                     watchlist_single=watchlist, approved_word=approved, verifier_enabled=verifier_enabled)
        judged, verified, cached = None, None, False
        gate = apply_name_policy(occurrence_id=occurrence_id, judge_decision=None, **gates)
        needs_model = gate.reading == NAME_POLICY_TABLE["no_judge"][1]
        count += 1
        if needs_model and name_judge is not None:
            name_judge.progress_update(f"candidate {count}/{total}: judging {finding.file}:{finding.line}")
            try:
                judged, cached = name_judge.decide(candidate, context, occurrence_id)
            except Exception as exc:
                _record_runtime_failure(name_judge, finding, "judge", exc)
            if judged is not None and watchlist and verifier_enabled:
                verified = _verify_candidate(name_judge=name_judge, name_verifier=name_verifier,
                                             judge_decision=judged, occurrence_id=occurrence_id,
                                             finding=candidate, snippet=context)
        targets = [finding]
        if judged is not None and judged.outcome in {"anonymize_whole", "anonymize_part"} and judged.person_texts:
            try:
                targets = _candidate_person_union_findings(finding=finding, snippet=context,
                                                           decision=judged, layout=layout)
            except ValueError:
                judged = None
        hidden, pending = _apply_review_answers(targets, review_answers, decisions=audit,
                                                file_sha256=file_sha256, layout=layout)
        kept.extend(hidden)
        for target in pending:
            target_id = source_occurrence_id(target, file_sha256)
            decision = apply_name_policy(occurrence_id=target_id, **gates,
                                         judge_decision=replace(judged, occurrence_id=target_id) if judged else None,
                                         verifier_decision=replace(verified, occurrence_id=target_id) if verified else None)
            gate = "watchlist_pair" if pair else "instruction_text" if instruction_text_requires_anonymization(context) else ""
            _record_decision(name_judge, target, judged, decision, cached, policy_gate=gate,
                             verifier_decision=verified, watchlist=watchlist, approved=approved, decisions=audit)
            if decision.outcome != "leave_unchanged":
                kept.append(target)
    return kept


def _verify_candidate(
    *,
    name_judge: NameJudge,
    name_verifier: object | None,
    judge_decision: Decision,
    occurrence_id: str,
    finding: scanner.Finding,
    snippet: str,
) -> Decision | None:
    if name_verifier is None or judge_decision.outcome != "propose_unchanged":
        return None
    try:
        return name_verifier.verify(
            occurrence_id=occurrence_id,
            candidate=finding.text,
            context=snippet,
        )
    except Exception as exc:  # pragma: no cover - defensive plug-in boundary
        _record_runtime_failure(name_judge, finding, "verifier", exc)
        return None


def _candidate_person_union_findings(
    *,
    finding: scanner.Finding,
    snippet: str,
    decision: Decision,
    layout=None,
) -> list[scanner.Finding]:
    """Return replacement spans for the union of candidate and person text.

    The decision keeps the model's copied text for audit, while this adapter
    computes offsets from immutable source text. The original candidate is
    included in the union, so a full-name expansion can never expose the
    original surname or given-name candidate.
    """

    line, candidate_start, candidate_end = unmark_candidate_line(snippet, finding.logical_candidate or finding.text)
    person_spans = locate_person_texts(
        candidate=finding.logical_candidate or finding.text,
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

    if finding.logical and layout is not None:
        return [hit for start, end in merged for hit in source_findings(
            replace(finding, start=start, end=end), finding.logical, layout)]

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




def _record_decision(
    name_judge: NameJudge,
    finding: scanner.Finding,
    judge_decision: Decision | None,
    policy_decision: Decision,
    cached: bool,
    policy_gate: str = "",
    verifier_decision: Decision | None = None,
    watchlist: bool = False, approved: bool = False,
    decisions: list[dict[str, object]] | None = None,
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
    (decisions if decisions is not None else name_judge.decisions).append(
        {
            "file": finding.file,
            "start": finding.start, "end": finding.end,
            "line": finding.line,
            "column": finding.column,
            "text": finding.text,
            "source": finding.source,
            "watchlist": watchlist or finding.source in {"watchlist", "watchlist_pair"}, "approved": approved,
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
            "prompt_version": name_judge.prompt_version if name_judge else "",
            "cached": cached,
            "error": error,
            "verifier_error": verifier_error,
            "context": finding.context,
            "logical_candidate": finding.logical_candidate, "logical_context": finding.logical_context,
            "review_line": finding.review_line, "review_key": finding.review_key,
            "judge_decision": (
                judge_decision.to_dict() if judge_decision is not None else None
            ),
            "verifier_decision": (
                verifier_decision.to_dict() if verifier_decision is not None else None
            ),
            "policy_decision": policy_decision.to_dict(),
        }
    )
