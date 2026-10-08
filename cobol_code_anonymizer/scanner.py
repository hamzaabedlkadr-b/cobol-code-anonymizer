"""PII scanning helpers for COBOL, copybooks, and JCL."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .cobol_layout import SourceLayout
from .policy import spacy_candidate_reason
from .candidates import (
    Detection,
    Occurrence,
    finding_fields,
    records_from_finding,
)
from .overlaps import resolve_overlaps
from .source_reader import read_source
from .logical_text import continued_literal_texts, extraction_texts, LogicalText, LogicalPiece
from .text_matching import line_column, source_findings, fold_watchlist_value, prepare_watchlist, IDENTIFIER_PART_RE, WATCHLIST_TOKEN_RE


TEXT_EXTENSIONS = {
    ".cbl",
    ".cob",
    ".cobol",
    ".cpy",
    ".jcl",
    ".proc",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".md",
    ".log",
    ".sql",
    ".dat",
    ".ctl",
}

DEFAULT_ENTITIES = {
    "NAME",
    "IBAN",
    "EMAIL",
    "CODICE_FISCALE",
    "MATRICOLA",
    "SUSPECTED_MATRICOLA",
    "PHONE",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)
CODICE_FISCALE_RE = re.compile(
    r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
    re.IGNORECASE,
)

MATRICOLA_LABELS = (
    "MATRICOLA",
    "MATR",
    "CODDIP",
    "COD-DIP",
    "CODICE-DIP",
    "CODICE-DIPENDENTE",
    "DIPENDENTE",
    "EMPLOYEE-ID",
    "EMP-ID",
)
MATRICOLA_STOP_VALUES = {
    "COMP",
    "DISPLAY",
    "HIGH-VALUES",
    "LOW-VALUES",
    "PIC",
    "PICTURE",
    "SPACE",
    "SPACES",
    "TO",
    "VALUE",
    "ZERO",
    "ZEROES",
    "ZEROS",
}

# Company matricola format: five or six digits, or seven starting with 5, 6, or 7.
MATRICOLA_VALUE = r"(?:[567][0-9]{6}|[0-9]{5,6})"
MATRICOLA_MOVE_RE = re.compile(
    rf"\bMOVE\s+['\"]?(?P<value>{MATRICOLA_VALUE})(?![A-Z0-9])['\"]?\s+TO\s+"
    rf"[\w-]*(?:{'|'.join(re.escape(label) for label in MATRICOLA_LABELS)})[\w-]*\b",
    re.IGNORECASE,
)
MATRICOLA_FIELD_VALUE_RE = re.compile(
    rf"\b[\w-]*(?:{'|'.join(re.escape(label) for label in MATRICOLA_LABELS)})[\w-]*\b"
    rf"[^\r\n]{{0,90}}\bVALUE\s+['\"]?(?P<value>{MATRICOLA_VALUE})(?![A-Z0-9])['\"]?",
    re.IGNORECASE,
)
MATRICOLA_KEY_VALUE_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(label) for label in MATRICOLA_LABELS)})\b"
    rf"\s*[:=]\s*['\"]?(?P<value>{MATRICOLA_VALUE})(?![A-Z0-9])['\"]?",
    re.IGNORECASE,
)
MATRICOLA_VALUE_RE = re.compile(rf"^{MATRICOLA_VALUE}$")
MATRICOLA_ANY_RE = re.compile(rf"(?<![A-Z0-9])(?P<value>{MATRICOLA_VALUE})(?![A-Z0-9])", re.IGNORECASE)

PHONE_LABEL_RE = re.compile(
    r"\b(?:TEL|TELEFONO|PHONE|CELL|CELLULARE)\b\s*[:=]?\s*"
    r"(?P<value>\+?\d[\d .()/-]{6,20}\d)",
    re.IGNORECASE,
)
QUOTED_LITERAL_RE = re.compile(
    r"'(?:''|[^'\r\n]){2,160}'|\"[^\"\r\n]{2,160}\""
)

ROSTER_FIELD_STOPWORDS = {
    "ID",
    "EMPLOYEE",
    "EMPLOYEEID",
    "MATR",
    "MATRICOLA",
    "NAME",
    "NOME",
    "COGNOME",
    "SURNAME",
    "FIRSTNAME",
    "LASTNAME",
}

# Honorifics that appear in HR exports but are not part of the name itself.
ROSTER_TITLES = {
    "DOTT",
    "DOTTSSA",
    "SSA",
    "DR",
    "DRSSA",
    "ING",
    "AVV",
    "RAG",
    "SIG",
    "SIGRA",
    "SIGNOR",
    "SIGNORA",
    "PROF",
    "GEOM",
    "ARCH",
    "CAV",
}

# Italian surname particles bind to the token that follows them, so
# "De Luca Maria" is a two-part name, not a three-part one.
SURNAME_PARTICLES = {
    "DE", "DEL", "DELL", "DELLA", "DELLE", "DELLO", "DEGLI", "DEI",
    "DI", "DA", "DAL", "DALLA", "DALLE", "DALLO",
    "LO", "LA", "LI", "LE", "SAN", "SANTA", "SANT",
    "VAN", "VON", "MC", "MAC", "O",
}

# Letters, plus apostrophes/hyphens *inside* a token: D'Amico and Jean-Pierre
# are one name token each, never two.
ROSTER_TOKEN_RE = re.compile(r"[^\W\d_](?:['’\-]?[^\W\d_])*", re.UNICODE)
# A trailing dot belongs to the token only for a one-letter initial. Ordinary
# sentence punctuation after a complete watchlist word must stay outside it.
PAIR_TOKEN_RE = re.compile(
    r"(?:[^\W\d_]\.(?![^\W\d_])|[^\W\d_]+(?:['’]{1,2}[^\W\d_]+)*)",
    re.UNICODE,
)
PAIR_GAP_RE = re.compile(r"(?:[ \t]+|,[ \t]*)")
FOLDED_WATCHLIST_TOKEN_RE = WATCHLIST_TOKEN_RE
WATCHLIST_FOLD_TRANSLATION = str.maketrans({"0": "O", "1": "I", "5": "S"})

@dataclass(frozen=True)
class Finding:
    file: str
    entity_type: str
    text: str
    start: int
    end: int
    line: int
    column: int
    confidence: float
    context: str
    source: str = ""
    logical_context: str = ""
    logical_candidate: str = ""
    logical: LogicalText | None = None
    logical_start: int = 0
    logical_end: int = 0
    review_line: str = ""
    review_key: str = ""

    def to_dict(self) -> dict[str, object]:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "logical"}
        data.pop("start")
        data.pop("end")
        data.pop("logical_context")
        data.pop("logical_candidate")
        for key in ("logical_start", "logical_end", "review_line", "review_key"):
            data.pop(key)
        return data

    def to_candidate_records(
        self,
        *,
        file_sha256: str,
        detector_version: str,
        region: str = "unknown",
        original_start: int | None = None,
        original_end: int | None = None,
        matched_entry: str | None = None,
        variant: str | None = None,
    ) -> tuple[Occurrence, Detection]:
        """Represent this finding with the new auditable record types.

        This is a migration adapter only.  It does not change the finding or
        run any additional detection, judging, overlap, or policy logic.
        """
        return records_from_finding(
            self,
            file_sha256=file_sha256,
            detector_version=detector_version,
            region=region,
            original_start=original_start,
            original_end=original_end,
            matched_entry=matched_entry,
            variant=variant,
        )

    @classmethod
    def from_candidate_records(
        cls,
        occurrence: Occurrence,
        detection: Detection,
    ) -> "Finding":
        """Rebuild the unchanged legacy finding after a record round-trip."""
        return cls(**finding_fields(occurrence, detection))


def read_text(path: Path) -> str:
    """Read a small auxiliary list using the legacy compatibility fallback.

    Production source programs are deliberately *not* read through this
    helper: :func:`scan_file` uses ``source_reader.read_source`` so unsafe
    decoding becomes a visible incomplete-file result.  This fallback remains
    temporarily for user-supplied watchlists and the pre-A4 writer.
    """

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def write_json(path: Path, findings: list[Finding]) -> None:
    path.write_text(
        json.dumps([finding.to_dict() for finding in findings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def iter_text_files(input_path: Path, skip_root: Path | list[Path] | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_text_candidate(input_path) else []
    skip_roots = normalize_skip_roots(skip_root)
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if any(is_relative_to(path, root) for root in skip_roots):
            continue
        if path.is_file() and is_text_candidate(path):
            files.append(path)
    return files


def iter_all_files(input_path: Path, skip_root: Path | list[Path] | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    skip_roots = normalize_skip_roots(skip_root)
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if any(is_relative_to(path, root) for root in skip_roots):
            continue
        if path.is_file():
            files.append(path)
    return files


def normalize_skip_roots(skip_root: Path | list[Path] | None) -> list[Path]:
    if skip_root is None:
        return []
    if isinstance(skip_root, Path):
        return [skip_root]
    return [root for root in skip_root if root is not None]


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or not path.suffix


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def relative_name(path: Path, input_path: Path) -> str:
    if input_path.is_file():
        return path.name
    return str(path.relative_to(input_path))





def context_for(text: str, start: int, end: int, radius: int = 75) -> str:
    prefix_start = max(0, start - radius)
    suffix_end = min(len(text), end + radius)
    return text[prefix_start:start] + "[[" + text[start:end] + "]]" + text[end:suffix_end]


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def load_names(
    extra_watchlists: list[Path] | None = None,
    include_default: bool = False,
) -> list[str]:
    names_path = Path(__file__).parent / "data" / "italian_names.txt"
    paths = ([names_path] if include_default else []) + list(extra_watchlists or [])
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in read_text(path).splitlines():
            value = " ".join(line.strip().split())
            if value and not value.startswith("#") and not value.isdecimal():
                names.add(value)
    return sorted(names, key=lambda value: (-len(value), value.upper()))


def load_employee_rosters(roster_paths: list[Path] | None = None) -> tuple[list[str], list[str]]:
    names: set[str] = set()
    matriculas: set[str] = set()
    for path in roster_paths or []:
        if not path.exists():
            continue
        for line in read_text(path).splitlines():
            line_names, line_matriculas = parse_employee_roster_line(line)
            names.update(line_names)
            matriculas.update(line_matriculas)
    return (
        sorted(names, key=lambda value: (-len(value), value.upper())),
        sorted(matriculas, key=lambda value: (-len(value), value)),
    )


def parse_employee_roster_line(line: str) -> tuple[set[str], set[str]]:
    if not line.strip() or line.lstrip().startswith("#"):
        return set(), set()

    matriculas = set()
    for match in re.finditer(rf"(?<![A-Z0-9]){MATRICOLA_VALUE}(?![A-Z0-9])", line, re.IGNORECASE):
        matriculas.add(match.group(0))

    without_ids = re.sub(rf"(?<![A-Z0-9]){MATRICOLA_VALUE}(?![A-Z0-9])", " ", line, flags=re.IGNORECASE)
    without_email = re.sub(EMAIL_RE, " ", without_ids)
    tokens = ROSTER_TOKEN_RE.findall(without_email)
    tokens = [
        token
        for token in tokens
        if token.upper().replace("-", "").replace("'", "").replace("’", "")
        not in ROSTER_FIELD_STOPWORDS | ROSTER_TITLES
    ]
    if not tokens:
        return set(), matriculas

    names = roster_name_variants(glue_surname_particles(tokens))
    return names, matriculas


def glue_surname_particles(tokens: list[str]) -> list[str]:
    """Join Italian surname particles to the token they belong to.

    ["De", "Luca", "Maria"] -> ["De Luca", "Maria"], so the name has two parts
    and both orderings can be generated. Without this, a roster line written
    surname-first never matches source text written given-name-first.
    """
    glued: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.upper().rstrip("'’")
        if key in SURNAME_PARTICLES and index + 1 < len(tokens):
            glued.append(f"{token} {tokens[index + 1]}")
            index += 2
        else:
            glued.append(token)
            index += 1
    return glued


def roster_name_variants(tokens: list[str]) -> set[str]:
    if len(tokens) == 1:
        return {tokens[0]} if len(tokens[0]) >= 2 else set()

    name = " ".join(tokens)
    names = {name}
    if len(tokens) == 2:
        names.add(f"{tokens[1]} {tokens[0]}")
        # A roster may be written either way round; without both initial forms,
        # one orientation silently fails to match.
        if len(tokens[0]) > 2:
            names.add(f"{tokens[0][0]} {tokens[1]}")
        if len(tokens[1]) > 2:
            names.add(f"{tokens[1][0]} {tokens[0]}")
    elif len(tokens[0]) > 2:
        names.add(" ".join([tokens[0][0], *tokens[1:]]))
    return names


def compile_name_regex(names: list[str], min_single_token_length: int = 3) -> re.Pattern[str] | None:
    patterns = []
    for name in names:
        pattern = name_pattern(name, min_single_token_length)
        if not pattern:
            continue
        patterns.append(pattern)
    if not patterns:
        return None
    return re.compile(rf"(?<![\w-])({'|'.join(patterns)})(?![\w-])", re.IGNORECASE | re.UNICODE)


def name_pattern(name: str, min_single_token_length: int) -> str | None:
    tokens = name.split()
    if not tokens:
        return None
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) < min_single_token_length:
            return None
        return re.escape(token)

    escaped_tokens = []
    for token in tokens:
        escaped = escape_name_token(token)
        if len(token) == 1:
            escaped += r"\.?"
        escaped_tokens.append(escaped)
    return r"\s+".join(escaped_tokens)


def escape_name_token(token: str) -> str:
    """Escape a name token, leaving apostrophes flexible.

    A roster entry written D'Amico must also match the typographic form
    D’Amico and COBOL's doubled-apostrophe escape D''Amico inside literals.
    """
    parts = re.split(r"['’]", token)
    return r"['’]{1,2}".join(re.escape(part) for part in parts)


def name_scan_ranges(text: str, scope: str) -> list[tuple[int, int]]:
    if scope == "all":
        return [(0, len(text))]

    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if is_name_context_line(line):
            ranges.append((offset, offset + len(line)))
        offset += len(line)

    for match in QUOTED_LITERAL_RE.finditer(text):
        ranges.append((match.start(), match.end()))

    return merge_ranges(ranges)


def free_text_scan_ranges(text: str, scope: str) -> list[tuple[int, int]]:
    if scope == "all":
        return [(0, len(text))]

    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if is_comment_line(line):
            ranges.append((offset, offset + len(line)))
        offset += len(line)

    for match in QUOTED_LITERAL_RE.finditer(text):
        ranges.append((match.start(), match.end()))

    return merge_ranges(ranges)


def find_watchlist_pair_spans(
    text: str,
    watchlist_values: tuple[str, ...] | list[str],
    *, words: set[str] | None = None, all_text: bool = False,
) -> list[tuple[int, int]]:
    """Return deterministic multi-word watchlist spans in comments/literals.

    A pair has adjacent single-word watchlist entries of at least three letters.
    The narrow separators keep
    this an auditable structural rule: spaces/tabs or one comma only. Pairs
    never cross a comment/literal boundary or a physical line.
    """

    watchlist_words = words if words is not None else prepare_watchlist(watchlist_values)[1]
    if not watchlist_words:
        return []

    spans: list[tuple[int, int]] = []
    # Pairs deliberately remain restricted to comments/literals even if the
    # caller asked other detectors to scan all code text.
    for range_start, range_end in ([(0, len(text))] if all_text else _watchlist_pair_scan_ranges(text)):
        tokens = list(PAIR_TOKEN_RE.finditer(text, range_start, range_end))
        index = 0
        while index < len(tokens):
            current = tokens[index]
            if not _is_watchlist_pair_word(current, watchlist_words):
                index += 1
                continue

            end_index = index
            while (
                end_index + 1 < len(tokens)
                and _pair_neighbours(tokens[end_index], tokens, end_index + 1)
                and _is_watchlist_pair_word(tokens[end_index + 1], watchlist_words)
            ):
                end_index += 1
            if end_index > index:
                spans.append((current.start(), tokens[end_index].end()))
                index = end_index + 1
                continue

            index += 1
    return spans


def _watchlist_pair_scan_ranges(text: str) -> list[tuple[int, int]]:
    """Return only comment and quoted-literal ranges for pair detection."""

    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if is_comment_line(line):
            ranges.append((offset, offset + len(line)))
        offset += len(line)
    ranges.extend((match.start(), match.end()) for match in QUOTED_LITERAL_RE.finditer(text))
    return merge_ranges(ranges)


def scan_watchlist_pairs(
    text: str,
    rel_file: str,
    watchlist_values: tuple[str, ...] | list[str],
    *, words: set[str] | None = None, all_text: bool = False,
) -> list[Finding]:
    """Emit high-confidence pair findings before regular NAME detectors."""

    findings: list[Finding] = []
    for start, end in find_watchlist_pair_spans(text, watchlist_values, words=words, all_text=all_text):
        line, column = line_column(text, start)
        findings.append(
            Finding(
                file=rel_file,
                entity_type="NAME",
                text=text[start:end],
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=0.99,
                context=context_for(text, start, end),
                source="watchlist_pair",
            )
        )
    return findings






def _pair_neighbours(
    current: re.Match[str],
    tokens: list[re.Match[str]],
    next_index: int,
) -> bool:
    """Check that two token matches have exactly one allowed short gap."""

    return (
        next_index < len(tokens)
        and PAIR_GAP_RE.fullmatch(current.string[current.end() : tokens[next_index].start()])
        is not None
    )


def _is_watchlist_pair_word(match: re.Match[str], words: set[str]) -> bool:
    return not match.group().endswith(".") and fold_watchlist_value(match.group()) in words


def _normalise_pair_token(value: str) -> str:
    folded = unicodedata.normalize(
        "NFKD",
        value.replace("''", "'").replace("’", "'").casefold(),
    )
    return "".join(character for character in folded if not unicodedata.combining(character))


def is_name_context_line(line: str) -> bool:
    stripped = line.strip()
    upper = line.upper()
    if not stripped:
        return False
    if stripped.startswith("*") or stripped.startswith("//*"):
        return True
    if len(line) > 6 and line[6] == "*":
        return True
    markers = (
        " AUTHOR",
        "DISPLAY ",
        " VALUE ",
        " STRING ",
        " ASSIGN TO ",
        " NOME",
        "COGNOME",
        "NOMINATIVO",
        "REFERENTE",
        "RESPONSABILE",
        "OPERATORE",
        "ANALISTA",
        "CONTATTARE",
    )
    return any(marker in upper for marker in markers)


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def offset_in_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start >= range_start and end <= range_end for range_start, range_end in ranges)


def is_inside_email_or_url(text: str, start: int, end: int) -> bool:
    token_start = start
    while token_start > 0 and not text[token_start - 1].isspace() and text[token_start - 1] not in "\"'<>(),;":
        token_start -= 1
    token_end = end
    while token_end < len(text) and not text[token_end].isspace() and text[token_end] not in "\"'<>(),;":
        token_end += 1
    token = text[token_start:token_end]
    return "@" in token or "://" in token




def build_presidio_analyzer(model_name: str, diagnostics: list[str]) -> object | None:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise RuntimeError("spaCy/Presidio is unavailable; install it or use --no-presidio") from exc

    try:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "it", "model_name": model_name}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["it"])
    except Exception as exc:
        raise RuntimeError(f"spaCy model {model_name!r} could not start; fix it or use --no-presidio: {exc}") from exc


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
    review_answers: object | None = None,
    name_judge: object | None = None,
    name_verifier: object | None = None,
    name_extractor: object | None = None,
    deterministic_names_enabled: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[Finding]:
    """Compatibility entry point; all orchestration is owned by pipeline.py."""

    # The import stays local because pipeline.py imports the detector helpers
    # in this module.  Keeping this small wrapper avoids breaking callers that
    # previously imported scan_path from scanner while leaving one real flow.
    from .pipeline import scan_path as run_pipeline

    return run_pipeline(
        input_path=input_path,
        entities=entities,
        extra_watchlists=extra_watchlists,
        employee_rosters=employee_rosters,
        include_default_names=include_default_names,
        name_scope=name_scope,
        skip_root=skip_root,
        use_presidio=use_presidio,
        presidio_model=presidio_model,
        diagnostics=diagnostics,
        not_complete_files=not_complete_files,
        identifier_review_decisions=identifier_review_decisions,
        resolved_review_files=resolved_review_files,
        review_answers=review_answers,
        name_judge=name_judge,
        name_verifier=name_verifier,
        name_extractor=name_extractor,
        deterministic_names_enabled=deterministic_names_enabled,
        progress=progress,
    )


def scan_file(
    path: Path,
    input_path: Path,
    entities: set[str],
    name_regex: re.Pattern[str] | None,
    roster_name_regex: re.Pattern[str] | None,
    roster_matricula_values: set[str],
    name_scope: str,
    presidio_analyzer: object | None = None,
    name_extractor: object | None = None,
    name_judge: object | None = None,
    name_verifier: object | None = None,
) -> list[Finding]:
    """Compatibility wrapper; production orchestration lives in pipeline.py."""

    source = read_source(path)
    rel_file = relative_name(path, input_path)
    findings = scan_text(
        source.text,
        rel_file,
        entities,
        name_regex,
        roster_name_regex,
        roster_matricula_values,
        name_scope,
        presidio_analyzer=presidio_analyzer,
        name_extractor=name_extractor,
    )
    if name_judge is None:
        return findings

    from .pipeline import review_name_findings

    return review_name_findings(
        findings,

        file_sha256=source.sha256,
        name_judge=name_judge,
        name_verifier=name_verifier,
    )


def scan_text(
    text: str,
    rel_file: str,
    entities: set[str],
    name_regex: re.Pattern[str] | None,
    roster_name_regex: re.Pattern[str] | None,
    roster_matricula_values: set[str],
    name_scope: str,
    presidio_analyzer: object | None = None,
    name_extractor: object | None = None,
    watchlist_pair_values: tuple[str, ...] | list[str] = (),
    layout: SourceLayout | None = None,
    pair_words: set[str] | None = None,
) -> list[Finding]:
    """Run detectors on already-decoded text without policy or model calls."""

    findings: list[Finding] = []

    if "EMAIL" in entities:
        findings.extend(regex_findings(text, rel_file, "EMAIL", EMAIL_RE, 0.98))
    if "IBAN" in entities:
        findings.extend(regex_findings(text, rel_file, "IBAN", IBAN_RE, 0.96))
    if "CODICE_FISCALE" in entities:
        findings.extend(regex_findings(text, rel_file, "CODICE_FISCALE", CODICE_FISCALE_RE, 0.96))
    if "MATRICOLA" in entities or "SUSPECTED_MATRICOLA" in entities:
        if roster_matricula_values:
            findings.extend(scan_roster_classified_matriculas(text, rel_file, roster_matricula_values))
        else:
            findings.extend(scan_matricola(text, rel_file))
    if "PHONE" in entities:
        findings.extend(regex_findings(text, rel_file, "PHONE", PHONE_LABEL_RE, 0.78, group="value"))
    if "NAME" in entities:
        name_findings: list[Finding] = []
        if layout is not None:
            entries = watchlist_pair_values if isinstance(watchlist_pair_values, dict) else prepare_watchlist(watchlist_pair_values)[0]
            pair_words = pair_words if pair_words is not None else prepare_watchlist(watchlist_pair_values)[1]
            for logical in extraction_texts(layout):
                local = []
                local.extend(scan_watchlist_pairs(logical.text, rel_file, (), words=pair_words, all_text=True))
                if name_regex:
                    local.extend(scan_watchlist_names(logical.text, rel_file, name_regex, "all"))
                local.extend(scan_folded_watchlist_names(logical.text, rel_file, entries, "all"))
                if roster_name_regex:
                    local.extend(scan_watchlist_names(logical.text, rel_file, roster_name_regex, "all", source="employee_roster"))
                if presidio_analyzer:
                    local.extend(scan_presidio_names(logical.text, rel_file, presidio_analyzer, "all"))
                for hit in resolve_overlaps(local).selected:
                    name_findings.extend(source_findings(hit, logical, layout))
            name_findings.extend(scan_identifier_watchlist_names(text, rel_file, entries, layout=layout))
            if name_extractor is not None:
                name_findings.extend(name_extractor.extract(text, rel_file, name_scope, deterministic=list(name_findings), layout=layout))
            findings.extend(resolve_overlaps(name_findings).selected)
            return findings
        # Add pairs before other NAME detectors. If another detector returns
        # the same span, overlap cleanup retains this stronger deterministic
        # source and the pipeline can bypass the judge.
        name_findings.extend(
            scan_watchlist_pairs(
                text,
                rel_file,
                watchlist_pair_values,
            )
        )
        # Identifier hits do not use the ordinary NAME route: the pipeline
        # records them as review-required and never creates a replacement.
        name_findings.extend(
            scan_identifier_watchlist_names(
                text,
                rel_file,
                watchlist_pair_values,
            )
        )
        if presidio_analyzer:
            name_findings.extend(
                scan_presidio_names(
                    text,
                    rel_file,
                    presidio_analyzer,
                    name_scope,
                )
            )
        if name_regex:
            name_findings.extend(
                scan_watchlist_names(
                    text,
                    rel_file,
                    name_regex,
                    name_scope,
                )
            )
            name_findings.extend(
                scan_folded_watchlist_names(
                    text,
                    rel_file,
                    watchlist_pair_values,
                    name_scope,
                )
            )
        if roster_name_regex:
            name_findings.extend(
                scan_watchlist_names(
                    text,
                    rel_file,
                    roster_name_regex,
                    name_scope,
                    source="employee_roster",
                    confidence=0.92,
                )
            )
        if name_extractor is not None:
            name_findings.extend(
                name_extractor.extract(
                    text,
                    rel_file,
                    name_scope,
                    deterministic=list(name_findings),
                    layout=layout,
                )
            )
        # Give the pipeline one stable detector span set. Policy may remove a
        # selected span but it never creates a replacement detector finding.
        name_findings = list(resolve_overlaps(name_findings).selected)
        findings.extend(name_findings)

    return findings


def regex_findings(
    text: str,
    rel_file: str,
    entity_type: str,
    regex: re.Pattern[str],
    confidence: float,
    group: str | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    for match in regex.finditer(text):
        start = match.start(group) if group else match.start()
        end = match.end(group) if group else match.end()
        value = text[start:end].strip()
        if not value:
            continue
        line, column = line_column(text, start)
        findings.append(
            Finding(
                file=rel_file,
                entity_type=entity_type,
                text=value,
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=confidence,
                context=context_for(text, start, end),
                source="regex",
            )
        )
    return findings


def scan_matricola(text: str, rel_file: str) -> list[Finding]:
    findings: list[Finding] = []
    for regex in (MATRICOLA_MOVE_RE, MATRICOLA_FIELD_VALUE_RE, MATRICOLA_KEY_VALUE_RE):
        findings.extend(regex_findings(text, rel_file, "MATRICOLA", regex, 0.84, group="value"))
    return [finding for finding in findings if is_probable_matricola_value(finding.text)]


def scan_roster_classified_matriculas(
    text: str,
    rel_file: str,
    roster_matriculas: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    for match in MATRICOLA_ANY_RE.finditer(text):
        start, end = match.start("value"), match.end("value")
        value = text[start:end]
        in_roster = value in roster_matriculas
        line, column = line_column(text, start)
        findings.append(
            Finding(
                file=rel_file,
                entity_type="MATRICOLA" if in_roster else "SUSPECTED_MATRICOLA",
                text=value,
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=0.99 if in_roster else 0.68,
                context=context_for(text, start, end),
                source="employee_roster" if in_roster else "not_in_employee_roster",
            )
        )
    return findings


def is_probable_matricola_value(value: str) -> bool:
    cleaned = value.strip().strip("\"'")
    upper = cleaned.upper()
    if upper in MATRICOLA_STOP_VALUES:
        return False
    return bool(MATRICOLA_VALUE_RE.fullmatch(cleaned))


def scan_presidio_names(text: str, rel_file: str, analyzer: object, scope: str) -> list[Finding]:
    """Keep bounded PERSON spans; policy decides their meaning."""
    try:
        results = analyzer.analyze(text=text, language="it", entities=["PERSON"])
    except Exception as exc:
        raise RuntimeError(f"spaCy detection failed: {exc}") from exc
    findings = []
    ranges = name_scan_ranges(text, scope)
    for result in results:
        if not 0 <= result.start < result.end <= len(text):
            raise RuntimeError("spaCy returned invalid source bounds")
        start, end = trim_span(text, result.start, result.end)
        if start >= end or not offset_in_ranges(start, end, ranges):
            continue
        if spacy_candidate_reason(text[start:end]):
            continue
        line, column = line_column(text, start)
        findings.append(Finding(rel_file, "NAME", text[start:end], start, end, line, column,
                                float(result.score), context_for(text, start, end), "presidio_spacy"))
    return findings


































def is_comment_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("*") or stripped.startswith("//*") or (len(line) > 6 and line[6] == "*")






def scan_watchlist_names(
    text: str,
    rel_file: str,
    name_regex: re.Pattern[str],
    scope: str,
    source: str = "watchlist",
    confidence: float = 0.7,
) -> list[Finding]:
    """Find exact watchlist spans without inferring adjacent name words."""

    raw: list[Finding] = []
    for start_range, end_range in name_scan_ranges(text, scope):
        segment = text[start_range:end_range]
        for match in name_regex.finditer(segment):
            start = start_range + match.start()
            end = start_range + match.end()
            if is_inside_email_or_url(text, start, end):
                continue
            value = text[start:end]
            line, column = line_column(text, start)
            raw.append(
                Finding(
                    file=rel_file,
                    entity_type="NAME",
                    text=value,
                    start=start,
                    end=end,
                    line=line,
                    column=column,
                    confidence=confidence,
                    context=context_for(text, start, end),
                    source=source,
                )
            )
    return list(resolve_overlaps(raw).selected)




def scan_folded_watchlist_names(
    text: str,
    rel_file: str,
    watchlist_values: tuple[str, ...] | list[str],
    scope: str,
) -> list[Finding]:
    """Find visual watchlist variants and exactly-two-entry glued tokens.

    The normal exact matcher remains authoritative for ordinary spellings. This
    supplemental matcher folds only ``0/O``, ``1/I``, ``5/S``, case, and
    accents. It deliberately requires a whole token, so suffixed identifiers
    such as ``SALA1`` and ``C0NTI01`` are not converted into name candidates.
    """

    entries = watchlist_values if isinstance(watchlist_values, dict) else prepare_watchlist(watchlist_values)[0]
    if not entries:
        return []

    findings: list[Finding] = []
    for range_start, range_end in name_scan_ranges(text, scope):
        segment = text[range_start:range_end]
        for match in FOLDED_WATCHLIST_TOKEN_RE.finditer(segment):
            raw = match.group()
            folded = fold_watchlist_value(raw)
            start = range_start + match.start()
            if folded in entries:
                findings.append(_folded_watchlist_finding(text, rel_file, start, start + len(raw)))
                continue
    return list(resolve_overlaps(findings).selected)


# Identifier review is deliberately narrower than general code scanning. A
# candidate here is never rewritten automatically: an identifier can have
# references outside the current batch, so only a reviewer can decide it is
# safe to rename or retain.
_COBOL_DATA_NAME_RE = re.compile(
    r"^\s*(?:\d{1,2}|FD|SD)\s+(?P<identifier>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9-]*)\b",
    re.IGNORECASE,
)
_COPYBOOK_NAME_RE = re.compile(
    r"^\s*COPY\s+(?P<identifier>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9-]*)\b",
    re.IGNORECASE,
)
_PROGRAM_ID_RE = re.compile(
    r"\bPROGRAM-ID\.\s*(?P<identifier>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9-]*)\b",
    re.IGNORECASE,
)
_PARAGRAPH_NAME_RE = re.compile(
    r"^\s*(?P<identifier>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9-]*)\.\s*$",
    re.IGNORECASE,
)
_IDENTIFIER_COMPONENT_RE = re.compile(
    r"[A-ZÀ-ÖØ-Þ]+(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])|"
    r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+|[A-ZÀ-ÖØ-Þ]+|\d+"
)


def scan_identifier_watchlist_names(
    text: str,
    rel_file: str,
    watchlist_values: tuple[str, ...] | list[str],
    *, layout: SourceLayout | None = None,
) -> list[Finding]:
    """Return watchlist components in COBOL identifiers for manual review.

    Only data names, paragraph names, COPY targets, and PROGRAM-ID values are
    examined. Hyphens and CamelCase boundaries split one identifier into
    components, and visual watchlist folding is reused. The returned source is
    ``identifier_watchlist`` so the pipeline records review-required and never
    adds a replacement span.
    """

    entries = watchlist_values if isinstance(watchlist_values, dict) else prepare_watchlist(watchlist_values)[0]
    if not entries:
        return []

    if layout is not None:
        findings = []
        for region in layout.regions_of_kind("code"):
            value = text[region.decoded_start:region.decoded_end]
            logical = LogicalText(value, (LogicalPiece(0, len(value), region.decoded_start, region.decoded_end),), "code")
            for match in IDENTIFIER_PART_RE.finditer(value):
                if fold_watchlist_value(match.group()) in entries:
                    template = Finding(rel_file, "NAME", match.group(), match.start(), match.end(), region.line,
                                       match.start() + 1, .7, "", "watchlist")
                    findings.extend(source_findings(template, logical, layout))
        return findings
    findings: list[Finding] = []
    offset = 0
    for physical_line in text.splitlines(keepends=True):
        line = physical_line.rstrip("\r\n")
        code_offset = _identifier_code_offset(line)
        code = line[code_offset:]
        stripped = code.lstrip()
        if not stripped or stripped.startswith(("*", "//*", "//")):
            offset += len(physical_line)
            continue
        code = code.split("*>", 1)[0]
        matches = [
            pattern.search(code)
            for pattern in (
                _COBOL_DATA_NAME_RE,
                _COPYBOOK_NAME_RE,
                _PROGRAM_ID_RE,
                _PARAGRAPH_NAME_RE,
            )
        ]
        for match in (item for item in matches if item is not None):
            identifier = match.group("identifier")
            identifier_start = offset + code_offset + match.start("identifier")
            for component in _IDENTIFIER_COMPONENT_RE.finditer(identifier):
                value = component.group()
                if fold_watchlist_value(value) not in entries:
                    continue
                start = identifier_start + component.start()
                end = identifier_start + component.end()
                line_number, column = line_column(text, start)
                findings.append(
                    Finding(
                        file=rel_file,
                        entity_type="NAME",
                        text=text[start:end],
                        start=start,
                        end=end,
                        line=line_number,
                        column=column,
                        confidence=0.70,
                        context=context_for(text, start, end),
                        source="identifier_watchlist",
                    )
                )
        offset += len(physical_line)
    return list(resolve_overlaps(findings).selected)


def _identifier_code_offset(line: str) -> int:
    """Return the code-area start for conventional fixed-format records."""

    if len(line) >= 7 and (bool(line[:6].strip()) or line[6] in " */D-d"):
        return 7
    return 0




def _folded_watchlist_finding(text: str, rel_file: str, start: int, end: int) -> Finding:
    line, column = line_column(text, start)
    return Finding(
        file=rel_file,
        entity_type="NAME",
        text=text[start:end],
        start=start,
        end=end,
        line=line,
        column=column,
        confidence=0.70,
        context=context_for(text, start, end),
        source="watchlist",
    )


def remove_overlaps(findings: list[Finding]) -> list[Finding]:
    """Compatibility wrapper around the overlap-group resolver.

    New code should retain :func:`resolve_overlaps` so later evidence can see
    every group member. Legacy callers still receive the unchanged selected
    finding list through this function.
    """

    return list(resolve_overlaps(findings).selected)
