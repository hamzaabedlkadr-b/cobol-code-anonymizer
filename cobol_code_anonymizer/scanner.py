"""PII scanning helpers for COBOL, copybooks, and JCL."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


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

# Company matricola format: six digits, optionally prefixed by 5, 6, or 7.
MATRICOLA_VALUE = r"[567]?\d{6}"
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

PHONE_LABEL_RE = re.compile(
    r"\b(?:TEL|TELEFONO|PHONE|CELL|CELLULARE)\b\s*[:=]?\s*"
    r"(?P<value>\+?\d[\d .()/-]{6,20}\d)",
    re.IGNORECASE,
)

NAME_STOPWORDS = {
    "AUTHOR",
    "CELL",
    "CELLULARE",
    "CODICE",
    "COGNOME",
    "COMMENTO",
    "EMAIL",
    "FISCALE",
    "IBAN",
    "MAIL",
    "MATRICOLA",
    "NOME",
    "NOMINATIVO",
    "OPERATORE",
    "REFERENTE",
    "RESPONSABILE",
    "TEL",
    "TELEFONO",
    "TEST",
}
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

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("start")
        data.pop("end")
        return data


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def write_json(path: Path, findings: list[Finding]) -> None:
    path.write_text(
        json.dumps([finding.to_dict() for finding in findings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def iter_text_files(input_path: Path, skip_root: Path | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_text_candidate(input_path) else []
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if skip_root and is_relative_to(path, skip_root):
            continue
        if path.is_file() and is_text_candidate(path):
            files.append(path)
    return files


def iter_all_files(input_path: Path, skip_root: Path | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files: list[Path] = []
    for path in input_path.rglob("*"):
        if skip_root and is_relative_to(path, skip_root):
            continue
        if path.is_file():
            files.append(path)
    return files


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


def line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    column = offset + 1 if last_newline == -1 else offset - last_newline
    return line, column


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
    include_default: bool = True,
) -> list[str]:
    names_path = Path(__file__).parent / "data" / "italian_names.txt"
    paths = ([names_path] if include_default else []) + list(extra_watchlists or [])
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in read_text(path).splitlines():
            value = " ".join(line.strip().split())
            if value and not value.startswith("#"):
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
    tokens = re.findall(r"[^\W\d_][^\W\d_'-]*", without_email, flags=re.UNICODE)
    tokens = [token for token in tokens if token.upper().replace("-", "") not in ROSTER_FIELD_STOPWORDS]
    if not tokens:
        return set(), matriculas

    names = roster_name_variants(tokens)
    return names, matriculas


def roster_name_variants(tokens: list[str]) -> set[str]:
    if len(tokens) == 1:
        return {tokens[0]} if len(tokens[0]) >= 2 else set()

    name = " ".join(tokens)
    names = {name}
    if len(tokens) == 2:
        names.add(f"{tokens[1]} {tokens[0]}")
    if len(tokens[0]) > 2:
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
        escaped = re.escape(token)
        if len(token) == 1:
            escaped += r"\.?"
        escaped_tokens.append(escaped)
    return r"\s+".join(escaped_tokens)


def compile_matricola_regex(matriculas: list[str]) -> re.Pattern[str] | None:
    patterns = [re.escape(value) for value in matriculas if MATRICOLA_VALUE_RE.fullmatch(value)]
    if not patterns:
        return None
    return re.compile(rf"(?<![A-Z0-9])({'|'.join(patterns)})(?![A-Z0-9])", re.IGNORECASE)


def name_scan_ranges(text: str, scope: str) -> list[tuple[int, int]]:
    if scope == "all":
        return [(0, len(text))]

    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if is_name_context_line(line):
            ranges.append((offset, offset + len(line)))
        offset += len(line)

    for match in re.finditer(r"'[^'\r\n]{2,160}'|\"[^\"\r\n]{2,160}\"", text):
        ranges.append((match.start(), match.end()))

    return merge_ranges(ranges)


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


def trim_name_stopwords(text: str, start: int, end: int) -> tuple[int, int]:
    while True:
        value = text[start:end].strip()
        if not value:
            return start, start
        words = value.split()
        first = words[0].strip(":,.;").upper()
        last = words[-1].strip(":,.;").upper()
        changed = False
        if first in NAME_STOPWORDS:
            start = text.find(words[0], start, end) + len(words[0])
            changed = True
        if last in NAME_STOPWORDS and start < end:
            end = text.rfind(words[-1], start, end)
            changed = True
        start, end = trim_span(text, start, end)
        if not changed:
            return start, end


def build_presidio_analyzer(model_name: str, diagnostics: list[str]) -> object | None:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        diagnostics.append(
            "Microsoft Presidio/spaCy are not installed; falling back to the bundled watchlist. "
            "Install with: python -m pip install -e . && python -m spacy download it_core_news_sm"
        )
        diagnostics.append(f"Import error: {exc}")
        return None

    try:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "it", "model_name": model_name}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["it"])
    except Exception as exc:
        diagnostics.append(
            f"Could not start Presidio with spaCy model {model_name!r}; "
            "falling back to the bundled watchlist."
        )
        diagnostics.append(f"Presidio error: {exc}")
        return None


def scan_path(
    input_path: Path,
    entities: set[str] | None = None,
    extra_watchlists: list[Path] | None = None,
    employee_rosters: list[Path] | None = None,
    include_default_names: bool = True,
    name_scope: str = "context",
    skip_root: Path | None = None,
    use_presidio: bool = True,
    presidio_model: str = "it_core_news_sm",
    diagnostics: list[str] | None = None,
) -> list[Finding]:
    selected = entities or DEFAULT_ENTITIES
    diag = diagnostics if diagnostics is not None else []
    roster_names, roster_matriculas = load_employee_rosters(employee_rosters)
    names = load_names(extra_watchlists, include_default=include_default_names)
    name_regex = compile_name_regex(names) if "NAME" in selected else None
    roster_name_regex = (
        compile_name_regex(roster_names, min_single_token_length=2) if "NAME" in selected else None
    )
    roster_matricola_regex = (
        compile_matricola_regex(roster_matriculas) if "MATRICOLA" in selected else None
    )
    if employee_rosters:
        diag.append(
            "Loaded employee roster entries: "
            f"{len(roster_names)} name variants, {len(roster_matriculas)} matriculas."
        )
    presidio_analyzer = (
        build_presidio_analyzer(presidio_model, diag)
        if use_presidio and "NAME" in selected
        else None
    )
    findings: list[Finding] = []
    roster_paths = {path.resolve() for path in employee_rosters or []}
    for path in iter_text_files(input_path, skip_root=skip_root):
        if path.resolve() in roster_paths:
            continue
        findings.extend(
            scan_file(
                path,
                input_path,
                selected,
                name_regex,
                roster_name_regex,
                roster_matricola_regex,
                name_scope,
                presidio_analyzer=presidio_analyzer,
            )
        )
    return remove_overlaps(findings)


def scan_file(
    path: Path,
    input_path: Path,
    entities: set[str],
    name_regex: re.Pattern[str] | None,
    roster_name_regex: re.Pattern[str] | None,
    roster_matricola_regex: re.Pattern[str] | None,
    name_scope: str,
    presidio_analyzer: object | None = None,
) -> list[Finding]:
    text = read_text(path)
    rel_file = relative_name(path, input_path)
    findings: list[Finding] = []

    if "EMAIL" in entities:
        findings.extend(regex_findings(text, rel_file, "EMAIL", EMAIL_RE, 0.98))
    if "IBAN" in entities:
        findings.extend(regex_findings(text, rel_file, "IBAN", IBAN_RE, 0.96))
    if "CODICE_FISCALE" in entities:
        findings.extend(regex_findings(text, rel_file, "CODICE_FISCALE", CODICE_FISCALE_RE, 0.96))
    if "MATRICOLA" in entities:
        findings.extend(scan_matricola(text, rel_file))
        if roster_matricola_regex:
            findings.extend(scan_roster_matriculas(text, rel_file, roster_matricola_regex))
    if "PHONE" in entities:
        findings.extend(regex_findings(text, rel_file, "PHONE", PHONE_LABEL_RE, 0.78, group="value"))
    if "NAME" in entities:
        name_findings: list[Finding] = []
        if presidio_analyzer:
            name_findings.extend(scan_presidio_names(text, rel_file, presidio_analyzer, name_scope))
        if name_regex:
            name_findings.extend(scan_watchlist_names(text, rel_file, name_regex, name_scope))
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
        findings.extend(remove_overlaps(name_findings))

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


def scan_roster_matriculas(
    text: str,
    rel_file: str,
    matricola_regex: re.Pattern[str],
) -> list[Finding]:
    findings: list[Finding] = []
    for match in matricola_regex.finditer(text):
        start, end = match.start(), match.end()
        value = text[start:end]
        line, column = line_column(text, start)
        findings.append(
            Finding(
                file=rel_file,
                entity_type="MATRICOLA",
                text=value,
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=0.99,
                context=context_for(text, start, end),
                source="employee_roster",
            )
        )
    return findings


def is_probable_matricola_value(value: str) -> bool:
    cleaned = value.strip().strip("\"'")
    upper = cleaned.upper()
    if upper in MATRICOLA_STOP_VALUES:
        return False
    return bool(MATRICOLA_VALUE_RE.fullmatch(cleaned))


def scan_presidio_names(
    text: str,
    rel_file: str,
    analyzer: object,
    scope: str,
) -> list[Finding]:
    findings: list[Finding] = []
    ranges = name_scan_ranges(text, scope)
    try:
        results = analyzer.analyze(text=text, language="it", entities=["PERSON"])
    except Exception:
        return []
    for result in results:
        start, end = trim_span(text, result.start, result.end)
        start, end = trim_name_stopwords(text, start, end)
        if (
            start >= end
            or not offset_in_ranges(start, end, ranges)
            or is_inside_email_or_url(text, start, end)
        ):
            continue
        value = text[start:end]
        if is_probable_name_false_positive(value):
            continue
        line, column = line_column(text, start)
        findings.append(
            Finding(
                file=rel_file,
                entity_type="NAME",
                text=value,
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=float(result.score),
                context=context_for(text, start, end),
                source="presidio_spacy",
            )
        )
    return findings


def is_probable_name_false_positive(value: str) -> bool:
    normalized = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if not normalized:
        return True
    if any(char.isdigit() for char in normalized):
        return True
    if any(char in normalized for char in ("=", "'", '"')):
        return True
    if "-" in normalized:
        return True
    words = normalized.upper().split()
    if len(words) == 1 and normalized.isupper():
        return True
    technical_words = {
        "CALL",
        "COMP",
        "DISPLAY",
        "DIVISION",
        "ELSE",
        "END",
        "IF",
        "MOVE",
        "PERFORM",
        "PIC",
        "SECTION",
        "THEN",
        "TO",
        "USING",
        "VALUE",
        "WHEN",
        *NAME_STOPWORDS,
    }
    return any(word in technical_words for word in words)


def scan_watchlist_names(
    text: str,
    rel_file: str,
    name_regex: re.Pattern[str],
    scope: str,
    source: str = "watchlist",
    confidence: float = 0.7,
) -> list[Finding]:
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
    raw = remove_overlaps(raw)
    return merge_adjacent_names(text, raw)


def merge_adjacent_names(text: str, findings: list[Finding]) -> list[Finding]:
    merged: list[Finding] = []
    pending: Finding | None = None
    for finding in sorted(findings, key=lambda item: (item.file, item.start, item.end)):
        if pending is None:
            pending = finding
            continue
        gap = text[pending.end : finding.start]
        same_file = pending.file == finding.file
        same_line = pending.line == finding.line
        if same_file and same_line and gap and gap.strip() == "":
            start, end = pending.start, finding.end
            line, column = line_column(text, start)
            pending = Finding(
                file=pending.file,
                entity_type="NAME",
                text=text[start:end],
                start=start,
                end=end,
                line=line,
                column=column,
                confidence=max(pending.confidence, finding.confidence),
                context=context_for(text, start, end),
                source=pending.source if pending.source == finding.source else "mixed",
            )
        else:
            merged.append(pending)
            pending = finding
    if pending is not None:
        merged.append(pending)
    return merged


def remove_overlaps(findings: list[Finding]) -> list[Finding]:
    priority = {
        "CODICE_FISCALE": 100,
        "IBAN": 95,
        "EMAIL": 90,
        "MATRICOLA": 80,
        "PHONE": 75,
        "NAME": 60,
    }
    ordered = sorted(
        findings,
        key=lambda item: (
            item.file,
            item.start,
            -priority.get(item.entity_type, 0),
            -(item.end - item.start),
        ),
    )
    kept: list[Finding] = []
    for finding in ordered:
        overlaps = [
            existing
            for existing in kept
            if existing.file == finding.file
            and not (finding.end <= existing.start or finding.start >= existing.end)
        ]
        if not overlaps:
            kept.append(finding)
            continue
        best = max(
            overlaps,
            key=lambda item: (priority.get(item.entity_type, 0), item.end - item.start),
        )
        finding_rank = (priority.get(finding.entity_type, 0), finding.end - finding.start)
        best_rank = (priority.get(best.entity_type, 0), best.end - best.start)
        if finding_rank > best_rank:
            kept = [item for item in kept if item not in overlaps]
            kept.append(finding)
    return sorted(kept, key=lambda item: (item.file, item.start, item.end))
