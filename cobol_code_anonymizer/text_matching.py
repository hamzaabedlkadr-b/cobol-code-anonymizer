"""Safe text-normalisation helpers for anchoring model answers.

The local models copy source text back to the application.  Their copies can
contain harmless presentation differences that must not turn a valid answer
into an error: prompt-only ``[[...]]`` markers, repeated whitespace, letter
case, or a source apostrophe written as either a Unicode apostrophe or COBOL's
doubled quote.  This module normalises only for comparison.  It never changes
the audited model answer or the source offsets used for replacement.
"""

from __future__ import annotations

import re
import hashlib
import json
import unicodedata
from bisect import bisect_right
from dataclasses import replace
from .source_reader import source_line_starts


def name_model_input(context: str, span: str) -> str:
    """Give both models the same marked source text and span."""
    return (
        "BEGIN_UNTRUSTED_COBOL\n"
        f"{context}\n"
        "END_UNTRUSTED_COBOL\n"
        f'Highlighted span (untrusted data): "{span}"'
    )


def name_word_spans(value: str) -> list[tuple[int, int]]:
    """Split name text into whole words, keeping apostrophes and hyphens."""
    return [(match.start(), match.end()) for match in
            re.finditer(r"[^\W\d_]+(?:['’\-]+[^\W\d_]+)*", value)]


def span_is_code(layout, start: int, end: int) -> bool:
    """Return whether a candidate touches a code region of the source layout."""
    index = max(0, bisect_right(layout.region_starts, start) - 1)
    for index in range(index, len(layout.regions)):
        region = layout.regions[index]
        if region.decoded_start >= end:
            break
        if region.kind == "code" and start < region.decoded_end:
            return True
    return False


def normalize_evidence_text(value: str) -> str:
    """Remove prompt markers and collapse whitespace for quote comparison."""

    return " ".join(value.replace("[[", "").replace("]]", "").split())


def evidence_quote_is_anchored(quote: str, context: str) -> bool:
    """Return whether a copied quote safely matches the supplied source line.

    Exact normalised matching is preferred. Small local models sometimes add
    quote marks, omit a sequence area, or make a one- or two-character copy
    error, so comparison also accepts edit distance at most two against one
    contiguous context fragment. This validates audit evidence only; it never
    supplies source offsets or makes a name decision.
    """

    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    candidates = [normalize_evidence_text(context)]
    stripped_lines = [
        line[7:] if len(line) > 7 else line
        for line in context.replace("[[", "").replace("]]", "").splitlines()
    ]
    candidates.append(normalize_evidence_text("\\n".join(stripped_lines)))
    for candidate in candidates:
        if normalized_quote in candidate:
            return True
        if _matches_within_edit_distance(normalized_quote, candidate, limit=2):
            return True
    return False


def _matches_within_edit_distance(quote: str, context: str, *, limit: int) -> bool:
    """Check short contiguous source fragments without a fuzzy whole-line match."""

    lower = max(1, len(quote) - limit)
    upper = min(len(context), len(quote) + limit)
    for size in range(lower, upper + 1):
        for start in range(0, len(context) - size + 1):
            if _edit_distance_at_most(quote, context[start : start + size], limit):
                return True
    return False


def _edit_distance_at_most(left: str, right: str, limit: int) -> bool:
    """Return whether two short strings differ by no more than the limit."""

    if abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        smallest = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            smallest = min(smallest, value)
        if smallest > limit:
            return False
        previous = current
    return previous[-1] <= limit


def normalize_person_text(value: str) -> str:
    """Canonicalise harmless copied-text differences for comparisons.

    Models occasionally collapse spacing while copying a name from a fixed
    width source line.  Space runs, source apostrophe spelling, and case are
    presentation details; source offsets are still always taken from the
    original text after a successful match.
    """

    return " ".join(
        value.replace("''", "'").replace("’", "'").casefold().split()
    )


def equivalent_person_occurrences(
    candidate: str,
    person_text: str,
) -> list[tuple[int, int]]:
    """Locate person text while accepting case and apostrophe equivalents.

    One apostrophe in a model answer can match an ASCII apostrophe, a Unicode
    right apostrophe, or the doubled ASCII quote used inside COBOL literals.
    Returned spans always refer to the original candidate, so downstream word
    boundary checks and replacements remain source accurate.
    """

    if not person_text:
        return []

    pieces: list[str] = []
    index = 0
    while index < len(person_text):
        character = person_text[index]
        if character.isspace():
            while index + 1 < len(person_text) and person_text[index + 1].isspace():
                index += 1
            pieces.append(r"\s+")
        elif character in {"'", "’"}:
            # Treat a doubled ASCII quote as one logical apostrophe token.
            if character == "'" and index + 1 < len(person_text):
                if person_text[index + 1] == "'":
                    index += 1
            pieces.append("(?:''|'|’)")
        else:
            pieces.append(re.escape(character))
        index += 1

    pattern = "".join(pieces)
    return [
        (match.start(1), match.end(1))
        for match in re.finditer(
            f"(?=({pattern}))",
            candidate,
            flags=re.IGNORECASE,
        )
    ]


_PERSON_WORD_RE = re.compile(r"[^\W_]+(?:(?:''|'|’)[^\W_]+)*", re.UNICODE)
WATCHLIST_TOKEN_RE = _PERSON_WORD_RE
IDENTIFIER_PART_RE = re.compile(r"[^\W\d_]+(?:(?:''|'|’)[^\W\d_]+)*", re.UNICODE)

def _word_matches_with_tolerance(source_word: str, copied_word: str) -> bool:
    """Compare one copied word without accepting short-word guesses."""

    source = normalize_person_text(source_word)
    copied = normalize_person_text(copied_word)
    if len(copied) <= 3:
        return source == copied
    return _edit_distance_at_most(source, copied, limit=2)


def tolerant_person_occurrences(line: str, person_text: str) -> list[tuple[int, int]]:
    """Find one source span when each copied name word is a near match.

    Exact and apostrophe-equivalent matching runs first. This fallback only
    aligns consecutive whole words and never invents a source spelling: its
    returned offsets always select the text that actually appears in the line.
    """

    copied_words = _PERSON_WORD_RE.findall(person_text)
    source_words = list(_PERSON_WORD_RE.finditer(line))
    if not copied_words or len(copied_words) > len(source_words):
        return []

    matches: list[tuple[int, int]] = []
    width = len(copied_words)
    for start_index in range(len(source_words) - width + 1):
        window = source_words[start_index : start_index + width]
        if all(
            _word_matches_with_tolerance(source_match.group(), copied_word)
            for source_match, copied_word in zip(window, copied_words)
        ):
            matches.append((window[0].start(), window[-1].end()))
    return matches


def output_span_to_source(start: int, end: int, replaced_spans: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    """Map an unchanged output span back to source; reject replacement interiors."""
    shift = 0
    for source_start, source_end, length in replaced_spans:
        output_start = source_start + shift
        output_end = output_start + length
        if start < output_end and output_start < end:
            return None
        if end <= output_start:
            break
        shift += length - (source_end - source_start)
    return start - shift, end - shift


def marked_source_line(text: str, start: int, end: int) -> str:
    """Mark a decoded source span with the reader's physical line bounds."""
    left, _ = source_line_bounds(text, start)
    _, right = source_line_bounds(text, max(start, end - 1))
    return f"{text[left:start]}[[{text[start:end]}]]{text[end:right]}"


def line_column(text: str, offset: int) -> tuple[int, int]:
    starts = source_line_starts(text)
    index = bisect_right(starts, offset) - 1
    return index + 1, offset - starts[index] + 1


def source_line_bounds(text: str, offset: int) -> tuple[int, int]:
    starts = source_line_starts(text)
    index = bisect_right(starts, offset) - 1
    end = starts[index + 1] if index + 1 < len(starts) else len(text)
    while end > starts[index] and text[end - 1] in "\r\n":
        end -= 1
    return starts[index], end


def source_occurrence_id(finding, sha: str) -> str:
    spans = finding.logical.source_spans(finding.logical_start, finding.logical_end) if finding.logical else ((finding.start, finding.end),)
    return hashlib.sha256(json.dumps([finding.file, sha, spans]).encode()).hexdigest()


def source_findings(finding, logical, layout):
    """Map a logical detection and its review key to physical source pieces."""
    source = layout.source.text
    word = logical.text[finding.start:finding.end]
    marked = logical.text[:finding.start] + "[[" + word + "]]" + logical.text[finding.end:]
    lines = dict.fromkeys(source_line_bounds(source, piece.source_start) for piece in logical.pieces)
    review_line = "\n".join(source[a + (6 if layout.format == "fixed_cobol" else 0):b] for a, b in lines)
    key = review_key(logical.kind, review_line)
    for start, end in logical.source_spans(finding.start, finding.end):
        while start < end and source[start].isspace():
            start += 1
        while start < end and source[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        line, column = line_column(source, start)
        yield replace(finding, text=source[start:end], start=start, end=end, line=line, column=column,
                      context=marked_source_line(source, start, end), logical=logical,
                      logical_start=finding.start, logical_end=finding.end,
                      logical_context=marked, logical_candidate=word, review_line=review_line, review_key=key)


def physical_replacement(finding, alias: str) -> str:
    """Write a joined name once, then remove its remaining source pieces."""
    if finding.logical:
        spans = finding.logical.source_spans(finding.logical_start, finding.logical_end)
        if spans and finding.start > spans[0][0]:
            return ""
    return alias


def fold_watchlist_value(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("''", "'").replace("’", "'")).upper().translate(str.maketrans("015", "OIS"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def prepare_watchlist(values):
    """Build token lookups once; pair words need at least three letters."""
    entries = {fold_watchlist_value(word): word for word in values if _PERSON_WORD_RE.fullmatch(word)}
    pairs = {key for key, word in entries.items() if sum(char.isalpha() for char in word) >= 3}
    return entries, pairs


def logical_bounds(logical, source_spans):
    """Map source fragments back into their owning logical record."""
    bounds = []
    for start, end in source_spans:
        piece = next((piece for piece in logical.pieces if piece.source_start <= start < end <= piece.source_end), None)
        if piece is None:
            return None
        bounds.append((piece.logical_start + start - piece.source_start, piece.logical_start + end - piece.source_start))
    return (bounds[0][0], bounds[-1][1]) if bounds else None


def review_key(kind: str, line: str) -> str:
    return hashlib.sha256(json.dumps([kind, line], ensure_ascii=False).encode()).hexdigest()

COBOL_KEYWORDS = frozenset(
    "ACCEPT ADD ALTER CALL CANCEL CLOSE COMPUTE CONFIGURATION CONTINUE COPY "
    "DATA DELETE DISPLAY DIVISION ELSE END-IF EVALUATE EXEC EXIT GO GOBACK "
    "IF INITIALIZE INSPECT INSTALLATION INTO INVOKE OPEN PERFORM PIC PICTURE "
    "MOVE PROCEDURE PROGRAM-ID READ REWRITE SECTION SELECT SET STOP STRING SUBTRACT "
    "THEN UNSTRING VALUE WHEN WORKING-STORAGE WRITE"
    .split()
)


def neighboring_model_context(finding, layout, marked_line):
    """Add adjacent text lines as context, never as anchoring targets."""
    if layout is None:
        return marked_line
    text = layout.source.text
    starts = source_line_starts(text)
    spans = finding.logical.source_spans(0, len(finding.logical.text)) if finding.logical else ((finding.start, finding.end),)
    first = bisect_right(starts, spans[0][0]) - 1
    last = bisect_right(starts, max(spans[-1][0], spans[-1][1] - 1)) - 1
    neighbors = []
    for index in (first - 1, last + 1):
        value = ""
        if 0 <= index < len(starts):
            left, right = source_line_bounds(text, starts[index])
            value = text[left + (6 if layout.format == "fixed_cobol" else 0):right]
        neighbors.append(value.replace("[[", "").replace("]]", ""))
    return "\n".join(line for line in (neighbors[0], marked_line, neighbors[1]) if line)
