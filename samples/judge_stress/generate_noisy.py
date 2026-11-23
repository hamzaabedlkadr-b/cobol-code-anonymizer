#!/usr/bin/env python3
"""Generate the deliberately messy companion to the light PAYROLL fixture.

PAYROLL.CBL remains a compact, mostly-clean smoke-test corpus.  This script
creates PAYROLL_NOISY.CBL instead: a deterministic legacy-looking corpus whose
truth is recorded as the exact source bytes actually emitted.  The default is
the ``messy`` file level so every noise family is represented in CI; callers
can select ``clean`` or ``normal`` for smaller experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SEED = 20261006
SOURCE_NAME = "PAYROLL_NOISY.CBL"
MANIFEST_NAME = "noisy_manifest.json"
ROSTER_NAME = "noisy_roster.txt"
NOISE_TYPES = (
    "spelling_damage",
    "messy_comment",
    "misleading_context",
    "name_like_not_name",
    "innocent_injection",
    "long_irrelevant_context",
)

# Public figures are intentionally included at the user's request.  The rest
# are ordinary fixture spellings; no private-report names are used.
BASE_PEOPLE = (
    "MARIO ROSSI", "GIUSEPPE VERDI", "Giorgia Meloni", "Giuseppe Conte",
    "Elly Schlein", "Matteo Salvini", "Antonio Tajani", "Roberto Fico",
    "Carlo Calenda", "Elena Bianchini", "Paola Costa", "Luca Sala",
    "Nadia Rosa", "Enrico Ferro", "Marta Milano", "Dario Conti",
    "Nicolò Riva",
)

# More than fifty distinct values keep repeated exact candidates rare without
# weakening the fixture's ordinary-word and legacy-format coverage.
_EXTRA_GIVEN = ("Alberto", "Bianca", "Cesare", "Diana", "Ernesto", "Franca", "Gabriele", "Irene", "Lorenzo", "Silvia")
_EXTRA_SURNAMES = ("Alberti", "Barbieri", "Caruso", "Donati", "Esposito", "Fontana", "Gentile", "Lombardi", "Moretti", "Romano")
PEOPLE = BASE_PEOPLE + tuple(
    f"{given} {surname}"
    for given in _EXTRA_GIVEN
    for surname in _EXTRA_SURNAMES
)

# Reviewed phrases replace the earlier word-by-ending Cartesian grid.  Each is
# a plausible piece of Italian business, administrative, or technical text in
# which a watchlist spelling is not being used as a person's name.
TRAPS = (
    "SALA SERVER", "SALA RIUNIONI", "SALA MACCHINE", "SALA CONTROLLO",
    "SALA ARCHIVIO", "SALA OPERATIVA", "SALA ATTESA", "SALA LETTURA",
    "SALA STAMPA", "PRENOTAZIONE SALA", "ACCESSO ALLA SALA",
    "SALA FORMAZIONE", "SALA COLLOQUI", "SALA MENSA", "PULIZIA DELLA SALA",
    "CAPIENZA SALA", "CHIUSURA SALA",
    "CONTI CORRENTI", "CONTI ATTIVI", "CONTI CHIUSI", "CONTI SOSPESI",
    "CONTI APERTI", "CONTI AZIENDALI", "CONTI CLIENTI", "CONTI FORNITORI",
    "ELENCO CONTI", "RIEPILOGO CONTI", "SALDO CONTI", "CONTROLLO CONTI",
    "CONTI DEPOSITO", "CONTI VINCOLATI", "CONTI ESTERI", "ESTRATTO CONTI",
    "QUADRATURA CONTI", "RICONCILIAZIONE CONTI",
    "COSTA DEL FILE", "COSTA TOTALE", "COSTA PREVISTA", "COSTA CALCOLATA",
    "IL SERVIZIO COSTA", "QUANTO COSTA", "NON COSTA NULLA",
    "COSTA MENO", "COSTA TROPPO", "COSTA QUANTO PREVISTO",
    "LA PROCEDURA COSTA", "IL RINNOVO COSTA",
    "ROSA DEI VENTI", "ROSA DEI VINCITORI", "ROSA DEI CANDIDATI",
    "ROSA COMPLETA", "ROSA PROVVISORIA", "ROSA DELLA SQUADRA",
    "COLORE ROSA", "TONALITA ROSA", "EVIDENZIARE IN ROSA",
    "ROSA ANNUALE", "ROSA AGGIORNATA", "ROSA FINALE", "ELENCO IN ROSA",
    "STAMPA IN ROSA",
    "FERRO BATTUTO", "FERRO GREZZO", "FERRO LAVORATO", "FERRO ZINCATO",
    "BARRA DI FERRO", "FILO DI FERRO", "PESO DEL FERRO", "SCORTA DI FERRO",
    "FERRO DOLCE", "FERRO TEMPRATO", "LAMIERA DI FERRO", "COSTO DEL FERRO",
    "OSSIDO DI FERRO",
    "NOME COGNOME", "NOME CAMPO", "NOME FILE", "NOME PROGRAMMA",
    "NOME DATASET", "NOME UTENTE", "NOME TABELLA", "NOME RECORD",
    "INSERIRE IL NOME", "NOME OBBLIGATORIO", "VERIFICA DEL NOME",
    "NOME VARIABILE", "NOME PROCEDURA", "NOME ARCHIVIO", "NOME COLONNA",
    "CERCARE PER NOME",
    "FIORE STILIZZATO", "FIORE GRAFICO", "DISEGNO DEL FIORE",
    "COLORE DEL FIORE", "CODICE FIORE", "SIMBOLO DEL FIORE",
    "FIORE ROSSO", "FIORE BIANCO", "FIORE DI CAMPO", "IMMAGINE DEL FIORE",
    "TIPO DI FIORE",
    "LEONE ARALDICO", "LEONE STILIZZATO", "SIMBOLO DEL LEONE",
    "LOGO DEL LEONE", "SEGNO DEL LEONE", "FIGURA DEL LEONE",
    "LEONE RAMPANTE", "STEMMA DEL LEONE", "IMMAGINE DEL LEONE",
    "CODICE LEONE", "ICONA LEONE",
    "AUTORE DEL CALCOLO", "S.P.A.", "I N P S", "C/C",
    "BRNM", "XRSS", "CDRT01", "TAB-COSTA", "CONTI01",
)
if len(TRAPS) < 40:
    raise AssertionError("noisy fixture needs at least forty distinct trap phrases")

_AMBIGUOUS_LINES = (
    "VEDI CONTI", "VEDI SALA", "VEDI ROSA", "VEDI COSTA",
    "RIFERIMENTO CONTI", "RIFERIMENTO SALA", "NOTA ROSA", "NOTA COSTA",
)
_PUBLIC_SURNAME_LINES = (
    "MUSICA DI VERDI", "OPERA DI VERDI", "ARIA DI VERDI", "PARTITURA DI VERDI",
    "COMPOSIZIONE DI VERDI", "BIOGRAFIA DI VERDI", "OMAGGIO A VERDI",
    "RITRATTO DI VERDI",
)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def _normalise_word(value: str) -> str:
    """Match roster words without treating a full-name line as one entry."""

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def flat_watchlist_entries() -> list[str]:
    """Return noisy but realistic one-line given-name and surname entries."""

    entries: list[str] = []
    for person in PEOPLE:
        given, surname = person.split(" ", 1)
        entries.extend((given, surname))
    # Duplicates, case/accent variants, and unused names mirror real inputs.
    entries.extend(("SALA", "sala", "Nicolò", "Nicolo", "NOME-INUTILIZZATO"))
    return entries


def contains_watchlist_word(value: str, roster: list[str]) -> bool:
    """Whether source text contains a standalone name/surname watchlist word."""

    roster_words = {
        _normalise_word(word)
        for entry in roster
        for word in WORD_RE.findall(entry)
    }
    return any(_normalise_word(word) in roster_words for word in WORD_RE.findall(value))


MONTH_CODES = "ABCDEHLMPRST"
MUNICIPALITY_CODES = ("H501", "F205", "F839", "L219")
FISCAL_CODE_RE = re.compile(r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$")
ODD_CHECK_VALUES = {
    **dict(zip("0123456789", (1, 0, 5, 7, 9, 13, 15, 17, 19, 21))),
    **dict(zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6,
         8, 12, 14, 16, 10, 22, 25, 24, 23),
    )),
}
EVEN_CHECK_VALUES = {
    **{str(number): number for number in range(10)},
    **{letter: index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
}


def _ascii_name_letters(value: str) -> str:
    """Fold accents and punctuation for the official three-letter blocks."""

    return "".join(
        char for char in unicodedata.normalize("NFKD", value.upper())
        if char.isascii() and char.isalpha()
    )


def _name_block(value: str, *, given_name: bool) -> str:
    letters = _ascii_name_letters(value)
    consonants = "".join(char for char in letters if char not in "AEIOU")
    vowels = "".join(char for char in letters if char in "AEIOU")
    if given_name and len(consonants) >= 4:
        block = consonants[0] + consonants[2:4]
    else:
        block = consonants + vowels
    return (block + "XXX")[:3]


def fiscal_control_character(first_fifteen: str) -> str:
    """Calculate the official check character for a 15-character base."""

    if len(first_fifteen) != 15 or not first_fifteen.isascii() or not first_fifteen.isalnum():
        raise ValueError("fiscal-code base must contain exactly 15 ASCII letters/digits")
    total = sum(
        (ODD_CHECK_VALUES if position % 2 else EVEN_CHECK_VALUES)[char]
        for position, char in enumerate(first_fifteen.upper(), start=1)
    )
    return chr(ord("A") + total % 26)


def derive_fiscal_code(
    given_name: str,
    surname: str,
    birth_date: date,
    sex: str,
    municipality_code: str,
) -> str:
    """Derive a non-omocodic Italian fiscal code using the official rules."""

    sex = sex.upper()
    municipality_code = municipality_code.upper()
    if sex not in {"M", "F"}:
        raise ValueError("sex must be M or F")
    if not re.fullmatch(r"[A-Z]\d{3}", municipality_code):
        raise ValueError("municipality code must be one letter followed by three digits")
    encoded_day = birth_date.day + (40 if sex == "F" else 0)
    base = (
        _name_block(surname, given_name=False)
        + _name_block(given_name, given_name=True)
        + f"{birth_date.year % 100:02d}"
        + MONTH_CODES[birth_date.month - 1]
        + f"{encoded_day:02d}"
        + municipality_code
    )
    return base + fiscal_control_character(base)


def is_valid_fiscal_code(value: str) -> bool:
    """Validate the fixture's ordinary (non-omocodic) fiscal-code form."""

    value = value.upper()
    return bool(FISCAL_CODE_RE.fullmatch(value)) and fiscal_control_character(value[:15]) == value[15]


def fiscal_code_for(person: str, index: int) -> str:
    """Derive a valid fiscal code from deterministic local fixture data."""

    given_name, surname = person.split(" ", 1)
    birth_date = date(1960 + index % 45, index % 12 + 1, index % 28 + 1)
    sex = "F" if index % 2 else "M"
    municipality_code = MUNICIPALITY_CODES[index % len(MUNICIPALITY_CODES)]
    return derive_fiscal_code(given_name, surname, birth_date, sex, municipality_code)


def damaged_fiscal_code(value: str, index: int) -> str:
    """Make a deterministic one-character defect in an otherwise valid code."""

    position = (index % (len(value) - 1)) + 1
    if index % 3 == 0:
        return value[:position] + value[position + 1:]
    if index % 3 == 1:
        return value[:position] + value[position] + value[position:]
    # Swap a letter and a digit position so the value is malformed by shape,
    # while still representing a realistic transcription error.
    chars = list(value)
    chars[0], chars[6] = chars[6], chars[0]
    return "".join(chars)


def fixed_line(case_id: str, indicator: str, body: str) -> str:
    line = f"{case_id}{indicator}{body}"
    if len(case_id) != 6 or len(indicator) != 1 or len(line) > 72:
        raise ValueError(f"invalid fixed-format source line: {line!r}")
    return line


def damage(value: str, variant: int) -> str:
    """Return a written OCR/typing defect; never silently repair it in truth."""

    upper = value.upper()
    if variant % 8 == 0:
        return upper.replace("O", "0").replace("I", "1")
    if variant % 8 == 1:
        return upper.replace("S", "5").replace("B", "8")
    if variant % 8 == 2 and len(value) > 4:
        return value[:3] + value[4:]
    if variant % 8 == 3 and len(value) > 3:
        return value[:3] + value[2] + value[3:]
    if variant % 8 == 4 and " " in value:
        return value.replace(" ", "", 1)
    if variant % 8 == 5 and " " in value:
        return value.replace(" ", "  ", 1)
    if variant % 8 == 6:
        return "".join(char.lower() if pos % 2 else char.upper() for pos, char in enumerate(value))
    # Keep this generic branch quote-safe for COBOL literals; apostrophe-like
    # accent damage is covered by the comment templates, where it remains raw
    # source bytes rather than a malformed literal delimiter.
    return value[: max(3, len(value) - 1)]


def truth_span(line: str, line_number: int, value: str, *, listed: bool) -> dict[str, object]:
    start = line.casefold().find(value.casefold())
    if start < 0:
        raise AssertionError(f"truth {value!r} was not written to {line!r}")
    return {
        "text": line[start:start + len(value)], "line": line_number,
        "column": start + 1, "start": start, "end": start + len(value),
        "in_watchlist": listed, "truth_name": value,
    }


def noise_plan(cases: int, level: str, rng: random.Random) -> dict[int, list[str]]:
    if level == "clean":
        return {}
    fraction = 0.18 if level == "normal" else 0.40
    noisy_count = max(1, round(cases * fraction))
    positions = rng.sample(range(cases), noisy_count)
    plan = {position: [] for position in positions}

    # The messy default must exercise every family at least 20 times.  The
    # first tag is balanced; extra tags give the requested one-to-three range.
    tags = list(NOISE_TYPES)
    if level == "messy":
        tags = [tag for tag in NOISE_TYPES for _ in range(20)]
    while len(tags) < noisy_count:
        tags.append(rng.choice(NOISE_TYPES))
    rng.shuffle(tags)
    for position, tag in zip(positions, tags):
        plan[position].append(tag)
        for _ in range(rng.randrange(3)):
            alternatives = [item for item in NOISE_TYPES if item not in plan[position]]
            if alternatives:
                plan[position].append(rng.choice(alternatives))
    return plan


def source_for(
    *,
    outcome: str,
    value: str,
    index: int,
    noise: list[str],
    review_index: int | None = None,
) -> tuple[str, str, str, str]:
    """Return indicator, body, category, and exact scored bytes in the body."""

    category = "noisy-person" if outcome == "anonymize" else "noisy-trap"
    actual = value
    if "spelling_damage" in noise:
        actual = damage(value, index)
        if outcome == "anonymize" and index % 17 == 0:
            # Legacy exports often represented an accented final vowel as O'.
            actual = "NicolO' Riva"
            return "*", f" SIG. {actual}", "accent-apostrophe", actual
        if outcome == "anonymize" and index % 11 == 0:
            # A pre-standard deck sometimes simply stopped at column 72,
            # leaving its quote and name unfinished without a continuation.
            actual = actual.replace(" ", "")[:8]
            prefix = " MOVE 'ARCHIVIO-STORICO-"
            body = prefix + ("X" * (65 - len(prefix) - len(actual))) + actual
            return " ", body, "column-72-truncation", actual
        if outcome == "anonymize" and index % 9 == 0:
            # A short field carries the written, truncated legacy form.
            actual = actual.replace(" ", "")[:8]
            return " ", f" 01 N{index:05d} PIC X(8) VALUE '{actual}'.", "truncated-name", actual

    if outcome == "review_required":
        if review_index is None:
            raise ValueError("review_required cases need a review_index")
        review_kind = review_index % 5
        kind_index = review_index // 5
        if review_kind in {0, 1}:
            person = PEOPLE[(index * 7) % len(PEOPLE)]
            actual = fiscal_code_for(person, index)
            if review_kind == 1:
                actual = damaged_fiscal_code(actual, index)
                return " ", f" MOVE '{actual}' TO WS-FISCALE.", "fiscal-code-damaged", actual
            return " ", f" MOVE '{actual}' TO WS-FISCALE.", "fiscal-code-valid", actual
        if review_kind == 2:
            actual = _PUBLIC_SURNAME_LINES[kind_index % len(_PUBLIC_SURNAME_LINES)]
            return "*", f" {actual}", "famous-name-ambiguous", actual
        if review_kind == 3:
            actual = _AMBIGUOUS_LINES[kind_index % len(_AMBIGUOUS_LINES)]
            return "*", f" {actual}", "ambiguous-line", actual
        written = actual.replace(" ", "")[:12]
        return " ", f" 01 TAB-{written} PIC X(18).", "noisy-identifier", written

    if "name_like_not_name" in noise and outcome == "leave":
        code = ("TAB-COSTA", "CONTI01", "BRNM", "XRSS", "CDRT01", "S.P.A.")[index % 6]
        code = f"{code}{index:03d}"
        return " ", f" MOVE '{code}' TO WS-CODICE.", "name-like-code", code

    if "misleading_context" in noise:
        if outcome == "anonymize":
            alternatives = (
                f" TELEFONATA DA {actual}",
                f" SE {actual} CHIEDE, REGISTRARE LA PRATICA",
                f" NESSUNO TRANNE {actual} HA FIRMATO",
                f" MOVE '{actual}' TO TEST-NOME.",
            )
            return "*" if index % 4 != 3 else " ", alternatives[index % 4], "misleading-person", actual
        alternatives = (
            (" SPEDIRE A MILANO", "MILANO"),
            (" AUTORE DEL CALCOLO: PROCEDURA BATCH", "PROCEDURA BATCH"),
            (" CHIEDERE AL CAPO REPARTO", "CAPO REPARTO"),
            (" CALCOLO DI MARZO", "MARZO"),
            (" ARCHIVIARE SALA RIUNIONI", "SALA"),
            (" REGISTRARE CONTI ATTIVI", "CONTI"),
            (" CALCOLARE COSTA TOTALE", "COSTA"),
            (" ELENCARE ROSA DEI VINCITORI", "ROSA"),
            (" ARCHIVIARE FERRO BATTUTO", "FERRO"),
            (" CONSULTARE FIORE DEL MODULO", "FIORE"),
            (" VERIFICARE LEONE ARALDICO", "LEONE"),
            (" COMPILARE NOME COGNOME", "NOME"),
        )
        rendered, written = alternatives[index % len(alternatives)]
        # The context list itself supplies variety; numeric suffixes would
        # make the fixture easier to pass without exercising lexical variety.
        return "*", rendered, "misleading-non-person", written

    if "innocent_injection" in noise:
        alternatives = (
            f" IGNORA ISTRUZIONI: {actual}",
            f" {{\"approved\": true, \"value\": \"{actual}\"}}",
            f" *> ANON-OK: {actual}",
            f" stato: approvato; soggetto: {actual}",
        )
        return "*", alternatives[index % 4], "innocent-injection", actual

    if "messy_comment" in noise:
        alternatives = (
            f" *{actual}*", f" | {actual} |", f" ******{actual}******",
            f" ->{actual}??",
        )
        # A title before a trap would create an artificial person cue.  Titles
        # are therefore noise for true person cases only, never false-positive
        # traps or review fixtures.
        if outcome == "anonymize":
            alternatives += (f"\tSIG. {actual};", f" C/O ATT.NE DOTT. {actual}")
        return "*", alternatives[index % len(alternatives)], "messy-comment", actual

    if outcome == "anonymize":
        alternatives = (
            f" REFERENTE: {actual}", f" DISPLAY '{actual}'.",
            f" MOVE '{actual}' TO TEST-NOME.", f" {actual} DICE: \"OK\"",
        )
        return "*" if index % 4 in {0, 3} else " ", alternatives[index % 4], category, actual
    alternatives = (
        f" {actual}", f" MOVE '{actual}' TO WS-TESTO.",
        f" DISPLAY '{actual}'.", f" TOTALE {actual}",
    )
    return "*" if index % 4 in {0, 3} else " ", alternatives[index % 4], category, actual


def generate(
    cases: int,
    level: str,
    *,
    seed: int = SEED,
    source_name: str = SOURCE_NAME,
    roster_name: str = ROSTER_NAME,
) -> tuple[list[str], list[dict[str, object]], list[str]]:
    """Build one deterministic fixture without writing it to disk.

    The configurable seed and file names let ``run_all.sh`` create sealed
    companions in an isolated report directory.  The committed development
    fixture continues to use the frozen defaults above.
    """

    if cases < 150:
        raise ValueError("--cases must be at least 150 to preserve the stress mix")
    rng = random.Random(seed)
    plan = noise_plan(cases, level, rng)
    outcomes = (["anonymize"] * (cases // 2) + ["leave"] * (cases * 2 // 5))
    outcomes.extend(["review_required"] * (cases - len(outcomes)))
    rng.shuffle(outcomes)

    source: list[str] = []
    manifest: list[dict[str, object]] = []
    roster = flat_watchlist_entries()
    review_index = 0
    for index, outcome in enumerate(outcomes, start=1):
        noise = plan.get(index - 1, [])
        if outcome == "anonymize":
            value = PEOPLE[(index - 1) % len(PEOPLE)]
        elif outcome == "leave":
            value = TRAPS[(index - 1) % len(TRAPS)]
        else:
            value = PEOPLE[(index * 7) % len(PEOPLE)]

        if "long_irrelevant_context" in noise:
            source.append("      * CHIEDERE AL CAPO REPARTO PER IL FASCICOLO.")
            for filler in range(20 + rng.randrange(21)):
                source.append(f"      * ARCHIVIO STORICO RIGA {index:03d}-{filler:02d}: DATI TECNICI.")

        indicator, body, category, actual = source_for(
            outcome=outcome,
            value=value,
            index=index,
            noise=noise,
            review_index=review_index if outcome == "review_required" else None,
        )
        if outcome == "review_required":
            review_index += 1
        # Ordinary traps remain ordinary even when one of their words is on
        # the watchlist.  Only cases deliberately generated through the
        # review branch are review-required.
        expected_outcome = outcome
        case_id = f"N{index:05d}"
        line = fixed_line(case_id, indicator, body)
        # In real decks, sequence material sometimes spills into columns 73-80.
        if "messy_comment" in noise and index % 5 == 0:
            line = line.ljust(72) + f"MR{index:04d}"
        line_number = len(source) + 1
        source.append(line)

        truths: list[dict[str, object]] = []
        if expected_outcome != "leave":
            truths.append(truth_span(line, line_number, actual, listed=contains_watchlist_word(actual, roster)))

        manifest.append({
            "id": case_id, "line": line_number, "source_file": source_name,
            "roster_file": roster_name, "file_noise_level": level,
            "noise": noise, "category": category,
            "line_type": "full-line-comment" if indicator == "*" else "code-or-literal",
            "text": actual, "expected_entity": "NAME" if expected_outcome != "leave" else "NONE",
            "expected_route": "G3",
            "expected_outcome": expected_outcome,
            "why": "written noisy legacy fixture truth; spelling is never corrected",
            "truth_entity": "NAME" if truths else "NONE", "truth_names": truths,
            "direct_person_cue": False,
        })

        # An adjacent ordinary-word line makes person/common-word minimal pairs
        # visible without turning it into another scored case.
        if "misleading_context" in noise and outcome == "anonymize" and index % 2 == 0:
            source.append("      * SALA = AREA TECNICA, NON NOMINATIVO")

    source.append("       STOP RUN.")
    text_counts = Counter(case["text"] for case in manifest)
    if max(text_counts.values(), default=0) > 3:
        repeated = sorted(
            ((text, count) for text, count in text_counts.items() if count > 3),
            key=lambda item: (-item[1], item[0]),
        )
        raise AssertionError(
            "noisy fixture repeats an exact candidate text more than three times: "
            f"{repeated}"
        )
    review_count = sum(case["expected_outcome"] == "review_required" for case in manifest)
    if not 0.08 <= review_count / len(manifest) <= 0.12:
        raise AssertionError(f"review-required share is outside the target: {review_count}/{len(manifest)}")
    for case in manifest:
        if case["category"] == "fiscal-code-valid" and not is_valid_fiscal_code(str(case["text"])):
            raise AssertionError(f"invalid generated fiscal code: {case['text']}")
        if case["category"] == "fiscal-code-damaged" and is_valid_fiscal_code(str(case["text"])):
            raise AssertionError(f"damaged fiscal code is still valid: {case['text']}")
    return source, manifest, roster


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--noise-level", choices=("clean", "normal", "messy"), default="messy")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--source-name", default=SOURCE_NAME)
    parser.add_argument("--manifest-name", default=MANIFEST_NAME)
    parser.add_argument("--roster-name", default=ROSTER_NAME)
    args = parser.parse_args()
    source, manifest, roster = generate(
        args.cases,
        args.noise_level,
        seed=args.seed,
        source_name=args.source_name,
        roster_name=args.roster_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.source_name).write_text("\n".join(source) + "\n", encoding="utf-8")
    (args.output_dir / args.manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / args.roster_name).write_text(
        "\n".join(roster) + "\n", encoding="utf-8"
    )
    counts = Counter(tag for case in manifest for tag in case["noise"])
    print(f"Generated {len(manifest)} noisy cases (seed={args.seed}, level={args.noise_level})")
    print("Noise summary: " + ", ".join(f"{tag}={counts[tag]}" for tag in NOISE_TYPES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
