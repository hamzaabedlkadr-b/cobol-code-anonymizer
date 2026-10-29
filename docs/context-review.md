# Watchlist-first contextual review

This workflow extracts candidates, asks a local Ollama model to classify each occurrence,
and produces reports. It does not anonymize or change source files. It runs independently
of the experimental name-judge redesign and requires no Presidio/spaCy model download.

## Set up once

Use Python 3.10 or newer. Install Ollama from https://ollama.com and keep its server running.
From this repository, install the command and download the default model:

```powershell
python -m pip install -e .
ollama pull qwen3.5:9b
```

The default model needs sufficient RAM/VRAM for a 9B model. To use another installed
Ollama model, add `--model MODEL_NAME`. If Ollama is not running, start it with
`ollama serve` in another terminal. The default server is http://127.0.0.1:11434.
`--host` can point to another server; source context will be sent to that server.

## Run with one command

```powershell
python -m cobol_code_anonymizer.review "C:\cobol_code\ted\batch" --watchlist "D:\utente_locale\Desktop\cobol-code-anonymizer\private_watchlists\company_workers.txt" --report-dir "C:\cobol_code\ted\review_reports"
```

After installation, `cobol-review` is an equivalent command:

```powershell
cobol-review "C:\cobol_code\ted\batch" --watchlist "D:\lists\company_workers.txt" --report-dir "C:\cobol_code\ted\review_reports"
```

On macOS/Linux, use the same arguments with your own paths and `python3` if needed.
Reports must be outside the input folder. Each run replaces the generated reports in
that report folder; copy manually edited reports elsewhere before rerunning.

The terminal prints extraction progress by file and review progress by occurrence:

```text
[Extract 1/42] PROGRAM.CBL
Extracted 204 occurrences from 42 files.
[Review 1/204] NAME | PROGRAM.CBL:15 ...
  1/204 CONFIRMED | NAME | PROGRAM.CBL:15
```

There are no replacement prompts. To extract candidates without a model call, add
`--extract-only`. To change the per-request timeout, add `--timeout 300`.

## Watchlist and grouping

Use a UTF-8 text file with one first name, surname, full name, or numeric matricola
per line. Mixed rows such as `Mario Rossi 00058` or `00058 Mario Rossi` are also
accepted: the full name remains one name entry, and the numeric ID is stored separately.
Comma, semicolon, tab, and pipe separated fields are supported (including quoted fields).
For example, `Mario;Rossi;00058` supplies separate name and surname entries plus an ID.
A header containing only name/surname/matricola column labels is skipped.
Blank lines and lines starting with `#` are ignored. Unsupported numeric entries fail
validation with a line number rather than silently being treated as names. Alphanumeric matricole consisting of one letter followed by digits, such as `D0023`,
are also accepted (including in mixed rows); they never enter the name list. Longer
letter prefixes attached to digits, such as `Mario00058`, still need a space or delimiter.

```text
Mario
Rossi
Sala
00058
5123456
```

Source decoding supports UTF-8 (including BOM) and falls back to Windows-1252.
UTF-16/binary inputs containing NUL bytes are reported as errors. Convert EBCDIC and
other encodings before running; they are not supported by this workflow.

Matching is case-insensitive and keeps original source spelling. It uses boundaries
and scans all source lines, including identifiers such as `WS-ROSSI-COD`. It does not
perform accent folding, fuzzy spelling correction, or discover unlisted names.
Names embedded in email addresses or URLs are not extracted as separate name candidates.

Adjacent watchlist hits separated only by horizontal whitespace are grouped.
Between matched names, a dotted initial (`D.`) or one of `De`, `Di`, `Del`, `Della`,
`Dei`, `Da`, `D'`, `D’` is also included. For `Mario D. Rossi`, the evidence lists
`Mario` and `Rossi` as watchlist matches and `D.` as an inferred connector.
Commas, other punctuation, hyphens, and newlines do not join separate hits.
An explicitly listed full name is matched as one entry.

## Structured candidates

- Matricole: every standalone five- or six-digit number, or seven digits starting
  with 5, 6, or 7. Leading zeros are preserved. Numeric-watchlist membership is
  recorded separately; dates, counts, and other matching numbers still reach the model.
- Alphanumeric matricole: one letter followed by digits, such as `D0023`, supplied in
  the watchlist. Exact standalone occurrences are extracted case-insensitively; the
  original spelling and leading zeros are preserved. Unlisted alphanumeric codes are
  not guessed to be employee IDs. Employee-watchlist membership is recorded separately.
- Email: the existing email syntax pattern.
- IBAN: the existing contiguous IBAN pattern, plus modulo-97 checksum and Italian
  length checks. Other countries' exact lengths and account existence are not checked.
- Codice fiscale: the existing standard syntax pattern; no checksum or omocodia handling.
- Phone: the existing pattern requiring a TEL/TELEFONO/PHONE/CELL/CELLULARE label.

Higher-specificity matches suppress embedded fragments: digits inside an IBAN or
phone candidate are not also exported as matricole. Invalid IBAN checksums remain
visible in the report as evidence rather than silently discarding the candidate.
Spaced IBANs, unlabelled phones, and personal data outside these patterns can be missed.

## Model review

The model receives the exact candidate, suspected type, validation evidence, matched
watchlist entries, and only the previous/current/next physical source lines. The current
occurrence is marked with `[[...]]`. It receives neither the whole source file nor the
whole watchlist nor the source path. At file boundaries an absent neighbor is empty.
Continued COBOL lines are not reconstructed by this version.

Decisions apply to occurrences, not globally to words:

- `CONFIRMED`: context supports the suspected personal-information type.
- `REJECTED`: context clearly supports an ordinary word/number, technical token,
  or a token clearly unrelated to the suspected type. A surname used in an example
  is still a name: `DEPUTATO ... (ES. FRATTINI)` should classify FRATTINI as `CONFIRMED`.
  Merely appearing after `ES.` or in a test does not justify rejecting a candidate.
- `UNCERTAIN`: ambiguous or insufficient context, or review failure.
- `NOT_REVIEWED`: extraction-only run.

The model cannot verify that a person, employee record, mailbox, or account exists.
The prompt treats source text as untrusted; model decisions still require evaluation.
JSON output is validated, including the exact candidate and occurrence ID. A failed
request or invalid answer is retried once; persistent failure becomes `UNCERTAIN`
with an error, and the command exits unsuccessfully after writing the reports.
There is no automatic replacement step and no decision is silently discarded.

## Reports

| File | Purpose |
|---|---|
| `report.txt` | Readable summary and separate confirmed/rejected/uncertain sections with context |
| `candidates.csv`, `candidates.json` | Original extraction evidence saved before model startup |
| `review.csv`, `review.json` | Every candidate plus decision, reason, model digest, and errors |
| `review_progress.jsonl` | Live completed answers, each with candidate, location, context, and reason; flushed after every answer |
| `run.json` | Completion state, counts, failed files, model configuration, watchlist hash |
| `review_cache.sqlite3` | Incremental local cache for interrupted/repeated runs |

Locations use one-based lines/columns; `end_column` is inclusive. JSON is the canonical
format for exact strings such as zero-prefixed employee IDs. When opening CSV in Excel,
import the value column as Text to retain its leading zeros. The blank `review_answer`
column is for your manual review notes; this command does not apply those answers.
Reports, explanations, and the cache contain private source text. Keep the whole report
folder local. Before a rerun, preserve human edits elsewhere.

Cache entries include source context/hash, extraction evidence, model digest, and prompt.
Changed files or model versions are re-reviewed. Press Ctrl+C once during model review to stop gracefully. The command writes
`report.txt`, `review.csv`, `review.json`, and `run.json` before exiting with code 130.
Completed decisions keep their classifications; remaining candidates are `NOT_REVIEWED`.
The live `review_progress.jsonl` file is also readable while the command runs. Avoid
pressing Ctrl+C again while the partial reports are being written.

Failed answers are not cached. Successful
answers are committed individually, so rerunning the identical command resumes completed
work without model calls for those candidates.

Exit code 0 means extraction/review completed (or extraction-only completed). Code 1 means
incomplete processing; inspect `run.json` and error columns. Code 2 means invalid arguments; code 130 means interrupted with partial reports saved.
Zero candidates does not establish that the source is free of private information.

## Validation

Run `python -m unittest discover -s tests -v`. Tests cover grouping, source positions,
leading zeros, overlapping structured values, IBAN checksum evidence, malformed model
answers, model startup/source errors, output-folder isolation, and an end-to-end command
with a simulated Ollama response, cache reuse, and invalidation after source changes.
These tests do not establish the real model's classification accuracy. Evaluate a labeled
sample from the target codebase before using the classifications to choose replacements.

Prompt v3 corrects the example-name policy. Older prompt answers remain in the cache
but are not reused after this update; candidates are re-reviewed under the new policy.
