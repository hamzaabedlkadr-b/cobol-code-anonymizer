# COBOL Code Anonymizer

Interactive anonymizer for COBOL source files, copybooks, and JCL folders.

Name detection uses Microsoft Presidio with spaCy's Italian model as the primary detector, plus a bundled Italian name/surname watchlist for uppercase COBOL comments, isolated surnames, and legacy-code edge cases.

It scans for:

- Italian names and surnames in COBOL/JCL comments, string literals, `DISPLAY`, `VALUE`, `STRING`, `AUTHOR`, and name-related lines
- IBAN values
- Email addresses
- Italian codice fiscale values
- Matricola / employee identifiers near fields like `MATRICOLA`, `CODDIP`, `COD-DIP`, `CODICE-DIPENDENTE`, using five or six digits, or seven digits starting with `5`, `6`, or `7`
- Phone numbers when they appear near labels like `TEL`, `TELEFONO`, `PHONE`, `CELL`

The tool writes anonymized copies to a new output folder. It does not modify the original files.

## Quick Start

```powershell
git clone https://github.com/hamzaabedlkadr-b/cobol-code-anonymizer.git
cd cobol-code-anonymizer
python -m pip install -e .
python -m spacy download it_core_news_sm
python -m cobol_code_anonymizer C:\path\to\cobol-folder --out-dir C:\path\to\anonymized-output
```

## Detection Modes

Use one short flag to select a mode. Leave the flag out for the baseline:

```powershell
# Baseline: spaCy/Presidio, watchlists, and employee roster
python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --out-dir anonymized

# LLM extraction only for names
python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --out-dir anonymized --llm

# Baseline detectors plus LLM extraction
python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --out-dir anonymized --union

# Union followed by active LLM judging
python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --out-dir anonymized --judge
```

Every command shows the findings and asks for replacements unless `--auto`,
`--scan-only`, or `--names-only` is used. The older explicit forms
`--mode baseline`, `--mode extraction-only`, `--mode union`, and
`--mode union-judge` remain supported.

Add `--explain` to any mode to show why each name was retained:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --union --explain
```

In union mode it reports whether the name came from the LLM extractor, a
baseline detector, or both. When only a baseline detector found a name, the
message says that the extractor returned no overlapping name; this is not
called a rejection because the extractor does not classify negative cases.
With `--judge`, the explanation also shows explicit judge decisions and the
reason for candidates removed as common words, places, organizations, or
technical terms. `--explain` only displays existing decisions and makes no
additional LLM calls.

`--llm` disables Presidio, all name watchlists, roster-based name matching, and
the unknown-name heuristic. An employee roster may still be supplied: its
numeric entries continue to classify matricole, but its names are not used.
Deterministic IBAN, fiscal-code, email, phone, and matricola detection is
unchanged in every mode.

The presets use the defaults in `cobol_code_anonymizer\llm.py`:

```python
NAME_EXTRACT_MODEL = "ministral-3:3b"
NAME_JUDGE_MODEL = "ministral-3:3b"
```

Change those two lines once to select the default model for each role. The
existing `--name-extract-model` and `--name-judge-model` flags remain available
for one-off comparisons.

The command prints the values it found and asks what to replace each one with:

```text
1/5 NAME 'Mario Rossi' (hits=2) [Nome001 Cognome001]:
2/5 MATRICOLA '5123456' (hits=1) [7280779]:
3/5 IBAN 'IT60X0542811101000000123456' [IT05I8806934850742747220794]:
```

Press `Enter` to accept the suggestion, type your own replacement, or type `skip` to leave that value unchanged. Type `all` to accept the current and all remaining suggestions without further prompts. Type `skip-all` to leave the current and all remaining values unchanged, keeping any replacements already chosen. To accept suggestions from the start, use `--auto`.

## Presidio And spaCy

Yes: the tool is designed to use Microsoft Presidio and spaCy for name detection.

- Microsoft Presidio entity: `PERSON`
- spaCy language model: `it_core_news_sm`
- Fallback/booster: bundled Italian names and surnames in `cobol_code_anonymizer/data/italian_names.txt`

If Presidio or the spaCy model is not installed, the command prints a warning and falls back to the bundled watchlist. To force watchlist-only mode:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --no-presidio
```

To use a different spaCy model:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --presidio-model it_core_news_lg
```

## Auto Mode

Use suggestions without prompts:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --out-dir anonymized --auto
```

## Scan Only

Create only a JSON report:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --scan-only --report-dir reports
```

Output:

- `anonymization_findings.json`

## Names Only Report

To see only detected names, with the folder, file, line, column, detector source, and context:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --names-only --report-dir reports
```

Output:

- `reports\names_findings.csv`
- `reports\names_findings.json`
- `reports\anonymization_findings.json`

Example terminal output:

```text
Name | Folder | File | Line | Column | Source
------------------------------------------------------------------------------
Mario Rossi | src\programs | src\programs\PDCBVC.CBL | 15 | 25 | presidio_spacy
Mattarella | jcl | jcl\JOB001.JCL | 4 | 18 | watchlist
```

## Unknown Surname Discovery

Presidio and watchlists can miss isolated uppercase surnames, especially in old COBOL comments such as `MAIL <SURNAME>`, `MODIFICHE DA <SURNAME>`, or `D'<SURNAME>`.

Use this review-first mode to find surname-like tokens even when they are not already in a list:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --names-only --detect-unknown-names --report-dir reports\unknown_names
```

The report marks these as:

```text
source = unknown_name_heuristic
```

Review the CSV before anonymizing:

```text
reports\unknown_names\names_findings.csv
```

After review, you have two safe options:

1. Add confirmed names/surnames to a private watchlist and rerun.
2. Run anonymization with `--detect-unknown-names` and answer the prompts manually, using `skip` for false positives.

Example interactive anonymization:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --detect-unknown-names --out-dir anonymized
```

Avoid combining `--detect-unknown-names` with `--auto` until you have reviewed the candidates, because unknown-name detection intentionally favors catching suspicious leftovers over perfect precision.

## Full-LLM Name Extraction

For opt-in direct extraction, use a local Ollama model:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --name-extract --report-dir reports
```

The extractor reads only the configured name scan ranges, sends chunked JSON records to Ollama, re-anchors every returned name inside the original source text, and writes:

- `reports\extraction_decisions.json`
- `reports\anonymization_findings.json`
- name CSV/JSON reports when `--names-only` is used

By default this is additive: extracted names are unioned with Presidio/spaCy, watchlists, unknown-name heuristics, and roster findings. Deterministic structured PII detection for IBAN, fiscal code, email, phone, and matricola is unchanged.

To change models per run:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --name-extract --name-extract-model ministral-3:3b
```

Extraction and judging can use different models:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --name-extract --name-extract-model ministral-3:3b --name-judge --name-judge-model gemma3:4b-it-qat
```

To change the defaults once, edit `NAME_EXTRACT_MODEL` and `NAME_JUDGE_MODEL` in
`cobol_code_anonymizer\llm.py`. Per-run flags still override those defaults
independently. You can also reuse `--ollama-host`, `--llm-timeout`, and tune
request size with `--name-extract-chunk-lines`.

Extraction-only thesis/evaluation runs should explicitly disable deterministic name detectors while leaving structured PII enabled:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --name-extract --no-presidio --no-default-name-watchlist --names-only --report-dir reports\extract_only
```

If an extractor canary, schema, transport, or chunk failure occurs, the run is marked incomplete and exits with code `1`. The findings report and extraction audit are still written, but replacement prompts, mappings, and anonymized files are blocked.

## CSV Review Workflow

If you prefer choosing replacements in a file:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --create-map replacement_map.csv
```

Edit the `replacement` column in `replacement_map.csv`, then run:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --map-file replacement_map.csv --out-dir anonymized --auto
```

Rows with an empty `replacement` value are filled with the deterministic suggestion when `--auto` is used. Without `--auto`, the tool prompts for missing replacements.

## Worker Identifier Non-Linkability

Names and matricole are anonymized per occurrence by default. If `Mario` appears twice, the two findings get different replacement suggestions, such as `Nome001` and `Nome002`, instead of sharing one replacement.

The same is true for matricole: two occurrences of `5123456` get separate valid replacement suggestions. This avoids leaking that two locations may refer to the same worker. Similar names and exact repeated names are not merged automatically.

If review explicitly confirms that two occurrences should share a replacement, set the same `replacement` value for those rows in `replacement_map.csv`. Use the `key` column in the CSV to keep occurrence-specific rows distinct.

## Name Scan Scope

By default, names are scanned only in likely human text areas to reduce false positives:

- COBOL/JCL comments
- quoted string literals
- `DISPLAY`, `VALUE`, `STRING`, `ASSIGN TO`
- name-related lines such as `NOME`, `COGNOME`, `NOMINATIVO`, `REFERENTE`, `RESPONSABILE`, `OPERATORE`

For a very broad scan:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --name-scope all
```

Broad scans catch more possible names but can also flag technical words or identifiers.

## Extra Watchlists

The package includes a large Italian first-name and surname watchlist. You can add your own list:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --watchlist my_names.txt
```

Use one name or surname per line. Numeric matricole in this file are classified as identifiers rather than names. Supported matricole have five or six digits, or seven digits starting with `5`, `6`, or `7`; their generated replacements preserve the digit count (a five-digit original such as `00058` gets a five-digit replacement). With numeric entries present, other matching numbers are reported as `SUSPECTED_MATRICOLA`.

## Company Roster Watchlist

If you have a list of people in the company, keep it as a private local file and pass it with `--watchlist`.

Recommended location:

```text
private_watchlists\company_people.txt
```

That folder is ignored by git so real employee names do not get committed.

Recommended format:

```text
# one entry per line; comments start with #
Mario Rossi
Rossi Mario
Rossi
Giulia Bianchi
Bianchi Giulia
Bianchi
```

Full names are safest. Surnames help catch COBOL comments and literals that contain only a last name, but they can create more false positives. Avoid adding common first names alone unless you really want a broad scan.

Example:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --watchlist private_watchlists\company_people.txt --names-only --report-dir reports
```

Then anonymize reviewed findings:

```powershell
python -m cobol_code_anonymizer C:\path\to\cobol-folder --watchlist private_watchlists\company_people.txt --out-dir anonymized
```

## Employee Roster Scan

Put the real company roster here:

```text
C:\Users\Lenovo\Desktop\Camera\control_flow\cobol-code-anonymizer\private_watchlists\company_workers.txt
```

Format: one first name, surname, or matricola per line.

```text
Mario
Rossi
5123456
Giulia
Bianchi
7654321
```

Matricola-format numbers found in the code but not in this file are reported as `SUSPECTED_MATRICOLA`.

Create the review table without the Presidio/spaCy model:

```powershell
cd C:\Users\Lenovo\Desktop\Camera\control_flow\cobol-code-anonymizer

python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --entities NAME MATRICOLA --no-presidio --no-default-name-watchlist --create-map reports\employee_roster_review\replacement_map.csv --report-dir reports\employee_roster_review
```

Edit this table:

```text
C:\Users\Lenovo\Desktop\Camera\control_flow\cobol-code-anonymizer\reports\employee_roster_review\replacement_map.csv
```

Use the `replacement` column to choose what each found value becomes. The `locations` column shows file and line, for example `PROGRAM.CBL:123`.

Create anonymized code files:

```powershell
cd C:\Users\Lenovo\Desktop\Camera\control_flow\cobol-code-anonymizer

python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.txt --entities NAME MATRICOLA --no-presidio --no-default-name-watchlist --map-file reports\employee_roster_review\replacement_map.csv --out-dir anonymized --auto
```

The anonymized code is written here:

```text
C:\Users\Lenovo\Desktop\Camera\control_flow\cobol-code-anonymizer\anonymized
```

Replace `C:\path\to\cobol-folder` with the folder that contains the COBOL, copybook, and JCL files you want to scan.

## Outputs

For a normal anonymization run, the output folder contains:

- anonymized copies of the input files
- `anonymization_findings.json`
- `replacement_map.csv`, including an occurrence-specific `key` column for non-linkable names and matricole

## Install As A Command

Optional:

```powershell
python -m pip install -e .
python -m spacy download it_core_news_sm
cobol-anonymizer C:\path\to\cobol-folder --out-dir anonymized
```
