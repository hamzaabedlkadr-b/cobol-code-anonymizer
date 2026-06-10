# COBOL Code Anonymizer

Interactive anonymizer for COBOL source files, copybooks, and JCL folders.

Name detection uses Microsoft Presidio with spaCy's Italian model as the primary detector, plus a bundled Italian name/surname watchlist for uppercase COBOL comments, isolated surnames, and legacy-code edge cases.

It scans for:

- Italian names and surnames in COBOL/JCL comments, string literals, `DISPLAY`, `VALUE`, `STRING`, `AUTHOR`, and name-related lines
- IBAN values
- Email addresses
- Italian codice fiscale values
- Matricola / employee identifiers near fields like `MATRICOLA`, `CODDIP`, `COD-DIP`, `CODICE-DIPENDENTE`, using the format `[567]?[0-9]{6}`
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

The command prints the values it found and asks what to replace each one with:

```text
1/5 NAME 'Mario Rossi' (hits=2) [Nome001 Cognome001]:
2/5 MATRICOLA '5123456' (hits=1) [7280779]:
3/5 IBAN 'IT60X0542811101000000123456' [IT05I8806934850742747220794]:
```

Press `Enter` to accept the suggestion, type your own replacement, or type `skip` to leave that value unchanged.

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

Use one name or surname per line.

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

For a file that contains employee names and matricole, use `--employee-roster`. The roster can be plain text, CSV, semicolon-separated, or tab-separated.

On this PC, the anonymizer repo is expected here:

```text
C:\Users\Lenovo\Desktop\Camera\control_flow\unused\anonymization\cobol-code-anonymizer
```

Put the real company roster in this private local path:

```text
C:\Users\Lenovo\Desktop\Camera\control_flow\unused\anonymization\cobol-code-anonymizer\private_watchlists\company_workers.csv
```

Inside the repo, that same file is referenced as:

```text
private_watchlists\company_workers.csv
```

Example roster lines:

```text
matricola;nome;cognome
5123456;Mario;Rossi
7654321;Giulia;Bianchi
998877;Luca Esposito
```

The tool extracts:

- names from the non-numeric text on each line
- matricole matching `[567]?[0-9]{6}`

Exact roster-only scan:

```powershell
cd C:\Users\Lenovo\Desktop\Camera\control_flow\unused\anonymization\cobol-code-anonymizer

python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.csv --entities NAME MATRICOLA --no-presidio --no-default-name-watchlist --scan-only --report-dir reports\employee_roster_scan
```

Then anonymize reviewed findings:

```powershell
cd C:\Users\Lenovo\Desktop\Camera\control_flow\unused\anonymization\cobol-code-anonymizer

python -m cobol_code_anonymizer C:\path\to\cobol-folder --employee-roster private_watchlists\company_workers.csv --entities NAME MATRICOLA --no-presidio --no-default-name-watchlist --out-dir anonymized
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
