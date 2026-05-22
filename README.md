# COBOL Code Anonymizer

Interactive anonymizer for COBOL source files, copybooks, and JCL folders.

Name detection uses Microsoft Presidio with spaCy's Italian model as the primary detector, plus a bundled Italian name/surname watchlist for uppercase COBOL comments, isolated surnames, and legacy-code edge cases.

It scans for:

- Italian names and surnames in COBOL/JCL comments, string literals, `DISPLAY`, `VALUE`, `STRING`, `AUTHOR`, and name-related lines
- IBAN values
- Email addresses
- Italian codice fiscale values
- Matricola / employee identifiers near fields like `MATRICOLA`, `CODDIP`, `COD-DIP`, `CODICE-DIPENDENTE`
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
2/5 MATRICOLA 'A123456' (hits=1) [MAT000002]:
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

## Outputs

For a normal anonymization run, the output folder contains:

- anonymized copies of the input files
- `anonymization_findings.json`
- `replacement_map.csv`

## Install As A Command

Optional:

```powershell
python -m pip install -e .
python -m spacy download it_core_news_sm
cobol-anonymizer C:\path\to\cobol-folder --out-dir anonymized
```
