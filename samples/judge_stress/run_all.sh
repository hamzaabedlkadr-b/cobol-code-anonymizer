#!/usr/bin/env bash
# Run every Stage 0 fixture with the same production detector configuration.
# This script makes local LLM calls. Codex never runs it; the user runs it
# manually after the unit suite passes.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: samples/judge_stress/run_all.sh RUN_LABEL [REFERENCE_ROOT]" >&2
  exit 2
fi

run_label=$1
reference_root=${2:-}
stress_dir=samples/judge_stress
report_base=${COBOL_STRESS_REPORT_ROOT:-reports/stress_runs}
run_root="$report_base/$run_label"
model=${COBOL_STRESS_MODEL:-ministral-3:3b}
use_presidio=${COBOL_STRESS_USE_PRESIDIO:-0}

if [[ -e "$run_root" ]]; then
  echo "Refusing to overwrite existing run: $run_root" >&2
  exit 2
fi
if [[ "$use_presidio" != 0 && "$use_presidio" != 1 ]]; then
  echo "COBOL_STRESS_USE_PRESIDIO must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$run_root/fixtures/sealed_a" "$run_root/fixtures/sealed_b"
.venv/bin/python "$stress_dir/generate_noisy.py" \
  --seed 20261107 --output-dir "$run_root/fixtures/sealed_a" \
  --source-name PAYROLL_NOISY.CBL --manifest-name manifest.json \
  --roster-name roster.txt > "$run_root/fixtures/sealed_a/generation.txt"
.venv/bin/python "$stress_dir/generate_noisy.py" \
  --seed 20261208 --output-dir "$run_root/fixtures/sealed_b" \
  --source-name PAYROLL_NOISY.CBL --manifest-name manifest.json \
  --roster-name roster.txt > "$run_root/fixtures/sealed_b/generation.txt"

detector_args=(--no-default-name-watchlist)
if [[ "$use_presidio" == 0 ]]; then
  # This is the current production setting. Set COBOL_STRESS_USE_PRESIDIO=1
  # only when the real production command enables Presidio/spaCy too.
  detector_args+=(--no-presidio)
fi

run_fixture() {
  local run_name=$1 source_file=$2 roster_file=$3 report_dir=$4
  local started finished status
  mkdir -p "$report_dir"
  started=$(date +%s)
  set +e
  .venv/bin/cobol-anonymizer "$source_file" \
    --mode union-judge \
    --name-extract-model "$model" \
    --watchlist "$roster_file" \
    "${detector_args[@]}" \
    --name-judge-model "$model" \
    --name-verifier-model "$model" \
    --out-dir "$report_dir/out" \
    --report-dir "$report_dir" \
    --auto > "$report_dir/run.log" 2>&1
  status=$?
  set -e
  finished=$(date +%s)
  echo $((finished - started)) > "$report_dir/runtime_seconds.txt"
  if [[ $status -eq 1 && -f "$report_dir/out_manifest.csv" ]]; then
    # Expected withheld fixture files are valid measurements, not completed releases.
    if .venv/bin/python -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); sys.exit(0 if rows and all(r["status"] in {"released", "withheld", "skipped"} for r in rows) and any(r["status"] == "withheld" for r in rows) else 1)' "$report_dir/out_manifest.csv"; then
      status=0
    fi
  fi
  if [[ $status -ne 0 ]]; then
    echo "$run_name failed with exit code $status; see $report_dir/run.log" >&2
    exit "$status"
  fi
}

analyze_fixture() {
  local run_name=$1 manifest=$2 primary=$3 repeat=$4 reference=$5
  local args=(
    --manifest "$manifest"
    --decisions "$primary/decisions.csv"
    --run-dir "$primary"
    --noise-runs "$primary" "$repeat"
  )
  if [[ -n "$reference" ]]; then
    args+=(--compare "$reference")
  fi
  if ! .venv/bin/python "$stress_dir/analyze_results.py" "${args[@]}" \
    > "$primary/analysis.txt" 2>&1; then
    echo "$run_name comparison failed; see $primary/analysis.txt" >&2
    return 1
  fi
  .venv/bin/python "$stress_dir/analyze_results.py" \
    --manifest "$manifest" \
    --decisions "$repeat/decisions.csv" \
    --run-dir "$repeat" > "$repeat/analysis.txt"
}

fixture_names=(clean noisy sealed_a sealed_b)
sources=(
  "$stress_dir/PAYROLL.CBL"
  "$stress_dir/PAYROLL_NOISY.CBL"
  "$run_root/fixtures/sealed_a/PAYROLL_NOISY.CBL"
  "$run_root/fixtures/sealed_b/PAYROLL_NOISY.CBL"
)
rosters=(
  "$stress_dir/roster.txt"
  "$stress_dir/noisy_roster.txt"
  "$run_root/fixtures/sealed_a/roster.txt"
  "$run_root/fixtures/sealed_b/roster.txt"
)
manifests=(
  "$stress_dir/manifest.json"
  "$stress_dir/noisy_manifest.json"
  "$run_root/fixtures/sealed_a/manifest.json"
  "$run_root/fixtures/sealed_b/manifest.json"
)

for index in "${!fixture_names[@]}"; do
  name=${fixture_names[$index]}
  primary="$run_root/$name"
  repeat="$run_root/${name}_repeat"
  run_fixture "$name" "${sources[$index]}" "${rosters[$index]}" "$primary"
  run_fixture "${name}_repeat" "${sources[$index]}" "${rosters[$index]}" "$repeat"
done

comparison_failed=0
for index in "${!fixture_names[@]}"; do
  name=${fixture_names[$index]}
  reference=""
  if [[ -n "$reference_root" ]]; then
    reference="$reference_root/$name"
  fi
  if ! analyze_fixture \
    "$name" "${manifests[$index]}" \
    "$run_root/$name" "$run_root/${name}_repeat" "$reference"; then
    comparison_failed=1
  fi
done

.venv/bin/python "$stress_dir/summarize_runs.py" \
  --run clean "${manifests[0]}" "$run_root/clean" \
  --run noisy "${manifests[1]}" "$run_root/noisy" \
  --run sealed_a "${manifests[2]}" "$run_root/sealed_a" \
  --run sealed_b "${manifests[3]}" "$run_root/sealed_b" \
  > "$run_root/summary.txt"

echo "Stage 0 reports: $run_root"
echo "Summary: $run_root/summary.txt"
exit "$comparison_failed"
