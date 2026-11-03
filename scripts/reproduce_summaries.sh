#!/usr/bin/env bash
set -euo pipefail

study_root="$(cd "$(dirname "$0")/.." && pwd)"
author_repo="${1:?Usage: scripts/reproduce_summaries.sh /path/to/lm_understand-logs-at-pinned-revision}"
python_bin="${REPRO_PYTHON:-python3}"

for raw in \
  "$study_root/results/raw/nonmath_probe.jsonl" \
  "$study_root/results/raw/no_thinking_probe.jsonl" \
  "$study_root/results/raw/long_cap_probe.jsonl"; do
  if [[ ! -f "$raw" ]]; then
    printf 'Missing local raw output: %s\nRun the local probes first; raw generations are intentionally not published.\n' "$raw" >&2
    exit 2
  fi
done

"$python_bin" "$study_root/scripts/verify_probe_pairs.py"
"$python_bin" "$study_root/scripts/analyze_author_results.py" --author-repo "$author_repo"
"$python_bin" "$study_root/scripts/summarize_nonmath_probe.py" \
  --input "$study_root/results/raw/nonmath_probe.jsonl" \
  --out "$study_root/results/thinking_probe_summary.csv" --mode thinking
"$python_bin" "$study_root/scripts/summarize_nonmath_probe.py" \
  --input "$study_root/results/raw/no_thinking_probe.jsonl" \
  --out "$study_root/results/no_thinking_probe_summary.csv" --mode no_thinking
"$python_bin" "$study_root/scripts/summarize_nonmath_probe.py" \
  --input "$study_root/results/raw/long_cap_probe.jsonl" \
  --out "$study_root/results/long_cap_probe_summary.csv" \
  --mode thinking --expected-items 2
