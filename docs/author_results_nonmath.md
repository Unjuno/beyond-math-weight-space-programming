# Published non-math results: independent re-score

Status: **author-artifact re-score**, not a fresh model inference run. The inputs are the public per-problem JSON files from `RikiyaT/lm_understand-logs` at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. `scripts/analyze_author_results.py` merges the disjoint shards and computes score differences on the same questions. Each question is a bootstrap cluster; multiple samples for one question are averaged before resampling. Pointwise 95% intervals are shown, plus Bonferroni-adjusted 95% intervals across the three p50c non-math primary outcomes.

| Non-math task | Base | 10 passes | 25 passes | 50 passes | Paired item bootstrap, p50c minus base (95% CI) | Items / samples per condition |
|---|---:|---:|---:|---:|---:|---:|
| IFEval strict instruction following | 21.0% | 49.0% | 39.9% | 62.3% | +41.3 pp [+37.3, +45.2]; adjusted [+36.5, +46.0] | 541 / 1,082 |
| GPQA Diamond science QA | 22.7% | 29.4% | 30.1% | 36.6% | +13.9 pp [+9.3, +18.4]; adjusted [+8.3, +19.4] | 198 / 792 |
| MMLU-Pro fixed random 1k | 40.3% | 40.0% | 44.7% | 45.5% | +5.2 pp [+2.3, +8.1]; adjusted [+1.6, +8.7] | 1,000 / 1,000 |

## What this supports

- The released evaluation records show a sizable improvement in satisfying explicit instruction constraints (IFEval).
- They show a smaller but positive effect on one fixed mixed-domain multiple-choice slice (MMLU-Pro), and a positive effect on science QA (GPQA Diamond).
- Together, the author-produced benchmark outcomes are consistent with an effect that reaches beyond math. This supports studying non-math task transfer; it does not establish broad transfer by itself.
- The source report attributes gains mainly to suppressing long reasoning loops. Therefore “more tasks completed under this sampling/token budget” is a better-supported description than “new general knowledge or capability was added.”

## What it does not support

- MMLU-Pro includes quantitative/STEM content; GPQA is science and can involve calculation. Neither is a pure non-math measure.
- IFEval checks explicit constraints. It does not certify truth, usefulness, quality, coding correctness, safety, or ordinary conversation quality.
- MMLU-Pro is a fixed 1,000-question subset of 12,032, with one sample per question. The paired interval describes this set, not every subject or future question.
- The intervals quantify prompt-level sampling variation in the released evaluation records. They do not include editor-search variability, benchmark-selection effects, software/backend differences, or unlogged trials.
- The assessment is not an independent generation rerun. Separate 24-item local exploratory probes are reported in `local_probe_results.md`.

## Reproduce the re-score

```sh
python scripts/analyze_author_results.py \
  --author-repo /path/to/lm_understand-logs
```

Raw author benchmark prompts/outputs remain in the public upstream repository and are not copied here.
