# Reproduction status — 2026-11-13

## Summary

We can reconstruct the released p50c edit exactly and locally observe a substantial change in completion behavior on a small paired sample. That makes a narrow claim plausible: **a targeted weight edit can sometimes change a model's behavior and improve answers on tested examples.** It does not establish the paper's full MATH-500 result, the reliability of discovering such edits, or a general capability gain.

The current evidence comes from distinct studies. Do not combine their scores as if they came from one experiment.

## Evidence by type

| Evidence | Frozen scope | Result | Interpretation |
| --- | --- | --- | --- |
| Author benchmark record re-score | Released IFEval, GPQA Diamond, and fixed MMLU-Pro per-item records | Recomputed aggregates agree with the published records | Verifies score arithmetic on supplied records, not original inference. |
| Published p50c edit reconstruction | Pinned author code, base revision, and p50c edit archive | Rebuilt L15/L19 BF16 tensor hashes match the published edit | Exact artifact reconstruction; not a fresh search. |
| Local MATH-500 checkpoint evaluation | Four selected items across two tranches; four generations per item and arm; thinking on; 32,768-token cap | Base 0/16 correct, 16/16 cap hits; p50c 11/16 correct, 3/16 cap hits | Promising local signal for the released edit on these items. The second tranche followed the first; total is exploratory, not a benchmark estimate. |
| Local non-math applied suite | 24 hand-authored tasks; one paired sample per task; full cap | EOS-before-cap 11/24→24/24; correctness mixed by task and grader | Supports changed termination. Does not establish broad non-math capability. |
| Earlier reconstructed edit search | Separate local search, 18 successful forward calls | Candidate edits were rejected; no edit retained | Negative result for that specific approximate search. Not the author trajectory. |
| Fresh 50-request-budget continuation | Newer run on TRAIN-only diagnostic data | 11 successful forward calls; paused during pass12 | Does not complete the search or show a causal effect of the proposed edit. |
| External WSP study | Hu et al., arXiv preprint 2611.09382 | Reports three Qwen3.5-0.8B MATH trajectories with held-out changes of +29.45, +5.00, and +27.30 points | Independent-group corroboration as reported by its authors; it edits different weights and is not our reproduction of p50c. |

## Local p50c MATH-500 result

The first tranche evaluated two selected test items: base 0/8 correct with 8/8 cap hits, and p50c 5/8 correct with 1/8 cap hit. The second, separately preregistered tranche evaluated two previously unused items: base 0/8 correct with 8/8 cap hits, and p50c 6/8 correct with 2/8 cap hits. Combined descriptive counts are 0/16 versus 11/16 correct and 16/16 versus 3/16 cap hits.

The result is informative because we generated and scored new outputs locally after rebuilding the exact published edit. It remains small and selection-sensitive: the first item in the initial tranche was elementary, and the second tranche was added after seeing the first. Evaluation used Transformers 5.20.2 / PyTorch 2.14.1 on Apple MPS; the paper reports vLLM. Local per-checkpoint seed initialization does not reproduce the author's random stream. The result cannot estimate performance over all 500 questions or confirm 48.1%.

Full conditions and per-tranche summaries: [method-fidelity audit](method_fidelity_audit_2026-11-11.md) and [second-tranche report](p50c_math500_followup_2026-11-12.md). Token-level generations remain local.

## Fresh 50-request-budget run: paused during pass12

This is separate from the older 18-call reconstructed edit search. The new run reached **11 successful forward calls out of the planned 50**. It nominated layer 20 `mlp.down_proj` input coordinate 0 from a small TRAIN-only exact-token activation comparison. That is a discovery candidate, not a verified mechanism.

The candidate rationale was specific but low-powered: a pass10 sample contained 20 same-problem original-weight generations on five preselected TRAIN questions. Ten were scored correct (nine completed, one reached the cap after emitting a correct answer), and ten were incorrect cap hits. Pass11 replayed exact token prefixes and compared 56 shared 8-gram contexts across four eligible questions. The coordinate-0 activation contrast had the same sign for correct-completed versus capped trajectories on all four; a fixed-offset background comparison matched only two of four. This nominates a testable coordinate, but the comparison is outcome-selected, involves four eligible questions, has no coordinate control in pass12, and does not establish causality.

Pass12 preregistered a half-scale attenuation of that one column with matched original/edited generations on four TRAIN problems. It was interrupted before completing the paired comparison. In the clean retry, one original-arm generation completed (5,088 tokens, scored correct); a second was interrupted at about 25,088/32,768 tokens. No edited-arm generation or pass12 response was recorded. A preceding interrupted attempt had seven original-arm rows; these remain locally archived and were not combined with the clean retry. **There is no pass12 treatment effect to report.** The experiment goal and worker are paused.

The author procedure is not fully reproduced: the local forward stack uses Transformers/MPS rather than the released run's original service; the original persistent editor kernel and orchestration are unavailable; and this run did not complete 50 calls. The [earlier 18-call search report](fresh_edit_search_results.md) describes a different approximate search and must not be merged with this one.

## External follow-up: what it adds

Hu et al.'s 2026-11-06 arXiv preprint reports 16 editing trajectories over seven benchmark families and two Qwen3.5 model sizes. For Qwen3.5-0.8B MATH, it reports held-out MATH-500 accuracy changing from 8.35% to 37.80% for its primary edit; two additional searches report 13.35% and 35.65%. Its primary edit changes 1,023 BF16 values in layer 23 `mlp.down_proj` column 507. It also reports that six of 16 searches retained the original checkpoint, separating successful candidates from unsuccessful discovery attempts.

This is a different edit and search from Takehi et al.'s p50c L15/L19 edit. It strengthens the case that WSP-like editing can sometimes produce held-out gains; it does not establish that the original p50c claim or search was reproduced. The arXiv paper provides detailed methods and aggregate trajectory tables, but the companion GitHub repository currently lists a PDF rather than raw inference logs. We have not independently rerun or audited that study's outputs. See [external-paper assessment](external_wsp_followup_2026-11.md).

## Claim boundary

Supported: an exact reconstruction of the public edit, author-record arithmetic checks, local behavioral changes under fixed-checkpoint evaluations, and independent published reports of other successful WSP edits.

Not supported: a full reproduction of p50c's 48.1% MATH-500 score; a fresh reproduction of its complete Astra search and causal validation; broad gains on arbitrary non-math tasks; or proof that a localized channel is the general cause of repetitive reasoning.

## Data handling

This public summary contains aggregate scores, methods, and status only. Full generations, activation arrays, model weights, local worker logs, the full Luna–Astra conversation, and the local `LUNA_MEMORY.md` remain outside Git. Selected sanitized event and decision records are tracked. No CI workflow was added.
