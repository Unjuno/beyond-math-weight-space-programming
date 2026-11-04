# Full-cap paired applied-task probe of the released p50c edit

## Scope and relation to the paper

Takehi et al.'s November 2026 research blog reports non-math scores for IFEval, GPQA Diamond, and a fixed 1,000-question MMLU-Pro subset. We previously re-scored released per-item records, which audits those records but does not regenerate the benchmark generations. This report covers a separate exploratory suite of 24 hand-authored tasks (8 structured extractions, 8 operational decisions, 8 grounded-writing tasks). It does not rerun those benchmarks or repeat the 50-pass edit search. The evaluated p50c checkpoint was fixed before this evaluation and was not tuned on the suite. Some facts/rubric criteria from the task set were exposed to a separate alpha100 editor after that candidate was frozen, so this is not editor-blind evidence.

## Protocol

- One paired generation per task and arm; same pinned Qwen3.5-0.8B checkpoint, prompt, seed, and input token IDs within each pair; randomized arm order.
- Public p50c tensors at MLP down-projection layers 15 and 19. Provenance and artifact hashes are in `results/heldout24_p50c_fullcap_manifest.json`.
- Thinking enabled, BF16 on Apple MPS with Transformers `AutoModelForImageTextToText`; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and 32,768-token cap. The author report describes vLLM for evaluation; our checkpoint evaluation uses a different MPS/Transformers backend. Evaluation code is available in the upstream release, while the complete editing-agent orchestration and forward-pass server are not fully available. These are separate limitations: this run is an approximate checkpoint evaluation, and the fresh edit search was not reproduced.
- Primary strict success requires EOS before cap and exact recursive JSON key/type/value match for structured tasks, or EOS before cap and all four writing rubric factors. Two model graders (Luna and Astra) independently rated anonymous writing answers; they are not human raters. Scores are retained separately.
- Paired bootstrap resamples task IDs (50,000 draws, fixed seed). With one sample per hand-authored task, intervals describe this suite only.

## Frozen exact-score results

| Task family | Luna frozen pre-audit first pass, base / edit | Astra writing-rater sensitivity, base / edit | Luna delta, 95% CI | Astra delta, 95% CI |
|---|---:|---:|---:|---:|
| Structured extraction (8) | 1/8 → 4/8 | 1/8 → 4/8 | +37.5 pp [+12.5, +75.0] | +37.5 pp [+12.5, +75.0] |
| Operational decisions (8) | 3/8 → 2/8 | 3/8 → 2/8 | −12.5 pp [−37.5, 0.0] | −12.5 pp [−37.5, 0.0] |
| Grounded writing (8) | 0/8 → 6/8 | 1/8 → 3/8 | +75.0 pp [+37.5, +100.0] | +25.0 pp [−25.0, +75.0] |
| **All tasks (24)** | **4/24 → 12/24** | **5/24 → 9/24** | **+33.3 pp [+12.5, +54.2]** | **+16.7 pp [−4.2, +37.5]** |

The Luna values above are the frozen **pre-audit first-pass scores**, not corrected gold-standard labels. Astra's second review identified two concrete Luna reading errors: w04-A has three actual lines (not one), and w05-A omits “December.” Other writing differences concern interpretation and coherence. Preserve both blind ratings; the frozen first pass remains useful for showing scorer sensitivity but is not error-free. EOS-before-cap completion was 11/24 base versus 24/24 edited. Exact typed JSON matches were 4/16 versus 6/16. Among structured answers that reached EOS, exact matches were 4/9 versus 6/16 (44.4% vs 37.5%); this completion-selected comparison is descriptive, not a causal accuracy estimand. EOS, parseability, answer length, and task correctness must not be conflated. The four writing factors agreed on 57/64 judgments; the overall writing-sensitive estimate is reviewer-dependent.

## Semantic audit and task-specification sensitivity

Astra blind-rated all 32 structured final answers without arm labels or raw generations. This post-hoc semantic sensitivity audit does not replace the frozen exact-match scoring. Under semantic correctness + no unsupported/contradictory content + gold-relative schema validity + EOS, results were 5/16 base versus 9/16 edited (+25 pp; paired 95% interval −6.25 to +56.25). If gold-only schema validity is ignored, the count is 6/16 versus 9/16 (+18.75 pp; interval −12.5 to +50). Excluding d08, the content-only count is 6/15 versus 9/15 (+20 pp; interval −13.3 to +53.3). For operational decisions, semantic scoring is 3/8 versus 3/8; after excluding d08 it is 3/7 versus 3/7. This differs from the strict exact score (3/8 vs 2/8), so the apparent one-item operational decline is not robust to semantic/gold ambiguity.

The audit exposed three task-specification problems that should be fixed before using this suite as a benchmark: x07 requests keys but does not state the `unknowns` type (gold requires an array); d07 says arrays are required while two gold values are strings; d08 does not define whether `verified_amount` refers to the invoice (£860) or approved PO (£680). The edited d08 answer’s £860 is present in the source, so calling it definitively false would overstate the evidence. Strict scores remain unchanged for transparency, and the semantic sensitivity is reported alongside them.

## Completion proxy, repetition, and reward-hacking audit

There is no task-score reward, online feedback, or evaluation-driven edit selection in this fixed-checkpoint run. We therefore found no evidence of deliberate reward hacking. We did find a clear **proxy-metric hazard**: EOS rises from 11/24 to 24/24, while exact structured matches rise only from 4/16 to 6/16 and semantic operational success is unchanged at 3/8. For example, edited outputs mark an in-transit parcel `LOST` (d04) and assert “No data loss” when the prompt says it is unconfirmed (d06). Ending normally is not the same as doing the task correctly.

Token-ID diagnostics show mean excess repeated 8-gram fraction 0.5755 base versus 0.1289 edited (paired change −0.4466; task-bootstrap 95% interval −0.5923 to −0.2999). Among base generations that hit the cap, the mean is 0.9210; edited generations had no cap hits. A crude flag for any 4-gram repeated at least three times triggers on all 24 items in both arms and is non-discriminating. These are descriptive output-pattern statistics, affected by different generation lengths and EOS; they do not establish the edit’s causal mechanism or prove that every flagged response is a reasoning loop.

### Item-level follow-up audit

The 13 base-cap → edited-EOS transitions were rebuilt from the local token-level JSONL and the committed sanitized answer/score files with `scripts/audit_completion_pairs.py`. All 13 had lower excess repeated 8-gram fractions after editing (individual deltas −0.566 to −0.871). The transition identities, token counts, pairwise repeat8 values, and structured strict grades are printed by that script. This is direct evidence that these particular edited generations contained less repeated 8-gram text and ended before the cap. It does not establish that the model internally exited a reasoning loop: the edit may alter answer length or terminate generation earlier, and the 8-gram measure is length-sensitive.

The task outcomes clarify why EOS alone was misleading. Four operational-decision generations changed from cap to EOS, yet none passed the strict task check; semantic blind scoring gives 3/8 success in both arms overall. Six writing generations changed from cap to EOS; the apparent writing gain depends on evaluator (Luna 0/8→6/8; Astra 1/8→3/8) and is not a stable quality estimate. In extraction, three specific transitions (x04–x06) went from cap-bound strict failures to EOS-bound strict successes; across all eight extraction items strict success was 1/8→4/8. These were the strongest task-level signal and motivated the targeted follow-up below. The follow-up recurred as a per-item absolute success pattern, but only x05 beat base in aggregate; x04 and x06 tied and individual paired outcomes changed. This is not an established general ability gain.

The supported mechanism-level statement is therefore narrower than “the edit fixed loops”: the fixed public edit shortened all 13 previously capped outputs, each with lower repeated 8-gram fraction; this coincided with three strict extraction wins, no strict operational-decision gain, and evaluator-sensitive writing outcomes. The targeted independent-seed follow-up below supports recurrence on the selected prompts. A length-matched intervention or other mechanism-specific control is still needed to distinguish changed stopping from robust loop suppression.

### Targeted fresh-seed follow-up of x04–x06

Because x04–x06 were the three exact extraction wins, a targeted follow-up was frozen before generation in [`p50c_extraction_followup_prereg.md`](p50c_extraction_followup_prereg.md). It compared the original checkpoint and public p50c patch on those same three prompts with three new paired seeds per prompt, keeping the 32,768-token cap and sampling settings fixed. The nine pairs finished in 3,918 seconds (65.3 min). All generations, configuration, tensor hashes, and checksums are retained locally; the sanitized per-pair score record is `results/p50c_extraction_followup_20261104_scores.json`, reproducible with `scripts/score_p50c_extraction_replication.py`.

| Prompt | Base strict successes | p50c strict successes | Base EOS | p50c EOS |
|---|---:|---:|---:|---:|
| x04 | 2/3 | 2/3 | 2/3 | 3/3 |
| x05 | 2/3 | 3/3 | 2/3 | 3/3 |
| x06 | 2/3 | 2/3 | 2/3 | 3/3 |
| **All nine pairs** | **6/9** | **7/9** | **6/9** | **9/9** |

The preregistered absolute threshold (p50c strict success on at least 2/3 seeds for each item) was met. This does not mean all three original paired wins replicated: x04 and x06 tied their base success counts, and only x05 improved (2/3→3/3). Across the nine paired seeds there were two edited-only successes and one base-only success, for a net gain of one. The larger and consistent shift was EOS (9/9 vs 6/9). This is targeted replication on three prompts selected because they previously favored the edit, not a general extraction estimate. The base was already correct on six of nine trials, so most outcomes were not edit-dependent.

The strict failures also distinguish content from format. On x04 seed 20261113, base returned the exact `awaiting purchase-order match`, while p50c returned `Pending`, causing a strict regression. On x06 seed 20261112, base produced no final answer; p50c supplied the correct milestones but as a top-level array instead of the required `{"milestones": [...]}` object, so it remained a strict failure. The latter is evidence of recovered requested content with schema failure, but any semantic reclassification would be post hoc; the strict score is unchanged. Sanitized final answers and per-seed metadata are available in `results/p50c_extraction_followup_20261104_answers.jsonl`; scoring is reproducible with `scripts/score_p50c_extraction_replication.py`. Raw token IDs and full reasoning outputs remain local.

To check the length confound descriptively, repeated 8-gram fraction was also computed over the first 1,024 generated tokens in each of the nine new pairs. It was lower under p50c in 8/9 pairs (mean paired change −0.0424; one pair higher). This fixed-window sensitivity suggests the original full-output difference is not solely due to having shorter outputs, but it still does not prove a hidden-state loop mechanism: early generated tokens can already differ in content and repetition.

## Interpretation and claim boundary

The strongest supported observation in this selected suite is a large shift in EOS completion and lower repeated-token fractions. Correctness evidence is mixed: exact extraction improves; strict operational exact match falls by one item, but its semantic-audit result is flat once rubric/specification issues are considered; writing estimates vary by model grader. Every broad semantic sensitivity interval includes zero. The data do **not** establish a robust general non-math capability gain, user-ready quality, or a loop-suppression mechanism. The defensible position is an exploratory fixed-checkpoint transfer study with a completion/repetition signal and grader- and rubric-sensitive task outcomes.

The hand-authored suite has one sample per task, is not a representative benchmark, includes the disclosed editor-exposure issue, and uses a backend different from the paper. It does not independently reproduce IFEval, GPQA Diamond, or MMLU-Pro. Full tasks and gold answers are in `results/heldout24_p50c_tasks.json`; sanitized final answers and termination metadata are in `results/heldout24_p50c_final_answers.jsonl`. Frozen Luna, Astra writing, and Astra structured semantic ratings are in the corresponding `results/heldout24_p50c_*ratings.json` files. Token-level generations and internal reasoning traces remain local and ignored by Git.

## Re-scoring committed evidence

The following commands deterministically re-score the published sanitized answers and regenerate the two anonymous packets without publishing raw token traces:

```sh
python scripts/score_transfer_eval.py \
  --tasks results/heldout24_p50c_tasks.json \
  --answers results/heldout24_p50c_final_answers.jsonl \
  --manifest results/heldout24_p50c_fullcap_manifest.json \
  --out /tmp/heldout24_scores.json \
  --blind-packets /tmp/heldout24_blind_packets.json \
  --blind-map /tmp/heldout24_blind_map.json \
  --blind-ratings results/heldout24_p50c_blind_ratings.json \
  --structured-ratings results/heldout24_p50c_astra_structured_semantic_ratings.json
python scripts/analyze_transfer_eval.py /tmp/heldout24_scores.json \
  --secondary-ratings results/heldout24_p50c_astra_blind_ratings.json \
  --out /tmp/heldout24_analysis.json
```

To repeat generation, use a new output path, then run `scripts/export_sanitized_transfer_answers.py` before scoring; the raw output contains token-level reasoning and is ignored by Git. A new generation requires fresh blind ratings; historical ratings must not be applied to it. No CI was used.
