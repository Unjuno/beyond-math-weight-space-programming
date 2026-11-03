# Local non-math probe: exploratory result

## Status and scope

Completed 2026-11-03 on the released Qwen3.5-0.8B base and author's p50c edit. This is a **24-item exploratory checkpoint comparison**, not the original benchmark reproduction, not a fresh weight-edit search, and not an independent human quality study.

The probe used 8 explicit instruction-constraint prompts, 8 structured extraction prompts, and 8 grounded-writing prompts from `configs/nonmath_probe.jsonl`. Base and edited model saw the same prompt and per-item seed. Decoding used BF16 on MPS, Transformers 5.20.2, temperature 0.6, top-p 0.95, top-k 20, custom presence penalty 1.5, with thinking disabled and a 512-token cap. Exact generated token IDs and raw outputs remain locally in ignored `results/raw/no_thinking_probe.jsonl`; the tracked summary is `results/no_thinking_probe_summary.csv`.

## Results

| Task family | Base pass | Edited pass |
|---|---:|---:|
| Explicit instruction constraints (8) | 6/8 (75%) | 5/8 (62.5%) |
| Structured extraction (8) | 2/8 (25%) | 1/8 (12.5%) |
| Grounded writing (8) | 3/8 (37.5%) | 4/8 (50%) |
| **All items (24)** | **11/24 (45.8%)** | **10/24 (41.7%)** |

The paired edited-minus-base pass-rate difference was **−4.2 percentage points**, item bootstrap 95% interval **[−16.7, +8.3] points**. This small probe does not show an improvement; the interval is too wide to establish a regression. Cap-hit rate was 0% in both arms, EOS completion 100%, repeated-token-4-gram indicator 20.8% in each, and mean output length 33.4 versus 34.1 tokens. It therefore gives no indication that this non-thinking short-cap comparison is measuring loop-rescue benefits.

The task-level checks are deterministic but hand-authored and not blinded. They evaluate formatting and selected content constraints, not full factuality or writing quality. Repeats/seeds are absent, prompts were designed locally, and the inference stack differs from the author's full evaluation. Do not compare these rates numerically with IFEval or call them generalized non-math capability.

## Interpretation and next question (Luna)

Luna interprets this as **no detected improvement on the selected short-form observable checks**. The observed per-family mix (two families lower, one higher) and wide paired interval do not identify a stable direction. Because thinking was disabled and neither arm hit the output cap, this comparison does not test the proposed loop-suppression mechanism. Luna asked Astra to explain that boundary and recommend the most informative next validation. Astra recommended a paired thinking-on base/edit run at 2,048 tokens, measuring answer-plus-check success over all items and separating final-answer arrival and cap rates. The initial 11 base rows were only a partial, censored sample: retrospective boundary inspection shows 3 of those 11 did emit `</think>`, so they must not be described as all censored. Luna reused them and completed all 24 base/edit pairs. The question, response, and Luna's explicit rationale are preserved append-only in `logs/luna_astra_dialogue.jsonl`; these explanations do not claim complete access to latent cognition.

## Thinking-on paired run

A 2,048-token-cap exploratory thinking-on run was completed for all 24 paired prompts. The local chat template places the opening `<think>` in the input prompt; generated output contains the remainder and sometimes a closing `</think>` followed by a final answer. We therefore re-scored using the recorded `thinking` flag and the closing boundary; raw JSONL is unchanged. Original raw records are preserved in ignored `results/raw/nonmath_probe.jsonl`; re-scored aggregates are in `results/thinking_probe_summary.csv`.

| Thinking-on metric (24 prompts) | Base | Edited | Edited − base, paired item bootstrap 95% interval |
|---|---:|---:|---:|
| Completed final answer + checks | 5/24 (20.8%) | 5/24 (20.8%) | 0.0 pp [−16.7, +16.7] |
| Final-answer boundary reached | 6/24 (25%) | 13/24 (54.2%) | +29.2 pp [+8.3, +50.0] |
| Hit 2,048-token cap | 18/24 (75%) | 12/24 (50%) | −25.0 pp [−45.8, −4.2] |
| Ended by EOS | 6/24 (25%) | 12/24 (50%) | +25.0 pp [+4.2, +45.8] |
| Mean generated tokens | 1,894 | 1,684 | −210 [−378, −57] |

The edit shifted the stopping behavior toward earlier EOS, fewer cap hits, and more final-answer boundaries in this setup, especially on structured extraction (cap 5/8→1/8). But only 5/24 prompts passed checks in either arm; the additional answer arrivals did not increase the number of successful tasks. There were 13 final-answer-boundary arrivals but 12 EOS completions in the edited arm; boundary arrival and completed generation are separate. This supports a changed answer-completion/termination effect under a short cap, **not** improved task accuracy, semantic quality, or added capability. Both arms also flag repeated 4-grams in all 24 generations; that crude metric cannot itself establish a reasoning loop. The cap censors outcomes, so no inference about what would happen beyond 2,048 tokens is justified.

## Scoring correction record

An intermediate summary incorrectly reported zero final-answer arrivals. The parser had treated the decoded generation as if it contained the opening `<think>` token, but that token is in the prompt; only the closing `</think>` may appear in generated text. We inspected the template and saved outputs, updated scoring to use the recorded thinking mode plus the generated closing boundary, and recomputed both summaries before publication. The raw generations and token IDs were not changed. The earlier intermediate count is superseded by the corrected table above. The 11-row initial sample also included three closed thinking sections; it was incomplete, not 11/11 censored.

## Additional 32,768-token-cap spot check

Following Astra's recommendation, Luna ran two thinking-enabled prompts under the author's 32,768-token cap, using the same base/edited checkpoints and decoding settings. IDs `inst03` and `inst05` were drawn from the 24 configured prompts with Python `random.Random(20261103).sample(...)`. Because this follow-up was conceived after examining the 24-item results, treat it as an exploratory spot check, not an unbiased confirmation sample.

| Prompt | Base tokens / check | Edited tokens / check |
|---|---:|---:|
| inst03 | 332 / pass | 283 / pass |
| inst05 | 1,161 / pass | 771 / pass |

Both arms completed both checks and ended with EOS; neither approached the cap. The base/edited mean lengths were 746.5/527 tokens. This spot check shows these two short instruction tasks do not require the 32,768-token budget, but it gives no evidence of a loop-rescue benefit or broad generalization. The raw records are local in ignored `results/raw/long_cap_probe.jsonl`; the small summary is `results/long_cap_probe_summary.csv`.

## Conclusion

The later 24-item full-cap applied suite is reported separately in [`heldout24_p50c_results.md`](heldout24_p50c_results.md). Strict exact scoring found extraction improved and operational-decision success declined by one item. A subsequent blinded semantic audit made the operational result flat (3/8 in both arms) and identified three prompt/gold specification ambiguities; the frozen exact scores remain unchanged. The writing-sensitive overall effect varied by grader (Luna 4/24→12/24; Astra 5/24→9/24, with the second paired interval spanning zero). EOS completion rose 11/24→24/24, but this cannot be equated with correctness. This is not a rerun of IFEval, GPQA Diamond, or MMLU-Pro.

In the paired no-thinking sample, the released edit did **not** improve these 24 custom non-math task checks; the interval remains compatible with small benefit or harm. In the paired thinking-on short-cap sample, the edit increased final-answer arrival and reduced cap hits, but did not increase successful checks (5/24 in each arm). Two post-hoc thinking-on prompts at the 32,768 cap both passed under both checkpoints and did not approach the cap. These are separate exploratory outcomes. The public author records provide separate evidence of improved reported IFEval, GPQA Diamond, and fixed-slice MMLU-Pro scores; our local probe neither confirms nor invalidates those benchmark results. Broader claims require larger, repeated, preregistered, independently authored benchmark evaluation with the original generation protocol and appropriate task-family analysis.
