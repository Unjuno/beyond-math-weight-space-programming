# Publication positioning and claim boundary

## Recommended description

An exploratory, paired fixed-checkpoint evaluation of a released Qwen3.5-0.8B weight edit on 24 hand-authored non-math tasks, accompanied by a re-score of the benchmark records from Takehi et al.'s 2026 research blog report. The local study observes substantially more EOS-before-cap completions and lower repeated 8-gram fractions under the edited checkpoint. Task correctness is mixed and sensitive to the writing grader and three imperfect structured-task specifications.

## Claims supported by the current evidence

- We reproduced the author's reported aggregate benchmark scores from released per-item records; this verifies those records and scoring, not the original generations.
- The reported benchmark values are IFEval 21.0%→62.3%, GPQA Diamond 22.7%→36.6%, and fixed-slice MMLU-Pro 40.3%→45.5%. Our local verification re-scores author records; it is not fresh inference on those benchmarks.
- On this specific paired 24-item suite, EOS-before-cap rose from 11/24 to 24/24, while strict exact structured matches rose from 4/16 to 6/16.
- A blind semantic sensitivity audit found structured success of 5/16 vs 9/16 with gold-relative schema strictness, or 6/16 vs 9/16 when that schema condition is relaxed; intervals are wide and include zero.
- The suite surfaced concrete completion-proxy failures and task-specification defects, which are useful evaluation findings in their own right.

## Claims not supported

- A robust or general improvement in non-mathematical capability, safety, factual calibration, coding, or user-facing quality.
- A causal mechanism that the weight edit suppresses reasoning loops. Repeat statistics and EOS are descriptive and do not isolate mechanism.
- A fresh reproduction of the paper's 50-pass Astra editing procedure or its IFEval, GPQA Diamond, or MMLU-Pro generations.
- A claim that all outcomes are correct because the answer terminated, parsed, or was shorter.

## Limitations that belong beside any headline

Only one sample per hand-authored task; two model graders rather than human raters; grader-sensitive writing outcomes; two known reading errors in Luna's frozen first-pass scores (w04-A line count, w05-A missing December); three task/gold specification ambiguities; some suite facts/rubric were exposed to a separate candidate editor after its freeze; fixed public p50c was not retuned on this suite; Apple MPS/Transformers backend differs from the author's vLLM evaluation path; raw full-length generations are not public. Exact strict scores are retained unchanged and semantic checks are explicitly post-hoc sensitivities.

## Separate approximate editing attempts

The repository preserves two distinct approximate local editing attempts on training data; neither is the released p50c search. In the initial alpha100 attempt, the edited run emitted a gold boxed answer earlier on one paired item but both generations hit 32,768 tokens without EOS; its repeated 8-gram fraction was higher (0.8668 vs 0.8539). A later fresh search selected and froze a layer 23 column-137 candidate before four untouched confirmation items. On those four items, base, candidate, and matched control each scored 0/4 strict correct and all 12 generations hit the 8,192-token cap without EOS; mean repeated 8-gram fraction was 0.4557, 0.4777, and 0.4745 respectively. Neither attempt demonstrates a successful reproduction or an edit benefit. The later search also used a different MPS/Transformers harness and substantially fewer passes than the authors' 50-pass procedure. See the [initial paired audit](../results/astra_clean_20261103/pass_07/paired_generation_audit.json) and the [fresh search report](fresh_edit_search_results.md).

## Short headline

**The released edit greatly increased normal termination on a small applied-task suite, but evidence that it improved non-math task correctness remains mixed and inconclusive.**
