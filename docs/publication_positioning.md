# Publication positioning and claim boundary

## Recommended description

An exploratory, paired fixed-checkpoint evaluation of a released Qwen3.5-0.8B weight edit on 24 hand-authored non-math tasks, accompanied by a re-score of the author's released benchmark records. The local study observes substantially more EOS-before-cap completions and lower repeated 8-gram fractions under the edited checkpoint. Task correctness is mixed and sensitive to the writing grader and three imperfect structured-task specifications.

## Claims supported by the current evidence

- We reproduced the author's reported aggregate benchmark scores from released per-item records; this verifies those records and scoring, not the original generations.
- On this specific paired 24-item suite, EOS-before-cap rose from 11/24 to 24/24, while strict exact structured matches rose from 4/16 to 6/16.
- A blind semantic sensitivity audit found structured success of 5/16 vs 9/16 with gold-relative schema strictness, or 6/16 vs 9/16 when that schema condition is relaxed; intervals are wide and include zero.
- The suite surfaced concrete completion-proxy failures and task-specification defects, which are useful evaluation findings in their own right.

## Claims not supported

- A robust or general improvement in non-mathematical capability, safety, factual calibration, coding, or user-facing quality.
- A causal mechanism that the weight edit suppresses reasoning loops. Repeat statistics and EOS are descriptive and do not isolate mechanism.
- A fresh reproduction of the paper's 50-pass Astra editing procedure or its IFEval, GPQA Diamond, or MMLU-Pro generations.
- A claim that all outcomes are correct because the answer terminated, parsed, or was shorter.

## Limitations that belong beside any headline

Only one sample per hand-authored task; two model graders rather than human raters; grader-sensitive writing outcomes; three task/gold specification ambiguities; some suite facts/rubric were exposed to a separate candidate editor after its freeze; fixed public p50c was not retuned on this suite; Apple MPS/Transformers backend differs from the paper's vLLM path; raw reasoning traces are not public. Exact strict scores are retained unchanged and semantic checks are explicitly post-hoc sensitivities.

## Short headline

**The released edit greatly increased normal termination on a small applied-task suite, but evidence that it improved non-math task correctness remains mixed and inconclusive.**
