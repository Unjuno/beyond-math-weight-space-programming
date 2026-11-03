# Non-math task protocol

## Primary question

Does the released Qwen3.5-0.8B weight edit improve tasks other than mathematics, and is the effect limited to output completion after suppressing reasoning loops?

## Author-reported starting point

Under thinking mode, 32,768-token cap, temperature 0.6, top-p 0.95, top-k 20 and presence penalty 1.5:

| Task | Base | p10c | p25c | p50c | Samples in author protocol |
|---|---:|---:|---:|---:|---|
| IFEval strict instruction following | 21.0 | 49.0 | 39.9 | 62.3 | 2 per prompt |
| GPQA Diamond science QA | 22.7 | 29.4 | 30.1 | 36.6 | 4 per question |
| MMLU-Pro broad knowledge, fixed 1k | 40.3 | 40.0 | 44.7 | 45.5 | 1 per question |

Author scores are accuracy means over samples. GPQA base final value is 22.7 (149-question interim figure 24.5 is not the final score). These published numbers are hypotheses for this investigation, not our own result.

## Evaluation arms

- **Artifact re-score:** independently recompute paired per-problem differences from released author records; inspect IDs, denominators, sample counts and grader outputs.
- **Fresh checkpoint evaluation:** same base and p50c edit, prompt IDs, chat template, tokenizer, inference backend, precision, decoding settings, seeds and token cap. Save raw outputs and token IDs. This sprint's paired 24-item local probes are exploratory: thinking disabled with a 512-token cap, and thinking enabled with a 2,048-token cap. Neither reproduces the author's thinking-enabled 32,768-token benchmark protocol.
- **New non-math tasks:** preregistered held-out instruction-following, coding, extraction/summarization and factual QA prompts. Do not use these prompts in editing or prompt selection. Score with deterministic checks where possible and blinded human scoring for semantic quality.
- **Loop controls:** report answer accuracy, successful completion, cap-hit, repeated n-gram/sequence rate, and output length separately. A shorter answer is not automatically a better answer.

## Interpretability / rationale trace

For every interaction, store append-only JSONL fields: timestamp, run/turn id, Luna question, Astra explicit rationale, exact tool request, executed code hash, forward inputs summary, returned output/metrics, proposed weight change, edit hash, Luna interpretation, uncertainty and next question. Retain the original upstream traces unchanged. Do not label the explicit rationale as complete internal cognition.

## Decision rules

- Call a task gain only when same-item paired evaluation favors the edit and uncertainty is reported.
- Call a broad non-math gain only after at least two distinct task families improve, with no material regressions on the preregistered companion checks.
- If loop/truncation improves but semantic answer quality does not, report a completion/reliability effect, not capability gain.
- Any shortened generation limit, changed prompt/template/backend or reduced sample count is a separate exploratory run, never a direct reproduction of the paper's score. For this local probe, a completed task requires a parsed final answer after the generated `</think>` boundary, EOS, no token-cap hit, and all deterministic item checks passing. Reasoning prose is never scored as a user-visible answer. Boundary arrival, EOS, cap-hit, and task-check success are reported separately.
- The local prompts cover only explicit instruction constraints, structured extraction, and grounded writing. Coding, broad factual calibration, safety/refusal, human-rated quality, and general non-math capability remain untested.
- A fresh Astra run is a separate goal from checkpoint evaluation; do not substitute Luna's interpretation or another model's advice for Astra's actual interaction trace.
