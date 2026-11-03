# Luna project memory — append only

This file preserves project-level decisions and links to prior local research. Do not replace or truncate it; append dated entries. Existing model files, author traces, previous results and Codex memory remain in place.

## Prior local Qwen work retained

- The separate `evil-qwen-repro` project contains Qwen3.5-4B MPS/BF16 neuron and SAE experiments, raw generations, evaluation reports and logs. Those files are not moved or deleted.
- Reusable protocol lessons: preserve exact input token IDs, keep model revision/backend/decoding fixed, separate weight edits from activation interventions, and separate output form from semantic correctness/knowledge/safety claims.
- Those results concern a different model and intervention. They are design precedent only, not evidence for Qwen3.5-0.8B weight-space programming.

## Current question

Does the published edit transfer to non-math tasks? Check instruction following (IFEval), science QA (GPQA), broad knowledge (MMLU-Pro), and a small fresh held-out set. Distinguish task families and report boundaries; do not turn three benchmark results into a universal capability claim.

## Luna → Astra trace rule

Keep the exact question, Astra's explicit response, code/tool actions, observed output, edit/checkpoint hashes, and Luna's interpretation as separate append-only records. Explicit rationale is an observable explanation, not proof of complete access to latent reasoning. Original Astra traces are upstream artifacts and stay immutable.

## 2026-11-03 checkpoint

- Public project repo: https://github.com/Unjuno/beyond-math-weight-space-programming
- Author-trace revision: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- Author-reported p50c non-math scores: IFEval 21.0→62.3, GPQA Diamond 22.7→36.6, MMLU-Pro 40.3→45.5.
- Luna's item-cluster bootstrap re-score is in `results/author_nonmath_paired.csv`.
- p50c author search: 20.3 hours; exact final edit is released. Four hours is enough for focused checkpoint comparison, not a promise to rediscover the edit.
