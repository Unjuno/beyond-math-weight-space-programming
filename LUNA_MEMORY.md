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
- Luna→Astra explicit rationale is persisted in `logs/luna_astra_dialogue.jsonl`; local question and subsequent reasoning are logged append-only. These observed explanations help interpret why Luna chose each action, but do not establish complete access to model cognition.
- Luna's initial local thinking-enabled attempt was stopped after 11 base outputs at a 2,048-token cap without `</think>`; those raw rows were retained, then reused in the completed paired 24-item thinking-on run after Astra advised matching all questions. The experiment scope and rationale are documented in `research_log.md` and `docs/local_probe_results.md`.
- Local no-thinking result: base 11/24, edited 10/24; paired delta -4.2 pp (95% item-bootstrap interval -16.7 to +8.3). Treat as exploratory, not evidence of general improvement or harm.
- Luna interprets the local result as no detected improvement on selected short-form observable checks. The family mix and wide interval do not show a stable direction; because thinking was disabled and neither arm hit the output cap, this does not test the proposed loop-suppression mechanism. Luna asked Astra to explain this boundary and suggest the most informative next check; preserve the exchange and both rationales append-only when Astra responds.
- Astra advised a matched 24-item thinking-on base/edit run at the 2,048-token cap, scoring answer-plus-check success over all prompts, with answer arrival and cap rate separated. It emphasized that current 11 base thinking outputs are censored and do not prove indefinite looping; the no-thinking result neither confirms nor refutes author IFEval. Exchange 3 and Luna's decision rationale are in the dialogue JSONL.
- Completed that matched thinking-on pair: after correcting the scorer to use the thinking flag and generated `</think>` boundary, answer+check passed 5/24 in both arms; final-answer arrival was 6/24 base vs 13/24 edited (delta +29.2pp, paired item bootstrap 95% [+8.3,+50.0]); cap hits 18/24 vs 12/24; EOS 6/24 vs 12/24; mean tokens 1,894 vs 1,684. This supports increased answer completion under a short cap but not increased task success. Summary is `results/thinking_probe_summary.csv`.
- Correction audit: an intermediate report incorrectly claimed zero final answers because the opening `<think>` token was supplied by the prompt and therefore absent from decoded generated text. Luna inspected the template and raw outputs, changed the parser to use the recorded thinking flag plus closing `</think>`, recomputed summaries, and corrected every report before push. Earlier interim values are superseded by the corrected aggregates; raw JSONL was not edited. The explicit question/rationale is in dialogue exchange 4.
- Further audit correction: the initial 11-row base sample was not 11/11 censored; three rows already had a generated `</think>` boundary. The first two inspected examples lacked it, which led to an overly broad interim statement. The complete paired counts are authoritative; the partial-sample characterization is superseded.
- After Astra suggested a small original-cap check, Luna drew inst03 and inst05 with seed 20261103 and tested both checkpoints at 32,768 max tokens. All 4 outputs passed, with 332/283 and 1,161/771 tokens; no cap hits. Since this was post-hoc and n=2, it only confirms these two tasks completed under either checkpoint. See `results/long_cap_probe_summary.csv`.
