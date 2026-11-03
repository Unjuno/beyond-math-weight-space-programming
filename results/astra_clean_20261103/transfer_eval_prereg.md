# Held-out non-math transfer evaluation (preregistered, prompts sealed)

This specification and encrypted task bundle were fixed while the clean editor was still running and before any fresh candidate edit was frozen. The editor must not inspect the bundle or its key before freeze.

## Task set

- 24 new applied tasks: 8 structured extraction, 8 operational decisions, and 8 grounded-writing tasks.
- Task prompts and answer keys are stored in `transfer_eval_tasks.fernet`; ciphertext SHA-256: `ceed5ef77a5df5645e6a635246237e584e075d6c6493b4b5c21d29ec0be5a4e2`.
- The suite is hand-authored and exploratory. It covers routine information handling, operational choices under explicit constraints, and short grounded communications. It is not a standardized benchmark or a representative sample of all non-math work.

## Paired execution

- Evaluate only after the candidate weight edit is frozen and its hash is recorded.
- For each task, generate one base and one edited response with the same native chat-template token IDs, seed, model/backend, thinking setting, decoding configuration, and token cap. Randomize which arm runs first per task; retain raw outputs and token IDs unchanged.
- Use the paper's thinking-enabled generation parameters: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and up to 32,768 new tokens. Explicit EOS is the tokenizer assistant end token; pad uses the tokenizer pad token. Any backend change, reduced cap, or missing pair is disclosed as a deviation and is not described as a direct reproduction.
- The primary task-success result requires a completed response (EOS before the cap) and a correct task answer. Report completion, cap hits, exact/semantic correctness, and token count separately.

## Scoring

- Extraction and operational-decision outputs must be valid JSON with exactly the requested keys and recursively type-correct values matching the sealed answer key. Compare JSON types strictly; booleans and numbers are not interchangeable.
- Grounded-writing outputs are scored blind to arm, generation order, seed, and token count. A grader sees only the task prompt, its fixed rubric, and one anonymous final answer. Record `factual_coverage`, `no_unsupported_or_inaccurate_claim`, `task_and_format_fulfilled`, and `coherent_and_usable` as booleans; all four must be true for task success. Preserve brief evidence plus missing-fact and violation IDs.
- Freeze all scores before revealing the anonymous arm map. Do not use exact keyword checks as the sole semantic writing score.

## Analysis and limits

- Primary comparison: paired item success-rate difference (edited minus base) across all 24 tasks, with paired bootstrap 95% interval resampling task IDs. Report the three task families separately with intervals; family groups have only eight tasks each.
- One sample per item and hand-authored tasks limit uncertainty estimates. A positive result supports only this selected suite and configuration; it does not establish broad non-math capability or generalize numerically to IFEval/GPQA/MMLU-Pro.
- Keep the tasks sealed until weight freeze. Publish task definitions and scoring records only after evaluation is complete; never publish the encryption key.
