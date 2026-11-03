# What the experiment can establish

## In scope

1. Whether the released edit reproduces the authors' claimed non-math effects on the reported tasks: instruction following (IFEval), graduate-level science QA (GPQA Diamond), and broad multiple-choice knowledge (MMLU-Pro).
2. Whether these effects remain when evaluation is paired by the same questions and when completion/truncation is measured separately from answer accuracy.
3. Whether gains extend to additional non-math tasks such as coding, summarization, extraction, and multi-constraint instructions, using held-out prompts and independent scoring.
4. Whether any apparent gain is explained mostly by reducing looping/timeout behavior rather than improving underlying knowledge or reasoning.
5. Whether edits damage other behavior: factual calibration, refusal/harmlessness, format adherence, language quality, or task robustness.

## Out of scope or not currently identifiable

- A public checkpoint and complete logs let us audit the **submitted final edit** and rerun evaluation. They do not prove Astra's whole latent reasoning. Public traces contain its explicit step-by-step explanations and executable actions; these are observable rationales, not a guarantee of complete access to internal thought.
- Running a different model to imitate Astra's edit is not a reproduction of the Astra procedure. The editor, harness, prompts, state handoffs, and no-training controls all matter.
- The original harness source is not released. Its behavior is described in prompts/logs but cannot be fully audited as executable code.
- Training-set exposure controls appear in the logs, but an external rerun cannot prove the original editor had no other data access beyond the logged/tool boundary.
- Gains on three non-math benchmarks do not establish cross-domain general intelligence. GPQA is science QA; MMLU-Pro is a fixed 1k multiple-choice sample; IFEval measures explicit instruction constraints. None alone measures all practical non-math work.
- A benchmark score does not prove reliable performance in production, absence of regressions, safety, truthfulness, robustness to paraphrase, or transfer to other model sizes/families.
- The authors reported no established math gain for non-looping Gemma 3 270M; the result supports a boundary hypothesis: removing an obstruction may reveal latent capability, while edits do not automatically create absent capability.

## Four-hour achievable endpoint

- Verify the author release revision and selected raw outputs.
- Rebuild/check the BF16 edit and its tensor hashes where resources allow.
- Independently re-score released per-problem results with paired comparisons and uncertainty intervals.
- Run a focused, held-out non-math checkpoint comparison locally if both models download and load within the time budget.
- Publish a task-by-task report, clearly labeling artifact audit, independent re-score and fresh inference as different evidence layers.

## Claims ladder

| Evidence achieved | Allowed conclusion |
|---|---|
| Author score table only | Authors report a task gain |
| Raw per-problem logs independently re-scored | Released evaluation artifacts support / fail to support the reported metric |
| Fresh paired evaluation on released checkpoints | The published edit changes performance on this new measured sample under stated settings |
| Fresh Astra editing run with audited harness + held-out evaluation | The editing procedure reproduced under those conditions |
| Multiple task families, seeds, models, controls | Broader transfer evidence, still limited to tested scope |
