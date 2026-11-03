# What the experiment can establish

## This sprint: measured scope

1. Re-score the released author records for IFEval, GPQA Diamond, and the fixed 1,000-item MMLU-Pro sample using paired item comparisons and bootstrap intervals. This audits the released evaluation artifacts; it does not regenerate the authors' model outputs.
2. Compare the released base and edited checkpoints on 24 custom prompts covering explicit instruction constraints, structured extraction, and grounded writing, under thinking-on/2,048-token and thinking-off/512-token conditions. Also report a post-hoc two-prompt thinking-on/32,768-token spot check.
3. Separate final-answer boundary, EOS, cap-hit, and deterministic task-check outcomes. The primary local success measure requires a final answer, EOS, no cap hit, and all item checks passing.
4. Preserve the Luna→Astra question, Astra's stated rationale, resulting action, observed evidence, and Luna's interpretation in the append-only project log. This records interpretable reasons for the choices; it is not a claim to expose full internal cognition.

## Not measured in this sprint

- Coding ability, factual calibration on open-ended QA, summarization quality, safety/refusal behavior, robustness to paraphrases, multilingual quality, or human-rated semantic quality were not evaluated. The extraction and grounded-writing prompts are narrow deterministic checks, not broad benchmarks for those capabilities.
- We did not run a fresh 50-pass editing search, test an Astra editor against the original harness, reproduce the authors' original generations, or evaluate other models.

## Out of scope or not currently identifiable

- A public checkpoint and complete logs let us audit the **submitted final edit** and rerun evaluation. They do not prove Astra's whole latent reasoning. Public traces contain its explicit step-by-step explanations and executable actions; these are observable rationales, not a guarantee of complete access to internal thought.
- Running a different model to imitate Astra's edit is not a reproduction of the Astra procedure. The editor, harness, prompts, state handoffs, and no-training controls all matter.
- The original harness source is not released. Its behavior is described in prompts/logs but cannot be fully audited as executable code.
- Training-set exposure controls appear in the logs, but an external rerun cannot prove the original editor had no other data access beyond the logged/tool boundary.
- Gains on three non-math benchmarks do not establish cross-domain general intelligence. GPQA is science QA; MMLU-Pro is a fixed 1k multiple-choice sample; IFEval measures explicit instruction constraints. None alone measures all practical non-math work.
- A benchmark score does not prove reliable performance in production, absence of regressions, safety, truthfulness, robustness to paraphrase, or transfer to other model sizes/families.
- The authors reported no established math gain for non-looping Gemma 3 270M; the result supports a boundary hypothesis: removing an obstruction may reveal latent capability, while edits do not automatically create absent capability.

## Four-hour sprint deliverables completed

- Pinned the author-log and model revisions; audited 488 tensor keys and confirmed 130,188 changed BF16 values in the two released edited tensors.
- Independently re-scored the released author per-problem records with paired comparisons and uncertainty intervals.
- Ran and verified 24 matched prompts for thinking-on and 24 for thinking-off, plus two post-hoc original-cap prompt pairs; saved complete raw outputs locally.
- Published task-by-task findings, the rationale and correction trail, reproduction scripts, pinned package versions, and an artifact-hash manifest.

## Claims ladder

| Evidence achieved | Allowed conclusion |
|---|---|
| Author score table only | Authors report a task gain |
| Raw per-problem logs independently re-scored | Released evaluation artifacts support / fail to support the reported metric |
| Fresh paired evaluation on released checkpoints | The published edit changes performance on this new measured sample under stated settings |
| Fresh Astra editing run with audited harness + held-out evaluation | The editing procedure reproduced under those conditions |
| Multiple task families, seeds, models, controls | Broader transfer evidence, still limited to tested scope |

## Practical boundary

This experiment shows that an already released weight edit can be inspected, replayed on available checkpoints, and assessed on bounded non-math tasks without changing the model weights. It does not establish that an agent can reliably perform arbitrary non-math work, nor that an editing procedure can discover this change again. On the local thinking-on sample, answer-boundary arrival rose from 6/24 to 13/24 while completed checked tasks stayed 5/24 in each arm. That supports a stopping/completion shift under this particular cap and prompt set, not broader capability.
