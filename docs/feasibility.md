# What the experiment can establish

## This sprint: measured scope

1. Re-score the released author records for IFEval, GPQA Diamond, and the fixed 1,000-item MMLU-Pro sample using paired item comparisons and bootstrap intervals. This audits the released evaluation artifacts; it does not regenerate the authors' model outputs.
2. Compare the released base and edited checkpoints on the original 24-prompt thinking-on/2,048-token and thinking-off/512-token probes, a post-hoc two-prompt original-cap spot check, and a fresh 24-task thinking-on/32,768-token applied suite covering extraction, operational decisions, and grounded writing.
3. Separate final-answer boundary, EOS, cap-hit, typed answer correctness, writing-rubric factors, and composite task success. The full-cap suite reports two independent blind model-rater results for the subjective writing component.
4. Preserve the Luna→Astra question, Astra's stated rationale, resulting action, observed evidence, and Luna's interpretation in the append-only project log. This records interpretable reasons for the choices; it is not a claim to expose full internal cognition.

## Not measured in this sprint

- Coding ability, open-domain factual calibration, safety/refusal behavior, paraphrase robustness, multilingual quality, or human-rated semantic quality were not evaluated. The new writing rubric was rated by two language-model graders, not humans. The prompts are narrow and hand-authored, not broad benchmarks for those capabilities.
- We did not run a fresh 50-pass editing search, test an Astra editor against the original harness, reproduce the authors' original generations, or evaluate other models.

## Out of scope or not currently identifiable

- The released checkpoint and published traces let us audit the **submitted final edit** and rerun evaluation. They do not prove Astra's whole latent reasoning. Published traces contain explicit explanations and executable actions; these are observable rationales, not a guarantee of complete access to internal thought.
- Running a different model to imitate Astra's edit is not a reproduction of the Astra procedure. The editor, harness, prompts, state handoffs, and no-training controls all matter.
- The published evaluation code is available and uses vLLM. The complete edit-search agent orchestration (prompts, sandbox and forward-pass server) is not released, so a fresh editing run cannot be checked for full procedural equivalence. Our local checkpoint evaluation instead used Transformers on Apple MPS, which is a separate backend difference.
- Training-set exposure controls appear in the logs, but an external rerun cannot prove the original editor had no other data access beyond the logged/tool boundary.
- Gains on three non-math benchmarks do not establish cross-domain general intelligence. GPQA is science QA; MMLU-Pro is a fixed 1k multiple-choice sample; IFEval measures explicit instruction constraints. None alone measures all practical non-math work.
- A benchmark score does not prove reliable performance in production, absence of regressions, safety, truthfulness, robustness to paraphrase, or transfer to other model sizes/families.
- The authors reported no established math gain for non-looping Gemma 3 270M; the result supports a boundary hypothesis: removing an obstruction may reveal latent capability, while edits do not automatically create absent capability.

## Four-hour sprint deliverables completed

- Pinned the author-log and model revisions; audited 488 tensor keys and confirmed 130,188 changed BF16 values in the two released edited tensors.
- Independently re-scored the released author per-problem records with paired comparisons and uncertainty intervals.
- Ran and verified 24 matched prompts for thinking-on and 24 for thinking-off, plus two post-hoc original-cap prompt pairs; saved complete raw outputs locally.
- Published task-by-task findings, the rationale and correction trail, reproduction scripts, pinned package versions, and an artifact-hash manifest.
- Continued beyond the initial four-hour plan to run 24 full-cap applied tasks (48 paired generations), freeze two anonymous writing-score sets, and calculate grader-specific paired uncertainty intervals. The new result is suggestive but grader-sensitive; see `heldout24_p50c_results.md`.
- Two separate approximate training-only editing attempts were made. The initial alpha100 attempt had one paired generation; the edited output emitted the gold answer earlier but both arms hit the 32,768-token cap without EOS and repetition worsened. A later fresh search screened dose and single-column interventions, froze a layer-23 candidate before confirmation, and evaluated candidate/base/control on four untouched items. All three arms were 0/4 strictly correct and all generations hit the 8,192-token cap without EOS; candidate repetition was slightly higher on average than base. These are unsuccessful/inconclusive pilots, not a reproduction; see [the fresh search report](fresh_edit_search_results.md). Their evidence remains separate from the released-p50c evaluation.

## Claims ladder

| Evidence achieved | Allowed conclusion |
|---|---|
| Author score table only | Authors report a task gain |
| Raw per-problem logs independently re-scored | Released evaluation artifacts support / fail to support the reported metric |
| Fresh paired evaluation on released checkpoints | The published edit changes performance on this new measured sample under stated settings |
| Fresh edit under an explicitly described, audited local harness + held-out evaluation | An independent editing experiment was run under those stated conditions; this alone does not establish equivalence to the authors' procedure |
| Demonstrated process equivalence to the author setup, then fresh edit + independent held-out evaluation | Evidence that the edit procedure was reproduced under the compared conditions |
| Multiple task families, seeds, models, controls | Broader transfer evidence, still limited to tested scope |

## Practical boundary

This experiment shows that an already released weight edit can be inspected, replayed on available checkpoints, and assessed on bounded non-math tasks without discovering or training an additional edit. It does not establish that an agent can reliably perform arbitrary non-math work, nor that an editing procedure can discover this change again. In the full-cap suite, edited EOS completion was higher (24/24 vs 11/24), while exact structured matches changed from 4/16 to 6/16. A blinded semantic audit makes operational success flat (3/8 vs 3/8) and identified specification ambiguities in x07, d07, and d08. The writing-sensitive all-task gain was +33.3 pp under Luna's frozen pre-audit first pass and +16.7 pp under Astra's independent blind rating; Astra's paired 95% interval includes zero. The first pass contains two known reading errors (w04-A line count and w05-A missing December), so retain it only as a frozen sensitivity analysis, not as a gold-standard label. Repeated 8-gram fractions also fell descriptively, but these metrics do not prove a causal loop-suppression mechanism. This supports a completion/repetition shift; a robust broad non-math capability claim remains unestablished. See `publication_positioning.md` for the recommended claim boundary.
