# Why Astra chose the p50c edit: observable trace

This is a reconstruction from Astra's published `p50c` prompts, explicit explanations, executed Python/forward calls, work notes and final edit manifest. It records the **reported decision path**; it is not evidence that the trace exposes all latent reasoning. The source logs are preserved upstream at the pinned [p50c run](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c).

## Observable decision path

1. **Identify the failure mode.** On its first 96 practice generations, 85 reached the 32,768-token cap. The run notes describe repeated reconsideration of already established steps. Astra treated non-termination as the main obstruction to observing what the model could answer.
2. **Form a mechanism hypothesis.** It compared matched productive and repetitive continuations and used the forward tool to inspect model activations/log probabilities. The recorded evidence implicated MLP outputs in layers 15 and 19 as strong writers of the repetition-associated direction.
3. **Try narrow edits and reject weak evidence.** The trace includes single-neuron ablations and rank-one direction removals. Astra explicitly treated small gains with large standard errors as inconclusive, and asked for independent held-out practice evaluation instead of keeping a candidate based only on shorter answers.
4. **Check the mechanism.** It used held-out paired practice problems, dose-response, layer isolation, and a norm-matched random direction control. The final explanation says the edit reduced repetition while retaining productive reasoning; correct edited answers still used substantial reasoning tokens. The trace also records some regressions and failure cases, so it did not conclude universal harmlessness.
5. **Choose a sparse final intervention.** The submitted p50c edit subtracts the repetition-associated direction from 64 MLP output columns at each of layers 15 and 19. It leaves token embeddings, output head, final norm, and other layers untouched. The edit changes 130,188 BF16 values out of 131,072 targeted entries.
6. **Evaluate withheld tasks.** Only after the edit was frozen did the authors evaluate math, instruction following, science QA and broad multiple choice. Astra was not shown those benchmark questions during editing. This separation matters when interpreting the non-math scores.

## How Luna uses this trace

The local extension asks a narrower question than the original Astra search: does this frozen released edit help non-math tasks, and does the effect depend on thinking mode? The Luna→Astra design questions and answers are appended in [`logs/luna_astra_dialogue.jsonl`](../logs/luna_astra_dialogue.jsonl); the experimental prompts, outputs, hashes and task scores stay separate. Astra's advice is not a substitute for an actual new p50c search.

## Interpretive boundary

The public trace makes the editor's stated diagnosis, tests and acceptance choices reviewable. It cannot prove that every latent cause was verbalized, that no behavior was omitted, or that another run would select the same edit. The original sandbox/harness implementation is not fully released, and the complete 50-pass search took 20.3 hours. Thus, the **final checkpoint edit and its stated justification** are auditable; full autonomous procedure reproducibility remains incomplete.
