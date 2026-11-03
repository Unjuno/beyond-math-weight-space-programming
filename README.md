# Beyond Math: Weight-Space Programming reproduction

Can direct weight editing improve **non-mathematical tasks** in Qwen3.5-0.8B, and where does that effect stop? This repository audits Takehi et al.'s released experiment and independently evaluates the released base/edited checkpoints, with particular attention to instruction following and general/science multiple choice.

## Why this study

The headline math gain may largely reflect ending a known reasoning loop. The paper also reports non-math results: IFEval 21.0→62.3, GPQA Diamond 22.7→36.6, and MMLU-Pro (fixed 1,000 questions) 40.3→45.5 after the 50-pass edit. These results suggest cross-task effects, but do not show a general improvement across arbitrary tasks. We will inspect paired per-problem records and separately evaluate tasks that test instruction following, factual knowledge, coding, and ordinary task completion.

## What we can and cannot establish

- **Can:** audit the released Astra prompts, responses, code, forward-pass summaries, exact final tensor edit and authors' per-problem benchmark records; rebuild the edited BF16 tensors from the pinned base revision; independently score public checkpoints on new non-math prompts; compare task accuracy, instruction adherence, answer completion, loops and truncation separately.
- **Cannot infer from artifact audit alone:** that Astra's stated rationale is its full internal reasoning; that an edit adds general capability rather than exposing existing capability by stopping loops; that performance transfers to models outside Qwen3.5-0.8B; that quality or safety is preserved on unmeasured tasks.
- **Cannot claim procedure reproduction** unless the original Astra model and the original harness constraints are available and we rerun the edit search. The original 50-pass exploration took 20.3 hours; a four-hour sprint can audit artifacts and run focused checkpoint evaluations, not promise an identical fresh search.

See [`docs/feasibility.md`](docs/feasibility.md), [`docs/experiment_protocol.md`](docs/experiment_protocol.md), and [`research_log.md`](research_log.md).

Our small local paired probes and their limits are in [`docs/local_probe_results.md`](docs/local_probe_results.md). They do not reproduce the author's benchmark protocol.

In a 24-item thinking-on probe (2,048-token cap), final-answer arrival increased from 6/24 to 13/24, but passing task checks remained 5/24 in both arms. In the separate thinking-disabled probe (512-token cap), checks passed 11/24 base versus 10/24 edited. These small custom probes indicate an output-completion shift in the thinking-on condition, without a measured increase in task success.

A separate post-hoc two-prompt check at the author's 32,768-token cap passed under both checkpoints and did not approach the cap; it is too small to establish a loop-rescue effect.

## Published conditions and results

The author's p50c run changed the BF16 values of 130,188 of 131,072 targeted weights in MLP output columns at layers 15 and 19. Reported 50-pass scores: IFEval 62.3%, GPQA Diamond 36.6%, MMLU-Pro 45.5%. Corresponding base scores: 21.0%, 22.7%, 40.3%. These are author-produced results, available for independent audit in the linked raw logs; they are not this repository's independent rerun.

## Working rules

- Keep raw upstream traces, edited weights, model caches and Luna's working memory intact. Do not rewrite original logs; append our own dated notes and preserve hashes.
- Record each Luna question to Astra, Astra's answer, proposed change, tool/code invocation, observed result and Luna's interpretation as separate fields. Label explanations as reported rationale, not access to hidden cognition.
- Public repository; no CI workflows. Heavy model artifacts and generated outputs stay local/ignored. Publish small reports, scripts and provenance.
- Every conclusion names the task, checkpoint, evaluation split, sample count, generation settings and remaining uncertainty.

## Sources

- [Takehi et al., Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [Complete author traces and Experiment 8 code](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)
- [Author-edited Qwen3.5-0.8B checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- [Base Qwen3.5-0.8B checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Hugging Face live demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo)
