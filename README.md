# Beyond Math: audit and checkpoint evaluation of a weight-space edit

Can direct weight editing improve **non-mathematical tasks** in Qwen3.5-0.8B, and where does that effect stop? This repository audits benchmark records from Takehi et al.'s 2026 research blog and evaluates the released base/edited checkpoints on separate local probes. It does not regenerate the authors' benchmark outputs or reproduce their fresh Astra edit search.

## Why this study

The author report claims that the 50-pass edit improved non-math results: IFEval 21.0→62.3, GPQA Diamond 22.7→36.6, and MMLU-Pro (fixed 1,000 questions) 40.3→45.5. We re-scored its released per-problem records and ran paired probes on released checkpoints. In a separate 24-task full-cap applied probe, EOS-before-cap rose 11/24→24/24, while strict exact structured matches rose 4/16→6/16. The overall composite was 4/24→12/24 under Luna's frozen first-pass writing rubric and 5/24→9/24 under Astra's second blind rating; the latter paired interval spans zero. A later blind semantic audit found task/gold ambiguities and rated operational semantic success 3/8 in both arms. This is a completion/repetition signal with mixed, grader- and rubric-sensitive correctness, not established general non-math capability. Local probes do not regenerate the author's benchmark results. See [publication positioning](docs/publication_positioning.md) and the [full-cap report](docs/heldout24_p50c_results.md).

## Paper results and what we independently checked

The authors' [research blog report](https://rikiyat.github.io/blog/weight-space-programming/) and this repository's checks are different evidence types:

| Benchmark | Author-reported base → p50c | Sampling in author report | Local check in this repository |
|---|---:|---:|---|
| IFEval strict instruction following | 21.0% → 62.3% | 2 samples per prompt | Re-scored the released per-problem records; no new generations |
| GPQA Diamond science QA | 22.7% → 36.6% | 4 samples per question | Re-scored the released per-problem records; no new generations |
| MMLU-Pro fixed 1,000-question subset | 40.3% → 45.5% | 1 sample per question | Re-scored the released per-problem records; no new generations |

The re-score reproduces aggregate values from the authors' published records, not the original inference run. The fresh local 24-task suite below uses different, hand-authored tasks and is not a benchmark replication. See [`docs/author_results_nonmath.md`](docs/author_results_nonmath.md) for paired per-item intervals and [`docs/heldout24_p50c_results.md`](docs/heldout24_p50c_results.md) for the local evaluation.

## What we can and cannot establish

- **Completed here:** audit the released Astra prompts, responses, code, forward-pass summaries, final tensor edit and author per-problem records; independently re-score paired records; compare the public base and edited checkpoints on short probes and a separate 24-task thinking-on applied suite covering extraction, operational decisions and grounded writing; measure typed answer correctness, rubric scores, EOS, cap and task success separately.
- **Can be done with the released artifacts:** rebuild/check the edited BF16 tensors from the pinned base revision and rerun the local probe with matching hardware/software. This is checkpoint evaluation, not a fresh editing search.
- **Not tested here:** coding, broad factual calibration, safety/refusal, paraphrase robustness, human-rated semantic quality, and general non-math work. The narrow extraction and grounded-writing checks do not establish those abilities.
- **Cannot infer from artifact audit alone:** that Astra's stated rationale is its full internal reasoning; that an edit adds general capability rather than exposing existing capability by stopping loops; that performance transfers to models outside Qwen3.5-0.8B; that quality or safety is preserved on unmeasured tasks.
- **Fresh editing procedure not reproduced:** we did not rerun the authors' 50-pass search under equivalent conditions. The original took 20.3 hours; the complete editing-agent orchestration and forward-pass server are not fully available, so a new search with a different local harness would establish only results under that declared setup.
- **Separate approximate edit attempt:** a training-only alpha100 candidate was explored with an incomplete local harness. Four edit strengths were compared in a small teacher-forced screen. In one paired full-generation training example, the candidate emitted the gold boxed answer earlier, but both generations hit the 32,768-token cap without EOS and the edited output's repeated 8-gram fraction was higher (0.8668 vs 0.8539). It had no held-out confirmation and is not a successful reproduction or evidence for the released p50c results; see the [paired audit](results/astra_clean_20261103/pass_07/paired_generation_audit.json) and [interpretation](results/astra_clean_20261103/pass_07/interpretation.json).

See [`docs/feasibility.md`](docs/feasibility.md), [`docs/experiment_protocol.md`](docs/experiment_protocol.md), and [`research_log.md`](research_log.md).

Our local paired probes and their limits are in [`docs/local_probe_results.md`](docs/local_probe_results.md) and the new full-cap applied-task report is [`docs/heldout24_p50c_results.md`](docs/heldout24_p50c_results.md). Neither is a rerun of the paper's IFEval, GPQA Diamond or MMLU-Pro protocol. Reproduction commands, pinned dependencies/checkpoints and hashes are in [`docs/local_setup.md`](docs/local_setup.md) and [`results/experiment_manifest.json`](results/experiment_manifest.json). Run `REPRO_PYTHON=.venv/bin/python scripts/reproduce_summaries.sh /path/to/pinned/lm_understand-logs` to verify the author-record analyses and regenerate the original summary tables.

In the full-cap applied suite, EOS-before-cap completion was 11/24 base and 24/24 edited. Strict typed-JSON task success was mixed by family: extraction 1/8→4/8, operational decisions 3/8→2/8. A blinded semantic audit found 3/8→3/8 operational success and exposed three task/gold specification ambiguities; exact scores remain frozen and unchanged. Grounded-writing task success ranged from 0/8→6/8 under Luna's first pass to 1/8→3/8 under Astra's second blind ratings; 57/64 rubric factors agreed, with two clear Luna reading errors noted in the report. Writing was judged by two model graders, not humans. The tasks are hand-authored, one sample per task, and some facts were exposed to the separate editor after its candidate freeze. The fixed public p50c patch was not selected or retuned on these tasks. See the report for hashes, scoring, and limits.

In a 24-item thinking-on probe (2,048-token cap), final-answer arrival increased from 6/24 to 13/24, but completed tasks (final answer, EOS, no cap hit, and every deterministic check passing) remained 5/24 in both arms. In the separate thinking-disabled probe (512-token cap), checks passed 11/24 base versus 10/24 edited. These small custom probes indicate an output-termination shift in the thinking-on condition, without a measured increase in task success.

A separate post-hoc two-prompt check at the author's 32,768-token cap passed under both checkpoints and did not approach the cap; it is too small to establish a loop-rescue effect.

## Published conditions and results

The author's p50c run changed the BF16 values of 130,188 of 131,072 targeted weights in MLP output columns at layers 15 and 19. Reported 50-pass scores: IFEval 62.3%, GPQA Diamond 36.6%, MMLU-Pro 45.5%. Corresponding base scores: 21.0%, 22.7%, 40.3%. These are author-produced results, available for independent audit in the linked raw logs; they are not this repository's independent rerun.

## Working rules

- Keep raw upstream traces, edited weights, model caches and Luna's working memory intact. Do not rewrite original logs; append our own dated notes and preserve hashes.
- Record each Luna question to Astra, Astra's answer, proposed change, tool/code invocation, observed result and Luna's interpretation as separate fields. Label explanations as reported rationale, not access to hidden cognition.
- Public repository; no CI workflows. Full raw token-level generations and model weights stay local/ignored. Sanitized final answers, aggregate scores, small audit records and provenance may be published when identified clearly.
- Every conclusion names the task, checkpoint, evaluation split, sample count, generation settings and remaining uncertainty.

## Sources

- [Takehi et al., Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [Pinned author logs and published Experiment 8 evaluation/edit code](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)
- [Author-edited Qwen3.5-0.8B checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- [Base Qwen3.5-0.8B checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Hugging Face live demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo)
