# Method-fidelity audit: published Qwen edit vs. the L12 directional screen

## Correction

The 24-generation experiment completed on 2026-11-10 was **not a reproduction of the paper's retained edit or its reported math evaluation**. It tested a later, locally designed L12 single-column intervention. The result T=0/6 is valid for that frozen continuation screen, but it cannot be used as a negative replication of the paper's p50c edit. Earlier wording that let those conclusions blur together was wrong.

## What the authors did

The Experiment 8 prompt asked Astra to diagnose math failures and make targeted weight changes. Astra had a persistent CPU Python kernel over the weights, 7,469 GSM8K and 7,275 MATH practice problems plus AIME answers, and a 50-forward-pass budget for generations or teacher-forced activation/log-probability reads. The p50c run used GPT-6 Astra at max reasoning, 158 turns, 80 Python runs, and about 20.3 hours. The experiment protocol froze thinking-on sampling and evaluated up to 32,768 new tokens. The published edit subtracted measured repetition directions from 64 `mlp.down_proj` output columns in each of layers 15 and 19. The final report describes matched productive/repetitive contexts, dose-response tests, layer-isolation tests, a norm-matched random control, and a paired confirmation on 1,152 practice problems. See the [p50c prompt](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt), [Experiment 8 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md), [p50c final report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/final_answer.md), and the [authors' blog](https://rikiyat.github.io/blog/weight-space-programming/).

## What the L12 screen did

The L12 study chose one observational candidate, `model.language_model.layers.12.mlp.down_proj.weight[:,1226]`, from two selected failure traces. It attenuated that single column by 10% and compared it with two signed-direction column controls. It ran 24 generations: three saved TRAIN prefixes, two seeds each, four arms, and a 4,096-token continuation cap. It did not reproduce the authors' identification process, layer targets, 64-column shared-direction edit, 50-pass Astra search, dose-response/layer-isolation package, 1,152-pair confirmation, or benchmark evaluation.

| Dimension | Authors' p50c | Our L12 screen |
| --- | --- | --- |
| Editor/search | Astra in the registered Python/forward-pass harness; 50 passes | Luna ran a predesigned local candidate screen; no equivalent 50-pass Astra search |
| Target | Shared repetition direction, MLP output columns in L15 and L19; 64 per layer | One L12 MLP down-projection input coordinate/weight column, index 1226 |
| Edit | Subtract the measured direction from 128 selected output columns total | Attenuate one column by 10%; C+/C− are signed controls |
| Causal checks | Matched productive/repetitive contexts, dose response, layer isolation, norm-matched random control | A narrow four-arm comparison of one preselected column; it does not validate the author's direction |
| Evaluation | Paired practice confirmation (1,152 problems) and benchmarks; thinking on, up to 32,768 tokens | Three saved-prefix continuations x two seeds; 24 generations, 4,096-token cap |

Both use the same Qwen3.5-0.8B family and the local base revision is pinned to `2fc06364715b967f1860aea9cf38778875588b17`, the revision named in the author logs. Shared model identity does not make the edit, search, or evaluation protocol equivalent.

## What is actually established in this repository

1. The public p50c tensor patch was rebuilt and its two edited tensor hashes were checked against the released artifact.
2. Author-released per-item records were re-scored to check the reported aggregates. That reproduces arithmetic on published records, not the original inference run.
3. The released p50c checkpoint was independently run on custom local probes, including a separate 24-task full-cap applied suite. Those outputs are useful checkpoint-evaluation evidence, but they are not a fresh Astra edit search or a rerun of MATH-500/GSM8K/IFEval/GPQA/MMLU-Pro as in the paper. The mixed local task-success results and their limits remain in [`heldout24_p50c_results.md`](heldout24_p50c_results.md).
4. The separate alpha100, L23, and latest L12 experiments tested other locally constructed candidates. None reproduces the authors' method.

## Correct conclusion

The exact released p50c patch is now rebuilt and bit-exact at both edited tensors. A small fresh paired evaluation on two randomly selected MATH-500 problems reproduces the **direction of the loop-termination effect**: base ended in a cap-hit verbatim loop on all 8 generations, while p50c did so on 1/8. The observed task correctness was mixed: 5/8 p50c generations were correct, with 4/4 on the first, elementary multiple-choice item and 1/4 on the second counting item. The latter had one wrong boxed answer (8 instead of 9), one unboxed response, one correct answer, and one cap hit. This is evidence that the published checkpoint can shorten/terminate looping outputs under matched local settings, not evidence that math accuracy rose to the paper's reported MATH-500 score.

The paper reports 48.1% on MATH-500 for p50c versus 7.9% for base, using 4 samples per problem across all 500 problems. Our two-item sample is far too small to estimate that benchmark score. The second problem was selected by a fixed seed without looking at candidate output, but it was added after seeing the first tranche and is therefore explicitly a sequential exploratory follow-up, not confirmatory evidence. The exact edit was evaluated; Astra's 50-pass editing trajectory was not rerun.

The earlier L12 screen remains a separate local variant. Its T=0/6 result does not test or refute p50c. Full conditions, selection rationale, results, failure modes, reward-hacking review, and artifact hashes for this run are in [the sanitized pilot summary](../results/p50c_math500_pilot_summary_20261111.json); raw generations stay local/ignored.

To reproduce the **editing method**, one must still reconstruct the authorized Astra harness and repeat the 50-pass search trajectory. Replaying the released final patch cannot establish that fresh search.

## Paired MATH-500 checkpoint pilot (2026-11-11)

### What was fixed before inference

- Base: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- Edit: `RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a178705bcab7644806d23c535517f35723b`.
- The base and edited checkpoints have the same 488 tensor keys. Only L15/L19 MLP `down_proj.weight` changed, with 130,188 BF16 values differing. The edited tensor hashes match the published values: L15 `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53`, L19 `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2`.
- Dataset: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) at commit `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; `test.jsonl` SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`.
- Each item was selected by a deterministic random rule before generating its paired base/edit outputs, then run with matched seeds `0–3`, thinking enabled, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and a 32,768-token cap. The second item was added after seeing the first tranche and is explicitly exploratory. Grading used the last boxed answer and math-verify 0.9.0.
- Initial tranche: Python `random.Random(20261111).randrange(500)` chose index 227, `test/precalculus/819.json`. The problem asks which shape is described by `z=c` in cylindrical coordinates; gold is plane. This is a simple item and should not be treated as a difficult reasoning test.
- Sequential follow-up: `random.Random(20261112)` chose index 107 without replacement, `test/number_theory/914.json`. It was registered after the initial result, before follow-up inference, and is marked exploratory. The problem asks for positive odd primes below 30; gold is 9.
- Local generation backend: Transformers 5.20.2 / PyTorch 2.14.1 on Apple MPS. The authors report vLLM, so the software backend differs even though the model, prompt template, sampling settings, token cap, and scorer were aligned.

### Results

| Randomly selected item | Arm | Correct / 4 | Cap hits / 4 | Boxed answers / 4 | Mean generated tokens | Terminal verbatim loop / 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Precalculus, level 3 | Base | 0 | 4 | 0 | 32,768 | 4 |
| Precalculus, level 3 | p50c | 4 | 0 | 4 | 7,966.5 | 0 |
| Number Theory, level 2 | Base | 0 | 4 | 0 | 32,768 | 4 |
| Number Theory, level 2 | p50c | 1 | 1 | 2 | 22,607 | 1 |

Across these 2 problems and 8 matched seeds, base was 0/8 correct, 8/8 cap-hit, and 8/8 ended in a repeated terminal word block. p50c was 5/8 correct, 1/8 cap-hit, and 1/8 ended in such a block. The terminal-loop diagnostic checks for a 1–200-word block repeated at least three times over at least 40 words. A separate repeated-4-gram statistic also fell on average from 0.932 in base to 0.651 in p50c, but it is a coarse diagnostic and not the primary correctness score.

### Reward-hacking and interpretation review

There was no score-driven editing or benchmark-specific tuning in this run: the p50c checkpoint and each problem-selection rule were fixed before generating that tranche's outputs, and every output was retained. The easy first item was answered correctly in all four edited samples. On the second item, edited outputs were mixed: one correct `9`, one incorrect `8`, one response with no boxed answer, and one cap-hit response with no boxed answer. Correctness required answer equivalence, so boxed formatting alone did not earn a positive score. One exact terminal loop remained in the edited arm, and repeated 4-grams remained common; therefore the edit did not eliminate all repetitive behavior. These checks find no direct reward-hacking signal in this tiny sample, but cannot establish its absence generally.

### Scope boundary

This is a paired checkpoint pilot, not the complete paper reproduction. It covers 2 of 500 MATH-500 items; its per-item and overall proportions are descriptive only. A rough estimate using the paper's published mean output lengths and our local throughput is on the order of several weeks for 500 items x 4 samples x 2 checkpoints, before scoring and setup. The full benchmark score and fresh 50-pass Astra edit search remain un-reproduced. The raw model generations and prompt records stay in ignored local paths; only sanitized counts, IDs, protocol, and hashes are published.
