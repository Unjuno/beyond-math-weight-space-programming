# Full-cap paired applied-task probe of the released p50c edit

## What this does and does not reproduce

The paper's non-math evaluation is IFEval (two samples per prompt), GPQA Diamond (four), and a fixed 1,000-question MMLU-Pro subset (one sample). Its reported base to 50-pass p50c scores are 21.0% to 62.3%, 22.7% to 36.6%, and 40.3% to 45.5%, respectively. This repository previously re-scored the authors' released per-item records; that verifies their scoring outputs but does not regenerate those benchmark generations.

This separate run tests the released p50c checkpoint on 24 hand-authored applied tasks: 8 information extractions, 8 operational decisions, and 8 grounded-writing tasks. It is an exploratory transfer probe, **not a rerun of the paper's benchmarks**. It also does not repeat the 50-pass edit search: the public checkpoint is fixed before evaluation. The task bundle was prepared while a separate local candidate was being edited; some task/rubric facts were later exposed to that editor after its candidate had been frozen. The p50c checkpoint was not selected or retuned using these tasks. The writing grader received anonymous A/B labels; its initial ratings were frozen before the arm mapping was revealed. The scores are Luna's rubric judgments, not an independent human study; a second blind rating audit is pending.

## Protocol

- Same pinned Qwen3.5-0.8B checkpoint, tokenizer, task prompt, per-task seed, and exact input token IDs for each base/edit pair; one sampled generation per arm per task.
- Public p50c edited tensors applied at MLP down-projection layers 15 and 19. Checkpoint, patch, tensor, task, and runner hashes are recorded in `results/heldout24_p50c_fullcap_manifest.json`.
- Thinking enabled; BF16 on Apple MPS with Transformers `AutoModelForImageTextToText`; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and 32,768 new-token cap. Per-task arm order was randomized. The paper used vLLM; the exact Experiment 8 persistent harness/backend is not released, so this inference run is approximate.
- Extraction and decision tasks require valid JSON with exactly the gold keys and strict recursive type/value equality. Writing requires all four preregistered rubric factors: full factual coverage, no unsupported/inaccurate claims, task and format met, coherent and usable. The primary task-success outcome also requires EOS before the cap. We report exact answer quality and EOS separately.
- Pair bootstrap resamples task IDs (50,000 draws, deterministic seed) for the edited-minus-base difference. With 24 hand-selected tasks and one sample per arm, the interval describes this task set only; family intervals each use eight items.

## Results

| Task family | Luna base / edit | Astra base / edit | Luna delta, 95% CI | Astra delta, 95% CI |
|---|---:|---:|---:|---:|
| Structured extraction (8) | 1/8 → 4/8 | 1/8 → 4/8 | +37.5 pp [+12.5, +75.0] | +37.5 pp [+12.5, +75.0] |
| Operational decisions (8) | 3/8 → 2/8 | 3/8 → 2/8 | −12.5 pp [−37.5, 0.0] | −12.5 pp [−37.5, 0.0] |
| Grounded writing (8) | 0/8 → 6/8 | 1/8 → 3/8 | +75.0 pp [+37.5, +100.0] | +25.0 pp [−25.0, +75.0] |
| **All tasks (24)** | **4/24 → 12/24** | **5/24 → 9/24** | **+33.3 pp [+12.5, +54.2]** | **+16.7 pp [−4.2, +37.5]** |

EOS-before-cap completion was 11/24 for base and 24/24 for edited. Exact typed JSON matches were 4/16 versus 6/16 across extraction and decisions. Thus the result reflects both a large completion difference and mixed correctness: extraction improved, operational-decision exact success declined by one item, and the writing result depends on the grader. Luna's first-pass and Astra's second blind ratings agree on 57/64 rubric factors (89.1%) and on 12/16 writing task-success outcomes. Astra's alternative overall paired interval includes zero. The EOS gain alone is not counted as task success.

## Interpretation and limits

On this selected applied suite, both reviewers found the same extraction and operational-decision outcomes, with extraction improving and decisions declining. The overall composite improvement is **grader-sensitive**: Luna's first pass gives +33.3 points, while Astra's independent blind writing score gives +16.7 points with an interval spanning zero. The rubric factors agreed on 57/64 judgments; four of the eight writing items changed pass/fail between reviewers. The second reviewer caught two concrete first-pass reading errors: the w04 answer has three actual lines, and the w05 answer omits “December.” The w02 request wording and w03 added actor remain semantic judgment differences. Full per-factor decisions and evidence are available in the two rating files. Each task ID was sampled once, so these intervals are coarse and apply only to this hand-authored suite.

These results are compatible with the paper's reported non-math transfer, but they do **not** independently reproduce its IFEval, GPQA Diamond, or MMLU-Pro percentages, and cannot establish broad non-math capability. The task suite is hand-authored, not a representative sample of work, and the backend differs from the paper. Full prompt/gold definitions and sanitized final answers are published for audit in `results/heldout24_p50c_tasks.json` and `results/heldout24_p50c_final_answers.jsonl`. Blind ratings are in `results/heldout24_p50c_blind_ratings.json` (Luna) and `results/heldout24_p50c_astra_blind_ratings.json` (Astra). Raw token-level generations, including long internal reasoning traces, remain local and ignored by Git; the sanitized artifact contains final answers and termination metadata only.

## Reproduction commands

With the pinned base checkpoint and public p50c patch installed locally, rerun paired generation with `scripts/run_transfer_eval.py` (use new output paths because the runner refuses to overwrite evidence):

```sh
python scripts/run_transfer_eval.py \
  --arm p50c \
  --model /path/to/Qwen3.5-0.8B \
  --patch-dir /path/to/p50c-patch \
  --tasks results/heldout24_p50c_tasks.json \
  --out /tmp/heldout24_p50c_raw.jsonl \
  --manifest /tmp/heldout24_p50c_manifest.json \
  --max-new-tokens 32768
```

The runner writes token-level output under the requested path; keep it local because long internal reasoning traces are included. To reproduce deterministic JSON scoring and blind writing packets on that new run, run:

```sh
python scripts/score_transfer_eval.py \
  --tasks results/heldout24_p50c_tasks.json \
  --raw /tmp/heldout24_p50c_raw.jsonl \
  --manifest /tmp/heldout24_p50c_manifest.json \
  --blind-ratings results/heldout24_p50c_blind_ratings.json \
  --out /tmp/heldout24_scores.json \
  --blind-packets /tmp/heldout24_blind_packets.json \
  --blind-map /tmp/heldout24_blind_map.json \
  --blind-ratings results/heldout24_p50c_blind_ratings.json
python scripts/analyze_transfer_eval.py /tmp/heldout24_scores.json \
  --secondary-ratings results/heldout24_p50c_astra_blind_ratings.json \
  --out /tmp/heldout24_analysis.json
```

The two blind rating files and their hashes are preserved; the arm map is embedded in the score artifact only after the Luna ratings were frozen. No CI was used.
