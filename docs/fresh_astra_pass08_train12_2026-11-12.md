# Fresh Astra search, pass 8: fixed 12-sample TRAIN baseline

## Result

The preregistered request completed 12 original-weight generations: six new TRAIN items (three GSM8K and three MATH), two deterministic samples per item, with no early stopping. All 12 generated 32,768 tokens, all 12 were scored incorrect, and all 12 were cap hits. The repeated-8-gram fraction averaged 0.9020 (range 0.8308–0.9647). The per-item result was 0/2 correct and 2/2 cap hits for each of the six items.

| TRAIN item | Samples | Correct | 32,768-token caps | Mean repeated-8-gram fraction |
|---|---:|---:|---:|---:|
| gsm8k_train_3502 | 2 | 0 | 2 | 0.9053 |
| gsm8k_train_7033 | 2 | 0 | 2 | 0.8592 |
| gsm8k_train_3449 | 2 | 0 | 2 | 0.9050 |
| math_train_geometry_99 | 2 | 0 | 2 | 0.9286 |
| math_train_prealgebra_304 | 2 | 0 | 2 | 0.9003 |
| math_train_number_theory_285 | 2 | 0 | 2 | 0.9139 |
| **Total / mean** | **12** | **0** | **12** | **0.9020** |

Every scored record has a corresponding unscored raw record with matching ID, sample, seed, and generated token IDs. All 12 activation archives are present and finite. Each contains 32 generated-token bins for residual stream layers 0–24 and MLP down-projection outputs 0–23, plus prompt-last and first-8192-token summaries. Raw generations, token IDs, and activation arrays remain local and ignored by Git.

## Frozen conditions and provenance

- Request: `0014_new_train_six_by_two_fullcap.json`; SHA-256 `a23289b5be1d22293a4164844577977bbce71bf44c413fa90921d70a52ead988`.
- Input TRAIN files and item selection are recorded in the local preregistration `results/astra_reproduction_20261111_fresh50/raw/editor_work/analysis/preregistration_0014_train_sample.json`. The sample was fixed before generation; a distinct six-item confirmation set was reserved and untouched.
- Base revision: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- Thinking enabled; temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5; 32,768-token cap; local MPS/Transformers; math-verify 0.9.0.
- The request used one successful generation call and consumed 19,585.8 seconds (5 h 26 min 26 s), exceeding the four-hour work target. No weights were changed.
- Scored JSONL SHA-256: `b05a5fb95a0389d98fe56990a030950b14349bd920319c025c9e9cee585a8a58`; unscored raw JSONL SHA-256: `3c249963933cd782897ae0f1cada01a3c0c072b592b55dbeba03c0f8ed9307f9`.

## Interpretation and limits

This is an unedited, small TRAIN diagnostic after the first seven successful forward calls. It establishes that all 12 selected samples exhausted the cap and had substantial repeated 8-gram overlap under this local backend. The 8-gram statistic is descriptive; it does not prove that repetition caused the failures. Twelve TRAIN generations do not estimate benchmark accuracy, identify a causal direction or layer, evaluate an edit, reproduce the authors' 50-pass Astra search, or reproduce the paper's paired math benchmark.

The local forward path uses Transformers/PyTorch on Apple MPS. The authors used a different serving stack. The exact author orchestration, persistent editor kernel, and original forward service remain unavailable. The separate released-p50c checkpoint pilot and its exact tensor hashes are documented in [`method_fidelity_audit_2026-11-11.md`](method_fidelity_audit_2026-11-11.md); those do not turn this fresh search diagnostic into a method reproduction.

The next analysis must examine the saved text and binned activation timecourses across these fixed failures, report whether any feature is shared across items/time, and keep observational similarity separate from a causal intervention. Do not make an edit from this result alone. The reserved confirmation IDs remain sealed until a candidate and analysis rule are fixed.
