# Independent paired p50c MATH-500 follow-up (2026-11-12)

## Result

The published p50c weight edit was freshly reconstructed and hash-checked before this run. On two previously unused, hash-selected MATH-500 items, the local paired run scored **base 0/8, p50c 6/8** using the pinned author answer extraction and `math-verify 0.9.0`.

| Frozen item ID | Base | p50c | Base cap hits | p50c cap hits |
| --- | ---: | ---: | ---: | ---: |
| `test/algebra/567.json` | 0/4 | 4/4 | 4/4 | 0/4 |
| `test/number_theory/357.json` | 0/4 | 2/4 | 4/4 | 2/4 |
| **Total, descriptive only** | **0/8** | **6/8** | **8/8** | **2/8** |

The base arm produced no boxed answer in any of the eight runs and hit the 32,768-token limit every time. The edit answered all four samples on the first item and two of four on the second. One edited response on the second item closed its thinking section but still had no boxed answer and was scored incorrect. The six correct edited answers were checked against the dataset reference by the author-compatible extraction plus symbolic answer verifier.

These observations are consistent with the published edit helping completion and correctness on these selected examples. They do **not** reproduce the reported 48.1% MATH-500 score: this is 2 of 500 items, and the local generation backend and RNG are not the authors' vLLM setup. The samples are descriptive, not a benchmark accuracy estimate or a statistical confirmation. The earlier 2026-11-11 pilot was a separate tranche; this follow-up's two item IDs and selection rule were frozen together before their problem bodies were written to the run folders.

## What was done and why

1. Retrieved all 500 rows at dataset revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, ranked IDs by SHA-256 of a fixed salt and ID, and excluded the two items already evaluated in the earlier pilot. The first two remaining IDs were frozen before generation. This avoids choosing examples based on model outputs.
2. Rebuilt the p50c edit from the pinned author artifacts and checked that L15 and L19 `down_proj.weight` hashes equal the published hashes. The full rebuild audit is documented in [the method-fidelity audit](method_fidelity_audit_2026-11-11.md).
3. Ran four base and four edited generations per item with thinking enabled, the author-reported temperature/top-p/top-k/presence-penalty settings, and a 32,768-token cap. Seed indices 0–3 were initialized separately for each checkpoint. The exact prompt suffix and author scorer were retained.
4. During earlier MPS attempts, indexing an MPS score tensor by generated token IDs failed. The runner was changed to apply the presence-penalty update to a validated CPU copy of the FP32 logits, then return scores to MPS. It rejects out-of-range IDs and non-finite logits rather than repairing them. All 16 completed generations have in-vocabulary token IDs; their raw file hashes and summaries match the frozen local manifests.
5. Kept generated token IDs, text, per-sample scores, preregistration, attempts, and logs under ignored `results/raw/math500_reproduction_20261112_followup/`. Only aggregate counts, IDs, protocol, and hashes are tracked here; raw generations are not in Git.

## Fidelity and limits

- **Reproduced:** reconstruction of the released p50c weight edit, including exact published L15/L19 tensor hashes; the reported sampling values, prompt suffix, cap, last-boxed-answer rule, and `math-verify 0.9.0` grading intent.
- **Approximated:** evaluation used local Transformers 5.20.2 / PyTorch 2.14.1 on Apple MPS. The paper reports vLLM. Separate local seed initialization does not reproduce vLLM's random stream or guarantee token-by-token matched samples. The custom logits processor implements the stated presence penalty but is not a vLLM execution.
- **Not reproduced:** the full 500-item benchmark evaluation and the fresh 50-pass Astra edit-search trajectory. Replaying the already released final edit is not the same as reproducing the search that selected it.
- **Time cost:** these 16 generations took 14,909.875 seconds of recorded generation time (4.14 hours; mean 932 seconds each). A straight-line estimate for 4,000 generations (500 items × 4 samples × 2 checkpoints) is about 1,035 hours, or 43 days on this local setup, before overhead. The full benchmark was therefore not completed.
- **Reward-hacking check:** each positive score required the last extracted boxed answer to verify against the corresponding reference. We saw no false positive in these 16 outputs under that scorer. This small check cannot establish the absence of evaluator gaming or semantic failures in general.

## Reproducibility records

- Dataset rows SHA-256: `8c64443b4679d04084c214ae2cd13788f384ae5e3b17c33086a8960e97aa686b`
- Frozen common preregistration SHA-256: `157aad9be41a0e85a5e7e1b6aa07da377a81d3c4b1483cfc0dc044dd8de69db4`
- Item 1 raw local JSONL SHA-256: `69fd533fc19254a91861d147055d04b784d23bf5b496199d46a8f7f09f210f0a`
- Item 2 raw local JSONL SHA-256: `55ffd150ab9972a1b36504fe53b92e729bd719ec4f8724d123f84b0fcda88c96`
- Runner: [`scripts/run_math_reproduction.py`](../scripts/run_math_reproduction.py)
- Frozen selection: [`scripts/preregister_math500_followup.py`](../scripts/preregister_math500_followup.py)
