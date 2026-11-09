# Human-operated L12 directional intervention (protocol revision 2)

Status: **design prepared, not yet executable or frozen**. No model weights were changed and no confirmation item body was opened. This is a distinct intervention design nested under the human-operated study; it does not erase or replace the failed matched-coordinate control result.

## Identity and separation

- Parent study: `human-astra-manual-20261105-v1`.
- New intervention: `human-astra-manual-20261105-l12c1226-direction-v1`.
- Base checkpoint SHA-256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Candidate tensor: `model.language_model.layers.12.mlp.down_proj.weight[:,1226]` (BF16, shape `[1024,3584]`).
- Prior same-layer matched-coordinate design remains `matched_control_not_found`; its thresholds will not be relaxed.
- This intervention asks whether attenuating the output contribution of the observed L12 input coordinate changes repeated-check behavior more than equal-component-magnitude signed-direction controls at that same coordinate. It does not test neuron specificity against other coordinates.

## Hypothesis, intervention, and controls

The two failure traces show an increased L12 coordinate-1226 activation on the same repeated phrase token IDs. Since this coordinate selects a column of the L12 MLP down-projection, a 10% attenuation of that column is predicted to reduce the repeated off-task arithmetic-check continuation on both fixed failure prefixes, while preserving correct answer completion on confirmation items. This is an observationally motivated causal hypothesis, not an established mechanism. Plausible harm includes disrupting useful arithmetic checks or answer generation.

Four arms use the same base checkpoint and differ only in that one BF16 column:

- `B`: unchanged base.
- `T`: target attenuation, `w+d`.
- `C+`: signed-direction control, `w+v`.
- `C-`: signed-direction control, `w-v`.

The controls preserve each target component's absolute BF16 change magnitude and therefore the exact Frobenius norm of the column change, but rotate its sign pattern. They are not orthogonal-neuron controls, and their resulting column norms need not match the target. No claim of full causal specificity follows from this design.

### Exact BF16 target construction

Set alpha to exactly `0.10`. For each finite BF16 component `w_k`, treat the BF16 value as an exact binary rational. Preserve signed zeros. Let `a=abs(w_k)`. Round exact `(9/10)*a` to BF16 using round-to-nearest, ties-to-even, yielding `b0`. Enumerate positive finite BF16 bit patterns from `b0` toward `a`, inclusive, in ascending magnitude. Select the first `b` for which `q=2*a-b` is exactly representable as finite BF16. Apply the original sign to form `t_k` from `b` and `m_k` from `q`; set `d_k=t_k-w_k`. Any NaN/Inf or failure to find a value is a hard stop.

Required gates, computed on the full column in float64 from exact stored BF16 values:

`0.09 <= ||d||2 / ||w||2 <= 0.11` and `cos(d,-w) >= 0.99`.

If either gate fails, stop. Do not adjust alpha or alter the search rule.

### Deterministic signed-direction control

Use control seed string `human-astra-manual-20261105-l12c1226-direction-v1|signed-control-v1`. For each trial `r=0..1023` and component `k=0..1023`, compute SHA-256 of UTF-8 string `{seed}|trial={r}|coordinate={k}`. Set `s_k=+1` if the low bit of byte 0 is zero, otherwise `-1`; set `v_k=s_k*d_k`. Choose the first trial satisfying both `abs(cos(v,w)) <= 0.1` and `abs(cos(v,d)) <= 0.1`. Stop if none qualifies. This procedure uses weights only and must be frozen before any intervention output is generated.

Because each target/control component is either `t_k` or `m_k`, all three modified columns are exactly BF16-representable. Re-read saved variants and verify using exact rational arithmetic that per-component absolute changes and total Frobenius change norm match exactly. Any extra tensor difference or round-trip mismatch is a hard stop.

## Fixed evaluation plan

### Focused TRAIN mechanism screen

Use only the three already observed discovery prefixes: `gsm8k_train_6445`, `math_train_algebra_542`, and `gsm8k_train_3185`. Continue each saved exact token prefix with all four arms, two paired seeds per prefix, and a 4,096 generated-token cap: 24 continuations. Seed rule: SHA-256 of `{intervention_id}|focused|{id}|sample={sample}`, first four bytes as unsigned big-endian integer, samples 0 and 1. The frozen seeds are 6445: 1084327040, 3346301434; algebra_542: 1677174321, 3447544442; 3185: 37864219, 2080160158. Require the pinned `math-verify` scorer before starting.

For **each** sample independently, advance only if all of these hold: (1) T gives a formally correct answer, think-close, and EOS on all three prefixes; (2) B is formally incorrect on both failure prefixes and formally correct with think-close/EOS on 3185; (3) each of C+ and C− is formally correct with think-close/EOS on 3185 and is formally incorrect on at least one of the two failure prefixes. An unfinished control with a formally correct answer does not count as an accuracy win. Both samples must pass; do not pool across seeds or use proxy-only results. Any missing scorer result, incomplete pair, or failed criterion stops this candidate without confirmation. Complete all 24 planned generations even if an outcome gate fails; stop only for infrastructure/scorer failure or time budget, preserving a partial run as incomplete. Each condition must load an independent checkpoint and reprocess the full prompt and saved prefix; no KV cache or edited state carries between arms. Fixed per-row execution order is: sample0/6445 B,T,C+,C−; sample0/algebra T,C+,C−,B; sample0/3185 C+,C−,B,T; sample1/6445 C−,B,T,C+; sample1/algebra B,T,C+,C−; sample1/3185 T,C+,C−,B. The order rotates to reduce systematic order effects and is frozen before intervention outputs. This TRAIN-only gate cannot establish confirmation accuracy. Do not select dose, direction, seed, or arm based on observed outputs.

### Confirmation

Select two IDs from each source's existing confirmation partition using ascending SHA-256 of `{intervention_id}|confirmation-selection-v1|{source_filename}|{id}`, breaking hash ties by ID lexicographic order. Selection was computed using IDs only; the selected IDs are:

| Source | ID | sample 0 seed | sample 1 seed |
|---|---|---:|---:|
| GSM8K | `gsm8k_train_5341` | 2517555310 | 2890724130 |
| GSM8K | `gsm8k_train_1178` | 8746557 | 3600332144 |
| MATH | `math_train_precalculus_207` | 864367089 | 2034957992 |
| MATH | `math_train_intermediate_algebra_231` | 4152760572 | 3381638749 |

Each item/sample is paired across B/T/C+/C-. Use generation cap 32,768 and samples 0 and 1, yielding 32 generations. All four item bodies remain unopened until the intervention, control construction, gates, scorer, prompts, decoding, and seeds are frozen and the TRAIN mechanism screen is complete. IDs, seed rules, and derived seeds may be public; do not publish item bodies or answers.

The confirmation arm order is balanced over pair positions and frozen as follows:

| ID | sample 0 order | sample 1 order |
|---|---|---|
| `gsm8k_train_5341` | B, T, C+, C− | T, C+, C−, B |
| `gsm8k_train_1178` | C+, C−, B, T | C−, B, T, C+ |
| `math_train_precalculus_207` | B, T, C+, C− | T, C+, C−, B |
| `math_train_intermediate_algebra_231` | C+, C−, B, T | C−, B, T, C+ |

Each of the four conditions occupies each run position twice. Every condition loads a fresh base checkpoint, applies only its registered column, and rebuilds generation state from that task's prompt; no model state or cache carries across arms.

The proposed accuracy gate is deliberately paired and strict: over the 8 item-seed pairs, T must exceed B, C+, and C- by at least two correct pairs against each arm; there must be at least one item for which both seeds change from B-wrong to T-correct; B-correct to T-wrong must be zero; B-correct completed answer to T-not-completed must be zero; every T-correct outcome must include a correct boxed answer, think-close, and EOS; and there may be no unscorable or incomplete pair. Report every paired transition, EOS/cap, answer accuracy, repeated-8-gram/exact-loop behavior, and token count separately. A proxy improvement cannot pass the accuracy gate.

If confirmation passes, a final holdout may be opened once only after a fresh freeze record. If confirmation fails or is unscorable, stop this candidate and do not open final holdout. No general capability, statistical superiority, or non-math-transfer claim is supported by this small set.

## Runtime and gates not yet cleared

Seven prior full-cap generations took 1,394.951–2,683.031 seconds (median 1,415.611; mean 1,749.175). For 32 confirmation generations, the corresponding all-cap range is 12.40–23.85 hours (median-rate estimate 12.58 hours; mean-rate estimate 15.55 hours). Linear scaling gives 1.16–2.24 hours for the 24 focused 4,096-token continuations (median 1.18; mean 1.46), before checkpoint reload/setup overhead. Combined, this projects to 13.56–26.08 hours (median 13.76; mean 17.01). These are empirical-rate projections, not guarantees; short-cap timing has not been measured. Early EOS can shorten runtime; a time limit does not authorize changing the frozen design.

The prior shared MPS environment still has no `math-verify`; it was left unchanged. For this new study, `math-verify==0.9.0` and its exact dependency versions are isolated under the ignored run-local `raw/scorer_vendor/math_verify_0_9_0/`. The upstream author artifacts specify `math-verify` but do not pin a package version, so 0.9.0 is this study's explicit scorer pin, not a claim about the author's exact environment. The reproducible validation script passed known-correct, known-incorrect, and missing-box cases; module hashes and results are recorded in `results/human_astra_manual_20261105/scorer_validation_summary.json` and the ignored raw validation record. Official package source: [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify). No task data or weights were used in scorer validation. The scorer gate is now cleared. No weight intervention has occurred. The column preparation and paired-generation tools have passed static review; the human must create the deterministic column artifacts and Luna must register their hashes before generation can start.

## Human operation sequence

The column-preparation command is the first human action. It reads the pinned base safetensors and writes only three 1,024-element BF16 column files plus a hash manifest under the ignored experiment raw directory; it does not change the base or write a full model. Run from the repository root:

```sh
REPRO_PYTHON=/absolute/path/to/the/existing/python-3.11-MPS-environment/bin/python
"$REPRO_PYTHON" scripts/prepare_human_astra_l12_directional_columns.py
```

Stop if it reports any failed numeric gate, non-finite value, missing mirror value, missing sign-control trial, or pre-existing output directory. Return the generated `manifest.json` hash and the printed norm/cosine values to Luna for an independent review and a pre-generation registration update. The evaluator will refuse to run until that exact artifact hash is entered into the parent manifest.

Only after Luna verifies and registers the column artifacts, the human may run the focused TRAIN screen:

```sh
"$REPRO_PYTHON" scripts/run_human_astra_directional_eval.py --stage focused --attempt 1
```

The runner loads a fresh base model for each scheduled arm, applies one column in memory, verifies the registered parameter hash and zero unrelated tensor changes, then generates and stores raw local evidence. It preserves the saved prefix in the presence-penalty history. After all 24 outputs, run the frozen gate scorer; only if both samples pass may Luna register the focused gate hash and the human start confirmation. The confirmation runner opens only the four registered task bodies after this pass. Do not run confirmation early or alter the script/IDs/seeds/gates after seeing focused outputs.

## Decision record

Astra recommended this separate direction-comparison design only after the prospectively frozen matched-coordinate selector found no eligible coordinate. Luna retains the original failure unchanged, because replacing or loosening its rule after observing the null selection would bias that design. This new design answers a narrower same-column directional question and cannot establish coordinate specificity. The revision to two paired seeds follows the user's explicit multi-seed requirement; the larger evaluation size is retained despite its substantial measured runtime rather than being reduced after observing model results.
