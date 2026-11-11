# Luna decision and outcome analysis — 2026-11-10

> **Method-fidelity correction (2026-11-11):** This report's focused result is not a reproduction of the authors' p50c method. It is the result of a distinct L12 coordinate-1226, 10%-attenuation continuation screen. See [`method_fidelity_audit_2026-11-11.md`](method_fidelity_audit_2026-11-11.md) for the exact differences. T=0/6 must not be described as a failure to reproduce the paper.

## What Luna saw and did

Luna inspected the registered experiment manifest, the pinned checkpoint identity, the frozen scripts and gates, prior local dialogue records, and the raw artifacts generated in this run. The new user instruction explicitly changed the operator from the human to Luna. Luna therefore ran the already registered column preparer, then independently checked each saved artifact's SHA-256, shape, BF16 dtype, and changed-element count. The full base checkpoint was not written or changed.

Luna ran all 24 predeclared focused generations using three saved TRAIN prefixes, two fixed seeds per prefix, and four arms. One initial CLI invocation used a string where the script required an integer attempt number; argument parsing rejected it before model loading or generation. Luna read the error and reran with the documented integer form. This was a preventable preflight omission and is included in the decision record.

After scoring all rows with the pinned scorer, Luna applied the frozen advancement gate without changing thresholds, seeds, examples, or scoring conditions. The gate failed, so Luna did not open confirmation bodies or run the 32-generation confirmation stage.

## Result

| Arm | Formally correct completed | EOS | At token cap | Mean repeat-8 fraction |
| --- | ---: | ---: | ---: | ---: |
| B | 1/6 | 1/6 | 5/6 | 0.7225 |
| T (target attenuation) | 0/6 | 0/6 | 6/6 | 0.7235 |
| C+ | 1/6 | 1/6 | 5/6 | 0.6417 |
| C− | 2/6 | 2/6 | 4/6 | 0.5486 |

T never produced a formally correct completed answer. It reached the 4096-token cap on all six rows. The predeclared focused gate therefore failed for both seeds. C− had two successes, C+ and B one each, but this is only a small exploratory signal from selected training-prefix continuations. It does not support a target-specific causal claim. The earlier matched-coordinate control search still found no eligible coordinate; this signed-direction comparison cannot establish coordinate specificity.

Item-seed completion pattern (a ✓ denotes formally correct completion):

| Prefix | Seed index | B | T | C+ | C− |
| --- | ---: |:---:|:---:|:---:|:---:|
| 6445 failure | 0 | — | — | — | — |
| Algebra 542 failure | 0 | — | — | — | — |
| 3185 comparison | 0 | — | — | — | ✓ |
| 6445 failure | 1 | — | — | — | — |
| Algebra 542 failure | 1 | — | — | — | ✓ |
| 3185 comparison | 1 | ✓ | — | ✓ | — |

The highest repeat-8 fractions occurred in the two preselected failure prefixes; all arms generally stayed in repetitive continuations there. The frozen exact-final-32-token-loop indicator was false on every row. The repeat-8 metric was descriptive and did not affect the gate. All four formally correct rows also had EOS and a generated think close; no formally correct-but-incomplete result or correctness-format exploit was detected. There is no direct evidence of reward hacking in this run.

## Luna's observable decision pattern

1. **Protocol following:** the explicit user instruction changed who operates the registered procedure. Luna changed that operational choice while keeping the preregistered treatment, comparator, scorer, and gates fixed.
2. **Evidence-first staging:** Luna checked artifact identity and file integrity before inference, then waited for all scheduled outputs before deciding.
3. **Fail-closed stopping:** when the predeclared gate failed, Luna stopped before confirmation-data exposure instead of searching for a friendlier sample or redefining success.
4. **Correction after error:** the malformed attempt argument was corrected from the tool error; the failed invocation produced no model output.
5. **Calibrated reporting:** a failed small continuation screen is reported as “not reproduced under this design,” not as proof that the intervention never works.

These are traits visible in recorded actions and explicit explanations. They do not reveal the model's complete latent reasoning. A weakness observed here is incomplete CLI-schema preflight, which caused one avoidable rejected command. Another practical risk is that Luna can only preserve the rationale it states and records; unrecorded internal processes cannot be reconstructed from this ledger.

## Reproducibility boundary

The sanitized numeric summary and code are versioned. All raw generations and the Luna–Astra decision ledger remain local-only. The full attempt took 4451.5 seconds (about 74 minutes). Confirmation and final holdout were not run. The report therefore closes the focused screen, not the broader original non-math transfer question.
