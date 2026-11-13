# Beyond Math: an evidence audit of weight-space editing

This repository separates four things that were easy to conflate: published benchmark records, reconstruction of the released p50c edit, our local evaluation of that fixed edit, and our own attempts to discover an edit. **The exact released edit is reconstructable and its tensor hashes match; small local runs show a promising completion/correctness signal. The full 500-item benchmark and the authors' 50-pass Astra search are not reproduced.**

## Current result at a glance

| Evidence | Result | What it supports |
| --- | --- | --- |
| Public p50c edit reconstruction | Rebuilt L15/L19 tensors match the released hashes; only those two tensors differ from base | The released edit artifact can be reconstructed exactly. This does not reproduce how Astra found it. |
| Local paired MATH-500 checkpoint evaluation | Across four selected items and 16 generations per arm: base 0/16 correct and 16/16 cap hits; p50c 11/16 correct and 3/16 cap hits | A promising descriptive signal on this small selected sample that the released edit can change termination and answers. It is not an estimate of the 500-item score. |
| Local non-math applied suite | On 24 hand-authored tasks, EOS-before-cap rose 11/24→24/24; correctness was mixed and grader-sensitive | Evidence for changed completion behavior; not a general non-math capability result. |
| Our fresh editing attempts | An earlier reconstructed search used 18 successful calls and retained no edit. A separate later run used 11/50 successful forward calls, then paused during pass 12 before a paired intervention result | We have not reproduced the authors' full editing trajectory or demonstrated a new causal edit. |
| New external WSP study | A separate-group arXiv preprint reports three Qwen3.5-0.8B MATH searches with held-out changes of +29.45, +5.00, and +27.30 points | Additional evidence that WSP-like editing can sometimes work. These are the preprint authors' results, not our independent rerun. |

The 16-generation MATH result pools two separately run tranches. The second was registered before its own inference but followed the first tranche, so the combined result is exploratory and descriptive. Both used local Transformers/PyTorch on Apple MPS; the paper reports vLLM. Separate local seed initialization also does not recreate the authors' RNG stream. See the [current evidence and reproduction status](docs/reproduction_status_2026-11-13.md).

## What was and was not reproduced

- **Reproduced:** arithmetic on the released author per-item benchmark records; reconstruction and hash check of the released p50c tensors.
- **Independently evaluated locally:** the released p50c checkpoint on four selected MATH-500 items, plus custom non-math probes. These are new local generations, not the full benchmark protocols.
- **Not reproduced:** the reported 48.1% MATH-500 result, fresh IFEval/GPQA/MMLU-Pro generations, or the complete original 50-pass Astra edit search under its original runtime.
- **Not established:** broad capability gain, a general loop-repair mechanism, or that shorter output alone means a task was solved.

The L12 coordinate-1226 screen is a separate local variant. Its 0/6 completed-correct result does not test or refute the published p50c edit. The older 18-call reconstructed search and the later 11/50-call run are separate experiments and are not pooled.

## Reports

- [Current evidence, limits, and pause point](docs/reproduction_status_2026-11-13.md)
- [Evidence ledger](docs/evidence_ledger.md)
- [Published p50c method-fidelity audit and local MATH runs](docs/method_fidelity_audit_2026-11-11.md)
- [Second preregistered MATH tranche](docs/p50c_math500_followup_2026-11-12.md)
- [Non-math author-record re-score](docs/author_results_nonmath.md)
- [24-task full-cap applied suite](docs/heldout24_p50c_results.md)
- [Publication claim boundary](docs/publication_positioning.md)
- [Earlier reconstructed edit-search result](docs/fresh_edit_search_results.md)
- [Earlier 18-call fidelity correction](docs/reproduction_fidelity_correction_2026-11-11.md)
- [New external WSP preprint assessment](docs/external_wsp_followup_2026-11.md)

## Data and reproduction

Full model weights, token-level generations, activation arrays, the full Luna–Astra conversation record, and the local `LUNA_MEMORY.md` stay outside Git. Selected sanitized event/decision records, aggregate results, method descriptions, hashes, and scripts are tracked for audit. No CI workflow is used. Setup and summary reproduction instructions are in [local setup](docs/local_setup.md); the tracked [experiment manifest](results/experiment_manifest.json) records pinned inputs.

## Sources

- Takehi et al., [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and [pinned Experiment 8 artifacts](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math).
- Hu et al., [Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations](https://arxiv.org/abs/2611.09382), arXiv preprint submitted 2026-11-06.
