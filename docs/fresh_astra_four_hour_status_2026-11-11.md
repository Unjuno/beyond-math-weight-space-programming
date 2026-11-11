# Fresh Astra search: four-hour status

This is an interim status of the new, TRAIN-only Astra-led editing run. It is separate from the released p50c checkpoint evaluation and from the earlier L12 screen.

## Protocol in progress

- Base: Qwen3.5-0.8B at the author-logged base revision `2fc06364715b967f1860aea9cf38778875588b17`; the local safetensors file hash was frozen in the local preregistration.
- Editor-visible data: a fixed, disjoint TRAIN partition only. MATH-500 test problems and scores were not mounted into the editor container.
- One original-weight generation on each of 20 preselected TRAIN IDs, `thinking` enabled, temperature 0.6, `top_p` 0.95, `top_k` 20, presence penalty 1.5, and a 32,768-token cap. Requested hidden-state and MLP activation summaries cover all model layers.
- Local inference uses Transformers/PyTorch on Apple MPS. The authors used vLLM on a GPU; exact backend parity is not established.
- Astra proposed the request; Luna reviewed and manually queued it. The request is still running, so no successful forward pass has yet been finalized in the run ledger and no edit has been made.

## Results recorded at four hours

Eleven of the 20 original-weight generations had been fully written when this status was frozen:

| TRAIN source | Completed | Correct finished answers | Incorrect cap hits |
| --- | ---: | ---: | ---: |
| GSM8K | 4 | 2 | 2 |
| MATH | 7 | 3 | 4 |
| **Total** | **11** | **5** | **6** |

Each cap hit used all 32,768 new tokens and was scored incorrect by the local answer extractor and math-verify grader. The four-hour snapshot averages 21,214 generated tokens per recorded item. Full outputs, token IDs, individual scores, and activation arrays remain local and are not included here.

This result is only a mixed TRAIN diagnostic baseline. It does not identify repetition as the cause of the cap hits; that requires examining the saved continuations. It does not establish an edit effect, reproduce the p50c edit, or evaluate any held-out math benchmark.

## Incomplete work

At the four-hour point, the 12th generation was still running at about 19,000 of 32,768 tokens. Nine original-weight items remained after it. The full 20-item baseline, activation analysis, Astra's candidate selection and causal checks, use of all 50 successful forward calls, held-out confirmation, and the fresh paired MATH-500 evaluation had not started or completed. Therefore this is **not a completed reproduction**.

The public p50c patch remains a distinct artifact: its L15/L19 tensor hashes have been reverified locally against the author's published hashes, and a two-item paired MATH-500 pilot is documented in [`method_fidelity_audit_2026-11-11.md`](method_fidelity_audit_2026-11-11.md). Neither result substitutes for the new edit search or the full benchmark.
