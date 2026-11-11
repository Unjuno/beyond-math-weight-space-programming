# Fresh Astra-led edit search: final result

## Outcome

The 2026-11-04 run executed a fresh, TRAIN-only edit search using the reconstructed forward-pass/edit/evaluation loop. It did **not** reproduce a retained edit or a positive accuracy result. The search was frozen negatively after 18 successful forward passes.

No candidate was eligible for held-out evaluation: every tested edit failed its predeclared causal or paired-accuracy gate. The editor confirmation split and external held-out tasks therefore remained unopened. Evaluating a rejected edit would have changed the experiment's acceptance rule and would be reward-hacking the search outcome.

## What failed, and what we learned

This was not a null observation. The reconstructed search failed for identifiable reasons:

1. The MLP candidate showed local TRAIN selectivity, but did not survive the matched-control causal screen. Local activation contrast was therefore not sufficient evidence for a useful edit.
2. The attention candidate produced one apparent wrong-to-right item, but also changed a previously correct item to wrong. The apparent improvement was accompanied by cap termination and worse repetition, so it was not a reliable task or stopping improvement.
3. The recurrent candidate was chosen from a descriptive failure-vs-success residual-drift difference, but changing its decay did not reduce that drift consistently in the two failure contexts. The observed correlation did not identify a causal editing target.
4. The released edited checkpoint separately showed a large completion/EOS shift without an increase in strict task success. This makes stopping and capability easy to conflate and is the main diagnostic result of the broader repository.

The remaining uncertainty is substantial: the exact author orchestrator, persistent editor kernel, forward backend, full 50-pass budget, and original benchmark generation protocol were unavailable. Therefore the negative result is evidence about this reconstructed 18-pass run, not proof that no successful edit exists.

## Search and decisions

| Candidate | Evidence | Decision |
|---|---|---|
| Layer 23 MLP column/neuron 46 | TRAIN teacher-forced selectivity and matched-control causal screen | Rejected |
| Layer 7 attention head 5 output attenuation | Six-item paired TRAIN screen: 1 wrong→right, 1 right→wrong, net 0; cutoffs 4→5; mean tokens 24,017.5→27,815.2 | Rejected |
| Layer 10 recurrent head 3 decay, multiplier 0.5 | Neither failure context met the required 5% residual-drift reduction | Rejected |
| Layer 10 recurrent head 3 decay, multiplier 0.25 | Neither failure context met the required 5% residual-drift reduction | Rejected |

The attention wrong→right item was graded correct because an early boxed answer was present, but the generation continued to the 32,768-token cap, lacked a thinking close, and had worse repetition. It is not a stopping rescue. The paired accuracy gate correctly rejected it because a previously correct algebra item became a cap-bound failure.

## Reconstructed-protocol deviations

- The authors' persistent editor kernel and original forward server were unavailable; this run used the local CPU editor kernel and MPS/BF16 Transformers forward adapter documented in the manifest.
- The author 50-pass budget was not exhausted; the search stopped after 18 successful passes and left 32 unused.
- The exact author orchestration and backend are therefore not reproduced. The run is the closest executable Astra-led procedure available from the released artifacts, with deviations and raw hashes retained locally.

**Correction (2026-11-11):** the initial task prompt explicitly says the editor session continues until all 50 forward passes are used. The 18-pass futility stop was therefore a deviation from that prompt, despite the local run's stopping rule. This file's earlier wording that early stopping was "permitted" was inaccurate. The search is a completed negative 18-pass approximation, not a completed execution of the specified 50-pass procedure. See [`reproduction_fidelity_correction_2026-11-11.md`](reproduction_fidelity_correction_2026-11-11.md).

## Boundary of the claim

This result establishes a negative fresh-search outcome under the reconstructed conditions. It does not establish that no edit exists, nor does it estimate the authors' reported non-math benchmark gains. It also does not provide a held-out candidate result, because no candidate passed the frozen acceptance gates.

Author-record rescoring and released-checkpoint evaluation remain separate evidence types and are not counted here as reproduction.
