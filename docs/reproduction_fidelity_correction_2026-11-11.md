# Reproduction fidelity correction — 2026-11-11

This addendum corrects the status of the November 4 Astra-led search and records adapter fixes needed before another search. It does not replace or rewrite the original logs.

## The 18-pass search did not follow the full instruction

The frozen editor prompt says that the session continues until all 50 forward passes are used. The local run's final state instead records 18 successful calls, 32 unused calls, no retained edit, and a futility stop. The forward ledger contains 19 attempts: 18 successes and one failed request. The separate `manifest.json` still says only pass 1 completed, so it is stale and must not be used as the final pass count. The final state and append-only forward ledger are authoritative.

The run remains useful evidence: under its reconstructed conditions, three candidate families failed their local gates and no edit was retained. It is a **negative 18-pass approximation**, not a full reproduction of the instructed 50-pass Astra search. The previous phrase “early stopping ... as permitted” was not supported by the initial prompt and is withdrawn. The editor-confirmation and external-held-out sets remained unopened.

## Paired standard-error correction

The forward adapter previously calculated the paired standard error over every item-by-sample observation. With multiple samples per item, that treats repeated samples from one problem as independent clusters and can understate uncertainty. The adapter now pairs records by `(problem_id, sample)`, verifies matching seeds, averages correctness differences within each problem, and calculates the standard error across problem-level means.

This bug did **not** affect the old six-item paired accuracy decision: that request had one generation per item. From the saved transition counts (one wrong-to-right, one right-to-wrong, four unchanged), the corrected item-level standard error is `0.2581988897`, matching the recorded value. The old rejection remains unchanged.

## Adapter corrections before a new search

The local forward adapter now:

- accepts a custom user prompt aligned one-to-one with training problem IDs;
- rejects duplicate IDs and verifies identical item/sample keys and seeds in paired arms;
- enforces the teacher-forced `topk` limit of 50 and estimates uncompressed result size against the author's 6 GiB per-pass limit before inference;
- accepts run, training-data, and base-model paths through explicit environment variables so conformance work and a fresh editor run can use isolated directories.

The file compiles. A deterministic synthetic calculation verified that two problems with four perfectly opposite paired sample differences have item-level `delta=0`, `SE=1`, and eight paired generations; a six-item, one-sample calculation returns `SE=0.2581988897`. This is a local calculation check, not evidence of model or backend parity.

## Adapter conformance result

A separately preregistered local check used two previously unused TRAIN IDs, no edit, and a 256-token cap. Two byte-identical `model="both"` requests each completed eight matched base/edited generations. All prompt token IDs, seeds, and generated token IDs matched across both arms and both requests. The 32 activation artifacts each had 60 requested arrays; all 1,440 array-to-array comparisons were bit-exact, with hidden vectors of width 1,024 and MLP vectors of width 3,584.

Two additional invalid teacher-forced requests were rejected before any model forward: `topk=51` and a request whose estimated uncompressed outputs were 7,516,590,080 bytes (>6 GiB). These are expected validation failures, not failed model runs. All raw requests, IDs, token outputs, and activation arrays remain local/ignored. The sanitized summary is in the local ignored path `results/raw/astra_adapter_conformance_20261111/conformance_summary.json`.

This establishes repeatability and request/output-shape behavior for this sequential MPS adapter on the tested inputs. The 256-token outputs all reached the cap and provide no correctness evidence. It does not establish parity with the author's vLLM service, which warns that batching can change sampling streams.

## Remaining reproduction work

The next substantive step is a **new Astra-guided editor run** from the original task, with a full 50-successful-forward-pass budget and access only to a frozen TRAIN partition. Keep MATH-500 test problem bodies and scores out of the editor context. Preserve each Luna action, Astra's explicit rationale/request, exact code/request hashes, response metrics, and stopping decision. Keep the full search distinct from both the old negative 18-pass run and the released-checkpoint MATH-500 pilot.

The exact author vLLM environment and orchestration are not released; this machine has only the M1 Max MPS/Transformers service and no hosted GPU credentials configured. The benchmark pilot's measured local throughput implies that 2,000 base generations alone may take roughly 556 hours if all hit the 32,768-token cap; this is a rough extrapolation from two problems, not a measured full-benchmark runtime. The full MATH-500 × four-sample, two-checkpoint result therefore remains unrun and cannot be inferred from the pilot.

Primary evidence: [`initial_prompt.md`](../results/astra_reproduction_20261104/initial_prompt.md), the local [`pass_ledger.jsonl`](../results/astra_reproduction_20261104/raw/editor_work/forward/pass_ledger.jsonl), [`final_search_state.json`](../results/astra_reproduction_20261104/raw/editor_work/final_search_state.json), and [`l7h5_final_rejection_audit.json`](../results/astra_reproduction_20261104/raw/editor_work/l7h5_final_rejection_audit.json). Raw logs remain local and ignored.
