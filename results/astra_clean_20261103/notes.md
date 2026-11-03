# Astra clean Experiment 8 method reproduction

This is a new, approximate method reproduction, not a re-scoring of a published edit. The editor has not read the released edit, p50c outcomes/trajectory, local prior probes, or held-out benchmark content. The sole p50c files read are the original task prompt and wrapper instructions, with explicit permission.

## Protocol fidelity

- Base: local unedited Qwen3.5-0.8B, SHA-256 in manifest.json. Original tensors preserved.
- Data: author-decontaminated GSM8K TRAIN and MATH TRAIN JSONL only. No AIME, test-set prompts/answers, or non-math benchmark data are used for editing.
- Thinking enabled, paper user suffix, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5. Pass 1 requests the original 32,768-token maximum.
- Only direct, non-gradient, partial-tensor weight edits are allowed. Final norm, embedding, output layer, and whole-tensor rescaling are rejected by the CPU edit kernel.
- Separate persistent processes: CPU edit kernel and MPS forward worker. Neither performs training. The editor does not simulate model computation.
- At most 50 successful forward-tool requests. Failed attempts are separately preserved and disclosed. No free smoke generations.
- Stable per-item/sample seeds, paired original/edited generation on identical items, and item-level paired standard errors.
- Training-only confirmation partitions are fixed before diagnosis; their prompts/answers are withheld from editor inspection until a candidate is frozen.

## Approximation and missing components

The released harness/sandbox.py belongs to a separate Go experiment, and the exact Experiment 8 persistent sandbox/forward service is not available here. The local CPU kernel uses software restrictions rather than the author's Linux container isolation. MPS BF16 Transformers inference replaces the original backend; original RNG algorithm/backend numerical parity is unavailable. Local SHA256-derived seeds reproduce pairing within this run, not the author's exact trajectories. The author's math-verify 0.9.0 grading dependencies were provisioned separately under /tmp/exp8-math-verify without changing the model venv. MLP generation windows are implementation intermediates; editor decisions will use only the original-tool-equivalent prompt activation and first-8,192-token mean until a separately counted teacher-forced pass authorizes token-level analysis.

The initial 32+32 diagnostic pools and fixed confirmation pools are local reproducibility choices, not claims about the author's exact selection. Batches/caps after pass 1 will be chosen from measured wall time and disclosed; reaching a pass-count checkpoint alone is not replication of the paper's statistical power or full benchmark evaluation.

## Decisions and execution

1. Fresh manifest and disjoint training ID partitions created before diagnostic outputs.
2. Pass 1: two GSM8K TRAIN and two MATH TRAIN IDs chosen by a fixed hash ordering, with all-layer MLP capture, original weights only, full 32,768-token cap.
3. The first attempt was aborted after 220.682 s because HF inherited model-config EOS 248044 rather than chat tokenizer EOS 248046. No completed generation, grade, or gain is claimed. Evidence retained in failed_attempt_001 and worker_attempt_001.log; failed-attempt ledger entry retained.
4. All four actual prompt token arrays match direct native chat-template tokenization. The first audit incorrectly compared a list against a BatchEncoding object; the corrected comparison and both token arrays are preserved.
5. The corrected attempt explicitly uses EOS 248046, PAD 248044, native direct-template tokenization, and token-progress telemetry. First measured 256 tokens took 10.604 s (24.14 tokens/s); this is a partial-item estimate, not the completed-pass cost.

## Next acceptance gates

Read actual TRAIN failures, choose a testable mechanism, localize with separately counted teacher-forced evidence, perform direct weight edits, test several strengths causally, and confirm a frozen candidate on untouched TRAIN IDs. Keep an edit only if accuracy improves clearly beyond paired uncertainty. If the evidence fails, report that the tested edit did not establish improvement and freeze the base as the retained choice. An executed intervention and a validated beneficial repair are distinct outcomes.

6. Absolute four-hour deadline confirmed as 2026-11-03 06:59:15 UTC (15:59:15 JST). At 8,448 tokens the first item took 393.796 s; context slowdown makes an all-cap four-item pass approximately 154 min by an unvalidated quadratic extrapolation. Keep this request unchanged; prioritize actual causal editing, a frozen candidate, and independent paired TRAIN confirmation over pass-count milestones. Non-math transfer is not evaluated by this editor.

7. Independent base provenance confirmed: Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17, SHA-256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696. This is the paper experimental base (a post-trained checkpoint), not the separate Qwen3.5-0.8B-Base repository.

8. Corrected request 001 was explicitly interrupted by parent after 1/4 complete samples. It consumes one pass; 3/4 requested items remain interrupted and are not scored. The completed TRAIN item took 2599.168188 s, reached 32768 tokens, and was incorrect/no extracted answer. It contains no EOS 248046, no endoftext 248044, no thinking close, and a repeated self-check cycle (tail 8gram recurrence 0.966683). First-item diagnosis supports in-thinking recurrence rather than missed EOS handling. Any second-item prefix is telemetry only. Exact signal time was not supplied; durable interruption-recording time is stored separately.

9. Pass2 TF localization completed in 47.455 s on 519/469 tokens. Common prefix399 is activation-identical in all24 layers; compare only next70 tokens. Layer18 columns1261,1405,1824,1784 nominate a sparse intervention; three direct weight variants and fixed approximate control are separately hashed. Passes3–6 score these variants in ~2 s each. Alpha1 selectively lowers the repeated phrase locally (mean delta -0.148861 for phrase tokens, +0.007623 for arithmetic tokens), but this is not proof of recurrence repair. Whole loop/reference/prefix deltas are -0.044785/-0.007743/+0.000784. No mathematically feasible four-column nonselective activity match exists; control null effects cannot establish specificity. Six passes consumed including partial diagnostic; no generated edited output and no non-math evaluation yet.

10. Contamination incident: collaboration.list_agents unexpectedly returned a sibling completed final answer containing non-math task facts/scoring details. This editor is now exposed to those details before formal freeze, despite not opening the encrypted bundle or key. Parent was immediately notified. Existing transfer set cannot be called hidden from this editor; request replacement with new post-freeze tasks. Candidate alpha100, hashes, dose summary, and generation draft predate exposure; no edits/forward requests were changed after exposure. Incident details/timing are append-only in audit_events/rationale/luna_notes.

11. Authoritative parent-enqueued pass7 uses original+alpha100, same TRAIN ID/seed, full32768 cap, and layer18 activation capture. It is running; the editor cap8192 draft was not used. An idle-status report was corrected after inspecting the actual differently named queue file. No duplicate or mutation of the running request. Parent conservatively marks the existing transfer bundle not editor-blind and plans new post-freeze tasks.

12. Parent-authorized read-only public route review completed at commit58254e1; only runner/verifier/summarizer code read, no task/answer/output/weight-value payloads. Public runner is p50c-specific and defaults to old modes/caps; alpha100 requires manifest-driven patch loading, explicit new task source/scorer, new output paths, and paired invariant checks. Broader goal includes public p50c transfer and fresh alpha100 as separate claims. Authorized code review reveals released tensor identifiers/hashes; existing alpha100 choice/pass7 predate that metadata and must not be retuned from it. Details in public_route_review.json. No task content is included in this trace.

13. Pass7 completed. Original exactly replays all32768 pass1 tokens and is incorrect; alpha100 emits a correct boxed95 but still hits32768 with no EOS/endoftext/thinking close. Both fail strict completion. Edited global8gram recurrence is0.866762 versus base0.853912 and an alternate self-check motif repeats. This supports local lexical/format influence, not repaired recurrence; the selected TRAIN boxed-score win is not independent math improvement. Candidate is not validated/retained. Recommended next step is a frozen-candidate eight-pair balanced untouched TRAIN pilot, with official boxed accuracy and strict completion kept separate. No held-out task payload inspected.
