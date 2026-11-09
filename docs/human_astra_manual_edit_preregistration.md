# Human-operated Astra-guided weight edit: preregistration draft

## Status

This document defines a new experiment boundary and decision rules. It is **not** evidence that an edit has been performed or evaluated. The human operator applies any exact Astra-proposed edit; Luna records the rationale and checks the resulting artifact. Astra must choose from evidence available in the designated discovery/training split. No model weights or held-out item bodies have been accessed for this experiment.

Experiment ID: `human-astra-manual-20261105-v1`

## Separation from prior work

This experiment starts from the verified Qwen3.5-0.8B base checkpoint and is separate from author-log rescoring, released-p50c checkpoint evaluation, the 2026-11-04 Astra-led negative search, and the earlier alpha100/L23 attempts. It must use a new output directory and new append-only records. Do not overwrite, reuse edited checkpoints, or count any prior outcomes as observations in this study.

The currently identified candidate source is the author TRAIN collection after the previous 512 pilot exclusions (11,385 IDs). The prior Astra-led editor reserved 352 further TRAIN IDs across diagnostic, tuning, and confirmation sets. Astra advised excluding all 352 conservatively, although this does not imply all 352 prompt bodies were actually read. Six IDs are explicitly present in prior successful forward-call requests/responses. The prior manifest's `practice_source_sha256` values describe the full upstream files, not these sliced TRAIN files; the new manifest therefore records hashes of the exact local TRAIN slices. A read-only audit must confirm both those slice hashes and the existing aggregate TRAIN-ID hash before any prompt body is used.

## Data boundary and task scope

Partition author TRAIN IDs independently within GSM8K and MATH using SHA-256 ordering salted by this experiment ID, after excluding the 352 reserved IDs. Intended allocation per source: discovery 70%, validation 10%, frozen confirmation 10%, final holdout 10%. The human-run dry run must emit counts and set hashes only. It must not print or store prompt/answer bodies.

- Discovery: candidate observation and selection only.
- Validation: limited candidate checks before candidate freeze; every look is logged.
- Confirmation: opened only after candidate and all acceptance rules are frozen; any tuning after opening invalidates it as confirmation.
- Final holdout: unopened until the complete edit, control, seeds, evaluator, and report rules are frozen; one evaluation only.

These are author mathematical TRAIN records, not guaranteed unseen during model pretraining. This source can support a mathematical candidate-discovery experiment. It cannot by itself support a claim that the edit improves non-mathematical tasks. A non-math transfer claim requires a separately sourced, prospectively frozen non-math task set with known exposure status; existing 24 applied probes are not eligible as a fresh unseen holdout because of prior exposure concerns documented in the repository.

## Candidate and control freeze

No edit target or direction is selected yet. Astra must first state a falsifiable, TRAIN-grounded causal hypothesis and specify, before any human edit:

1. exact tensor key and element/column/head coordinates;
2. base tensor hash and exact arithmetic operation, dose, and resulting tensor hash;
3. expected target effect and plausible collateral harm;
4. one matched control with its selection rule frozen independently of outcome;
5. a reverse-dose or sham condition where applicable;
6. a single candidate-selection rule using discovery data and a single validation rule;
7. exact confirmation/final item IDs (or deterministic selection hashes), seeds, prompt serialization, tokenizer, decoding, and scorer, frozen without viewing those item bodies.

The human performs the exact operation only after reading Astra's instruction back and confirming the tensor/key/dose against the registered request. Store before/after hashes and an elementwise diff summary; abort on any unregistered or collateral change. Preserve the base checkpoint read-only and write the edit to a unique experiment-specific path.

## Initial discovery baseline

Before proposing an edit, run the original checkpoint on four fixed discovery items, balanced two GSM8K and two MATH, once each. Astra selected their IDs by a deterministic second SHA-256 ranking inside the frozen discovery partitions; Luna independently verified the ranking, membership, and seeds using IDs only, without inspecting task text. The exact IDs, order, and seeds are frozen in the new run manifest.

Use the fixed task instruction and full-cap settings in the manifest. Only original weights are loaded. Save exact prompt/token IDs, generated token IDs/text, gold and parsed last-boxed answer, math-verify outcome, EOS/think-close/cap status, length, repeat8 diagnostics, seed, timings, and environment/hash metadata. Raw traces and prompts remain local and ignored by Git. Record separate progress every 256 new tokens; interrupted generations remain infrastructure failures and are not counted as model errors.

The existing Python 3.11/MPS environment has PyTorch and Transformers but lacks `math-verify`. A request to add that scorer to the existing environment is pending. Baseline traces may be captured while this is unresolved, but correctness and strict-composite labels stay pending; do not choose conditional backups or freeze an edit from unscored correctness.

One conditional backup stage of four more fixed discovery items (two per source, already ordered and seeded in the manifest) is allowed only if the first four do not yield both (a) the same concrete failure motif on two distinct items and (b) at least one correct comparison item without that motif. No replacement items are permitted. After eight total items, if that criterion is unmet, stop with no candidate and do not extend the search. This selection is for exploratory TRAIN discovery only and cannot estimate failure prevalence or general task performance.

## Outcomes and anti-proxy rules

Primary outcome: exact benchmark-defined mathematical answer accuracy on each paired item/seed, scored by the pinned `math-verify` procedure and a fixed last-boxed-answer parser. Report item-level paired transitions and uncertainty; do not use a completion threshold as a substitute for accuracy. Separately report a strict completed-task composite requiring a correct answer, requested format, and EOS before cap, so correctness and termination remain inspectable as distinct components.

Always report separately: valid answer accuracy; EOS/think-close and cap status; generated token count; repeated 8-gram fraction and any exact-loop diagnostic; malformed/missing final answers; control-arm outcomes; and the number of attempted, failed, interrupted, and completed generations. A shorter response, EOS increase, lower repetition proxy, or teacher-forced likelihood shift alone is not a success. Any false answer that terminates cleanly remains an error. Keep all failures and do not extend the search after the frozen futility rule.

Candidate discovery begins with a few prospectively selected discovery items under original weights. Astra inspects saved token IDs, scores, repetition locations, and only predeclared activations; it must identify a concrete failure pattern in two distinct examples before proposing one hypothesis. Previously rejected L23/N46, L7/H5, and L10/H3 targets are ineligible without new independent evidence. If no pattern localizes, stop without editing. Before intervention outcomes, preregister measurable predictions, tolerable harm to correct reasoning, and numeric pass/fail thresholds. For a loop-suppression hypothesis, test repeated continuation in two contexts as well as reference selectivity; shorter output alone fails the gate.

Candidate retention requires the frozen confirmation criterion to pass for task accuracy, the matched control not to show the same effect, and no predeclared material regression. Decide from paired item/seed transitions against both original and control, not an edit-only absolute threshold. With a small sample, report every transition and make no statistical-superiority or general-capability claim. If confirmation fails, stop the candidate. Open final holdout only for a candidate that passes confirmation; evaluate once and report all arms. If no candidate passes, record a negative experiment and do not open final holdout.

For transfer to non-mathematical tasks, math TRAIN is only the edit-discovery source; it cannot supply non-math task evidence. A separate evaluation custodian must author or select a new exposure-audited non-math set, freeze task/gold hashes, and keep item bodies from Astra and the editor until the candidate and evaluation settings are frozen. Do not reuse x04-x06 or paraphrase them as independent tasks. If this protected set is unavailable, report that non-math transfer was not tested.

## Human/Astra/Luna roles and recordkeeping

- Before each human operation, Astra states the evidence, prediction, exact operation, change scope, expected runtime, verification, and stop condition. Astra reads eligible evidence, states its rationale, proposes the candidate and control, and interprets returned evidence.
- The human physically executes only the exact approved operations, reads the result back, and returns command output or artifact hashes. The human does not improvise tensor edits or reveal unopened item bodies.
- Luna maintains the append-only Astra dialogue/action ledger, verifies hashes and split boundaries, preserves raw logs and weights locally, and reports deviations and uncertainty. Luna does not substitute its own edit for Astra's proposal.

Record each interaction with timestamp, experiment/turn ID, exact question, Astra's stated rationale, exact human instruction, human-reported action/output, code/request hash, input summary, returned evidence, Luna's acceptance/rejection and reason, uncertainty, and next step. These are observable decision rationales, not a claim to expose complete internal cognition.

## Current gate

The read-only split audit passed on 2026-11-05 after correcting a hash-scope mismatch: the prior manifest hashes the full upstream practice files, while this experiment reads their sliced editor-TRAIN subsets. The current manifest now pins both the exact local subset file hashes and the upstream hashes as separate provenance. The confirmed eligible counts are GSM8K 5,594 and MATH 5,439; discovery counts are 3,917 and 3,810; validation, confirmation, and final holdout counts are 559 and 543 per source. All split hashes are recorded in the run manifest. No discovery prompts, confirmation items, final-holdout items, or weights were accessed during the ID-only audit. The next step is to request Astra's discovery-only selection and instructions; no edit may occur until its candidate-specific causal plan, control, predictions, and stop conditions are frozen.

## Deviations and limitations

The prior local TRAIN source and model were involved in earlier math studies; reserving IDs reduces item overlap but does not remove model pretraining exposure or all conceptual contamination. Existing p50c and applied-task outcomes cannot be counted as evidence for this experiment. A completely fresh non-math editing study is not yet operationally specified because a clean non-math discovery/training source has not been identified.

## 2026-11-09 discovery and localization status

The split-audited run completed its eight fixed discovery generations (four initial plus the one permitted backup stage) and stopped item discovery. Astra and Luna identified an output-level repeated arithmetic-check motif on two initial examples, then ran a bounded full-prefix activation observation on those two and one successful comparison. L12 MLP `down_proj` input coordinate 1226 rose during the same exact repeated phrase tokens in both failure traces; this is post-hoc discovery evidence and does not show causality. The activation summary and raw hashes are in the run manifest.

Astra's pre-capture float64 matched-coordinate control rule was applied to seven frozen token groups across all 3,584 L12 input coordinates. No coordinate other than 1226 met the combined column norm, all-window activity RMS, and weight-column direction filters. Record the outcome as `matched_control_not_found`; do not relax the rule. No candidate dose, control, confirmation evaluation, weight edit, or held-out access is frozen or completed. Astra has been asked whether a separately labeled same-coordinate directional comparison can support a different, narrower causal question. Any such route must be preregistered as a new exploratory intervention design and retain this failed matched-coordinate result.

The captured GSM8K backup answer `770` matches its stored gold string and arithmetic, but the pinned `math-verify` package was unavailable. Its formal scorer field remains null. No claim that the fixed scorer ran is permitted.
