# Pass 9 preregistration: token-matched thinking contexts

## Question

Do any residual-stream or MLP output activation differences align across exact repeated thinking phrases when comparing completed answers with cap-hit continuations, after restricting candidate selection to tokens and generation positions?

This is exploratory localization on already observed TRAIN outputs. It is not an independent confirmation and cannot establish that an observed direction causes repetition or task failure.

## Frozen source streams

- Completed group: all five correct, EOS-terminated outputs from attempt 0001 (`gsm8k_train_4288`, `gsm8k_train_897`, `math_train_prealgebra_1180`, `math_train_algebra_1314`, `math_train_geometry_50`).
- Cap-hit group: all 12 outputs from the six preregistered items in pass 0008. All 12 are incorrect cap hits.
- No item, seed, or phrase is selected or removed after inspecting activation values.

## Frozen token-only selection rule

1. Restrict each generated token sequence to the first 8,192 generated positions and the span after `<think>` through (but not including) `</think>`. For an unfinished output, the thinking span ends at the available sequence boundary.
2. Enumerate exact token-ID 8-grams whose decoded text contains at least two ASCII alphanumeric word groups.
3. Cross-group pairs must have the identical 8-gram and start positions at most 512 generated tokens apart. For each success/failure stream pair, rank candidate tuples by SHA-256 of the canonical stream keys, token IDs, and positions; retain at most four pairs.
4. Within each failure stream, compare the first and next occurrence of an identical 8-gram when their starts are at most 512 tokens apart; rank by SHA-256 and retain at most two pairs.
5. For each selected cross-group match, include a background window at a fixed +256-token offset when it remains inside that stream's thinking span. Do not substitute another offset when unavailable.
6. Cap the union of selected token positions at 256 per stream. Sort and merge adjacent selected positions into nonoverlapping replay ranges.
7. Capture residual stream layers 0–24 and MLP module outputs 0–23 by forcing the saved token IDs through the original model. Verify exact token replay. No layer, phrase, or position is selected from activation magnitude.

## Interpretation rules

- Report phrase-matched success/failure comparisons, within-failure first/repeated occurrences, and background windows separately.
- Aggregate at the problem level; token positions and repeated samples are not independent problems.
- If no token-matched pair exists, preserve that result and do not relax the rule.
- A direction is only an observational candidate. No edit is justified without a later causal intervention, dose response, and control direction. The reserved confirmation items stay untouched.

## Astra advice and Luna decision

Astra advised token-only exact-phrase matching at nearby positions, deterministic hash-based selection before activation inspection, a separate within-generation repeat comparison, and a background position control. It also advised comparing all layers rather than preselecting L15/L19, and keeping observational localization distinct from causality.

Luna chose the complete five-success group and all 12 pass-8 failures to avoid hand-picking individual examples; limited phrase search to a fixed 8,192-token thinking span; froze hash selection and per-stream capture budgets; and replayed only selected positions to reduce memory. This selection is outcome-conditioned and exploratory because the success group comes from an earlier observed tranche.
