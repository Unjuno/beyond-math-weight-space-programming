# External follow-up on weight-space programming

## Source and scope

Junru Hu and colleagues submitted *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations* to arXiv on 2026-11-06. The study reports 16 editing trajectories across seven benchmark families and Qwen3.5-0.8B / Qwen3.5-9B. It uses GPT-6 Astra, with up to 50 successful forward requests and 300 Python analyses per search. Development data are separated from held-out evaluation data in the paper's protocol.

Primary sources: [arXiv abstract](https://arxiv.org/abs/2611.09382), [full paper](https://arxiv.org/html/2611.09382v1), [project page](https://wsp-scope.github.io/), [companion repository](https://github.com/WSP-Scope/WSP-Scope-Project).

## What the paper reports

For Qwen3.5-0.8B MATH, the primary layer-23 `mlp.down_proj` column-507 edit changes 1,023 stored BF16 values and reports MATH-500 accuracy of 8.35%→37.80% (+29.45 points). Two additional MATH search trajectories report 8.35%→13.35% (+5.00) and 8.35%→35.65% (+27.30). The study also reports full held-out gains on some non-math tasks and says six of 16 trajectories retain the original checkpoint when a candidate is not validated well enough.

The paper separates candidate discovery, development scores, held-out scores, and final checkpoint selection. It also reports a counterexample where a large development improvement for a Qwen3.5-9B math candidate did not carry over to held-out evaluation. Those distinctions make it useful for assessing both feasibility and search reliability.

## How it relates to our result

This is a separate research group and separate editing trajectory, but the outcomes are reported by the paper's authors. It supports the broader proposition that an agent can sometimes make a localized weight edit with held-out benefit. It does not reproduce Takehi et al.'s p50c edit: the reported primary target is layer 23 column 507, while p50c changes repetition-direction components in layers 15 and 19. It also does not verify our local scores or finish our paused search.

The paper is an arXiv preprint, not a result we have independently re-executed. Its methods and aggregate trajectory table are detailed, but the companion GitHub repository currently exposes a PDF and not the raw per-generation logs or complete executable inference stack. We therefore treat its numerical results as externally reported evidence, not as independently audited raw evidence.

## Calibrated conclusion

Combined with our exact reconstruction of p50c and small local paired evaluation (0/16→11/16 correct, 16/16→3/16 cap hits), the new study makes **“weight-space programming can sometimes work”** a reasonable evidence-based claim. The limits remain substantial: our local p50c evaluation is only four selected items, the newer paper's runs are not independently audited here, and neither result establishes that a fresh run will reliably discover an effective edit or that the gains generalize to arbitrary tasks and models.
