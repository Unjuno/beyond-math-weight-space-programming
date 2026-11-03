# Luna research log

Append-only working notes. Preserve prior notes and source logs. This is Luna's record of questions, choices and interpretation for this project; it complements, but does not replace, raw Astra traces.

## 2026-11-03 — Initial framing

- User requested an experiment about what weight-space programming can and cannot do, public repo, minimal CI, and four-hour completion.
- Scope clarification: primary interest is whether changes generalize to tasks beyond math, and the boundary of that generalization.
- Interpretability requirement: Luna asks Astra why it chose an edit; retain the question, explicit rationale, actions and evidence. Explicit rationale is observable, not a claim of full latent reasoning access.
- Prior local Qwen research: Qwen3.5-4B MPS/BF16 work established useful controls (fixed input token IDs, same runner/config, thinking and sampling recorded, separate activation intervention from checkpoint weight edit). It is procedural precedent only; it is not evidence about this Qwen3.5-0.8B edit.
- Upstream author log revision observed: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. The upstream p50c run is reported at 20.3 hours, with 50 forward passes, 158 turns and 80 Python runs. Exact original fresh-search completion is not a four-hour promise.
- Next: independently analyze non-math per-problem logs (IFEval, GPQA Diamond, MMLU-Pro), compare paired item outcomes and uncertainty, then conduct a local held-out checkpoint evaluation if model downloads and inference permit.
