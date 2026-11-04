# Fresh local edit search: result and reproduction status

## Result in one sentence

We attempted a fresh, approximate weight-space edit search on training data, froze a candidate before confirmation, and found no confirmed benefit: base, candidate, and matched control were all strictly correct on 0/4 untouched items, and all 12 generations stopped at the 8,192-token cap without EOS.

This is a negative small pilot, not a reproduction of the paper's 50-pass method or its reported outcomes.

## What the first experiment did

The first experiment began with four TRAIN prompts to diagnose prolonged repetitive reasoning. On one GSM8K TRAIN prompt, the base model generated 32,768 tokens without closing its thinking section or emitting EOS. A 70-token continuation sampled from the repeated segment was compared by teacher forcing to the author's reference solution. Four layer-18 columns were then screened at several attenuation strengths with a control. The selected `alpha100` intervention produced the gold boxed answer earlier in a paired generation, but both base and edited runs still hit the 32,768-token cap; no EOS or thinking close appeared, and repeated 8-gram fraction increased from 0.8539 to 0.8668. This showed a local change in continuation, not repaired task completion or a faithful reproduction of the authors' edit-agent procedure.

Afterward, work shifted toward rescoring released author records and evaluating the already edited p50c checkpoint. Those are useful audits of released artifacts and checkpoint behavior, but they do not test whether a fresh search can rediscover an edit. The continuation experiment below was run to address that separate question.

## Fresh search protocol

- **Base:** Qwen3.5-0.8B, local checkpoint SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- **Training data:** GSM8K TRAIN and MATH TRAIN from pinned source commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; file SHA-256 values are in the manifest.
- **Search:** compared failure/reference teacher-forced log probability on training diagnostics; screened layer 23 target columns with attenuation and amplification doses, single-column interventions, and norm-matched controls. This was a local direct weight edit, not gradient training.
- **Freeze:** candidate `single137_double`, layer 23, down-projection column 137 multiplied by 2.0. SHA-256 `c61513042ffab0c018584ed264cd899f45836644ef274f6f0185a4e650527bc`. Matched control `single316_double`, column 316 multiplied by 2.0, SHA-256 `ad7a0ac4d14a59dcc18c5d3e8fab1dff1a23d831f6cfd1a27877ea8ee8e8d27e`. The candidate selection was exploratory: mean diagnostic contrast shift was -0.001351 across four TRAIN items; the control's was more negative at -0.002804.
- **Confirmation:** four untouched IDs, fixed per-item seed shared across arms, thinking enabled, max 8,192 generated tokens: `gsm8k_train_5362`, `gsm8k_train_7343`, `math_train_algebra_164`, and `math_train_prealgebra_36`. Candidate and control were frozen before their prompts/answers were opened.

## Confirmation outcomes

| Arm | Strict correct | Cap hit | EOS / endoftext | Any boxed substring | Mean repeated 8-gram fraction |
|---|---:|---:|---:|---:|---:|
| Base | 0/4 | 4/4 | 0 / 0 | 2/4 | 0.4557 |
| Frozen candidate | 0/4 | 4/4 | 0 / 0 | 2/4 | 0.4777 |
| Matched control | 0/4 | 4/4 | 0 / 0 | 2/4 | 0.4745 |

No boxed substring parsed as a strict final answer. All arms failed in the same decisive way: no generation completed within the cap and no task received a strict correct grade. The candidate did not improve the observed stopping, answer-format, or repetition outcomes; its average repetition fraction was modestly higher than base. Because there were no successful completions in any arm, this small confirmation cannot estimate a meaningful task-accuracy difference. It is still direct evidence against claiming that this frozen candidate fixed the observed failure mode on these four items.

## Reward-hacking / proxy audit

The candidate did not gain by emitting a parseable answer and then relying on a weak stopping proxy: strict parsed correctness stayed 0/4, cap hits stayed 4/4, and EOS/endoftext counts stayed zero. Boxed-substring counts were identical across arms and every such output failed strict parsing. Repetition also did not improve. These observations show no hidden proxy-only success in this confirmation. They are not evidence that deliberate reward hacking occurred.

## What was and was not reproduced

**Done:** a fresh local edit search was executed on training data; dose and control interventions were screened; a candidate and control were frozen before a disjoint confirmation split was accessed; paired generations were completed and analyzed; interrupted control work was resumed without overwriting the partial raw evidence.

**Not reproduced:** the authors' complete 50-pass agent search under its original setup, the reported edit, or the non-math benchmark gains. This local continuation recorded 34 forward passes in total (including the seven inherited from the initial experiment), used four confirmation items and an 8,192-token cap, and ran MPS/BF16 Transformers with reconstructed seeds. The pinned release does publish the exact prompts, Astra's turn-by-turn Python code/outputs and forward-call summaries; it omits the persistent Qwen editor kernel, orchestration, and forward-pass server implementation. The continuation also substituted manual edit choices for Astra as the editing agent, so it does not reproduce the central decision-making loop.

## Reproducibility artifacts

The full local run is preserved under `results/astra_method_continuation_20261103/` (about 1.6 GB), including raw generations, token IDs, activation traces, edit variants, interruption records, manifests, and analysis scripts. The raw tensors and full reasoning traces are not committed because of size and the project's raw-reasoning publication boundary. Compact, public analysis files are tracked here:

- `results/astra_method_continuation_20261103/manifest.json`
- `results/astra_method_continuation_20261103/edit_selection.json`
- `results/astra_method_continuation_20261103/frozen_candidate.json`
- `results/astra_method_continuation_20261103/dose_contrast_summary.json`
- `results/astra_method_continuation_20261103/all_dose_contrast_summary.json`
- `results/astra_method_continuation_20261103/single_column_screen_summary.json`
- `results/astra_method_continuation_20261103/heldout_confirmation_analysis.json`
- `results/astra_method_continuation_20261103/attempt_008_interruption.json` and `attempt_035_interruption.json`

The public JSON summaries contain metrics and hashes, not full generated reasoning. The base checkpoint and exact dataset revisions are identified by hashes in the manifest; the dataset source is the pinned [`exp8_math` commit](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math). The paper's reported results and method description are in the [author's research post](https://rikiyat.github.io/blog/weight-space-programming/).

## Conclusion

The answer to “did we reproduce it?” is **no**. We did perform a new approximate search and a frozen-candidate confirmation, rather than only replaying or rescoring the released result. That experiment did not reproduce a useful edit effect. The full source protocol and agent traces are available; a faithful new run still needs to make Astra the actual editor and rebuild the missing persistent CPU editor kernel and forward service. This earlier pilot materially reduced the pass count, confirmation sample, and generation cap. Do not present it as successful or merge its evidence with author-record rescoring or the p50c transfer probe.
