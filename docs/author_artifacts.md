# Author release provenance

- Blog: https://rikiyat.github.io/blog/weight-space-programming/
- Pinned author logs and published Experiment 8 evaluation/edit code: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math
- Edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Base checkpoint: https://huggingface.co/Qwen/Qwen3.5-0.8B

Pinned upstream log commit inspected: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

Experiment 8 used GPT-6 Astra (`gpt-6-astra`) via Codex CLI, reasoning effort max; one conversation per run, with fresh handoffs every 40 turns or harness restart. Codex tools were off. Astra had a persistent CPU-only Python kernel and a forward-pass tool with 10, 25 or 50 pass budgets; practice sets were GSM8K train, MATH train, and AIME answers. No gradient descent/training, no network execution inside the weight-editing kernel, and no internet were allowed. Evaluation sets were kept from Astra.

The source is the authors' 2026 research blog and its pinned artifact release. The release contains the actual Experiment 8 first prompt, later Astra prompts/handoffs, turn answers/explanations, Python code and outputs, forward requests and returned summaries, run configuration, evaluation code and exact final edit. It does **not** include the Experiment 8 orchestrator, persistent Qwen Python-kernel implementation, or forward-pass server implementation; `harness/sandbox.py` is for the Go experiment, not Qwen. Some intermediate activation/generated-text work folders, per-sample generation lengths, and imported IFEval/multiple-choice grading components are also absent. Therefore the method and Astra's logged actions can be reconstructed in detail, while exact tool/runtime parity requires a new implementation. The exact final edit and rebuild script are published. The blog reports that p50c took 20.3 hours (158 turns, 80 Python runs, four conversations). For p50c, 131,072 entries were targeted, but after BF16 rounding 130,188 values differ.

The author's primary math outcome is accuracy of the last `\\boxed{}` answer graded with `math-verify`; EOS, cap-hit, and repetition are useful separate diagnostics, not requirements in that primary score. Author notice `002_tool_facts.txt` also says that samples from different forward calls are not reproducible even with identical requests because batching changes RNG state, so paired comparisons belong within one `model: "both"` forward call.

Important rationale boundary: Astra's logged explanations and actions allow us to reconstruct the reported decision path and interrogate it. They do not provide a complete, independently verified account of all internal reasoning.
