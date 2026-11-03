# Author release provenance

- Blog: https://rikiyat.github.io/blog/weight-space-programming/
- Complete traces and Experiment 8 code: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math
- Edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Base checkpoint: https://huggingface.co/Qwen/Qwen3.5-0.8B

Pinned upstream log commit inspected: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

Experiment 8 used GPT-6 Astra (`gpt-6-astra`) via Codex CLI, reasoning effort max; one conversation per run, with fresh handoffs every 40 turns or harness restart. Codex tools were off. Astra had a persistent CPU-only Python kernel and a forward-pass tool with 10, 25 or 50 pass budgets; practice sets were GSM8K train, MATH train, and AIME answers. No gradient descent/training, no network execution inside the weight-editing kernel, and no internet were allowed. Evaluation sets were kept from Astra.

The author README says p50c took 20.3 hours (158 turns, 80 Python runs, four conversations). It also records that the agent harness source, work folders/activations, some generation-length details and some graders are not included. The exact final edit and rebuild script are published. For p50c, 131,072 entries were targeted, but after BF16 rounding 130,188 values differ.

Important rationale boundary: Astra's logged explanations and actions allow us to reconstruct the reported decision path and interrogate it. They do not provide a complete, independently verified account of all internal reasoning.
