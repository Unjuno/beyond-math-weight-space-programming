# Local run environment and commands

Captured on the author's MacBook Pro M1 Max (64 GB unified memory), macOS MPS:

- Python 3.11.15
- PyTorch 2.14.1
- Transformers 5.20.2
- NumPy 2.4.6
- safetensors 0.8.0
- `torch.backends.mps.is_available() == True`
- Transformers fell back to reference PyTorch implementations for `causal_conv1d` and `flash-linear-attention`; the run is correct but slower than optimized kernels.

Model files are local and ignored by Git. The base and author-edited checkpoint each occupy about 1.6 GB. Exact Hugging Face revisions and file hashes are recorded in `results/experiment_manifest.json`.

## Prepare an environment and checkpoints

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-mps.txt
.venv/bin/python scripts/prepare_models.py
```

The listed package versions were present in the environment used for this experiment. A clean installation from this requirements file was not independently verified. Checkpoint download requires about 3.2 GB plus cache space.

## Checkpoint audit

```sh
.venv/bin/python scripts/audit_author_checkpoint.py \
  --base model-cache/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors \
  --edited model-cache/Qwen3.5-0.8B-edited/model.safetensors-00001-of-00001.safetensors
```

## Rebuild exact p50c patch

```sh
.venv/bin/python /path/to/lm_understand-logs/exp8_math/code/apply_edit.py \
  model-cache/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors \
  /path/to/lm_understand-logs/exp8_math/runs/p50c/edit \
  model-cache/p50c-patch
```

## Run the local probe

```sh
.venv/bin/python scripts/run_nonmath_probe.py \
  --model model-cache/Qwen3.5-0.8B \
  --patch-dir model-cache/p50c-patch \
  --prompts configs/nonmath_probe.jsonl \
  --max-new-tokens 2048
```

The runner appends JSONL records and skips completed `(task, condition)` pairs on resume. It verifies both edited tensor hashes before edited inference. Do not interpret a shortened cap as a direct reproduction of the paper's 32,768-token benchmark conditions.

The tracked probe config has 24 prompts. Recreate each saved arm by setting its condition and cap as follows (remove `results/raw/*.jsonl` only when intentionally starting a new run; existing raw records are preserved for this study):

```sh
# Thinking on, 2,048 tokens, both checkpoints.
.venv/bin/python scripts/run_nonmath_probe.py --model model-cache/Qwen3.5-0.8B --patch-dir model-cache/p50c-patch --prompts configs/nonmath_probe.jsonl --max-new-tokens 2048 --mode thinking --out results/raw/nonmath_probe.jsonl

# Thinking off, 512 tokens, both checkpoints.
.venv/bin/python scripts/run_nonmath_probe.py --model model-cache/Qwen3.5-0.8B --patch-dir model-cache/p50c-patch --prompts configs/nonmath_probe.jsonl --max-new-tokens 512 --mode no_thinking --out results/raw/no_thinking_probe.jsonl

# Selected post-hoc 32,768-token spot-check prompts.
.venv/bin/python scripts/run_nonmath_probe.py --model model-cache/Qwen3.5-0.8B --patch-dir model-cache/p50c-patch --prompts configs/nonmath_probe.jsonl --max-new-tokens 32768 --mode thinking --prompt-ids inst03,inst05 --out results/raw/long_cap_probe.jsonl
```

The runner loads the base checkpoint for each condition and applies the pinned p50c tensor patch for the edited arm. Existing `(prompt, condition)` rows are skipped, so use a new output path for an independent repetition. Exact decoding parameters and archived raw hashes are in the manifest.

To reproduce the author-record re-score and all three local summary tables, checkout the pinned public trace repository and run:

```sh
git clone https://github.com/RikiyaT/lm_understand-logs.git /tmp/wsp-author-logs-reproduction
git -C /tmp/wsp-author-logs-reproduction checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
REPRO_PYTHON=.venv/bin/python scripts/reproduce_summaries.sh /tmp/wsp-author-logs-reproduction
```

The summary script first verifies paired IDs, seeds, input tokens, thinking flags, and caps. The generated model outputs are deliberately not in Git; their local SHA-256 values are in the manifest. The author re-score source data is fetched from the pinned public repository above.
