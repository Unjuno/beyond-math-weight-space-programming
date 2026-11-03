# Local run environment and commands

Captured on the author's MacBook Pro M1 Max (64 GB unified memory), macOS MPS:

- Python 3.11.15
- PyTorch 2.14.1
- Transformers 5.20.2
- NumPy 2.4.6
- safetensors 0.8.0
- `torch.backends.mps.is_available() == True`
- Transformers fell back to reference PyTorch implementations for `causal_conv1d` and `flash-linear-attention`; the run is correct but slower than optimized kernels.

Model files are local and ignored by Git. The base and author-edited checkpoint each occupy about 1.6 GB. The Hugging Face base revision currently resolves to `2fc06364715b967f1860aea9cf38778875588b17`, matching the revision used in the author logs.

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
