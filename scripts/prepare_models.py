#!/usr/bin/env python3
"""Download the exact public base and p50c checkpoints used in this study."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = (
    ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "model-cache/Qwen3.5-0.8B"),
    ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b", "model-cache/Qwen3.5-0.8B-edited"),
)

def main():
    for repo_id, revision, local_dir in MODELS:
        path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
        print(f"{repo_id}@{revision} -> {Path(path)}")

if __name__ == "__main__":
    main()
