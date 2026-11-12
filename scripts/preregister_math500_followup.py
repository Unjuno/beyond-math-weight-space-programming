#!/usr/bin/env python3
"""Freeze a hash-ranked MATH-500 follow-up sample before local inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


DATASET = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
EXCLUDE = {"test/precalculus/819.json", "test/number_theory/914.json"}
SALT = "p50c-independent-paired-followup-20261112-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_rows() -> list[dict]:
    out = []
    for offset in range(0, 500, 100):
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "test",
                                       "offset": offset, "length": 100, "revision": REVISION})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=45) as response:
            payload = json.load(response)
        if payload.get("num_rows_total") != 500:
            raise RuntimeError(f"Unexpected dataset row count: {payload.get('num_rows_total')}")
        out.extend(entry["row"] for entry in payload["rows"])
    if len(out) != 500 or len({r["unique_id"] for r in out}) != 500:
        raise RuntimeError("Dataset rows are incomplete or contain duplicate IDs")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--items", type=int, default=2)
    args = ap.parse_args()
    rows = fetch_rows()
    ordered_ids = [r["unique_id"] for r in rows]
    ranked = sorted((r for r in rows if r["unique_id"] not in EXCLUDE),
                    key=lambda r: sha((SALT + "\0" + r["unique_id"]).encode()))
    selected = ranked[:args.items]
    canonical_data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                              for r in rows).encode()
    id_bytes = "\n".join(ordered_ids).encode()
    root = args.out_root
    root.mkdir(parents=True, exist_ok=False)
    common = {
        "study": "p50c independently selected paired MATH-500 follow-up",
        "status": "selection_frozen_pre_generation",
        "created_at": "2026-11-12 Asia/Tokyo",
        "dataset": {"repo": DATASET, "revision": REVISION, "rows": len(rows),
                    "canonical_rows_sha256": sha(canonical_data), "ordered_ids_sha256": sha(id_bytes)},
        "selection": {"algorithm": "ascending SHA-256(SALT + NUL + unique_id)", "salt": SALT,
                      "excluded_previously_run_ids": sorted(EXCLUDE),
                      "ranked_candidate_ids_sha256": sha("\n".join(r["unique_id"] for r in ranked).encode()),
                      "selected_ids": [r["unique_id"] for r in selected],
                      "selected_record_sha256": {r["unique_id"]: sha(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                                                                 separators=(",", ":")).encode())
                                                  for r in selected}},
        "protocol": {"base_revision": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
                     "edit_revision": "RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a178705bcab7644806d23c535517f35723b",
                     "thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 1.5, "max_new_tokens": 32768, "samples_per_arm": 4,
                     "local_seed_schedule": [0, 1, 2, 3],
                     "prompt_suffix": "Please reason step by step, and put your final answer within \\boxed{}.",
                     "scorer": "Pinned author exp8_math/code/mathqa.py; last boxed answer + math-verify 0.9.0",
                     "backend": "Transformers/PyTorch on Apple MPS; author used vLLM; local seeds do not reproduce vLLM RNG"},
        "limitations": ["Two items are a small follow-up, not a full MATH-500 rerun.",
                        "Task IDs are frozen before item bodies are written to the local run folders.",
                        "This set is from the public test benchmark and is not hidden from the model's pretraining."],
    }
    common_path = root / "preregistration.json"
    common_path.write_text(json.dumps(common, ensure_ascii=False, indent=2) + "\n")
    common_hash = sha(common_path.read_bytes())
    for i, row in enumerate(selected, start=1):
        item_dir = root / f"item_{i:02d}"
        item_dir.mkdir()
        item = {"id": row["unique_id"], "problem": row["problem"], "gold": row["answer"],
                "subject": row["subject"], "level": row["level"]}
        (item_dir / "selected_problem.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        item_prereg = {"study": common["study"], "status": "preregistered_before_generation",
                       "parent_preregistration": "../preregistration.json", "parent_sha256": common_hash,
                       "selected_index": i, "selected_id": row["unique_id"],
                       "selected_record_sha256": common["selection"]["selected_record_sha256"][row["unique_id"]],
                       "samples_per_checkpoint": 4, "max_new_tokens": 32768,
                       "protocol": common["protocol"]}
        (item_dir / "preregistration.json").write_text(json.dumps(item_prereg, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out_root": str(root), "parent_manifest_sha256": common_hash,
                      "selected_ids": common["selection"]["selected_ids"],
                      "dataset_sha256": common["dataset"]["canonical_rows_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
