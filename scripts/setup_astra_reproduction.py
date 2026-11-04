#!/usr/bin/env python3
"""Create a fresh, prior-run-disjoint train/editor and held-out split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/wsp-source/exp8_math/practice")
RUN = ROOT / "results/astra_reproduction_20261104"
RAW = RUN / "raw"
RUN_ID = "astra-repro-20261104-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prior_ids() -> set[str]:
    used: set[str] = set()
    old_roots = [ROOT / "results/astra_clean_20261103",
                 ROOT / "results/astra_method_continuation_20261103"]
    for old_root in old_roots:
        if not old_root.exists():
            continue
        for p in old_root.rglob("*.jsonl"):
            if any(part in {"requests", "responses", "edit_requests", "edit_responses"} for part in p.parts):
                continue
            try:
                for row in read_jsonl(p):
                    ident = row.get("id")
                    if isinstance(ident, str):
                        used.add(ident)
                    for k in ("problem_ids", "ids"):
                        vals = row.get(k)
                        if isinstance(vals, list):
                            used.update(x for x in vals if isinstance(x, str))
            except (OSError, json.JSONDecodeError):
                continue
        manifest = old_root / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                for vals in data.get("splits", {}).values():
                    used.update(x for x in vals if isinstance(x, str))
            except (OSError, json.JSONDecodeError):
                pass
    return used


def ranked(rows, partition: str):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{RUN_ID}|{partition}|{r['id']}".encode()).digest())


def main():
    (RAW / "train").mkdir(parents=True, exist_ok=True)
    (RAW / "heldout").mkdir(parents=True, exist_ok=True)
    all_ids = prior_ids()
    split_counts = {}
    train_ids, heldout_ids, excluded_ids = set(), set(), set()
    source_hashes = {}
    for filename in ("gsm8k_train.jsonl", "math_train.jsonl"):
        rows = list(read_jsonl(SOURCE / filename))
        source_hashes[filename] = sha256(SOURCE / filename)
        eligible = []
        for row in rows:
            ident = row["id"]
            if ident in all_ids:
                excluded_ids.add(ident)
            else:
                eligible.append(row)
        ordered = ranked(eligible, filename)
        hold_n = max(256, round(len(ordered) * 0.2))
        held = ordered[:hold_n]
        train = ordered[hold_n:]
        def write(name, data):
            with (RAW / name).open("w") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write("train/" + filename, train)
        write("heldout/" + filename, held)
        train_ids.update(r["id"] for r in train)
        heldout_ids.update(r["id"] for r in held)
        split_counts[filename] = {"available": len(rows), "prior_ids_excluded": len(rows) - len(eligible),
                                  "editor_train": len(train), "heldout": len(held)}
    assert not (train_ids & heldout_ids)
    assert not (train_ids & all_ids) and not (heldout_ids & all_ids)
    manifest = {
        "run_id": RUN_ID,
        "created_utc": "2026-11-04",
        "source_revision": "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f",
        "base_model": str(ROOT / "model-cache/Qwen3.5-0.8B"),
        "base_model_sha256": sha256(ROOT / "model-cache/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"),
        "practice_source_sha256": source_hashes,
        "split_method": "SHA256-ranked deterministic 80/20 per source; all IDs present in prior local edit-search ledgers/manifests excluded before ranking",
        "split_counts": split_counts,
        "prior_run_ids_excluded": len(excluded_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256("\n".join(sorted(heldout_ids)).encode()).hexdigest(),
        "heldout_data_visible_to_editor": False,
        "raw_data_local_only": True,
        "paper_settings": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                           "presence_penalty": 1.5, "max_new_tokens": 32768,
                           "primary_metric": "last boxed answer graded by math-verify"},
        "method_status": "fresh Astra-led run not yet started",
        "fidelity_deviations": ["Original persistent harness and forward server are absent; a new local adapter is being reconstructed.",
                                "Forward backend will be pinned and explicitly reported after parity gates."]
    }
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "counts": split_counts, "prior_ids_excluded": len(excluded_ids),
                      "train_ids_sha256": manifest["train_ids_sha256"],
                      "heldout_ids_sha256": manifest["heldout_ids_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
