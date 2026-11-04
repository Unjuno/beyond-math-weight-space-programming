#!/usr/bin/env python3
"""Read-only ID/count/hash audit for the human-operated Astra edit study.

Reads row IDs from the prior experiment's author-TRAIN JSONL files and the
prior reserved-ID ledger. It never prints prompt/answer bodies and writes no
files. Run from any working directory with Python 3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/astra_reproduction_20261104"
RUN_ID = "human-astra-manual-20261105-v1"
EXPECTED_TRAIN_IDS = 11_385
EXPECTED_RESERVED_IDS = 352
EXPECTED_ELIGIBLE_IDS = 11_033
EXPECTED_BY_SOURCE = {
    "gsm8k_train.jsonl": {"eligible": 5_594, "discovery": 3_917, "other": 559},
    "math_train.jsonl": {"eligible": 5_439, "discovery": 3_810, "other": 543},
}


def digest_ids(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def read_ids(path: Path) -> tuple[set[str], str]:
    ids: list[str] = []
    source_hash = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"Missing string id in {path.name}:{line_number}")
            ids.append(identifier)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path.name}")
    return set(ids), source_hash.hexdigest()


def main() -> None:
    manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    partitions = json.loads(
        (OLD / "raw/editor_work/train_partitions.json").read_text(encoding="utf-8")
    )
    reserved = set().union(*(set(values) for values in partitions.values()))
    if len(reserved) != EXPECTED_RESERVED_IDS:
        raise ValueError(f"STOP: expected {EXPECTED_RESERVED_IDS} reserved IDs, got {len(reserved)}")

    sources: dict[str, set[str]] = {}
    for name in EXPECTED_BY_SOURCE:
        ids, source_hash = read_ids(OLD / "raw/train" / name)
        expected_hash = manifest["practice_source_sha256"][name]
        if source_hash != expected_hash:
            raise ValueError(f"STOP: source file SHA-256 mismatch for {name}")
        sources[name] = ids
    all_ids = set().union(*sources.values())
    if sum(map(len, sources.values())) != len(all_ids):
        raise ValueError("STOP: source ID sets overlap")
    if len(all_ids) != EXPECTED_TRAIN_IDS:
        raise ValueError(f"STOP: expected {EXPECTED_TRAIN_IDS} TRAIN IDs, got {len(all_ids)}")
    if digest_ids(all_ids) != manifest["train_ids_sha256"]:
        raise ValueError("STOP: TRAIN ID hash does not match prior manifest")
    if not reserved <= all_ids:
        raise ValueError("STOP: prior reserved IDs are missing from TRAIN sources")

    print("run_id:", RUN_ID)
    print("prior_train_ids:", len(all_ids))
    print("prior_train_ids_sha256:", digest_ids(all_ids))
    print("reserved_ids:", len(reserved))
    allocated: set[str] = set()
    total_eligible = 0

    for name, ids in sources.items():
        eligible = ids - reserved
        ordered = sorted(
            eligible,
            key=lambda identifier: hashlib.sha256(
                f"{RUN_ID}|{name}|{identifier}".encode()
            ).digest(),
        )
        n = len(ordered) // 10
        splits = {
            "validation": set(ordered[:n]),
            "confirmation": set(ordered[n : 2 * n]),
            "final_holdout": set(ordered[2 * n : 3 * n]),
            "discovery": set(ordered[3 * n :]),
        }
        expected = EXPECTED_BY_SOURCE[name]
        if len(eligible) != expected["eligible"]:
            raise ValueError(f"STOP: unexpected eligible count for {name}: {len(eligible)}")
        if len(splits["discovery"]) != expected["discovery"]:
            raise ValueError(f"STOP: unexpected discovery count for {name}")
        for label in ("validation", "confirmation", "final_holdout"):
            if len(splits[label]) != expected["other"]:
                raise ValueError(f"STOP: unexpected {label} count for {name}")
        if set().union(*splits.values()) != eligible:
            raise ValueError(f"STOP: partition does not cover eligible IDs for {name}")
        if any(splits[a] & splits[b] for a in splits for b in splits if a < b):
            raise ValueError(f"STOP: partitions overlap for {name}")

        print(f"{name}: eligible={len(eligible)}")
        for label in ("discovery", "validation", "confirmation", "final_holdout"):
            members = splits[label]
            if members & (reserved | allocated):
                raise ValueError(f"STOP: overlap with reserved or prior allocation in {name}/{label}")
            allocated.update(members)
            print(f"  {label}: n={len(members)} sha256={digest_ids(members)}")
        total_eligible += len(eligible)

    if total_eligible != EXPECTED_ELIGIBLE_IDS or len(allocated) != EXPECTED_ELIGIBLE_IDS:
        raise ValueError("STOP: final eligible/allocation count mismatch")
    print("DRY_RUN_OK: read-only; no files, prompts, answers, or weights changed")


if __name__ == "__main__":
    main()
