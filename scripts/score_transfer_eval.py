#!/usr/bin/env python3
"""Score a frozen paired transfer run and produce blinded writing packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def parse_json_answer(text):
    if not text:
        return None
    cleaned = re.sub(r"<\|(?:im_end|end_of_turn|fim_suffix)\|>$", "", text.strip())
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(task_id):
    return int.from_bytes(hashlib.sha256(f"transfer-eval-v1|{task_id}".encode()).digest()[:4], "big")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--raw", type=Path)
    source_group.add_argument("--answers", type=Path, help="sanitized final-answer records committed for public re-scoring")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blind-packets", type=Path, required=True)
    p.add_argument("--blind-map", type=Path, required=True)
    p.add_argument("--blind-ratings", type=Path)
    p.add_argument("--structured-ratings", type=Path)
    a = p.parse_args()
    tasks = {t["id"]: t for t in json.loads(a.tasks.read_text())["tasks"]}
    source_path = a.raw or a.answers
    rows = [json.loads(x) for x in source_path.read_text().splitlines() if x.strip()]
    manifest = json.loads(a.manifest.read_text())
    if manifest.get("arm") != "p50c" or manifest.get("max_new_tokens") != 32768:
        raise SystemExit("manifest is not the full-cap public p50c run")
    if manifest.get("task_file_sha256") != sha(a.tasks):
        raise SystemExit("task file hash does not match the run manifest")
    if a.raw and manifest.get("raw_output_sha256") != sha(a.raw):
        raise SystemExit("raw output hash does not match the run manifest")
    if a.answers and manifest.get("sanitized_final_answers_sha256") != sha(a.answers):
        raise SystemExit("sanitized final-answer hash does not match the run manifest")
    if manifest.get("base_safetensors_sha256") != "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696":
        raise SystemExit("unexpected base checkpoint hash")
    expected_patches = {
        "15": "ec2a000f74d6b1f14c983747d79285c82cba29b97cda6bc8dd76c68dc0ef0483",
        "19": "ea580672ab639ba06e4d55a201ab9b9c144bc340d633159d1fdb2615b0bbcea5",
    }
    expected_tensors = {
        "15": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
        "19": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
    }
    if manifest.get("patch_file_sha256") != expected_patches or manifest.get("patched_tensor_sha256") != expected_tensors:
        raise SystemExit("public p50c patch/tensor hashes do not match the pinned edit")
    if manifest.get("sampling") != {
        "thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5
    }:
        raise SystemExit("sampling settings do not match the declared protocol")
    expected_families = {"extraction": 8, "operational_decision": 8, "grounded_writing": 8}
    if {family: sum(t["family"] == family for t in tasks.values()) for family in expected_families} != expected_families:
        raise SystemExit("unexpected task-family composition")
    if set(manifest.get("task_order", [])) != set(tasks):
        raise SystemExit("manifest task IDs do not match the task file")
    if len(rows) != 48 or len(tasks) != 24:
        raise SystemExit(f"expected 48 rows and 24 tasks; got {len(rows)} and {len(tasks)}")
    grouped = {}
    for r in rows:
        grouped.setdefault(r["id"], []).append(r)
    if set(grouped) != set(tasks) or any(len(v) != 2 for v in grouped.values()):
        raise SystemExit("missing, extra, or duplicate task rows")
    scores = []
    writing = []
    blind_map = {}
    structured_blind_map = {}
    rng = random.Random(20261105)
    structured_rng = random.Random(20261106)
    for task_id, task in tasks.items():
        pair = {r["condition"]: r for r in grouped[task_id]}
        if set(pair) != {"base", "edited"}:
            raise SystemExit(f"incomplete conditions for {task_id}")
        b, e = pair["base"], pair["edited"]
        if any(r["family"] != task["family"] for r in pair.values()):
            raise SystemExit(f"task-family mismatch for {task_id}")
        if b["seed"] != e["seed"] or (a.raw and b["input_token_ids"] != e["input_token_ids"]):
            raise SystemExit(f"pair mismatch for {task_id}")
        if b["seed"] != manifest["seed_by_task"][task_id] or b["seed"] != stable_seed(task_id):
            raise SystemExit(f"seed mismatch for {task_id}")
        if task["family"] == "grounded_writing":
            labels = ["A", "B"]
            rng.shuffle(labels)
            blind_map[task_id] = {labels[0]: "base", labels[1]: "edited"}
            for cond, label in zip(("base", "edited"), labels):
                row = pair[cond]
                writing.append({
                    "task_id": task_id,
                    "anonymous_label": label,
                    "prompt": task["prompt"],
                    "rubric": task["rubric"],
                    "final_answer": re.sub(r"<\|(?:im_end|end_of_turn|fim_suffix)\|>$", "", row["final_answer"].strip()),
                    "eos_before_cap": row["ended_with_eos"] and not row["cap_hit"],
                    "scores": {
                        "factual_coverage": None,
                        "no_unsupported_or_inaccurate_claim": None,
                        "task_and_format_fulfilled": None,
                        "coherent_and_usable": None,
                    },
                    "evidence": "",
                    "missing_fact_ids": [],
                    "violation_ids": [],
                })
        else:
            structured_labels = ["A", "B"]
            structured_rng.shuffle(structured_labels)
            structured_blind_map[task_id] = {structured_labels[i]: condition for i, condition in enumerate(("base", "edited"))}
            gold = task["gold"]
            for cond, row in pair.items():
                parsed = parse_json_answer(row["final_answer"])
                correct = parsed is not None and strict_equal(parsed, gold)
                score = {
                    "task_id": task_id,
                    "family": task["family"],
                    "condition": cond,
                    "json_valid": parsed is not None,
                    "typed_exact_match": correct,
                    "eos_before_cap": row["ended_with_eos"] and not row["cap_hit"],
                    "cap_hit": row["cap_hit"],
                    "generated_tokens": row["generated_tokens"],
                    "task_success": bool(correct and row["ended_with_eos"] and not row["cap_hit"]),
                }
                scores.append(score)
    a.blind_packets.write_text(json.dumps(writing, ensure_ascii=False, indent=2) + "\n")
    a.blind_map.write_text(json.dumps(blind_map, indent=2) + "\n")
    writing_scores = "pending blind rating"
    blind_ratings_sha256 = None
    if a.blind_ratings:
        ratings = json.loads(a.blind_ratings.read_text())
        blind_ratings_sha256 = sha(a.blind_ratings)
        seen = set()
        writing_scores = []
        for rating in ratings:
            key = (rating["task_id"], rating["anonymous_label"])
            if key in seen or key[0] not in blind_map or key[1] not in blind_map[key[0]]:
                raise SystemExit(f"duplicate or unknown blind rating: {key}")
            seen.add(key)
            factor_names = (
                "factual_coverage", "no_unsupported_or_inaccurate_claim",
                "task_and_format_fulfilled", "coherent_and_usable"
            )
            if any(type(rating.get(k)) is not bool for k in factor_names):
                raise SystemExit(f"blind rubric values must be JSON booleans: {key}")
            packet = next(x for x in writing if (x["task_id"], x["anonymous_label"]) == key)
            factors = {k: rating[k] for k in factor_names}
            writing_scores.append({
                "task_id": key[0], "family": "grounded_writing",
                "condition": blind_map[key[0]][key[1]], "blind_label": key[1],
                "eos_before_cap": packet["eos_before_cap"], "scores": factors,
                "all_rubric_factors_true": all(factors.values()),
                "task_success": all(factors.values()) and packet["eos_before_cap"],
                "evidence": rating.get("evidence", ""),
            })
        if len(seen) != 16:
            raise SystemExit(f"expected 16 frozen blind ratings, got {len(seen)}")
    structured_semantic_scores = "pending semantic audit"
    structured_ratings_sha256 = None
    if a.structured_ratings:
        ratings = json.loads(a.structured_ratings.read_text())
        structured_ratings_sha256 = sha(a.structured_ratings)
        seen = set()
        structured_semantic_scores = []
        factor_names = (
            "semantic_content_correct", "no_unsupported_or_contradictory_content", "json_schema_valid"
        )
        for rating in ratings:
            key = (rating["task_id"], rating["anonymous_label"])
            if key in seen or key[0] not in structured_blind_map or key[1] not in structured_blind_map[key[0]]:
                raise SystemExit(f"duplicate or unknown structured semantic rating: {key}")
            seen.add(key)
            if any(type(rating.get(k)) is not bool for k in factor_names):
                raise SystemExit(f"semantic rubric values must be JSON booleans: {key}")
            condition = structured_blind_map[key[0]][key[1]]
            machine = next(s for s in scores if s["task_id"] == key[0] and s["condition"] == condition)
            vals = {k: rating[k] for k in factor_names}
            structured_semantic_scores.append({
                "task_id": key[0], "family": machine["family"], "condition": condition,
                "blind_label": key[1], "eos_before_cap": machine["eos_before_cap"],
                "scores": vals,
                "semantic_task_success": all(vals.values()) and machine["eos_before_cap"],
                "content_success_ignoring_gold_schema": vals["semantic_content_correct"] and vals["no_unsupported_or_contradictory_content"] and machine["eos_before_cap"],
                "failure_category": rating.get("failure_category", ""),
                "evidence": rating.get("evidence", ""),
            })
        if len(seen) != 32:
            raise SystemExit(f"expected 32 frozen structured semantic ratings, got {len(seen)}")
    a.out.write_text(json.dumps({
        "protocol": "transfer_eval_prereg.md",
        "task_file_sha256": sha(a.tasks),
        "evidence_source": "raw_generation" if a.raw else "sanitized_final_answers",
        "raw_file_sha256": sha(a.raw) if a.raw else None,
        "sanitized_final_answers_sha256": sha(a.answers) if a.answers else None,
        "manifest_sha256": sha(a.manifest),
        "structured_scores": scores,
        "writing_scores_frozen_sha256": blind_ratings_sha256,
        "condition_map_revealed_after_score_freeze": bool(a.blind_ratings),
        "blind_arm_map_revealed_after_score_freeze": blind_map if a.blind_ratings else None,
        "writing_scores": writing_scores,
        "structured_semantic_ratings_frozen_sha256": structured_ratings_sha256,
        "structured_blind_arm_map_revealed_after_score_freeze": structured_blind_map if a.structured_ratings else None,
        "structured_semantic_scores": structured_semantic_scores,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"verified {len(rows)} rows / {len(grouped)} pairs; structured scores={len(scores)}; blinded writing answers={len(writing)}")


if __name__ == "__main__":
    main()
