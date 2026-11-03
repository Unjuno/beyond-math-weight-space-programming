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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blind-packets", type=Path, required=True)
    p.add_argument("--blind-map", type=Path, required=True)
    p.add_argument("--blind-ratings", type=Path)
    a = p.parse_args()
    tasks = {t["id"]: t for t in json.loads(a.tasks.read_text())["tasks"]}
    rows = [json.loads(x) for x in a.raw.read_text().splitlines() if x.strip()]
    manifest = json.loads(a.manifest.read_text())
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
    rng = random.Random(20261105)
    for task_id, task in tasks.items():
        pair = {r["condition"]: r for r in grouped[task_id]}
        if set(pair) != {"base", "edited"}:
            raise SystemExit(f"incomplete conditions for {task_id}")
        b, e = pair["base"], pair["edited"]
        if b["seed"] != e["seed"] or b["input_token_ids"] != e["input_token_ids"]:
            raise SystemExit(f"pair mismatch for {task_id}")
        if b["seed"] != manifest["seed_by_task"][task_id]:
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
            packet = next(x for x in writing if (x["task_id"], x["anonymous_label"]) == key)
            factors = {k: bool(rating[k]) for k in (
                "factual_coverage", "no_unsupported_or_inaccurate_claim",
                "task_and_format_fulfilled", "coherent_and_usable"
            )}
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
    a.out.write_text(json.dumps({
        "protocol": "transfer_eval_prereg.md",
        "task_file_sha256": sha(a.tasks),
        "raw_file_sha256": sha(a.raw),
        "manifest_sha256": sha(a.manifest),
        "structured_scores": scores,
        "writing_scores_frozen_sha256": blind_ratings_sha256,
        "condition_map_revealed_after_score_freeze": bool(a.blind_ratings),
        "blind_arm_map_revealed_after_score_freeze": blind_map if a.blind_ratings else None,
        "writing_scores": writing_scores,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"verified {len(rows)} rows / {len(grouped)} pairs; structured scores={len(scores)}; blinded writing answers={len(writing)}")


if __name__ == "__main__":
    main()
