#!/usr/bin/env python3
"""Verify completeness and paired prompt/seed/token invariants of saved probes."""
import argparse
import json
from collections import Counter
from pathlib import Path

def verify(path, expected_items, conditions, thinking, cap):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by = {(row["id"], row["condition"]): row for row in rows}
    if len(by) != len(rows):
        raise ValueError(f"{path}: duplicate (id, condition) rows")
    counts = Counter(row["condition"] for row in rows)
    if counts != Counter({condition: expected_items for condition in conditions}):
        raise ValueError(f"{path}: condition counts {dict(counts)}; expected {expected_items} each for {conditions}")
    task_ids = sorted({row["id"] for row in rows})
    if len(task_ids) != expected_items:
        raise ValueError(f"{path}: got {len(task_ids)} prompt IDs; expected {expected_items}")
    for task_id in task_ids:
        base, edited = (by[(task_id, condition)] for condition in conditions)
        if base["seed"] != edited["seed"]:
            raise ValueError(f"{path}: seed mismatch for {task_id}")
        if base["input_token_ids"] != edited["input_token_ids"]:
            raise ValueError(f"{path}: input-token mismatch for {task_id}")
        if base["thinking"] is not thinking or edited["thinking"] is not thinking:
            raise ValueError(f"{path}: thinking flag mismatch for {task_id}")
    for row in rows:
        cap_hit = row["generated_tokens"] >= cap
        if row["cap_hit"] != cap_hit:
            raise ValueError(f"{path}: cap flag inconsistent for {(row['id'], row['condition'])}")
        if "final_answer_reached" in row and row["final_answer_reached"] != bool(row.get("final_answer", "").strip()):
            raise ValueError(f"{path}: final-answer flag inconsistent for {(row['id'], row['condition'])}")
    print(f"{path}: {len(rows)} rows; {expected_items} matched prompt pairs; cap={cap}; IDs/seeds/input tokens verified")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinking", type=Path, default=Path("results/raw/nonmath_probe.jsonl"))
    parser.add_argument("--no-thinking", type=Path, default=Path("results/raw/no_thinking_probe.jsonl"))
    parser.add_argument("--spotcheck", type=Path, default=Path("results/raw/long_cap_probe.jsonl"))
    args = parser.parse_args()
    verify(args.thinking, 24, ("base_thinking", "edited_thinking"), True, 2048)
    verify(args.no_thinking, 24, ("base_no_thinking", "edited_no_thinking"), False, 512)
    verify(args.spotcheck, 2, ("base_thinking", "edited_thinking"), True, 32768)

if __name__ == "__main__":
    main()
