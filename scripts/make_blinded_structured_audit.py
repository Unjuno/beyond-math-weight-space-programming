#!/usr/bin/env python3
"""Create anonymous A/B packets for semantic review of structured answers."""
import argparse
import json
import random
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--packets", type=Path, required=True)
    p.add_argument("--map", type=Path, required=True)
    a = p.parse_args()
    tasks = {t["id"]: t for t in json.loads(a.tasks.read_text())["tasks"] if t["family"] != "grounded_writing"}
    rows = [json.loads(line) for line in a.answers.read_text().splitlines() if line.strip()]
    grouped = {}
    for r in rows:
        if r["id"] in tasks:
            grouped.setdefault(r["id"], {})[r["condition"]] = r
    if len(tasks) != 16 or set(grouped) != set(tasks) or any(set(v) != {"base", "edited"} for v in grouped.values()):
        raise SystemExit("expected 16 complete structured-task pairs")
    rng = random.Random(20261106)
    packets, mapping = [], {}
    for task_id, task in tasks.items():
        labels = ["A", "B"]
        rng.shuffle(labels)
        mapping[task_id] = {labels[i]: condition for i, condition in enumerate(("base", "edited"))}
        for label, condition in zip(labels, ("base", "edited")):
            row = grouped[task_id][condition]
            answer = re.sub(r"<\|(?:im_end|end_of_turn|fim_suffix)\|>$", "", row["final_answer"].strip())
            packets.append({
                "task_id": task_id,
                "anonymous_label": label,
                "task_family": task["family"],
                "prompt": task["prompt"],
                "gold": task["gold"],
                "final_answer": answer,
                "semantic_rubric": {
                    "all_requested_facts_and_decision_are_correct": "Allow capitalization, whitespace, equivalent numeric spelling, and harmless paraphrase. Do not overlook missing dates, changed numbers, reversed decisions, or unsupported certainty.",
                    "no_unsupported_or_contradictory_content": "Judge meaning, not keyword occurrence.",
                    "json_schema_valid": "Separately assess whether the answer is parseable JSON with exactly the required keys and types. Do not let a correct value in invalid structure pass the applied task.",
                },
                "ratings": {
                    "semantic_content_correct": None,
                    "no_unsupported_or_contradictory_content": None,
                    "json_schema_valid": None,
                },
                "evidence": "",
            })
    a.packets.write_text(json.dumps(packets, ensure_ascii=False, indent=2) + "\n")
    a.map.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"created {len(packets)} anonymous records for {len(tasks)} pairs")


if __name__ == "__main__":
    main()
