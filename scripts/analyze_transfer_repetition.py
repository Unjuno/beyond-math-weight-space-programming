#!/usr/bin/env python3
"""Compute descriptive repeated-token diagnostics from local raw generations."""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def excess_repeat_fraction(ids, n):
    total = max(0, len(ids) - n + 1)
    if total == 0:
        return 0.0
    counts = Counter(tuple(ids[i : i + n]) for i in range(total))
    return sum(count - 1 for count in counts.values() if count > 1) / total


def any_ngram_at_least(ids, n, threshold):
    total = max(0, len(ids) - n + 1)
    if total == 0:
        return False
    counts = Counter(tuple(ids[i : i + n]) for i in range(total))
    return max(counts.values(), default=0) >= threshold


def summarize(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    by_task = defaultdict(dict)
    for line in a.raw.read_text().splitlines():
        r = json.loads(line)
        by_task[r["id"]][r["condition"]] = {
            "cap_hit": r["cap_hit"],
            "repeat4_any_three": any_ngram_at_least(r["generated_token_ids"], 4, 3),
            "repeat8": excess_repeat_fraction(r["generated_token_ids"], 8),
            "tail2048_repeat8": excess_repeat_fraction(r["generated_token_ids"][-2048:], 8),
        }
    if len(by_task) != 24 or any(set(pair) != {"base", "edited"} for pair in by_task.values()):
        raise SystemExit("expected 24 complete paired raw generations")
    result = {"definition": "repeat8 is excess repeated 8-gram occurrences / all 8-gram positions; tail uses last 2048 generated tokens; repeat4 diagnostic flags any 4-gram occurring at least three times.", "arms": {}}
    for arm in ("base", "edited"):
        rows = [pair[arm] for pair in by_task.values()]
        result["arms"][arm] = {
            "n": len(rows),
            "eos_before_cap": sum(not r["cap_hit"] for r in rows),
            "cap_hits": sum(r["cap_hit"] for r in rows),
            "any_repeat4gram_three_or_more": sum(r["repeat4_any_three"] for r in rows),
            "repeat8_all": summarize([r["repeat8"] for r in rows]),
            "repeat8_cap_only": summarize([r["repeat8"] for r in rows if r["cap_hit"]]),
            "repeat8_eos_only": summarize([r["repeat8"] for r in rows if not r["cap_hit"]]),
            "tail2048_repeat8_all": summarize([r["tail2048_repeat8"] for r in rows]),
        }
    deltas = [pair["edited"]["repeat8"] - pair["base"]["repeat8"] for pair in by_task.values()]
    rng = random.Random(392)
    sims = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(50000))
    result["paired_repeat8_delta_edited_minus_base"] = {
        "mean": statistics.mean(deltas),
        "bootstrap_95_interval": [sims[1249], sims[48749]],
        "draws": 50000,
        "seed": 392,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if a.out:
        a.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
