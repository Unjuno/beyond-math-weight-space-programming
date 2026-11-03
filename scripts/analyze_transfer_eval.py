#!/usr/bin/env python3
"""Summarize paired binary task success with deterministic task bootstrap CIs."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def interval(scores, ids, draws, seed):
    deltas = [int(scores[i]["edited"]) - int(scores[i]["base"]) for i in ids]
    rng = random.Random(seed)
    boot = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(draws))
    return sum(deltas) / len(deltas), boot[int(0.025 * draws)], boot[min(draws - 1, int(0.975 * draws))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", type=Path)
    p.add_argument("--secondary-ratings", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--draws", type=int, default=50000)
    p.add_argument("--seed", type=int, default=391)
    a = p.parse_args()
    report = json.loads(a.scores.read_text())
    writing_rows = report.get("writing_scores", [])
    if not isinstance(writing_rows, list):
        writing_rows = []
    families = {r["task_id"]: r["family"] for r in report["structured_scores"]}
    families.update({r["task_id"]: "grounded_writing" for r in writing_rows})

    def summarize(task_scores):
        groups = {"overall": list(task_scores)}
        for family in ("extraction", "operational_decision", "grounded_writing"):
            ids = [task_id for task_id in task_scores if families[task_id] == family]
            if ids:
                groups[family] = ids
        result = {}
        for group, ids in groups.items():
            delta, low, high = interval(task_scores, ids, a.draws, a.seed)
            result[group] = {
                "n": len(ids),
                "base_successes": sum(task_scores[i]["base"] for i in ids),
                "edited_successes": sum(task_scores[i]["edited"] for i in ids),
                "delta": delta,
                "bootstrap_95_interval": [low, high],
            }
        return result

    luna_tasks = defaultdict(dict)
    for r in report["structured_scores"] + writing_rows:
        luna_tasks[r["task_id"]][r["condition"]] = bool(r["task_success"])
    expected_count = 24 if len(writing_rows) == 16 else 16
    if len(luna_tasks) != expected_count or any(set(pair) != {"base", "edited"} for pair in luna_tasks.values()):
        raise SystemExit(f"expected {expected_count} complete paired task scores")
    result = {"draws": a.draws, "seed": a.seed, "luna_first_pass_groups": summarize(luna_tasks)}
    if not writing_rows:
        result["writing_review_status"] = "pending blind rating; no writing task-success composite reported"
    if a.secondary_ratings:
        mapping = report["blind_arm_map_revealed_after_score_freeze"]
        secondary = json.loads(a.secondary_ratings.read_text())
        astra_tasks = defaultdict(dict)
        agree = 0
        factor_count = 0
        for r in secondary:
            condition = mapping[r["task_id"]][r["anonymous_label"]]
            factors = [bool(r[k]) for k in (
                "factual_coverage", "no_unsupported_or_inaccurate_claim",
                "task_and_format_fulfilled", "coherent_and_usable"
            )]
            first = next(x for x in report["writing_scores"] if x["task_id"] == r["task_id"] and x["blind_label"] == r["anonymous_label"])
            for key, value in zip(("factual_coverage", "no_unsupported_or_inaccurate_claim", "task_and_format_fulfilled", "coherent_and_usable"), factors):
                agree += value == first["scores"][key]
                factor_count += 1
            packet = next(x for x in report["writing_scores"] if x["task_id"] == r["task_id"] and x["blind_label"] == r["anonymous_label"])
            astra_tasks[r["task_id"]][condition] = bool(all(factors) and packet["eos_before_cap"])
        if len(astra_tasks) != 8 or any(set(pair) != {"base", "edited"} for pair in astra_tasks.values()):
            raise SystemExit("expected 16 secondary blind ratings over 8 writing tasks")
        # Override writing tasks with Astra ratings for the Astra sensitivity result.
        astra_all = {i: dict(v) for i, v in luna_tasks.items()}
        astra_all.update({i: dict(v) for i, v in astra_tasks.items()})
        result["astra_second_blind_rater_groups"] = summarize(astra_all)
        result["writing_factor_agreement"] = {"agree": agree, "total": factor_count, "rate": agree / factor_count}
    structured = report.get("structured_semantic_scores")
    if isinstance(structured, list):
        strict_semantic = defaultdict(dict)
        content_only = defaultdict(dict)
        for r in structured:
            strict_semantic[r["task_id"]][r["condition"]] = bool(r["semantic_task_success"])
            content_only[r["task_id"]][r["condition"]] = bool(r["content_success_ignoring_gold_schema"])
        if len(strict_semantic) != 16 or any(set(pair) != {"base", "edited"} for pair in strict_semantic.values()):
            raise SystemExit("expected complete paired semantic ratings for 16 structured tasks")
        result["astra_structured_semantic_audit"] = {
            "all_three_factors_and_eos": summarize(strict_semantic),
            "content_and_no_unsupported_plus_eos_ignoring_gold_schema": summarize(content_only),
            "content_plus_eos_ignoring_gold_schema_excluding_ambiguous_d08": summarize({i: dict(v) for i, v in content_only.items() if i != "d08"}),
            "all_three_factors_and_eos_excluding_ambiguous_d08": summarize({i: dict(v) for i, v in strict_semantic.items() if i != "d08"}),
        }
    result["d08_gold_sensitivity_excluding_task"] = {
        "luna_first_pass": summarize({i: dict(v) for i, v in luna_tasks.items() if i != "d08"})
    }
    if a.secondary_ratings:
        result["d08_gold_sensitivity_excluding_task"]["astra_second_blind_rater"] = summarize({i: dict(v) for i, v in astra_all.items() if i != "d08"})
    rendered = json.dumps(result, indent=2) + "\n"
    if a.out:
        a.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
