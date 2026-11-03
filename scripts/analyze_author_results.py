#!/usr/bin/env python3
"""Recompute paired non-math accuracy and bootstrap CIs from author per-item logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SETS = {
    "ifeval": ("ifeval", "strict prompt-level instruction following"),
    "gpqa": ("gpqa", "GPQA Diamond science QA"),
    "mmlupro1k": ("mmlupro1k", "MMLU-Pro fixed 1k"),
}
RUNS = ("unedited", "p10c", "p25c", "p50c")


def collect(root: Path, run: str, dataset: str) -> dict[str, list[int]]:
    files = sorted((root / "eval" / run / "parts").glob(f"*/{dataset}_per_problem.json"))
    if not files:
        raise FileNotFoundError(f"no {dataset} per-problem files in {root}/eval/{run}/parts")
    merged: dict[str, list[int]] = {}
    for path in files:
        for row in json.loads(path.read_text()):
            key = row["id"]
            if key in merged:
                raise ValueError(f"duplicate problem id {key} in {run}/{dataset}")
            merged[key] = [int(x) for x in row["correct"]]
    return merged


def bootstrap_delta(base: np.ndarray, edit: np.ndarray, seed: int, draws: int = 20_000) -> tuple[tuple[float, float], tuple[float, float]]:
    if base.shape != edit.shape:
        raise ValueError(f"paired item mismatch {base.shape} != {edit.shape}")
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws, dtype=np.float64)
    # Resample problem clusters. Each item contributes its mean over its recorded samples.
    for start in range(0, draws, 500):
        n = min(500, draws - start)
        idx = rng.integers(0, len(base), size=(n, len(base)))
        diffs[start : start + n] = (edit[idx] - base[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    adj_lo, adj_hi = np.quantile(diffs, [0.05 / 6, 1 - 0.05 / 6])
    return (float(lo), float(hi)), (float(adj_lo), float(adj_hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-repo", type=Path, required=True, help="checkout of RikiyaT/lm_understand-logs")
    ap.add_argument("--out", type=Path, default=Path("results/author_nonmath_paired.csv"))
    ap.add_argument("--draws", type=int, default=20_000)
    args = ap.parse_args()
    exp = args.author_repo / "exp8_math"
    out = []
    for short, (dataset, label) in SETS.items():
        values = {run: collect(exp, run, dataset) for run in RUNS}
        ids = sorted(values["unedited"])
        for run in RUNS:
            if set(values[run]) != set(ids):
                raise ValueError(f"unpaired problem IDs: {short} {run}")
        vectors = {
            run: np.asarray([np.mean(values[run][qid]) for qid in ids], dtype=np.float64)
            for run in RUNS
        }
        base = vectors["unedited"]
        base_acc = float(np.mean(np.concatenate([np.asarray(values["unedited"][q], dtype=float) for q in ids])))
        total_samples = sum(len(values["unedited"][q]) for q in ids)
        out.append({
            "task": short, "description": label, "run": "unedited", "items": len(ids),
            "samples": total_samples, "accuracy": base_acc, "delta_vs_base": 0.0,
            "paired_item_bootstrap_95_low": 0.0, "paired_item_bootstrap_95_high": 0.0,
            "p50_three_task_bonferroni_95_low": "", "p50_three_task_bonferroni_95_high": "",
        })
        for j, run in enumerate(RUNS[1:], start=1):
            edit_acc = float(np.mean(np.concatenate([np.asarray(values[run][q], dtype=float) for q in ids])))
            delta = float(np.mean(vectors[run] - base))
            (low, high), (adj_low, adj_high) = bootstrap_delta(base, vectors[run], 20261103 + 100 * j + len(ids), args.draws)
            out.append({
                "task": short, "description": label, "run": run, "items": len(ids),
                "samples": sum(len(values[run][q]) for q in ids), "accuracy": edit_acc,
                "delta_vs_base": delta, "paired_item_bootstrap_95_low": low,
                "paired_item_bootstrap_95_high": high,
                "p50_three_task_bonferroni_95_low": adj_low if run == "p50c" else "",
                "p50_three_task_bonferroni_95_high": adj_high if run == "p50c" else "",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(
            f"{row['task']:10s} {row['run']:8s} n={row['items']:4d} "
            f"acc={row['accuracy']:.4f} delta={row['delta_vs_base']:+.4f} "
            f"paired95=[{row['paired_item_bootstrap_95_low']:+.4f},"
            f" {row['paired_item_bootstrap_95_high']:+.4f}]"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
