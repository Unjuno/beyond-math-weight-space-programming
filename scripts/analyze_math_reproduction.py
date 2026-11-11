#!/usr/bin/env python3
"""Create a sanitized summary of local paired MATH-500 reproduction outputs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_verbatim_loop(text: str) -> tuple[bool, int | None]:
    """Author-compatible terminal loop check: 1..200 words repeated >=3 times, >=40 words."""
    words = text.split()[:-1]
    n = len(words)
    for period in range(1, min(200, n) + 1):
        repetitions = max(3, (40 + period - 1) // period)
        if repetitions * period > n:
            continue
        block = words[n - period:]
        if all(words[n - (j + 1) * period:n - j * period] == block
               for j in range(1, repetitions)):
            return True, period
    return False, None


def diagnose(row: dict) -> dict:
    ids = row["generated_token_ids"]
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    repeated_excess = sum(count - 1 for count in collections.Counter(grams).values() if count > 1)
    repeated_fraction = repeated_excess / max(1, len(grams))
    verbatim, period = terminal_verbatim_loop(row["output"])
    return {
        "arm": row["arm"], "seed": row["seed"],
        "generated_tokens": row["generated_tokens"], "seconds": round(row["seconds"], 1),
        "cap_hit": row["cap_hit"], "thinking_closed": row["thinking_closed"],
        "boxed_answer_present": row["last_boxed"] is not None,
        "math_verify_correct": bool(row["math_verify_correct"]),
        "terminal_verbatim_loop": verbatim, "terminal_loop_period_words": period,
        "repeated_4gram_excess_fraction": round(repeated_fraction, 4),
    }


def summarize(root: Path) -> dict:
    manifest_path = root / "preregistration.json"
    summary_path = root / "summary.json"
    raw_path = root / "raw_generations.jsonl"
    manifest = json.loads(manifest_path.read_text())
    scored = json.loads(summary_path.read_text())
    if manifest.get("status") != "scored":
        raise ValueError(f"{root}: run is not scored")
    if sha256(raw_path) != manifest.get("raw_generations_sha256"):
        raise ValueError(f"{root}: raw generation hash mismatch")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    if len(rows) != 8:
        raise ValueError(f"{root}: expected 8 generations, got {len(rows)}")
    diagnostics = [diagnose(row) for row in rows]
    per_arm = {}
    for arm in ("base", "edited"):
        group = [row for row in diagnostics if row["arm"] == arm]
        if len(group) != 4:
            raise ValueError(f"{root}: expected 4 {arm} generations")
        per_arm[arm] = {
            "n": len(group),
            "correct": sum(row["math_verify_correct"] for row in group),
            "accuracy": sum(row["math_verify_correct"] for row in group) / len(group),
            "boxed_answers": sum(row["boxed_answer_present"] for row in group),
            "cap_hits": sum(row["cap_hit"] for row in group),
            "thinking_closed": sum(row["thinking_closed"] for row in group),
            "terminal_verbatim_loops": sum(row["terminal_verbatim_loop"] for row in group),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in group) / len(group),
            "mean_repeated_4gram_excess_fraction": sum(
                row["repeated_4gram_excess_fraction"] for row in group
            ) / len(group),
        }
    paired = []
    for seed in range(4):
        pair = {row["arm"]: row for row in diagnostics if row["seed"] == seed}
        if set(pair) != {"base", "edited"}:
            raise ValueError(f"{root}: unmatched pair at seed {seed}")
        paired.append({
            "seed": seed,
            "base_correct": pair["base"]["math_verify_correct"],
            "edited_correct": pair["edited"]["math_verify_correct"],
            "base_cap_hit": pair["base"]["cap_hit"],
            "edited_cap_hit": pair["edited"]["cap_hit"],
        })
    return {
        "stage": manifest.get("stage", "initial preregistered tranche"),
        "selection_seed": manifest.get("selection_seed", 20261111),
        "selected_index": manifest["selected_index"],
        "selected_id": manifest["selected_id"],
        "subject": json.loads((root / "selected_problem.json").read_text())["subject"],
        "level": json.loads((root / "selected_problem.json").read_text())["level"],
        "dataset_revision": manifest["dataset_source_revision"],
        "dataset_file_sha256": manifest["dataset_file_sha256"],
        "raw_generation_sha256_local_only": sha256(raw_path),
        "scorer": scored["scorer"],
        "per_arm": per_arm,
        "paired": paired,
        "per_sample_diagnostics": diagnostics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tranches = [summarize(root) for root in args.study_dirs]
    arms = {}
    for arm in ("base", "edited"):
        group = [row for tranche in tranches for row in tranche["per_sample_diagnostics"]
                 if row["arm"] == arm]
        arms[arm] = {
            "n": len(group),
            "correct": sum(row["math_verify_correct"] for row in group),
            "boxed_answers": sum(row["boxed_answer_present"] for row in group),
            "cap_hits": sum(row["cap_hit"] for row in group),
            "thinking_closed": sum(row["thinking_closed"] for row in group),
            "terminal_verbatim_loops": sum(row["terminal_verbatim_loop"] for row in group),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in group) / len(group),
        }
    result = {
        "study": "Published p50c checkpoint paired MATH-500 pilot",
        "evidence_scope": "Two selected problems, 4 matched seeds per checkpoint per problem; descriptive only.",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "published_edit_revision": "d1671a178705bcab7644806d23c535517f35723b",
        "published_edit_tensor_sha256": {
            "L15": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
            "L19": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
        },
        "dataset": {"repo": "HuggingFaceH4/MATH-500",
                    "revision": tranches[0]["dataset_revision"],
                    "test_jsonl_sha256": tranches[0]["dataset_file_sha256"]},
        "protocol": {
            "thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "max_new_tokens": 32768,
            "matched_seed_schedule": [0, 1, 2, 3],
            "scorer": "last boxed answer + math-verify 0.9.0",
            "local_backend": "Transformers 5.20.2 / PyTorch 2.14.1 on Apple MPS",
            "backend_difference": "Authors report vLLM; local generation backend differs.",
        },
        "descriptive_total": arms,
        "tranches": tranches,
        "limits": [
            "Two problems are far too few to estimate MATH-500 benchmark accuracy.",
            "The second item was a sequential follow-up fixed after the first result; it is exploratory, not confirmatory.",
            "The fresh 50-pass Astra weight-editing trajectory was not rerun; this evaluates the exact published edit.",
            "Raw generations remain local and are identified only by SHA-256.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "descriptive_total": arms}, indent=2))


if __name__ == "__main__":
    main()
