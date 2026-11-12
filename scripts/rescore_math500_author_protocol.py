#!/usr/bin/env python3
"""Re-score saved MATH-500 outputs with the pinned author's mathqa.py."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-code-dir", type=Path, required=True,
                    help="exp8_math/code directory from the pinned author log checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("study_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.author_code_dir.resolve()))
    # mathqa.py imports an optional loop-diagnostic helper from the absent
    # private worker package. The public math scoring path does not call it.
    # Fail loudly if that unrelated helper is invoked during this rescore.
    mcqa_stub = types.ModuleType("mcqa")
    def unavailable_loop_helper(*_args, **_kwargs):
        raise RuntimeError("unrelated worker loop helper was unexpectedly called")
    mcqa_stub.loop_signals = unavailable_loop_helper
    sys.modules.setdefault("mcqa", mcqa_stub)
    import mathqa as author_mathqa

    reports = []
    for study_dir in args.study_dirs:
        item = json.loads((study_dir / "selected_problem.json").read_text())
        raw_path = study_dir / "raw_generations.jsonl"
        raw_bytes = raw_path.read_bytes()
        rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
        arm_results = {}
        local_disagreements = []
        for row in rows:
            if row["id"] != item["id"]:
                raise RuntimeError(f"ID mismatch in {study_dir}: {row['id']} != {item['id']}")
            pred, box_count = author_mathqa.extract(row["output"])
            correct = bool(author_mathqa.is_correct(pred, item["gold"]))
            row.setdefault("local_math_verify_correct", row.get("math_verify_correct"))
            row["author_protocol_last_boxed"] = pred
            row["author_protocol_box_count"] = box_count
            row["author_protocol_math_verify_correct"] = correct
            arm = row["arm"]
            arm_results.setdefault(arm, []).append(row)
            if row.get("local_math_verify_correct") != correct:
                local_disagreements.append({"arm": arm, "sample_index": row["sample_index"],
                                            "local": row.get("local_math_verify_correct"),
                                            "author": correct, "author_boxed": pred,
                                            "think_closed": row.get("thinking_closed"),
                                            "cap_hit": row.get("cap_hit")})
        arm_summary = {}
        for arm, subset in arm_results.items():
            arm_summary[arm] = {
                "n": len(subset),
                "correct": sum(r["author_protocol_math_verify_correct"] for r in subset),
                "cap_hits": sum(bool(r.get("cap_hit")) for r in subset),
                "thinking_closed": sum(bool(r.get("thinking_closed")) for r in subset),
                "boxed": sum(r["author_protocol_last_boxed"] is not None for r in subset),
            }
        reports.append({"study_dir": str(study_dir), "item_id": item["id"], "gold": item["gold"],
                        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "author_mathqa_sha256": hashlib.sha256((args.author_code_dir / "mathqa.py").read_bytes()).hexdigest(),
                        "arms": arm_summary, "local_author_disagreements": local_disagreements})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scorer_source": "pinned author exp8_math/code/mathqa.py",
                                    "studies": reports}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "studies": [
        {"id": r["item_id"], "arms": r["arms"], "disagreements": len(r["local_author_disagreements"])}
        for r in reports]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
