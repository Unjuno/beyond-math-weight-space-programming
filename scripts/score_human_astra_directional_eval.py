#!/usr/bin/env python3
"""Score frozen paired outputs and enforce preregistered stopping gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
RAW = RUN / "raw/directional_control_v1"
RAW_ROOT = RAW
PARENT = RUN / "manifest.json"
FOCUSED_IDS = ["gsm8k_train_6445", "math_train_algebra_542", "gsm8k_train_3185"]
ARMS = ["B", "T", "Cplus", "Cminus"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def grouped(rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    out = {}
    for row in rows:
        key = (row["id"], int(row["sample"]), row["arm"])
        if key in out:
            raise ValueError(f"Duplicate paired outcome row: {key}")
        out[key] = row
    return out


def metric_row(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "id", "source", "sample", "seed", "arm", "correct_math_verify",
        "completed_correct", "ended_with_eos", "think_closed_in_continuation",
        "cap_hit", "tokens", "repeat8_fraction", "tail_2048_repeat8_fraction",
        "exact_final_32_token_loop", "seconds", "model_load_seconds",
        "edit_and_integrity_check_seconds", "scoring_error",
    )}


def gate_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(canonical)


def score_focused(rows: list[dict], source_hash: str, runner_hash: str) -> dict:
    expected = {(ident, sample, arm) for ident in FOCUSED_IDS for sample in (0, 1) for arm in ARMS}
    actual = grouped(rows)
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    reasons = []
    if missing:
        reasons.append("incomplete_24_generation_set")
    if extra:
        reasons.append("unexpected_rows")
    if any(row.get("correct_math_verify") is None or row.get("scoring_error") for row in rows):
        reasons.append("unscorable_row")

    per_sample = {}
    for sample in (0, 1):
        sample_rows = {}
        for ident in FOCUSED_IDS:
            for arm in ARMS:
                row = actual.get((ident, sample, arm))
                if row is not None:
                    sample_rows[(ident, arm)] = row
        passes = True
        for ident in FOCUSED_IDS:
            if (ident, "T") not in sample_rows or (ident, "B") not in sample_rows:
                passes = False
                continue
        def r(ident, arm):
            return sample_rows.get((ident, arm))
        failure_ids = FOCUSED_IDS[:2]
        success_id = FOCUSED_IDS[2]
        for ident in failure_ids:
            base, target = r(ident, "B"), r(ident, "T")
            if base is None or base.get("correct_math_verify") is not False:
                passes = False
            if target is None or target.get("completed_correct") is not True:
                passes = False
        for arm in ARMS:
            row = r(success_id, arm)
            if row is None or row.get("completed_correct") is not True:
                passes = False
        for control in ("Cplus", "Cminus"):
            if not any(r(ident, control) is not None and r(ident, control).get("correct_math_verify") is False for ident in failure_ids):
                passes = False
        per_sample[str(sample)] = {
            "status": "pass" if passes else "fail",
            "target_completed_correct_all_three": all(r(ident, "T") is not None and r(ident, "T").get("completed_correct") is True for ident in FOCUSED_IDS),
            "base_failure_prefixes_formally_incorrect": all(r(ident, "B") is not None and r(ident, "B").get("correct_math_verify") is False for ident in failure_ids),
            "base_3185_completed_correct": r(success_id, "B") is not None and r(success_id, "B").get("completed_correct") is True,
            "both_controls_3185_completed_correct": all(r(success_id, arm) is not None and r(success_id, arm).get("completed_correct") is True for arm in ("Cplus", "Cminus")),
            "each_control_formally_incorrect_on_at_least_one_failure": all(any(r(ident, arm) is not None and r(ident, arm).get("correct_math_verify") is False for ident in failure_ids) for arm in ("Cplus", "Cminus")),
        }
    if missing or extra or any(row.get("scoring_error") or row.get("correct_math_verify") is None for row in rows):
        passed = False
    else:
        passed = all(item["status"] == "pass" for item in per_sample.values())
    if not passed and not reasons:
        reasons.append("preregistered_focused_gate_failed")
    payload = {
        "intervention_id": "human-astra-manual-20261105-l12c1226-direction-v1",
        "stage": "focused",
        "status": "pass_both_samples" if passed else "fail_candidate_stopped_or_incomplete",
        "expected_rows": 24,
        "observed_rows": len(rows),
        "missing_rows": [list(key) for key in missing],
        "extra_rows": [list(key) for key in extra],
        "failure_reasons": reasons,
        "per_sample_gate": per_sample,
        "outcomes_by_arm": {
            arm: {
                "correct_count": sum(row.get("correct_math_verify") is True for row in rows if row.get("arm") == arm),
                "completed_correct_count": sum(row.get("completed_correct") is True for row in rows if row.get("arm") == arm),
                "eos_count": sum(row.get("ended_with_eos") is True for row in rows if row.get("arm") == arm),
                "cap_hit_count": sum(row.get("cap_hit") is True for row in rows if row.get("arm") == arm),
                "mean_tokens": (sum(row.get("tokens", 0) for row in rows if row.get("arm") == arm) / max(1, sum(row.get("arm") == arm for row in rows))),
                "mean_repeat8_fraction": (sum(row.get("repeat8_fraction", 0.0) for row in rows if row.get("arm") == arm) / max(1, sum(row.get("arm") == arm for row in rows))),
            } for arm in ARMS
        },
        "all_pair_metrics": [metric_row(actual[key]) for key in sorted(actual)],
        "results_jsonl_sha256": source_hash,
        "runner_sha256": runner_hash,
        "confirmation_bodies_opened": False,
        "final_holdout_opened": False,
    }
    payload["gate_sha256"] = gate_digest(payload)
    return payload


def score_confirmation(rows: list[dict], source_hash: str, runner_hash: str) -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    ids = [item["id"] for item in parent["directional_intervention_v1"]["confirmation"]["items"]]
    expected = {(ident, sample, arm) for ident in ids for sample in (0, 1) for arm in ARMS}
    actual = grouped(rows)
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    reasons = []
    if missing:
        reasons.append("incomplete_32_generation_set")
    if extra:
        reasons.append("unexpected_rows")
    if any(row.get("correct_math_verify") is None or row.get("scoring_error") for row in rows):
        reasons.append("unscorable_row")

    complete = not missing and not extra and not any(row.get("correct_math_verify") is None or row.get("scoring_error") for row in rows)
    counts = {arm: sum(actual[key].get("correct_math_verify") is True for key in expected if key in actual and key[2] == arm) for arm in ARMS}
    accuracy_advantages = {arm: counts["T"] - counts[arm] for arm in ("B", "Cplus", "Cminus")}
    item_both_seed_rescue = []
    for ident in ids:
        if all(actual.get((ident, s, "B"), {}).get("correct_math_verify") is False and actual.get((ident, s, "T"), {}).get("correct_math_verify") is True for s in (0, 1)):
            item_both_seed_rescue.append(ident)
    base_to_target_accuracy_regressions = [
        [ident, sample] for ident in ids for sample in (0, 1)
        if actual.get((ident, sample, "B"), {}).get("correct_math_verify") is True
        and actual.get((ident, sample, "T"), {}).get("correct_math_verify") is False
    ]
    base_completed_to_target_incomplete = [
        [ident, sample] for ident in ids for sample in (0, 1)
        if actual.get((ident, sample, "B"), {}).get("completed_correct") is True
        and actual.get((ident, sample, "T"), {}).get("completed_correct") is not True
    ]
    target_correct_not_completed = [
        [ident, sample] for ident in ids for sample in (0, 1)
        if actual.get((ident, sample, "T"), {}).get("correct_math_verify") is True
        and actual.get((ident, sample, "T"), {}).get("completed_correct") is not True
    ]
    checks = {
        "target_exceeds_each_comparison_by_at_least_two_correct_pairs": all(value >= 2 for value in accuracy_advantages.values()),
        "at_least_one_item_both_seeds_base_wrong_target_correct": bool(item_both_seed_rescue),
        "zero_base_correct_to_target_incorrect": not base_to_target_accuracy_regressions,
        "zero_base_completed_correct_to_target_incomplete": not base_completed_to_target_incomplete,
        "all_target_correct_are_completed_correct": not target_correct_not_completed,
        "all_pairs_complete_and_scorable": complete,
    }
    passed = complete and all(checks.values())
    if not passed and not reasons:
        reasons.append("preregistered_confirmation_gate_failed")
    paired_transitions = []
    for ident in ids:
        for sample in (0, 1):
            pair = {arm: actual.get((ident, sample, arm)) for arm in ARMS}
            if all(pair[arm] is not None for arm in ARMS):
                paired_transitions.append({
                    "id": ident,
                    "sample": sample,
                    "correct_by_arm": {arm: pair[arm].get("correct_math_verify") for arm in ARMS},
                    "completed_correct_by_arm": {arm: pair[arm].get("completed_correct") for arm in ARMS},
                    "target_correct_vs": {arm: {
                        "target_correct_only": pair["T"].get("correct_math_verify") is True and pair[arm].get("correct_math_verify") is False,
                        "comparison_correct_only": pair["T"].get("correct_math_verify") is False and pair[arm].get("correct_math_verify") is True,
                    } for arm in ("B", "Cplus", "Cminus")},
                })
    outcomes_by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        outcomes_by_arm[arm] = {
            "correct_count": sum(row.get("correct_math_verify") is True for row in arm_rows),
            "completed_correct_count": sum(row.get("completed_correct") is True for row in arm_rows),
            "eos_count": sum(row.get("ended_with_eos") is True for row in arm_rows),
            "think_closed_count": sum(row.get("think_closed_in_continuation") is True for row in arm_rows),
            "cap_hit_count": sum(row.get("cap_hit") is True for row in arm_rows),
            "mean_tokens": sum(row.get("tokens", 0) for row in arm_rows) / max(1, len(arm_rows)),
            "mean_repeat8_fraction": sum(row.get("repeat8_fraction", 0.0) for row in arm_rows) / max(1, len(arm_rows)),
        }
    payload = {
        "intervention_id": "human-astra-manual-20261105-l12c1226-direction-v1",
        "stage": "confirmation",
        "status": "pass_confirmation_final_holdout_may_be_considered" if passed else "fail_or_incomplete_stop_candidate",
        "expected_rows": 32,
        "observed_rows": len(rows),
        "missing_rows": [list(key) for key in missing],
        "extra_rows": [list(key) for key in extra],
        "failure_reasons": reasons,
        "correct_counts_by_arm": counts,
        "outcomes_by_arm": outcomes_by_arm,
        "target_accuracy_advantages_vs_each_arm": accuracy_advantages,
        "acceptance_checks": checks,
        "items_with_both_seed_rescue": item_both_seed_rescue,
        "base_correct_to_target_incorrect": base_to_target_accuracy_regressions,
        "base_completed_correct_to_target_incomplete": base_completed_to_target_incomplete,
        "target_correct_but_not_completed": target_correct_not_completed,
        "paired_transitions": paired_transitions,
        "all_pair_metrics": [metric_row(actual[key]) for key in sorted(actual)],
        "results_jsonl_sha256": source_hash,
        "runner_sha256": runner_hash,
        "final_holdout_opened": False,
    }
    payload["gate_sha256"] = gate_digest(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("focused", "confirmation"), required=True)
    ap.add_argument("--attempt", type=int, required=True)
    args = ap.parse_args()
    run_dir = RAW_ROOT / f"{args.stage}_attempt_{args.attempt:03d}"
    results_path = run_dir / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    rows = read_rows(results_path)
    runner = ROOT / "scripts/run_human_astra_directional_eval.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    if args.stage == "focused":
        out = RAW / "focused_gate_v1.json"
        if out.exists():
            raise FileExistsError(f"Refuse to overwrite frozen gate record: {out}")
        report = score_focused(rows, sha256_file(results_path), sha256_file(runner))
    else:
        out = run_dir / "confirmation_gate_v1.json"
        if out.exists():
            raise FileExistsError(f"Refuse to overwrite prior gate record: {out}")
        report = score_confirmation(rows, sha256_file(results_path), sha256_file(runner))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate_sha256": report["gate_sha256"],
                      "failure_reasons": report["failure_reasons"], "report": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
