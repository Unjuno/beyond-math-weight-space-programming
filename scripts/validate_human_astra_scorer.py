#!/usr/bin/env python3
"""Validate and hash the isolated, pinned math-verify scorer for this study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
VENDOR = RUN / "raw/scorer_vendor/math_verify_0_9_0"
EXPECTED = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUN / "raw/scorer_validation_v1/validation.json")
    args = ap.parse_args()
    if not VENDOR.is_dir():
        raise FileNotFoundError(f"Isolated scorer vendor directory missing: {VENDOR}")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite prior validation: {out}")
    sys.path.insert(0, str(VENDOR))

    import math_verify
    from math_verify import parse, verify

    # This script lives in scripts/, so the frozen local extraction helper is importable.
    from astra_clean_grading import extract_answer

    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"Scorer package version mismatch: {versions}")

    gold = r"\boxed{770}"
    correct = r"\boxed{770}"
    incorrect = r"\boxed{771}"
    missing_box = "770"
    checks = {
        "correct_last_box_extraction": extract_answer(correct) == "770",
        "known_correct_pair": bool(verify(parse(gold), parse(correct))) is True,
        "known_incorrect_pair": bool(verify(parse(gold), parse(incorrect))) is False,
        "missing_box_is_unparsed": extract_answer(missing_box) is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Scorer sanity check failed: {checks}")

    module_dir = Path(math_verify.__file__).resolve().parent
    module_hashes = {
        str(path.relative_to(module_dir)): digest(path)
        for path in sorted(module_dir.rglob("*.py"))
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_run_id": "human-astra-manual-20261105-v1",
        "intervention_id": "human-astra-manual-20261105-l12c1226-direction-v1",
        "scorer_import_path": str(Path(math_verify.__file__).resolve()),
        "isolated_vendor_path": str(VENDOR),
        "versions": versions,
        "version_check": "passed",
        "sanity_checks": checks,
        "math_verify_module_sha256": module_hashes,
        "validation_script_sha256": digest(Path(__file__).resolve()),
        "shared_environment_modified": False,
        "task_data_opened": False,
        "weights_modified": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
