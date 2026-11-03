#!/usr/bin/env python3
"""Export final answers and termination metadata, excluding token-level traces."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.out.exists() and not a.force:
        raise SystemExit(f"refusing to overwrite {a.out}; choose a new path or pass --force")
    rows = [json.loads(x) for x in a.raw.read_text().splitlines() if x.strip()]
    if len(rows) != 48:
        raise SystemExit(f"expected 48 paired generations, got {len(rows)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            answer = r["final_answer"].strip()
            for suffix in ("<|im_end|>", "<|end_of_turn|>", "<|fim_suffix|>"):
                if answer.endswith(suffix):
                    answer = answer[: -len(suffix)].rstrip()
                    break
            record = {k: r[k] for k in ("id", "family", "condition", "seed", "generated_tokens", "cap_hit", "ended_with_eos")}
            record["eos_before_cap"] = bool(r["ended_with_eos"] and not r["cap_hit"])
            record["final_answer"] = answer
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = json.loads(a.manifest.read_text())
    if manifest.get("raw_output_sha256") != sha(a.raw):
        raise SystemExit("raw generation hash does not match manifest")
    manifest["sanitized_final_answers_sha256"] = sha(a.out)
    manifest["sanitized_answers_exclude"] = ["generated_token_ids", "input_token_ids", "full_output", "thinking text"]
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported 48 sanitized answers; sha256={manifest['sanitized_final_answers_sha256']}")


if __name__ == "__main__":
    main()
