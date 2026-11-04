#!/usr/bin/env python3
"""Run the preregistered base-only discovery baseline for the human/Astra study.

Only fixed discovery IDs from the run manifest can be evaluated. It writes
complete raw traces and append-only 256-token progress records under an ignored
run-local directory. It never edits model weights or reads held-out files.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
MANIFEST_PATH = RUN / "manifest.json"
OLD = ROOT / "results/astra_reproduction_20261104"
BASE = ROOT / "model-cache/Qwen3.5-0.8B"
TENSOR_FILE = BASE / "model.safetensors-00001-of-00001.safetensors"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CAP = 32768
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_ids(ids: set[str]) -> str:
    return digest_bytes("\n".join(sorted(ids)).encode())


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def expected_seed(run_id: str, ident: str) -> int:
    material = f"{run_id}|baseline-v1|sample-0|{ident}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def validate_plan(manifest: dict) -> list[dict]:
    if manifest["data"]["split_audit_status"] != "passed_read_only_by_luna_2026-11-05":
        raise ValueError("STOP: split audit is not marked passed")
    if manifest["data"]["prompt_bodies_opened_for_this_run"]:
        raise ValueError("STOP: discovery baseline was already opened; use a new preregistered attempt")
    if manifest["editing"]["weights_modified"]:
        raise ValueError("STOP: baseline runner must not be used after any weight edit")
    settings = manifest["discovery_baseline"]["settings"]
    if settings["max_new_tokens"] != CAP or settings["instruction"] != INSTRUCTION:
        raise ValueError("STOP: baseline runner does not match frozen cap/instruction")
    if settings["model_arm"] != "original checkpoint only" or settings["thinking"] is not True:
        raise ValueError("STOP: baseline runner does not match frozen model arm/thinking mode")
    for key in ("temperature", "top_p", "top_k", "presence_penalty"):
        if settings[key] != SAMPLING[key]:
            raise ValueError(f"STOP: baseline runner does not match frozen sampling setting {key}")

    run_id = manifest["run_id"]
    partitions = json.loads((OLD / "raw/editor_work/train_partitions.json").read_text(encoding="utf-8"))
    reserved = set().union(*(set(values) for values in partitions.values()))
    if len(reserved) != manifest["data"]["previously_reserved_ids_to_exclude"]:
        raise ValueError("STOP: reserved-ID count differs from run manifest")
    planned = [item for item in manifest["discovery_baseline"]["items"] if item["stage"] == "initial"]
    if len(planned) != 4:
        raise ValueError("STOP: exactly four initial baseline items must be frozen")

    rows: dict[str, dict] = {}
    all_source_ids: set[str] = set()
    for source in ("gsm8k_train.jsonl", "math_train.jsonl"):
        path = OLD / "raw/train" / source
        expected_source_hash = manifest["data"]["source_files_sha256_expected"][source]
        if digest_file(path) != expected_source_hash:
            raise ValueError(f"STOP: source TRAIN slice hash mismatch: {source}")
        source_rows = list(read_jsonl(path))
        ids = {row["id"] for row in source_rows}
        if len(ids) != len(source_rows):
            raise ValueError(f"STOP: duplicate IDs in {source}")
        all_source_ids.update(ids)
        eligible = ids - reserved
        ranked = sorted(
            eligible,
            key=lambda ident: hashlib.sha256(f"{run_id}|{source}|{ident}".encode()).digest(),
        )
        n = len(ranked) // 10
        discovery_ids = ranked[3 * n :]
        baseline_ranked = sorted(
            discovery_ids,
            key=lambda ident: hashlib.sha256(
                f"{run_id}|baseline-v1|{source}|{ident}".encode()
            ).digest(),
        )
        source_planned = [item["id"] for item in planned if item["source"] == source]
        if source_planned != baseline_ranked[:2]:
            raise ValueError(f"STOP: frozen initial IDs do not match baseline-v1 selection rule for {source}")
        for item in source_rows:
            if item["id"] in source_planned:
                if item["id"] in reserved:
                    raise ValueError(f"STOP: reserved item selected: {item['id']}")
                rows[item["id"]] = item

    if len(all_source_ids) != manifest["data"]["source_ids_expected"]:
        raise ValueError("STOP: prior TRAIN source ID total differs from run manifest")
    if digest_ids(all_source_ids) != manifest["data"]["split_counts_and_hashes"]["prior_train_ids_sha256"]:
        raise ValueError("STOP: prior TRAIN aggregate ID hash differs from run manifest")
    if not reserved <= all_source_ids:
        raise ValueError("STOP: some reserved IDs are missing from prior TRAIN source")

    ordered_plan = sorted(planned, key=lambda item: item["order"])
    for item in ordered_plan:
        ident = item["id"]
        if ident not in rows:
            raise ValueError(f"STOP: frozen discovery item missing from TRAIN slices: {ident}")
        seed = expected_seed(run_id, ident)
        if item["seed"] != seed:
            raise ValueError(f"STOP: seed mismatch for {ident}: expected {seed}")
    return [{**item, "row": rows[item["id"]]} for item in ordered_plan]


def repeat8_fraction(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 8]) for i in range(max(0, len(ids) - 7))]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def tokenizer_hashes(path: Path) -> dict[str, str]:
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
    result = {}
    for file in sorted(path.iterdir()):
        if file.is_file() and file.name in names:
            result[file.name] = digest_file(file)
    return result


def completed_ids(root: Path) -> set[str]:
    found: set[str] = set()
    for prior in sorted(root.glob("attempt_*/original.jsonl")):
        for record in read_jsonl(prior):
            if record.get("run_id") != "human-astra-manual-20261105-v1":
                raise ValueError(f"Unexpected run ID in prior output: {prior}")
            ident = record.get("id")
            if not isinstance(ident, str) or ident in found:
                raise ValueError(f"Missing or duplicate completed ID in prior outputs: {prior}")
            found.add(ident)
    return found


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1, help="append-only attempt number")
    args = ap.parse_args()
    if args.attempt < 1:
        raise ValueError("attempt must be >= 1")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    plan = validate_plan(manifest)
    if not TENSOR_FILE.is_file():
        raise FileNotFoundError(f"Base checkpoint is missing: {TENSOR_FILE}")

    expected_base = manifest["base_model"]["expected_sha256_from_prior_manifest"]
    base_hash = digest_file(TENSOR_FILE)
    if base_hash != expected_base:
        raise ValueError(f"STOP: base checkpoint hash mismatch: {base_hash}")

    base_out = RUN / "raw/discovery_baseline_v1"
    done = completed_ids(base_out) if base_out.exists() else set()
    todo = [item for item in plan if item["id"] not in done]
    if not todo:
        raise ValueError("All frozen initial discovery items already have completed outputs")

    attempt_name = f"attempt_{args.attempt:03d}"
    outdir = base_out / attempt_name
    if outdir.exists():
        raise FileExistsError(f"Refuse to overwrite existing attempt directory: {outdir}")
    outdir.mkdir(parents=True)
    progress_dir = outdir / "progress"
    progress_dir.mkdir()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps({"status": "loading_local_model", "base_sha256": base_hash,
                      "attempt": attempt_name, "items_to_run": [item["id"] for item in todo]}), flush=True)
    import torch
    import transformers
    from transformers import AutoTokenizer, LogitsProcessor, Qwen3_5ForConditionalGeneration, StoppingCriteria
    from astra_clean_grading import extract_answer
    try:
        from math_verify import parse, verify
        scorer_available = True
        scorer_import_error = None
    except ImportError as exc:
        parse = verify = None
        scorer_available = False
        scorer_import_error = f"{type(exc).__name__}: {exc}"

    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to("mps")
    model.generation_config.use_cache = True
    tokenizer_files = tokenizer_hashes(BASE)
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "mps",
        "dtype": "bfloat16",
        "attention": "eager",
        "base_sha256": base_hash,
        "runner_sha256": digest_file(Path(__file__)),
        "tokenizer_file_sha256": tokenizer_files,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": digest_bytes((tokenizer.chat_template or "").encode()),
        "math_verify_scorer_available": scorer_available,
        "math_verify_import_error": scorer_import_error,
    }

    class PresencePenalty(LogitsProcessor):
        def __init__(self, prompt_length: int, amount: float):
            self.prompt_length = prompt_length
            self.amount = amount

        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.prompt_length :]
            if generated.numel():
                scores[0, torch.unique(generated[0])] -= self.amount
            return scores

    class Progress(StoppingCriteria):
        def __init__(self, prompt_length: int, progress_path: Path, ident: str, cap: int):
            self.prompt_length = prompt_length
            self.progress_path = progress_path
            self.ident = ident
            self.cap = cap
            self.started = time.monotonic()
            self.last_checkpoint = self.started
            self.next_checkpoint = 256

        def __call__(self, input_ids, scores, **kwargs):
            count = int(input_ids.shape[1] - self.prompt_length)
            if count >= self.next_checkpoint:
                now = time.monotonic()
                generated = input_ids[0, self.prompt_length :].detach().cpu().tolist()
                interval = now - self.last_checkpoint
                rate = (count - (self.next_checkpoint - 256)) / max(interval, 1e-9)
                remaining = max(0, self.cap - count)
                append_jsonl(self.progress_path, {
                    "id": self.ident,
                    "generated_tokens": count,
                    "elapsed_seconds": round(now - self.started, 3),
                    "interval_seconds": round(interval, 3),
                    "tokens_per_second": round(rate, 3),
                    "eta_to_cap_seconds_at_recent_rate": round(remaining / rate, 1) if rate > 0 else None,
                    "generated_ids_prefix_sha256": digest_bytes(
                        bytes().join(int(token).to_bytes(4, "big") for token in generated)
                    ),
                })
                self.last_checkpoint = now
                self.next_checkpoint = ((count // 256) + 1) * 256
                print(json.dumps({
                    "progress": self.ident,
                    "tokens": count,
                    "rate_tokens_per_second": round(rate, 2),
                    "eta_to_cap_seconds": round(remaining / rate, 1) if rate > 0 else None,
                }), flush=True)
            return False

    answers_path = outdir / "original.jsonl"
    append_jsonl(outdir / "attempt_manifest.jsonl", {
        "run_id": manifest["run_id"],
        "attempt": attempt_name,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items": [{"id": x["id"], "seed": x["seed"], "stage": x["stage"]} for x in todo],
        "sampling": SAMPLING,
        "max_new_tokens": CAP,
        "instruction": INSTRUCTION,
        "runtime": runtime,
    })
    total_start = time.monotonic()
    prior_measured_seconds = 4692.3
    print(json.dumps({
        "run_id": manifest["run_id"],
        "items": [x["id"] for x in plan],
        "historical_four_item_reference_seconds": prior_measured_seconds,
        "historical_all_four_capped_reference_seconds": 4 * 2677.3,
        "note": "reference only; update ETA from observed token progress",
    }), flush=True)

    for plan_item in todo:
        ident = plan_item["id"]
        row = plan_item["row"]
        seed = plan_item["seed"]
        prompt = row["problem"] + "\n" + INSTRUCTION
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to("mps")
        prompt_length = int(input_ids.shape[-1])
        prompt_rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        progress_path = progress_dir / f"{ident}.jsonl"
        progress = Progress(prompt_length, progress_path, ident, CAP)
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=CAP,
                do_sample=True,
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                top_k=SAMPLING["top_k"],
                logits_processor=[PresencePenalty(prompt_length, SAMPLING["presence_penalty"])],
                stopping_criteria=[progress],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
        seconds = time.monotonic() - start
        generated_ids = output[0, prompt_length:].detach().cpu().tolist()
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        prediction = extract_answer(text)
        if not scorer_available:
            correct = None
            grading_error = scorer_import_error
        else:
            try:
                correct = bool(verify(parse("\\boxed{" + str(row["answer"]) + "}"),
                                      parse("\\boxed{" + str(prediction) + "}"))) if prediction is not None else False
                grading_error = None
            except Exception as exc:  # retain scorer/parser errors as explicit evidence
                correct = False
                grading_error = f"{type(exc).__name__}: {exc}"
        eos_id = tokenizer.eos_token_id
        ended_with_eos = bool(generated_ids and generated_ids[-1] == eos_id)
        cap_hit = len(generated_ids) >= CAP
        tail = generated_ids[-2048:]
        window_metrics = [
            {"start": start_idx, "end_exclusive": min(start_idx + 256, len(generated_ids)),
             "repeat8_fraction": repeat8_fraction(generated_ids[start_idx : start_idx + 256])}
            for start_idx in range(0, len(generated_ids), 256)
        ]
        record = {
            "run_id": manifest["run_id"],
            "attempt": attempt_name,
            "id": ident,
            "source": plan_item["source"],
            "sample": 0,
            "seed": seed,
            "model": "original",
            "base_sha256": base_hash,
            "prompt": prompt,
            "rendered_prompt": prompt_rendered,
            "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
            "prompt_token_ids_sha256": digest_bytes(
                bytes().join(int(token).to_bytes(4, "big") for token in input_ids[0].detach().cpu().tolist())
            ),
            "text": text,
            "generated_token_ids": generated_ids,
            "generated_token_ids_sha256": digest_bytes(
                bytes().join(int(token).to_bytes(4, "big") for token in generated_ids)
            ),
            "prediction_last_boxed": prediction,
            "gold": row["answer"],
            "correct_math_verify": correct,
            "math_verify_scorer_available": scorer_available,
            "format_valid_box": prediction is not None,
            "grading_error": grading_error,
            "ended_with_eos": ended_with_eos,
            "think_closed": "</think>" in text,
            "cap_hit": cap_hit,
            "strict_completed_success": (
                bool(correct and prediction is not None and ended_with_eos and "</think>" in text and not cap_hit)
                if correct is not None else None
            ),
            "tokens": len(generated_ids),
            "repeat8_fraction": repeat8_fraction(generated_ids),
            "tail_2048_repeat8_fraction": repeat8_fraction(tail),
            "repeat8_by_256_token_window": window_metrics,
            "seconds": round(seconds, 3),
            "runtime": runtime,
        }
        append_jsonl(answers_path, record)
        print(json.dumps({k: record[k] for k in (
            "id", "seed", "tokens", "correct_math_verify", "ended_with_eos",
            "think_closed", "cap_hit", "repeat8_fraction", "seconds"
        )}), flush=True)

    print(json.dumps({
        "completed_this_attempt": len(todo),
        "completed_before_attempt": len(done),
        "elapsed_seconds": round(time.monotonic() - total_start, 1),
        "raw_output": str(answers_path),
        "progress_dir": str(progress_dir),
        "note": "raw prompts, answers and token IDs are local-only; weights unchanged",
    }), flush=True)


if __name__ == "__main__":
    main()
