#!/usr/bin/env python3
"""Run frozen human/Astra directional arms; preserve all raw outputs locally.

Each arm loads an independent copy of the immutable base checkpoint and applies
exactly one registered BF16 column in memory. The base safetensors are never
written. Confirmation item text is accessed only after the focused TRAIN gate
has passed and is frozen in an append-only result file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
OLD = ROOT / "results/astra_reproduction_20261104"
PARENT_MANIFEST = RUN / "manifest.json"
BASE_DIR = ROOT / "model-cache/Qwen3.5-0.8B"
BASE_TENSOR = BASE_DIR / "model.safetensors-00001-of-00001.safetensors"
VENDOR = RUN / "raw/scorer_vendor/math_verify_0_9_0"
VALIDATION = RUN / "raw/scorer_validation_v1/validation.json"
ARM_DIR = RUN / "raw/directional_control_v1/columns_v1"
ARM_MANIFEST = ARM_DIR / "manifest.json"
RAW_ROOT = RUN / "raw/directional_control_v1"
INTERVENTION_ID = "human-astra-manual-20261105-l12c1226-direction-v1"
TENSOR_KEY = "model.language_model.layers.12.mlp.down_proj.weight"
COORDINATE = 1226
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
FOCUSED_CAP = 4096
CONFIRMATION_CAP = 32768
SOURCES = ("gsm8k_train.jsonl", "math_train.jsonl")
FOCUSED_IDS = (
    "gsm8k_train_6445",
    "math_train_algebra_542",
    "gsm8k_train_3185",
)
FOCUSED_PREFIX_LENGTHS = {
    "gsm8k_train_6445": 4341,
    "math_train_algebra_542": 3031,
    "gsm8k_train_3185": 2944,
}
FOCUSED_SEEDS = {
    "gsm8k_train_6445": [1084327040, 3346301434],
    "math_train_algebra_542": [1677174321, 3447544442],
    "gsm8k_train_3185": [37864219, 2080160158],
}
FOCUSED_ORDER = [
    (0, "gsm8k_train_6445", ["B", "T", "Cplus", "Cminus"]),
    (0, "math_train_algebra_542", ["T", "Cplus", "Cminus", "B"]),
    (0, "gsm8k_train_3185", ["Cplus", "Cminus", "B", "T"]),
    (1, "gsm8k_train_6445", ["Cminus", "B", "T", "Cplus"]),
    (1, "math_train_algebra_542", ["B", "T", "Cplus", "Cminus"]),
    (1, "gsm8k_train_3185", ["T", "Cplus", "Cminus", "B"]),
]
CONFIRMATION_ROWS = [
    ("gsm8k_train.jsonl", "gsm8k_train_5341", [2517555310, 2890724130]),
    ("gsm8k_train.jsonl", "gsm8k_train_1178", [8746557, 3600332144]),
    ("math_train.jsonl", "math_train_precalculus_207", [864367089, 2034957992]),
    ("math_train.jsonl", "math_train_intermediate_algebra_231", [4152760572, 3381638749]),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_ids_hash(ids: list[int]) -> str:
    return sha256_bytes(b"".join(int(token).to_bytes(4, "big") for token in ids))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def repeat8_fraction(ids: list[int]) -> float:
    if len(ids) < 8:
        return 0.0
    seen = set()
    repeats = 0
    for i in range(len(ids) - 7):
        gram = tuple(ids[i : i + 8])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / (len(ids) - 7)


def frozen_scorer():
    if not VALIDATION.is_file():
        raise RuntimeError("STOP: scorer validation record is missing")
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if report.get("versions", {}).get("math-verify") != "0.9.0" or not all(report.get("sanity_checks", {}).values()):
        raise RuntimeError("STOP: isolated scorer version/sanity validation failed")
    if report.get("shared_environment_modified") is not False:
        raise RuntimeError("STOP: scorer provenance says shared environment changed")
    sys.path.insert(0, str(VENDOR))
    from math_verify import parse, verify
    from astra_clean_grading import extract_answer
    return parse, verify, extract_answer


def validate_frozen_design(stage: str, args) -> tuple[dict, dict, list[dict]]:
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    design = parent["directional_intervention_v1"]
    if parent["data"]["split_audit_status"] != "passed_read_only_by_luna_2026-11-05":
        raise RuntimeError("STOP: parent split audit not passed")
    if parent["base_model"]["verified_sha256"] != design["candidate"]["base_sha256"]:
        raise RuntimeError("STOP: candidate/base checkpoint hash mismatch in manifest")
    settings = parent["discovery_baseline"]["settings"]
    for key, expected in {"thinking": True, "temperature": 0.6, "top_p": 0.95,
                          "top_k": 20, "presence_penalty": 1.5,
                          "precision": "bfloat16", "attention_implementation": "eager",
                          "instruction": INSTRUCTION}.items():
        if settings.get(key) != expected:
            raise RuntimeError(f"STOP: parent-frozen setting mismatch for {key}")
    expected_base = parent["base_model"]["expected_sha256_from_prior_manifest"]
    if sha256_file(BASE_TENSOR) != expected_base:
        raise RuntimeError("STOP: live base checkpoint hash differs from registered base")
    if not ARM_MANIFEST.is_file():
        raise RuntimeError("STOP: human has not prepared the registered BF16 arm columns")
    arm_manifest = json.loads(ARM_MANIFEST.read_text(encoding="utf-8"))
    expected_arm_manifest_hash = design.get("column_manifest_sha256")
    actual_arm_manifest_hash = sha256_file(ARM_MANIFEST)
    if expected_arm_manifest_hash != actual_arm_manifest_hash:
        raise RuntimeError("STOP: prepared arm manifest is not frozen in parent manifest")
    if arm_manifest.get("base_checkpoint_sha256") != expected_base or arm_manifest.get("base_checkpoint_modified") is not False:
        raise RuntimeError("STOP: arm artifacts do not attest to the immutable base")
    if arm_manifest.get("status") != "prepared_columns_only_not_applied_to_model":
        raise RuntimeError("STOP: unexpected preparation artifact status")
    scorer_record = design.get("scorer", {})
    if scorer_record.get("status") != "isolated_pin_validated":
        raise RuntimeError("STOP: isolated scorer pin is not registered as validated")
    if scorer_record.get("raw_validation_sha256") != sha256_file(VALIDATION):
        raise RuntimeError("STOP: scorer validation hash does not match the frozen manifest")
    expected_confirmation = [(row["source"], row["id"], row["seeds"]) for row in design["confirmation"]["items"]]
    if expected_confirmation != CONFIRMATION_ROWS:
        raise RuntimeError("STOP: evaluator confirmation IDs/seeds differ from registration")

    if stage == "confirmation":
        gate_path = RAW_ROOT / "focused_gate_v1.json"
        if not gate_path.is_file():
            raise RuntimeError("STOP: focused TRAIN gate result missing; confirmation items remain unopened")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "pass_both_samples" or gate.get("gate_sha256") != design.get("focused_gate_sha256"):
            raise RuntimeError("STOP: focused gate is not frozen/passed")

    scorer = frozen_scorer()
    return parent, arm_manifest, list(scorer)


def load_focus_records() -> dict[str, dict]:
    pinned_hashes = {
        "attempt_002": "9adcc018de80b35569968d1564bfd73027e539c2d829417d480afe61ae59ebe9",
        "attempt_003": "6bbfd351926090da3bac763b0674fd8cdf2cd84c0fb9b0c2a6cfb45306354733",
    }
    for attempt, expected_hash in pinned_hashes.items():
        path = RUN / f"raw/discovery_baseline_v1/{attempt}/original.jsonl"
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"STOP: discovery raw trace hash mismatch: {attempt}")
    paths = {
        "gsm8k_train_6445": RUN / "raw/discovery_baseline_v1/attempt_002/original.jsonl",
        "math_train_algebra_542": RUN / "raw/discovery_baseline_v1/attempt_002/original.jsonl",
        "gsm8k_train_3185": RUN / "raw/discovery_baseline_v1/attempt_003/original.jsonl",
    }
    found = {}
    for ident, path in paths.items():
        rows = [row for row in read_jsonl(path) if row.get("id") == ident and row.get("base_sha256") == "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"]
        if len(rows) != 1:
            raise RuntimeError(f"STOP: expected one original base trace for {ident}, found {len(rows)}")
        row = rows[0]
        expected_prefix = FOCUSED_PREFIX_LENGTHS[ident]
        if len(row["generated_token_ids"]) < expected_prefix or not row.get("prompt_token_ids"):
            raise RuntimeError(f"STOP: baseline trace too short or missing prompt IDs: {ident}")
        found[ident] = row
    return found


def load_confirmation_rows(parent: dict) -> list[dict]:
    """Only called after the confirmation-stage focused gate check."""
    run_id = parent["run_id"]
    old = json.loads((OLD / "raw/editor_work/train_partitions.json").read_text(encoding="utf-8"))
    reserved = set().union(*(set(values) for values in old.values()))
    result = []
    selected_by_source = {}
    for source, ident, seeds in CONFIRMATION_ROWS:
        selected_by_source.setdefault(source, []).append((ident, seeds))
    source_cache = {}
    for source, selected in selected_by_source.items():
        if source not in source_cache:
            expected_source_hash = parent["data"]["source_files_sha256_expected"][source]
            source_path = OLD / "raw/train" / source
            digest = hashlib.sha256()
            ids = []
            id_pattern = re.compile(rb'^\s*\{\s*"id"\s*:\s*"([^"\\]+)"')
            with source_path.open("rb") as stream:
                for line_no, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    match = id_pattern.match(raw_line)
                    if match is None:
                        raise RuntimeError(f"STOP: cannot extract ID-only JSONL key at {source}:{line_no}")
                    ids.append(match.group(1).decode("utf-8"))
            if digest.hexdigest() != expected_source_hash:
                raise RuntimeError(f"STOP: source hash mismatch: {source}")
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"STOP: duplicate source IDs: {source}")
            source_cache[source] = {"ids": set(ids), "path": source_path}
        ids = source_cache[source]["ids"]
        eligible = ids - reserved
        ordered = sorted(eligible, key=lambda x: hashlib.sha256(f"{run_id}|{source}|{x}".encode()).digest())
        n = len(ordered) // 10
        confirmation = set(ordered[n : 2 * n])
        source_selection = sorted(
            confirmation,
            key=lambda x: (hashlib.sha256(f"{INTERVENTION_ID}|confirmation-selection-v1|{source}|{x}".encode()).digest(), x),
        )[:2]
        registered_selection = [entry[1] for entry in CONFIRMATION_ROWS if entry[0] == source]
        if source_selection != registered_selection:
            raise RuntimeError(f"STOP: deterministic confirmation-ID selection mismatch for {source}")
        if not set(registered_selection) <= confirmation:
            raise RuntimeError(f"STOP: registered confirmation ID outside its split: {source}")
        selected_set = set(registered_selection)
        found = {}
        # Only the four registered rows are decoded into full task/answer objects.
        with source_cache[source]["path"].open("rb") as stream:
            id_pattern = re.compile(rb'^\s*\{\s*"id"\s*:\s*"([^"\\]+)"')
            for line_no, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                match = id_pattern.match(raw_line)
                if match is None:
                    raise RuntimeError(f"STOP: cannot extract ID-only JSONL key at {source}:{line_no}")
                ident = match.group(1).decode("utf-8")
                if ident in selected_set:
                    row = json.loads(raw_line)
                    if row.get("id") != ident or ident in found:
                        raise RuntimeError(f"STOP: malformed/duplicate selected row: {ident}")
                    found[ident] = row
        for ident, seeds in selected:
            if ident not in found:
                raise RuntimeError(f"STOP: registered confirmation body not found: {ident}")
            result.append({"source": source, "id": ident, "seeds": seeds, "row": found[ident]})
    return result


def verify_seed_rule(stage: str) -> None:
    for sample in (0, 1):
        for ident, expected in FOCUSED_SEEDS.items():
            material = f"{INTERVENTION_ID}|focused|{ident}|sample={sample}".encode()
            actual = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
            if actual != expected[sample]:
                raise RuntimeError(f"STOP: focused seed mismatch for {ident} sample {sample}")
    for _, ident, seeds in CONFIRMATION_ROWS:
        for sample, expected in enumerate(seeds):
            material = f"{INTERVENTION_ID}|confirmation|{ident}|sample={sample}".encode()
            actual = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
            if actual != expected:
                raise RuntimeError(f"STOP: confirmation seed mismatch for {ident} sample {sample}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("focused", "confirmation"), required=True)
    ap.add_argument("--attempt", type=int, required=True)
    args = ap.parse_args()
    if args.attempt < 1:
        raise ValueError("--attempt must be >= 1")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("STOP: this study is frozen to its Python 3.11 MPS runtime")

    parent, arm_manifest, scorer = validate_frozen_design(args.stage, args)
    verify_seed_rule(args.stage)
    parse, verify, extract_answer = scorer

    # Complete runtime preflight before confirmation item bodies can be decoded.
    torch = __import__("torch")
    import transformers
    from importlib.metadata import version
    from safetensors import safe_open
    from transformers import AutoTokenizer, LogitsProcessor, Qwen3_5ForConditionalGeneration, StoppingCriteria
    from astra_clean_grading import extract_answer as _extract_answer
    if _extract_answer is not extract_answer:
        raise RuntimeError("STOP: scorer/parser import mismatch")
    if torch.__version__ != "2.14.1" or transformers.__version__ != "5.20.2" or version("safetensors") != "0.8.0":
        raise RuntimeError("STOP: runtime package versions differ from the frozen MPS environment")
    if not torch.backends.mps.is_available():
        raise RuntimeError("STOP: MPS is unavailable")

    if args.stage == "focused":
        tasks = load_focus_records()
        schedule = [("focused", sample, ident, order, tasks[ident]) for sample, ident, order in FOCUSED_ORDER]
        cap = FOCUSED_CAP
    else:
        # validate_frozen_design checked the frozen focused gate before this reads any item body.
        confirmation = load_confirmation_rows(parent)
        schedule = []
        order_cycle = [
            ["B", "T", "Cplus", "Cminus"],
            ["T", "Cplus", "Cminus", "B"],
            ["Cplus", "Cminus", "B", "T"],
            ["Cminus", "B", "T", "Cplus"],
        ]
        for row_index, task in enumerate(confirmation):
            for sample in (0, 1):
                order = order_cycle[(row_index * 2 + sample) % len(order_cycle)]
                schedule.append(("confirmation", sample, task["id"], order, task))
        cap = CONFIRMATION_CAP

    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
    output_dir = RAW_ROOT / f"{args.stage}_attempt_{args.attempt:03d}"
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    progress_dir = output_dir / "progress"
    progress_dir.mkdir()
    ledger = output_dir / "events.jsonl"
    results = output_dir / "results.jsonl"
    manifest = {
        "intervention_id": INTERVENTION_ID,
        "stage": args.stage,
        "attempt": args.attempt,
        "base_checkpoint_sha256": sha256_file(BASE_TENSOR),
        "column_manifest_sha256": sha256_file(ARM_MANIFEST),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "tokenizer_files": {
            path.name: sha256_file(path)
            for path in sorted(BASE_DIR.glob("tokenizer*")) if path.is_file()
        },
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": "mps",
        "dtype": "bfloat16",
        "sampling": SAMPLING,
        "max_new_tokens": cap,
        "instruction": INSTRUCTION,
        "task_count": len(schedule),
        "schedule": [{"sample": s, "id": ident, "arm_order": order} for _, s, ident, order, _ in schedule],
    }
    (output_dir / "attempt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    class PresencePenalty(LogitsProcessor):
        def __init__(self, penalty_start: int, amount: float):
            self.penalty_start = penalty_start
            self.amount = amount
        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.penalty_start :]
            if generated.numel():
                scores[0, torch.unique(generated[0])] -= self.amount
            return scores

    class Progress(StoppingCriteria):
        def __init__(self, path: Path, prompt_length: int, item_id: str, arm: str, sample: int):
            self.path, self.prompt_length, self.item_id, self.arm, self.sample = path, prompt_length, item_id, arm, sample
            self.started = time.monotonic()
            self.last = self.started
            self.next_tokens = 256
        def __call__(self, input_ids, scores, **kwargs):
            count = int(input_ids.shape[1] - self.prompt_length)
            if count >= self.next_tokens:
                now = time.monotonic()
                new_ids = input_ids[0, self.prompt_length :].detach().cpu().tolist()
                record = {
                    "id": self.item_id, "sample": self.sample, "arm": self.arm,
                    "generated_tokens": count, "elapsed_seconds": round(now - self.started, 3),
                    "interval_seconds": round(now - self.last, 3),
                    "generated_token_ids_prefix": new_ids,
                    "generated_token_ids_prefix_sha256": token_ids_hash(new_ids),
                }
                append_jsonl(self.path, record)
                self.last = now
                self.next_tokens = ((count // 256) + 1) * 256
                print(json.dumps({"progress": self.item_id, "sample": self.sample, "arm": self.arm,
                                  "tokens": count, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            return False

    def apply_arm(model, arm: str) -> dict:
        param_map = dict(model.named_parameters())
        if TENSOR_KEY not in param_map:
            raise RuntimeError(f"STOP: model parameter not found: {TENSOR_KEY}")
        parameter = param_map[TENSOR_KEY]
        if parameter.dtype != torch.bfloat16 or list(parameter.shape) != [1024, 3584]:
            raise RuntimeError("STOP: loaded model tensor shape/dtype mismatch")
        with safe_open(str(BASE_TENSOR), framework="pt", device="cpu") as base_file:
            registered_base = base_file.get_tensor(TENSOR_KEY)[:, COORDINATE].contiguous()
        before = parameter.detach().cpu().contiguous()
        if not torch.equal(before[:, COORDINATE], registered_base):
            raise RuntimeError("STOP: loaded base tensor differs from base safetensors")
        before_bytes = before.view(torch.uint16).numpy().tobytes()
        column_hash = sha256_bytes(before[:, COORDINATE].contiguous().view(torch.uint16).numpy().tobytes())
        if column_hash != arm_manifest["base_column_bitpattern_sha256"]:
            raise RuntimeError("STOP: loaded base column hash differs from prepared arms")
        if arm == "B":
            replacement = before[:, COORDINATE].clone()
        else:
            arm_file = ARM_DIR / f"{arm}.safetensors"
            if sha256_file(arm_file) != arm_manifest["arms"][arm]["file_sha256"]:
                raise RuntimeError(f"STOP: column file hash mismatch for {arm}")
            with safe_open(str(arm_file), framework="pt", device="cpu") as saved:
                replacement = saved.get_tensor("column")
            if replacement.dtype != torch.bfloat16 or list(replacement.shape) != [1024]:
                raise RuntimeError(f"STOP: prepared column tensor invalid for {arm}")
            with torch.no_grad():
                parameter[:, COORDINATE].copy_(replacement.to(parameter.device))
        after = parameter.detach().cpu().contiguous()
        restored = after.clone()
        restored[:, COORDINATE] = before[:, COORDINATE]
        if not torch.equal(restored, before):
            raise RuntimeError("STOP: unintended parameter elements changed")
        changed_column_hash = sha256_bytes(after[:, COORDINATE].contiguous().view(torch.uint16).numpy().tobytes())
        expected_column_hash = (
            arm_manifest["base_column_bitpattern_sha256"]
            if arm == "B" else arm_manifest["arms"][arm]["column_bitpattern_sha256"]
        )
        if changed_column_hash != expected_column_hash:
            raise RuntimeError("STOP: applied column hash differs from the registered variant")
        return {
            "arm": arm,
            "base_parameter_sha256": sha256_bytes(before_bytes),
            "edited_parameter_sha256": sha256_bytes(after.view(torch.uint16).numpy().tobytes()),
            "base_column_sha256": column_hash,
            "edited_column_sha256": changed_column_hash,
            "unexpected_parameter_changes": 0,
            "full_checkpoint_written": False,
        }

    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    total_started = time.monotonic()
    completed = 0
    for stage_name, sample, ident, order, task_obj in schedule:
        if args.stage == "focused":
            row = task_obj
            prompt_ids = list(row["prompt_token_ids"])
            prefix_ids = list(row["generated_token_ids"][: FOCUSED_PREFIX_LENGTHS[ident]])
            input_token_ids = prompt_ids + prefix_ids
            penalty_start = len(prompt_ids)
            gold = row["gold"]
            source = row["source"]
        else:
            row = task_obj["row"]
            messages = [{"role": "user", "content": row["problem"] + "\n" + INSTRUCTION}]
            enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=True, return_dict=True, return_tensors="pt")
            input_token_ids = enc["input_ids"][0].tolist()
            penalty_start = len(input_token_ids)
            gold = row["answer"]
            source = task_obj["source"]
        if not input_token_ids:
            raise RuntimeError(f"STOP: empty input token sequence for {ident}")

        for arm in order:
            seed = FOCUSED_SEEDS[ident][sample] if args.stage == "focused" else task_obj["seeds"][sample]
            event = {"event": "generation_started", "id": ident, "source": source,
                     "sample": sample, "arm": arm, "seed": seed, "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            append_jsonl(ledger, event)
            print(json.dumps(event), flush=True)
            model = None
            load_started = time.monotonic()
            try:
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    BASE_DIR, local_files_only=True, dtype=torch.bfloat16,
                    attn_implementation="eager",
                ).eval().to("mps")
                model_load_seconds = time.monotonic() - load_started
                model.generation_config.use_cache = True
                edit_started = time.monotonic()
                applied = apply_arm(model, arm)
                edit_verify_seconds = time.monotonic() - edit_started
                input_ids = torch.tensor([input_token_ids], dtype=torch.long, device="mps")
                prompt_length = len(input_token_ids)
                torch.manual_seed(seed)
                torch.mps.manual_seed(seed)
                progress_path = progress_dir / f"{sample}_{ident}_{arm}.jsonl"
                progress = Progress(progress_path, prompt_length, ident, arm, sample)
                started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids, max_new_tokens=cap, do_sample=True,
                        temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                        top_k=SAMPLING["top_k"],
                        logits_processor=[PresencePenalty(penalty_start, SAMPLING["presence_penalty"])],
                        stopping_criteria=[progress], eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, use_cache=True,
                    )
                seconds = time.monotonic() - started
                output_ids = generated[0, prompt_length:].detach().cpu().tolist()
                text = tokenizer.decode(output_ids, skip_special_tokens=False)
                prediction = extract_answer(text)
                if prediction is None:
                    correct = False
                    scoring_error = None
                else:
                    try:
                        correct = bool(verify(parse(r"\boxed{" + str(gold) + "}"), parse(r"\boxed{" + str(prediction) + "}")))
                        scoring_error = None
                    except Exception as exc:
                        correct = None
                        scoring_error = f"{type(exc).__name__}: {exc}"
                ended_eos = bool(output_ids and output_ids[-1] == tokenizer.eos_token_id)
                think_closed = "</think>" in text
                cap_hit = len(output_ids) >= cap
                exact_loop = len(output_ids) >= 32 and output_ids[-32:] == output_ids[-64:-32]
                record = {
                    "intervention_id": INTERVENTION_ID,
                    "stage": args.stage,
                    "id": ident,
                    "source": source,
                    "sample": sample,
                    "seed": seed,
                    "arm": arm,
                    "base_checkpoint_sha256": sha256_file(BASE_TENSOR),
                    "column_manifest_sha256": sha256_file(ARM_MANIFEST),
                    "applied_edit": applied,
                    "input_token_ids": input_token_ids,
                    "input_token_ids_sha256": token_ids_hash(input_token_ids),
                    "continuation_text": text,
                    "continuation_token_ids": output_ids,
                    "continuation_token_ids_sha256": token_ids_hash(output_ids),
                    "gold": gold,
                    "prediction_last_boxed_in_continuation": prediction,
                    "correct_math_verify": correct,
                    "scoring_error": scoring_error,
                    "format_valid_box_in_continuation": prediction is not None,
                    "ended_with_eos": ended_eos,
                    "think_closed_in_continuation": think_closed,
                    "cap_hit": cap_hit,
                    "completed_correct": bool(correct and prediction is not None and think_closed and ended_eos and not cap_hit),
                    "tokens": len(output_ids),
                    "repeat8_fraction": repeat8_fraction(output_ids),
                    "tail_2048_repeat8_fraction": repeat8_fraction(output_ids[-2048:]),
                    "exact_final_32_token_loop": exact_loop,
                    "seconds": round(seconds, 3),
                    "model_load_seconds": round(model_load_seconds, 3),
                    "edit_and_integrity_check_seconds": round(edit_verify_seconds, 3),
                }
                append_jsonl(results, record)
                append_jsonl(ledger, {"event": "generation_completed", "id": ident, "sample": sample,
                                      "arm": arm, "tokens": len(output_ids), "seconds": round(seconds, 3),
                                      "correct": correct, "eos": ended_eos, "cap_hit": cap_hit})
                completed += 1
                print(json.dumps({"completed": completed, "total": len(schedule) * 4, "id": ident,
                                  "sample": sample, "arm": arm, "tokens": len(output_ids),
                                  "correct": correct, "eos": ended_eos, "cap_hit": cap_hit,
                                  "elapsed_total_seconds": round(time.monotonic() - total_started, 1)}), flush=True)
                if scoring_error:
                    raise RuntimeError(f"Formal scorer failed for {ident}/{sample}/{arm}: {scoring_error}")
            except Exception as exc:
                append_jsonl(ledger, {"event": "generation_failed_or_interrupted", "id": ident,
                                      "sample": sample, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    print(json.dumps({"status": "all_scheduled_generations_completed", "stage": args.stage,
                      "completed": completed, "expected": len(schedule) * 4,
                      "raw_results": str(results), "elapsed_seconds": round(time.monotonic() - total_started, 1)}), flush=True)


if __name__ == "__main__":
    import torch
    main()
