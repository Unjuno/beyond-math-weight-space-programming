#!/usr/bin/env python3
"""Paired, full-cap applied non-math evaluation of a released Qwen patch.

This runner deliberately keeps gold answers and writing rubrics out of raw
generation records. It refuses to resume or overwrite an existing output.
Task definitions are loaded from a local file excluded from Git until scoring
is frozen. The published p50c patch and locally generated alpha100 patch are
different experimental arms and must use separate output paths/manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_len: int, penalty: float = 1.5):
        self.prompt_len = prompt_len
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len :]
        if generated.numel():
            for row in range(scores.shape[0]):
                scores[row, torch.unique(generated[row])] -= self.penalty
        return scores


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def stable_seed(task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"transfer-eval-v1|{task_id}".encode()).digest()[:4], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--arm", choices=["p50c", "alpha100"], required=True)
    args = parser.parse_args()

    if args.max_new_tokens != 32768:
        raise ValueError("Primary reproduction route requires max_new_tokens=32768")
    if args.out.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation evidence")
    task_doc = json.loads(args.tasks.read_text())
    tasks = task_doc["tasks"]
    if len(tasks) != 24 or len({t["id"] for t in tasks}) != 24:
        raise ValueError("Expected 24 uniquely identified preregistered tasks")
    if {t["family"] for t in tasks} != {"extraction", "operational_decision", "grounded_writing"}:
        raise ValueError("Unexpected task-family set")

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    ).to("mps").eval()
    model.generation_config.use_cache = True

    if args.arm == "p50c":
        expected_tensor_hashes = {
            15: "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
            19: "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
        }
        expected_patch_file_hashes = {}
    else:
        expected_tensor_hashes = {}
        expected_patch_file_hashes = {
            18: "5042c800d051f92fc0d22a26a7694ccb7bc40a982c4fd46a703226f3398b9134"
        }
    names = {
        layer: f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        for layer in sorted(set(expected_tensor_hashes) | set(expected_patch_file_hashes))
    }
    original = {layer: model.get_parameter(name).detach().cpu().clone() for layer, name in names.items()}
    patch = {}
    for layer, name in names.items():
        patch_file = args.patch_dir / f"{name}.npy"
        import numpy as np

        patch[layer] = torch.from_numpy(np.load(patch_file))
        file_hash = sha256(patch_file)
        if layer in expected_patch_file_hashes and file_hash != expected_patch_file_hashes[layer]:
            raise RuntimeError(f"patch file hash mismatch at layer {layer}: {file_hash}")
        model.get_parameter(name).data.copy_(patch[layer].to("mps", dtype=torch.bfloat16))
        got = tensor_sha256(model.get_parameter(name))
        if layer in expected_tensor_hashes and got != expected_tensor_hashes[layer]:
            raise RuntimeError(f"edited tensor hash mismatch at layer {layer}: {got}")

    # Randomized per-task arm order is fixed and logged before generating.
    task_order = list(tasks)
    random.Random(20261103).shuffle(task_order)
    arm_rng = random.Random(20261104)
    arm_map = {t["id"]: arm_rng.choice(["base", "edited"]) for t in task_order}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_manifest = {
        "created_at": started,
        "arm": args.arm,
        "base_model_path": str(args.model),
        "base_safetensors_sha256": sha256(args.model / "model.safetensors-00001-of-00001.safetensors"),
        "task_file_sha256": sha256(args.tasks),
        "patch_file_sha256": {str(layer): sha256(args.patch_dir / f"{name}.npy") for layer, name in names.items()},
        "patched_tensor_sha256": {str(layer): tensor_sha256(model.get_parameter(name)) for layer, name in names.items()},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "task_order": [t["id"] for t in task_order],
        "arm_order_by_task": arm_map,
        "seed_by_task": {t["id"]: stable_seed(t["id"]) for t in tasks},
        "sampling": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
        "max_new_tokens": args.max_new_tokens,
        "backend": "Transformers AutoModelForImageTextToText, BF16, Apple MPS; attention backend left at Transformers default",
        "limitation": "The paper's exact Experiment 8 persistent harness/backend is unreleased; this is an approximate inference reproduction.",
        "completion_criteria": "A task succeeds only with EOS before cap and correct typed JSON, or EOS before cap and all four blind grounded-writing rubric factors true.",
        "raw_output_path": str(args.out),
    }
    args.manifest.write_text(json.dumps(run_manifest, indent=2) + "\n")
    end_turn = tok.convert_tokens_to_ids("<|im_end|>")
    for task in task_order:
        seed = stable_seed(task["id"])
        messages = [{"role": "user", "content": task["prompt"]}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to("mps")
        for condition in [arm_map[task["id"]], "edited" if arm_map[task["id"]] == "base" else "base"]:
            if condition == "base":
                for layer, name in names.items():
                    model.get_parameter(name).data.copy_(original[layer].to("mps", dtype=torch.bfloat16))
            else:
                for layer, name in names.items():
                    model.get_parameter(name).data.copy_(patch[layer].to("mps", dtype=torch.bfloat16))
            torch.manual_seed(seed)
            torch.mps.manual_seed(seed)
            t0 = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                    logits_processor=LogitsProcessorList([PresencePenalty(input_ids.shape[-1])]),
                    eos_token_id=[tok.eos_token_id, end_turn],
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            generated = output[0, input_ids.shape[-1] :].detach().cpu().tolist()
            text = tok.decode(generated, skip_special_tokens=False)
            record = {
                "id": task["id"], "family": task["family"], "condition": condition,
                "seed": seed, "thinking": True,
                "input_token_ids": input_ids[0].detach().cpu().tolist(),
                "generated_token_ids": generated, "output": text,
                "final_answer": text.rsplit("</think>", 1)[1].strip() if "</think>" in text else "",
                "generated_tokens": len(generated),
                "cap_hit": len(generated) >= args.max_new_tokens,
                "ended_with_eos": bool(generated and generated[-1] in {tok.eos_token_id, end_turn}),
                "thinking_closed": "</think>" in text,
                "seconds": time.time() - t0,
            }
            with args.out.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                import os

                os.fsync(f.fileno())
            print(f"{task['id']} {condition} tokens={len(generated)} eos={record['ended_with_eos']} cap={record['cap_hit']} sec={record['seconds']:.1f}", flush=True)
    run_manifest["finished_at"] = time.time()
    run_manifest["elapsed_seconds"] = run_manifest["finished_at"] - started
    run_manifest["raw_output_sha256"] = sha256(args.out)
    args.manifest.write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"completed 24 paired items in {run_manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
