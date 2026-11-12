#!/usr/bin/env python3
"""Run a preregistered paired MATH-500 p50c checkpoint spot evaluation on MPS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len, self.penalty = prompt_len, penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len:]
        if generated.numel():
            for row in range(scores.shape[0]):
                scores[row, torch.unique(generated[row])] -= self.penalty
        return scores


def sha256_tensor(tensor: torch.Tensor) -> str:
    bits = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(bits.tobytes()).hexdigest()


def last_boxed(text: str) -> str | None:
    # Port of the pinned author's mathqa.last_boxed: the last boxed takes
    # precedence over fbox; an unbraced "\boxed 5" is also accepted.
    start = text.rfind(r"\boxed")
    key = r"\boxed"
    if start < 0:
        start = text.rfind(r"\fbox")
        key = r"\fbox"
    if start < 0:
        return None
    rest = text[start + len(key):]
    candidate = rest.lstrip()
    if not candidate.startswith("{"):
        if key == r"\boxed" and rest.startswith(" "):
            return rest.strip().split("$")[0].split("\n")[0].strip() or None
        return None
    depth = 0
    for i, char in enumerate(candidate):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[1:i].strip() or None
    return None


def author_answer_text(text: str) -> str:
    # The pinned scorer uses the final answer after </think> when available;
    # if generation hits the cap before </think>, it scores the entire output.
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--patch-dir", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    study = args.study_dir
    item = json.loads((study / "selected_problem.json").read_text())
    manifest_path = study / "preregistration.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "preregistered_before_generation":
        raise RuntimeError("preregistration is not in the frozen pre-generation state")
    if args.max_new_tokens != manifest["max_new_tokens"] or args.samples != manifest["samples_per_checkpoint"]:
        raise ValueError("run settings differ from preregistration")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("mps").eval()
    model.generation_config.use_cache = True
    patch_targets = {
        15: "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
        19: "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
    }
    prompt = item["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to("mps")
    prompt_len = input_ids.shape[-1]
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    out_path = study / "raw_generations.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                old = json.loads(line)
                done.add((old["arm"], old["sample_index"]))

    for arm in ("base", "edited"):
        if arm == "edited":
            for layer, expected in patch_targets.items():
                name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
                target = torch.from_numpy(np.load(args.patch_dir / f"{name}.npy")).to(
                    device="mps", dtype=torch.bfloat16
                )
                weight = model.get_parameter(name)
                weight.data.copy_(target)
                actual = sha256_tensor(weight)
                if actual != expected:
                    raise RuntimeError(f"layer {layer} edit hash mismatch: {actual}")
        for sample_idx in range(args.samples):
            if (arm, sample_idx) in done:
                continue
            seed = sample_idx
            torch.manual_seed(seed)
            torch.mps.manual_seed(seed)
            start = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                    logits_processor=LogitsProcessorList([PresencePenalty(prompt_len, 1.5)]),
                    eos_token_id=list(eos_ids), pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )[0, prompt_len:].detach().cpu().tolist()
            text = tok.decode(generated, skip_special_tokens=False)
            thinking_closed = "</think>" in text
            answer_text = author_answer_text(text)
            record = {
                "id": item["id"], "arm": arm, "sample_index": sample_idx,
                "seed": seed, "thinking": True, "prompt_token_count": prompt_len,
                "generated_token_ids": generated, "generated_tokens": len(generated),
                "output": text, "answer_text": answer_text,
                "last_boxed": last_boxed(answer_text), "thinking_closed": thinking_closed,
                "cap_hit": len(generated) >= args.max_new_tokens,
                "ended_with_eos": bool(generated and generated[-1] in eos_ids),
                "seconds": time.time() - start,
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            print(f"{arm} sample={sample_idx} tokens={len(generated)} cap={record['cap_hit']} "
                  f"boxed={record['last_boxed'] is not None} sec={record['seconds']:.1f}", flush=True)

    manifest["status"] = "generation_complete"
    manifest["raw_generations_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest["raw_generation_rows"] = args.samples * 2
    manifest["runtime_environment"] = {
        "device": "Apple MPS", "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": __import__("platform").python_version(),
        "backend_note": "Transformers PyTorch/MPS; paper used vLLM, so backend differs.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "raw_sha256": manifest["raw_generations_sha256"]}))

    try:
        import math_verify
    except ImportError:
        extra = os.environ.get("MATHVERIFY_PKGS", "/tmp/mathverify-pkgs-repro")
        sys.path.insert(0, extra)
        import math_verify
    records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    gold = item["gold"]
    per_arm = {}
    for arm in ("base", "edited"):
        subset = [r for r in records if r["arm"] == arm]
        results = []
        for r in subset:
            pred = r["last_boxed"]
            ok = False
            if pred:
                try:
                    g = math_verify.parse(r"\boxed{" + gold + "}", parsing_timeout=5)
                    p = math_verify.parse(r"\boxed{" + pred + "}", parsing_timeout=5)
                    ok = bool(g and p and math_verify.verify(g, p, timeout_seconds=5))
                except Exception:
                    ok = False
            r["math_verify_correct"] = bool(ok)
            results.append(r)
        per_arm[arm] = {
            "n": len(results), "correct": sum(r["math_verify_correct"] for r in results),
            "accuracy": (sum(r["math_verify_correct"] for r in results) / len(results)) if results else None,
            "cap_hits": sum(r["cap_hit"] for r in results),
            "finished_thinking": sum(r["thinking_closed"] for r in results),
            "mean_generated_tokens": (sum(r["generated_tokens"] for r in results) / len(results)) if results else None,
        }
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "study": manifest["study"], "selected_id": item["id"],
        "gold_answer": gold, "scorer": "math-verify 0.9.0; last boxed answer",
        "per_arm": per_arm, "paired_correctness": [
            {"sample_index": i,
             "base": next((r["math_verify_correct"] for r in records if r["arm"] == "base" and r["sample_index"] == i), None),
             "edited": next((r["math_verify_correct"] for r in records if r["arm"] == "edited" and r["sample_index"] == i), None)}
            for i in range(args.samples)
        ],
        "interpretation_limit": "One selected MATH-500 problem with four samples per arm; descriptive pilot only.",
    }
    (study / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "scored"
    manifest["raw_generations_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest["summary_sha256"] = hashlib.sha256((study / "summary.json").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "per_arm": per_arm}))


if __name__ == "__main__":
    main()
