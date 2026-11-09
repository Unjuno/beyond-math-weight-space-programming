#!/usr/bin/env python3
"""Bounded teacher-forced localization on three frozen discovery traces.

This read-only measurement consumes only full prompt/token prefixes from the
human/Astra study's three predeclared traces. It writes local-only compact
activation summaries and per-observed-token log probabilities. It never edits
weights or opens validation/confirmation/final-heldout examples.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
BASE = ROOT / "model-cache/Qwen3.5-0.8B"
TENSOR = BASE / "model.safetensors-00001-of-00001.safetensors"
RAW = RUN / "raw/discovery_baseline_v1"
OUT = RUN / "raw/activation_localization_v1"
EXPECTED_BASE = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

CASES = {
    "gsm8k_train_6445": {
        "attempt": "attempt_002",
        "windows": {"transition": (4096, 4608), "cycle": (4608, 5120)},
    },
    "math_train_algebra_542": {
        "attempt": "attempt_002",
        "windows": {"transition": (2816, 3328), "cycle": (3328, 3840)},
    },
    "gsm8k_train_3185": {
        "attempt": "attempt_003",
        "windows": {
            "ordinary_check": (384, 896),
            "reinterpretation": (1152, 1664),
            "answer_transition": (2944, 3456),
        },
    },
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_case(ident: str, attempt: str) -> dict:
    path = RAW / attempt / "original.jsonl"
    matches = [json.loads(line) for line in path.read_text().splitlines() if json.loads(line).get("id") == ident]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one saved raw row for {ident}; found {len(matches)}")
    row = matches[0]
    if row["base_sha256"] != EXPECTED_BASE:
        raise ValueError(f"Base hash mismatch in {ident}")
    if sha_file(TENSOR) != EXPECTED_BASE:
        raise ValueError("Current base checkpoint hash mismatch")
    if len(row["generated_token_ids"]) < max(end for _, end in CASES[ident]["windows"].values()):
        raise ValueError(f"Saved trace is too short for frozen windows: {ident}")
    return row


def main() -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite localization output: {OUT}")
    if sha_file(TENSOR) != EXPECTED_BASE:
        raise ValueError("STOP: base checkpoint hash mismatch")
    OUT.mkdir(parents=True)
    osenv = __import__("os").environ
    osenv["HF_HUB_OFFLINE"] = "1"
    osenv["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to("mps")
    backbone = model.model.language_model
    layers = backbone.layers
    row_records = []

    for case_ix, (ident, spec) in enumerate(CASES.items(), start=1):
        row = read_case(ident, spec["attempt"])
        prompt_ids = row["prompt_token_ids"]
        generated = row["generated_token_ids"]
        max_end = max(end for _, end in spec["windows"].values())
        all_ids = prompt_ids + generated[:max_end]
        input_ids = torch.tensor([all_ids], dtype=torch.long, device="mps")
        prompt_len = len(prompt_ids)
        groups = {
            name: np.arange(prompt_len + start, prompt_len + end, dtype=np.int64)
            for name, (start, end) in spec["windows"].items()
        }
        captured: dict[str, dict[str, dict[str, object]]] = {}
        handles = []

        def make_hook(layer_idx: int, key: str, which: str):
            def hook(module, args, output=None):
                x = args[0] if which == "mlp_in" else output
                if isinstance(x, tuple):
                    x = x[0]
                if x is None or x.ndim != 3:
                    return
                for group, positions in groups.items():
                    local = positions[(positions >= 0) & (positions < x.shape[1])]
                    if not len(local):
                        continue
                    selected = x[0, torch.as_tensor(local, device=x.device)].detach().float().cpu().numpy()
                    entry = captured.setdefault(group, {}).setdefault(key, {})
                    entry["sum"] = selected.sum(axis=0) if "sum" not in entry else entry["sum"] + selected.sum(axis=0)
                    entry["sumsq"] = (selected * selected).sum(axis=0) if "sumsq" not in entry else entry["sumsq"] + (selected * selected).sum(axis=0)
                    entry["count"] = int(entry.get("count", 0)) + int(selected.shape[0])
            return hook

        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(li, f"residual_{li}", "out")))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(li, f"mlp_down_input_{li}", "mlp_in")))
            handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(li, f"mlp_down_output_{li}", "out")))

        t0 = time.monotonic()
        with torch.inference_mode():
            out = backbone(input_ids=input_ids, use_cache=False, output_hidden_states=False, return_dict=True)
            selected_logits = {}
            for group, positions in groups.items():
                # Position g-1 predicts observed generated token g.
                pred_positions = positions - 1
                pred_positions = pred_positions[pred_positions >= 0]
                hs = out.last_hidden_state[0, torch.as_tensor(pred_positions, device="mps")]
                logits = model.lm_head(hs).float()
                target_ids = input_ids[0, torch.as_tensor(pred_positions + 1, device="mps")]
                logp = torch.log_softmax(logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)
                selected_logits[group] = logp.detach().cpu().numpy()
        for handle in handles:
            handle.remove()

        arrays = {}
        compact = {"id": ident, "attempt": spec["attempt"], "prompt_tokens": prompt_len,
                   "teacher_forced_generated_prefix_tokens": max_end,
                   "windows": {}, "elapsed_seconds": round(time.monotonic() - t0, 3)}
        for group, vectors in captured.items():
            gsummary = {}
            for key, entry in vectors.items():
                count = entry["count"]
                mean = entry["sum"] / count
                variance = np.maximum(0.0, entry["sumsq"] / count - mean * mean)
                arrays[f"{group}__{key}__mean"] = mean.astype(np.float32)
                gsummary[key] = {
                    "count": count,
                    "mean_l2": float(np.linalg.norm(mean)),
                    "mean_token_l2": float(np.sqrt(np.maximum(0.0, entry["sumsq"] / count).sum())),
                    "mean_coordinate_std": float(np.sqrt(variance).mean()),
                }
            lp = selected_logits[group]
            arrays[f"{group}__observed_token_logprobs"] = lp.astype(np.float32)
            gsummary["observed_next_token_logprob"] = {
                "count": int(lp.size), "mean": float(lp.mean()), "median": float(np.median(lp)),
                "p10": float(np.quantile(lp, .1)), "p90": float(np.quantile(lp, .9)),
            }
            compact["windows"][group] = gsummary
        npz = OUT / f"{case_ix:02d}_{ident}.npz"
        np.savez_compressed(npz, **arrays)
        compact["summary_npz"] = npz.name
        compact["summary_npz_sha256"] = sha_file(npz)
        row_records.append(compact)
        (OUT / "progress.jsonl").open("a").write(json.dumps({"id": ident, "status": "completed", "seconds": compact["elapsed_seconds"]}) + "\n")
        print(json.dumps({"id": ident, "windows": list(compact["windows"]), "seconds": compact["elapsed_seconds"]}), flush=True)

    manifest = {
        "run_id": "human-astra-manual-20261105-v1",
        "measurement": "teacher_forced_full_prefix_layer_activation_and_observed_token_logprob",
        "base_sha256": EXPECTED_BASE,
        "script_sha256": sha_file(Path(__file__)),
        "tokenizer_name": str(BASE),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "device": "mps", "dtype": "bfloat16", "attention": "eager", "use_cache": False,
        "raw_inputs": {
            ident: {"attempt": spec["attempt"], "raw_sha256": sha_file(RAW / spec["attempt"] / "original.jsonl"), "windows": spec["windows"]}
            for ident, spec in CASES.items()
        },
        "rows": row_records,
        "claim_boundary": "Associational discovery measurement only; no weights changed; no causal mechanism claim.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "items": len(row_records), "output": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
