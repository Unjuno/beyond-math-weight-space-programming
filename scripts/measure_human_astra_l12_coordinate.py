#!/usr/bin/env python3
"""Measure one Astra-nominated L12 MLP coordinate on frozen full-prefix traces.

This is an observational follow-up only. It captures token-level L12
down_proj-input coordinate 1226, exact token IDs, and the original output
column norm on the same already-open discovery traces. It does not edit weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/human_astra_manual_20261105"
BASE = ROOT / "model-cache/Qwen3.5-0.8B"
TENSOR = BASE / "model.safetensors-00001-of-00001.safetensors"
RAW = RUN / "raw/discovery_baseline_v1"
OUT = RUN / "raw/l12_coordinate_observation_v1"
BASE_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
LAYER = 12
COORD = 1226
CASES = {
    "gsm8k_train_6445": ("attempt_002", {"transition": (4096, 4608), "cycle": (4608, 5120)}),
    "math_train_algebra_542": ("attempt_002", {"transition": (2816, 3328), "cycle": (3328, 3840)}),
    "gsm8k_train_3185": ("attempt_003", {"ordinary_check": (384, 896), "reinterpretation": (1152, 1664), "answer_transition": (2944, 3456)}),
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite: {OUT}")
    if sha_file(TENSOR) != BASE_SHA256:
        raise ValueError("STOP: base checkpoint hash mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to("mps")
    module = model.model.language_model.layers[LAYER].mlp.down_proj
    weight = module.weight.detach().float().cpu().numpy()
    if weight.ndim != 2 or weight.shape[1] <= COORD:
        raise ValueError(f"Invalid nominated column: layer {LAYER} down_proj shape {weight.shape}")
    column_norm = float(np.linalg.norm(weight[:, COORD]))
    column_norms = np.linalg.norm(weight, axis=0)
    percentile = float((column_norms < column_norm).mean())
    OUT.mkdir(parents=True)
    rows = []

    for ident, (attempt, windows) in CASES.items():
        raw_path = RAW / attempt / "original.jsonl"
        matches = [json.loads(line) for line in raw_path.read_text().splitlines() if json.loads(line).get("id") == ident]
        if len(matches) != 1:
            raise ValueError(f"Expected one raw row for {ident}; got {len(matches)}")
        row = matches[0]
        if row["base_sha256"] != BASE_SHA256:
            raise ValueError(f"Raw trace base mismatch: {ident}")
        max_end = max(end for start, end in windows.values())
        if len(row["generated_token_ids"]) < max_end:
            raise ValueError(f"Trace too short for frozen coordinate windows: {ident}")
        all_ids = row["prompt_token_ids"] + row["generated_token_ids"][:max_end]
        input_ids = torch.tensor([all_ids], dtype=torch.long, device="mps")
        prompt_length = len(row["prompt_token_ids"])
        selected = {name: np.arange(prompt_length + start, prompt_length + end, dtype=np.int64)
                    for name, (start, end) in windows.items()}
        captured = {}

        def capture(module, args):
            x = args[0]
            if x.ndim != 3:
                raise ValueError(f"Unexpected L12 MLP input shape: {tuple(x.shape)}")
            if x.shape[-1] <= COORD:
                raise ValueError(f"Coordinate {COORD} outside activation width {x.shape[-1]}")
            for name, positions in selected.items():
                indices = torch.as_tensor(positions, device=x.device)
                values = x[0, indices, COORD].detach().float().cpu().numpy()
                captured[name] = values

        handle = module.register_forward_pre_hook(capture)
        with torch.inference_mode():
            model.model.language_model(input_ids=input_ids, use_cache=False,
                                      output_hidden_states=False, return_dict=True)
        handle.remove()
        arrays = {"coordinate": np.asarray([COORD], dtype=np.int64),
                  "layer": np.asarray([LAYER], dtype=np.int64),
                  "down_proj_column_norms": column_norms.astype(np.float32)}
        stats = {}
        for name, values in captured.items():
            token_start, token_end = windows[name]
            token_ids = np.asarray(row["generated_token_ids"][token_start:token_end], dtype=np.int64)
            arrays[f"{name}__activation"] = values.astype(np.float32)
            arrays[f"{name}__token_ids"] = token_ids
            stats[name] = {
                "generated_token_interval": [token_start, token_end],
                "n": int(values.size),
                "activation_mean": float(values.mean()),
                "activation_std": float(values.std()),
                "activation_mean_abs": float(np.abs(values).mean()),
                "activation_rms": float(np.sqrt(np.mean(values * values))),
                "max_abs": float(np.abs(values).max()),
                "token_ids_sha256": hashlib.sha256(token_ids.astype(">i4").tobytes()).hexdigest(),
            }
        out_path = OUT / f"{ident}.npz"
        np.savez_compressed(out_path, **arrays)
        rows.append({"id": ident, "attempt": attempt, "raw_file_sha256": sha_file(raw_path),
                     "windows": stats, "npz": out_path.name, "npz_sha256": sha_file(out_path)})
        print(json.dumps({"id": ident, "windows": {k: {q: v for q, v in s.items() if q != "token_ids_sha256"}
                                                    for k, s in stats.items()}}, flush=True)

    manifest = {
        "run_id": "human-astra-manual-20261105-v1",
        "measurement": "L12 MLP down_proj input coordinate time series",
        "candidate_status": "observational_coordinate_only_not_an_edit_candidate",
        "base_sha256": BASE_SHA256,
        "script_sha256": sha_file(Path(__file__)),
        "layer": LAYER,
        "coordinate": COORD,
        "down_proj_weight_shape": list(weight.shape),
        "column_l2_norm": column_norm,
        "column_norm_percentile": percentile,
        "device": "mps", "dtype": "bfloat16", "attention": "eager", "use_cache": False,
        "raw_vectors_local_only": True,
        "items": rows,
        "claim_boundary": "No weight change; activation association only; content, token, and position confounds remain.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(OUT), "column_l2_norm": column_norm,
                      "column_norm_percentile": percentile}), flush=True)


if __name__ == "__main__":
    main()
