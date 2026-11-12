#!/usr/bin/env python3
"""Local Qwen forward adapter for the fresh Astra-led Experiment 8 rerun.

It accepts the author's JSON request shapes, uses only the run's editor-train
IDs, and writes raw outputs under the mounted /work/forward directory.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("WSP_RUN_DIR", ROOT / "results/astra_reproduction_20261104/raw/editor_work")).resolve()
TRAIN = Path(os.environ.get("WSP_TRAIN_DIR", ROOT / "results/astra_reproduction_20261104/raw/train")).resolve()
BASE = Path(os.environ.get("WSP_BASE_DIR", ROOT / "model-cache/Qwen3.5-0.8B")).resolve()
TENSOR_FILE = BASE / "model.safetensors-00001-of-00001.safetensors"
OUT = RUN / "forward"
REQUESTS = OUT / "requests"
RESPONSES = OUT / "responses"
PASSES = OUT / "passes"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'


def append(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def train_rows():
    rows = {}
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        if not (TRAIN / name).exists():
            continue
        with (TRAIN / name).open() as f:
            for line in f:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def answer(text):
    from astra_clean_grading import extract_answer
    return extract_answer(text)


def grade(pred, gold):
    if pred is None:
        return False
    from math_verify import parse, verify
    return bool(verify(parse("\\boxed{" + str(gold) + "}"), parse("\\boxed{" + str(pred) + "}")))


def paired_accuracy_summary(original, edited, item_ids, np):
    def keyed(records):
        result = {}
        for record in records:
            key = (record["id"], record["sample"])
            if key in result:
                raise ValueError(f"Duplicate paired generation key: {key}")
            result[key] = record
        return result

    original_by_key, edited_by_key = keyed(original), keyed(edited)
    if original_by_key.keys() != edited_by_key.keys():
        raise ValueError("Original and edited outputs do not have matching item/sample keys")
    differences_by_item = collections.defaultdict(list)
    observation_diffs = []
    for key in sorted(original_by_key):
        left, right = original_by_key[key], edited_by_key[key]
        if left["seed"] != right["seed"]:
            raise ValueError(f"Paired seed mismatch for {key}")
        delta = int(right["correct"]) - int(left["correct"])
        differences_by_item[key[0]].append(delta)
        observation_diffs.append(delta)
    if set(differences_by_item) != set(item_ids):
        raise ValueError("Paired outputs do not cover exactly the requested item IDs")
    item_deltas = [float(np.mean(differences_by_item[ident])) for ident in item_ids]
    return {
        "delta": float(np.mean(item_deltas)),
        "se": float(np.std(item_deltas, ddof=1) / np.sqrt(len(item_deltas))) if len(item_deltas) > 1 else None,
        "wrong_to_right": sum(d == 1 for d in observation_diffs),
        "right_to_wrong": sum(d == -1 for d in observation_diffs),
        "n_items": len(item_deltas),
        "n_paired_generations": len(observation_diffs),
        "item_mean_differences": dict(zip(item_ids, item_deltas)),
    }


def main():
    import numpy as np
    import torch
    import math_verify  # fail before model loading or inference if the pinned scorer is unavailable
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, LogitsProcessor, StoppingCriteria

    if not grade("2", "2") or grade("2", "3"):
        raise RuntimeError("math_verify preflight did not accept the matching answer and reject the mismatch")

    torch.set_grad_enabled(False)
    torch.set_num_threads(6)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for p in (OUT, REQUESTS, RESPONSES, PASSES):
        p.mkdir(parents=True, exist_ok=True)
    rows = train_rows()
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to("mps")
    params = dict(model.named_parameters())
    original = safe_open(str(TENSOR_FILE), framework="pt", device="cpu")
    layers = model.model.language_model.layers
    embeds = model.model.language_model.embed_tokens
    active = set()
    ledger_path = OUT / "pass_ledger.jsonl"

    class Presence(LogitsProcessor):
        def __init__(self, start):
            self.start = start
        def __call__(self, input_ids, scores):
            # vLLM presence_penalty acts on generated output IDs; its separate
            # repetition penalty is the setting that also includes prompt IDs.
            ids = torch.unique(input_ids[0, self.start:])
            if ids.numel():
                scores[0, ids] -= 1.5
            return scores

    class Progress(StoppingCriteria):
        def __init__(self, start, ident, variant):
            self.start, self.ident, self.variant, self.t0 = start, ident, variant, time.time()
        def __call__(self, input_ids, scores, **kwargs):
            n = input_ids.shape[1] - self.start
            if n and n % 512 == 0:
                print(json.dumps({"progress_id": self.ident, "model": self.variant,
                                  "tokens": n, "seconds": time.time() - self.t0}), flush=True)
            return False

    class Capture:
        def __init__(self, requested_layers=(), mlp_layers=None, hidden=False, teacher=False,
                     bin_size=None, max_generated_tokens=None, capture_ranges=None,
                     mlp_output_layers=None):
            self.layers = set(requested_layers)
            self.mlp_layers = set(mlp_layers or ())
            self.mlp_output_layers = set(mlp_output_layers or ())
            self.hidden = bool(hidden)
            self.teacher = teacher
            self.bin_size = bin_size
            self.max_generated_tokens = max_generated_tokens
            self.capture_ranges = capture_ranges
            if capture_ranges is not None:
                if teacher or not isinstance(capture_ranges, (list, tuple)) or not capture_ranges:
                    raise ValueError("Invalid generation capture ranges")
                for pair in capture_ranges:
                    if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                            any(type(i) is not int for i in pair) or pair[0] < 0 or pair[1] <= pair[0]):
                        raise ValueError("Invalid generation capture range")
                self.capture_positions = [pos for start, end in capture_ranges for pos in range(start, end)]
                if len(set(self.capture_positions)) != len(self.capture_positions):
                    raise ValueError("Generation capture ranges must not overlap")
                self.capture_slots = {pos: i for i, pos in enumerate(self.capture_positions)}
            if bin_size is not None:
                if teacher or type(bin_size) is not int or bin_size != 1024:
                    raise ValueError("Generation activation bin_size must be 1024")
                if type(max_generated_tokens) is not int or not 0 < max_generated_tokens <= 32768:
                    raise ValueError("Binned capture requires a valid generation token cap")
                self.n_bins = (max_generated_tokens + bin_size - 1) // bin_size
            self.data = {}
            self.handles = []
            self.calls = collections.defaultdict(int)
            self.answer_counts = collections.defaultdict(int)
            if self.hidden:
                if 0 in self.layers:
                    self.handles.append(embeds.register_forward_hook(self._hook("hidden_0", input_mode=False)))
                for i in self.layers:
                    if i > 0:
                        self.handles.append(layers[i - 1].register_forward_hook(self._hook(f"hidden_{i}", input_mode=False)))
            if self.mlp_layers:
                if any(i < 0 or i >= len(layers) for i in self.mlp_layers):
                    raise ValueError("MLP activation layers must be 0..23")
                for i in self.mlp_layers:
                    self.handles.append(layers[i].mlp.down_proj.register_forward_pre_hook(self._hook(f"mlp_{i}", input_mode=True)))
            if self.mlp_output_layers:
                if any(i < 0 or i >= len(layers) for i in self.mlp_output_layers):
                    raise ValueError("MLP output layers must be 0..23")
                for i in self.mlp_output_layers:
                    self.handles.append(layers[i].mlp.register_forward_hook(self._hook(f"mlp_out_{i}", input_mode=False)))

        def _hook(self, key, input_mode):
            def capture(module, args, output=None):
                x = args[0] if input_mode else output
                if isinstance(x, tuple):
                    x = x[0]
                if x is None or not hasattr(x, "detach"):
                    if self.bin_size is not None:
                        raise RuntimeError(f"Missing activation tensor for {key}")
                    return
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                if x.ndim != 3:
                    if self.bin_size is not None:
                        raise RuntimeError(f"Unexpected activation shape for {key}: {x.shape}")
                    return
                call = self.calls[key]
                self.calls[key] += 1
                d = self.data.setdefault(key, {"prompt": None, "sum": None, "count": 0})
                z = x.detach().float()
                if self.capture_ranges is not None:
                    if z.shape[0] != 1:
                        raise RuntimeError("Selected-position capture requires batch size one")
                    if call == 0:
                        d["prompt"] = z[0, -1].clone()
                        d["selected"] = torch.zeros((len(self.capture_positions), z.shape[-1]),
                                                     dtype=torch.float32, device=z.device)
                        d["selected_seen"] = [False] * len(self.capture_positions)
                        return
                    if z.shape[1] != 1:
                        raise RuntimeError("Selected-position capture requires cached one-token decoding")
                    position = call - 1
                    slot = self.capture_slots.get(position)
                    if slot is not None:
                        d["selected"][slot].copy_(z[0, 0])
                        d["selected_seen"][slot] = True
                    return
                if self.bin_size is not None:
                    self._capture_bins(d, z, call)
                    return
                if self.teacher:
                    d["sum"] = z[0].cpu().numpy()
                    d["count"] = int(z.shape[1])
                    return
                if call == 0:
                    d["prompt"] = z[0, -1].cpu().numpy()
                    # The prefill is prompt-only. Subsequent cached inputs
                    # correspond to generated tokens being processed.
                    return
                remaining = 8192 - self.answer_counts[key]
                if remaining <= 0:
                    return
                take = min(remaining, z.shape[1])
                part = z[0, :take].sum(0).cpu().numpy()
                d["sum"] = part if d["sum"] is None else d["sum"] + part
                d["count"] += take
                self.answer_counts[key] += take
            if input_mode:
                return lambda module, args: capture(module, args)
            return lambda module, args, output: capture(module, args, output)

        def _capture_bins(self, d, z, call):
            # Observational buffers only: no host transfers or scalar reads here.
            if z.shape[0] != 1:
                raise RuntimeError("Binned capture requires batch size one")
            if call == 0:
                # Entire first call is prompt prefill, even for a one-token prompt.
                d["prompt"] = z[0, -1].clone()
                d["sum"] = torch.zeros_like(z[0, -1])
                d["bin_sum"] = torch.zeros((self.n_bins, z.shape[-1]),
                                           dtype=torch.float32, device=z.device)
                d["bin_sumsq"] = torch.zeros_like(d["bin_sum"])
                d["bin_count"] = [0] * self.n_bins
                d["seen"] = 0
                return
            if z.shape[1] != 1:
                raise RuntimeError("Binned capture requires one-token cached decode calls")
            position = d["seen"]
            if position >= self.max_generated_tokens:
                raise RuntimeError("Binned capture exceeded the generation token cap")
            bin_index = position // self.bin_size
            value = z[0, 0]
            d["bin_sum"][bin_index].add_(value)
            d["bin_sumsq"][bin_index].addcmul_(value, value)
            d["bin_count"][bin_index] += 1
            d["seen"] += 1
            if d["count"] < 8192:
                d["sum"].add_(value)
                d["count"] += 1

        def save(self, path, generated_tokens=None, final_token_is_eos=False):
            arrays = {}
            if self.capture_ranges is not None:
                expected_keys = ({f"hidden_{i}" for i in self.layers} if self.hidden else set())
                expected_keys |= {f"mlp_{i}" for i in self.mlp_layers}
                expected_keys |= {f"mlp_out_{i}" for i in self.mlp_output_layers}
                if set(self.data) != expected_keys:
                    raise RuntimeError("Selected-position capture did not observe every selected hook")
                for name, d in self.data.items():
                    if not all(d.get("selected_seen", [])):
                        raise RuntimeError(f"Selected-position capture incomplete for {name}")
                    arrays[name + "_selected_values"] = d["selected"].cpu().numpy()
                    arrays[name + "_prompt_last"] = d["prompt"].cpu().numpy()
                arrays["capture_positions"] = np.asarray(self.capture_positions, dtype=np.int64)
                arrays["capture_ranges"] = np.asarray(self.capture_ranges, dtype=np.int64)
                np.savez_compressed(path, **arrays)
                self.close()
                return
            if self.bin_size is not None:
                if type(generated_tokens) is not int or not 0 < generated_tokens <= self.max_generated_tokens:
                    raise RuntimeError("Missing or invalid generated-token count for binned capture")
                expected_keys = ({f"hidden_{i}" for i in self.layers} if self.hidden else set())
                expected_keys |= {f"mlp_{i}" for i in self.mlp_layers}
                expected_keys |= {f"mlp_out_{i}" for i in self.mlp_output_layers}
                if set(self.data) != expected_keys:
                    raise RuntimeError("Binned capture did not observe every selected hook")
                # Cached decoding normally leaves the final sampled token unprocessed.
                # This backend can forward a sampled EOS once before stopping.
                expected = generated_tokens if final_token_is_eos else generated_tokens - 1
                for name, d in self.data.items():
                    if d.get("seen") != expected:
                        raise RuntimeError(f"Consumed-token alignment mismatch for {name}: "
                                           f"{d.get('seen')} != {expected}")
                arrays["answer_bin_size"] = np.asarray(self.bin_size, dtype=np.int64)
                arrays["answer_bin_token_start"] = np.arange(self.n_bins, dtype=np.int64) * self.bin_size
                arrays["generated_token_count"] = np.asarray(generated_tokens, dtype=np.int64)
            for name, d in self.data.items():
                prompt, total = d["prompt"], d["sum"]
                if self.bin_size is not None:
                    # The generation has finished before any device-to-host copy.
                    prompt = prompt.cpu().numpy()
                    total = total.cpu().numpy()
                    arrays[name + "_answer_bin_sum"] = d["bin_sum"].cpu().numpy()
                    arrays[name + "_answer_bin_sumsq"] = d["bin_sumsq"].cpu().numpy()
                    arrays[name + "_answer_bin_count"] = np.asarray(d["bin_count"], dtype=np.int64)
                if prompt is not None:
                    arrays[name + "_prompt_last"] = prompt
                if total is not None and d["count"]:
                    arrays[name + ("_teacher_tokens" if self.teacher else "_answer_mean_first8192")] = total if self.teacher else total / d["count"]
            if arrays:
                np.savez_compressed(path, **arrays)
            self.close()

        def close(self):
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def apply(which, edit_dir):
        nonlocal active
        edit_root = Path(edit_dir)
        paths = list(edit_root.glob("*.npy")) if which == "edited" else []
        now = {p.stem for p in paths}
        for name in active | now:
            if name in now:
                tensor = torch.from_numpy(np.load(edit_root / (name + ".npy"), allow_pickle=False))
            else:
                tensor = original.get_tensor(name)
            if name not in params:
                raise ValueError(f"No HF parameter for edited tensor {name}")
            params[name].copy_(tensor.to(device="mps", dtype=params[name].dtype))
        active = now

    def template(problem):
        messages = [{"role": "user", "content": problem + "\n" + INSTRUCTION}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                      add_generation_prompt=True, enable_thinking=True)["input_ids"]
        return rendered, ids

    def gen_request(req, pass_no, outdir):
        ids = req.get("problems", [])
        if not ids or any(i not in rows for i in ids):
            raise ValueError("Only IDs in the editor's disjoint TRAIN partition are available")
        if len(ids) != len(set(ids)):
            raise ValueError("A generation request may not repeat problem IDs")
        custom_prompts = req.get("prompts")
        if custom_prompts is not None and (
            not isinstance(custom_prompts, list)
            or len(custom_prompts) != len(ids)
            or any(not isinstance(prompt, str) or not prompt for prompt in custom_prompts)
        ):
            raise ValueError("prompts must contain one non-empty user message per problem ID")
        n = int(req.get("n", 1))
        model_choice = req["model"]
        variants = ["original", "edited"] if model_choice == "both" else [model_choice]
        if model_choice not in {"original", "edited", "both"}:
            raise ValueError("model must be original, edited, or both")
        if len(ids) * n > (256 if model_choice == "both" else 512):
            raise ValueError("Generation request exceeds author API limits")
        cap = int(req.get("max_tokens", 32768))
        if not 0 < cap <= 32768:
            raise ValueError("max_tokens must be 1..32768")
        acts = req.get("activations", {}) or {}
        requested_layers = acts.get("layers", [])
        mlp = acts.get("mlp", False)
        bin_size = acts.get("bin_size")
        if bin_size is not None:
            if type(bin_size) is not int or bin_size != 1024:
                raise ValueError("activations.bin_size must be 1024 when supplied")
            if not requested_layers and not mlp:
                raise ValueError("Activation bins require selected residual or MLP hooks")
        if any(i < 0 or i > 24 for i in requested_layers):
            raise ValueError("Activation layers must be 0..24")
        arms = {}
        for variant in variants:
            apply(variant, req.get("edit_dir", str(RUN / "edited")))
            records = []
            target = outdir / (variant + ".jsonl")
            raw_target = outdir / (variant + "_unscored.jsonl")
            if target.exists() or raw_target.exists():
                raise RuntimeError("Refuse to overwrite a generation artifact")
            for problem_index, ident in enumerate(ids):
                row = rows[ident]
                question = custom_prompts[problem_index] if custom_prompts is not None else row["problem"]
                prompt, input_ids = template(question)
                input_ids = input_ids.to("mps")
                prompt_len = input_ids.shape[1]
                for sample in range(n):
                    seed = int.from_bytes(hashlib.sha256(f"{ident}|{sample}".encode()).digest()[:4], "little")
                    torch.manual_seed(seed)
                    torch.mps.manual_seed(seed)
                    capture = Capture(requested_layers,
                                      mlp_layers=range(len(layers)) if mlp else (),
                                      hidden=bool(requested_layers), bin_size=bin_size,
                                      max_generated_tokens=cap)
                    start = time.time()
                    try:
                        with torch.inference_mode():
                            output = model.generate(input_ids=input_ids, max_new_tokens=cap, do_sample=True,
                                                    temperature=.6, top_p=.95, top_k=20,
                                                    logits_processor=[Presence(prompt_len)],
                                                    stopping_criteria=[Progress(prompt_len, ident, variant)],
                                                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                                    use_cache=True)
                    finally:
                        if bin_size is not None:
                            capture.close()
                    gen_ids = output[0, prompt_len:].cpu().tolist()
                    text = tok.decode(gen_ids, skip_special_tokens=False)
                    cap_hit = len(gen_ids) >= cap and gen_ids[-1] != tok.eos_token_id
                    grams = [tuple(gen_ids[j:j + 8]) for j in range(max(0, len(gen_ids) - 7))]
                    raw_rec = {"id": ident, "source": row["source"], "sample": sample, "seed": seed,
                               "model": variant, "prompt": prompt, "prompt_token_ids": input_ids[0].cpu().tolist(),
                               "text": text, "token_ids": gen_ids, "gold": row["answer"],
                               "tokens": len(gen_ids), "cutoff": cap_hit,
                               "repeat_8gram_fraction": 1 - len(set(grams)) / max(1, len(grams)),
                               "seconds": time.time() - start}
                    append(raw_target, raw_rec)
                    try:
                        pred = answer(text)
                        correct = grade(pred, row["answer"])
                    except Exception as exc:
                        capture.save(outdir / f"{variant}_{ident}_{sample}_activations.npz",
                                     generated_tokens=len(gen_ids),
                                     final_token_is_eos=bool(gen_ids and gen_ids[-1] == tok.eos_token_id))
                        raise RuntimeError(f"Unscored generation preserved in {raw_target}: {type(exc).__name__}: {exc}") from exc
                    capture.save(outdir / f"{variant}_{ident}_{sample}_activations.npz",
                                 generated_tokens=len(gen_ids),
                                 final_token_is_eos=bool(gen_ids and gen_ids[-1] == tok.eos_token_id))
                    failure = "correct" if correct else "cutoff" if cap_hit else "no_box" if pred is None else "wrong_answer"
                    rec = {**raw_rec, "prediction": pred, "correct": correct, "failure": failure}
                    append(target, rec)
                    records.append(rec)
                    print(json.dumps({k: rec[k] for k in ("id", "model", "tokens", "correct", "failure", "seconds")}), flush=True)
            arms[variant] = records
        summary = {v: {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                       "accuracy": float(np.mean([r["correct"] for r in rs])),
                       "mean_tokens": float(np.mean([r["tokens"] for r in rs])),
                       "cut": sum(r["cutoff"] for r in rs),
                       "failure_counts": dict(collections.Counter(r["failure"] for r in rs))} for v, rs in arms.items()}
        response = {"summary": summary}
        if len(variants) == 2:
            response["paired"] = paired_accuracy_summary(
                arms["original"], arms["edited"], ids, np
            )
        response["short_texts"] = [{"id": r["id"], "model": r["model"], "text_excerpt": r["text"][:300]} for rs in arms.values() for r in rs]
        return response

    def replay_request(req, pass_no, outdir):
        if req.get("model") not in {"original", "edited"}:
            raise ValueError("replay model must be original or edited")
        apply(req["model"], req.get("edit_dir", str(RUN / "edited")))
        sequences = req.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("replay requires a non-empty sequences list")
        acts = req.get("activations", {}) or {}
        requested_layers = acts.get("layers", [])
        mlp_output_layers = acts.get("mlp_outputs", [])
        mlp_input_bins = acts.get("mlp", False)
        bin_size = acts.get("bin_size")
        if any(type(i) is not int or i < 0 or i > 24 for i in requested_layers):
            raise ValueError("Replay residual layers must be 0..24")
        if any(type(i) is not int or i < 0 or i >= len(layers) for i in mlp_output_layers):
            raise ValueError("Replay MLP output layers must be 0..23")
        if type(mlp_input_bins) is not bool:
            raise ValueError("Replay activations.mlp must be boolean")
        if bin_size is not None:
            if type(bin_size) is not int or bin_size != 1024:
                raise ValueError("Replay activation bin_size must be 1024")
            if not requested_layers and not mlp_input_bins and not mlp_output_layers:
                raise ValueError("Replay activation bins require a selected residual or MLP hook")
        elif mlp_input_bins:
            raise ValueError("Replay MLP input capture requires activation bins")
        records = []

        class ForceSavedTokens(LogitsProcessor):
            def __init__(self, prompt_len, expected_ids):
                self.prompt_len = prompt_len
                self.expected_ids = expected_ids

            def __call__(self, input_ids, scores):
                position = input_ids.shape[1] - self.prompt_len
                if position < 0 or position >= len(self.expected_ids):
                    raise RuntimeError(f"Unexpected replay position {position}")
                scores.fill_(float("-inf"))
                scores[:, self.expected_ids[position]] = 0
                return scores

        for ix, item in enumerate(sequences):
            ident = item.get("id")
            if ident not in rows:
                raise ValueError(f"Replay ID is not in the editor TRAIN partition: {ident}")
            expected = item.get("expected_token_ids")
            ranges = item.get("capture_ranges")
            context_prefix = item.get("prefix_generated_token_ids", [])
            source_attempt = item.get("source_attempt")
            source_sample = item.get("source_sample")
            if (not isinstance(expected, list) or not expected or
                    any(type(token_id) is not int for token_id in expected)):
                raise ValueError("expected_token_ids must be a non-empty integer list")
            if (not isinstance(context_prefix, list) or
                    any(type(token_id) is not int for token_id in context_prefix)):
                raise ValueError("prefix_generated_token_ids must be an integer list")
            if bin_size is not None:
                if ranges is not None:
                    raise ValueError("Replay cannot combine activation bins with selected-position ranges")
                replay_count = len(expected)
                if replay_count > 32768:
                    raise ValueError("Binned replay is limited to the original 32768-token generation cap")
            else:
                if not isinstance(ranges, list) or not ranges:
                    raise ValueError("Replay requires capture_ranges when bins are not requested")
                if any(not isinstance(pair, list) or len(pair) != 2 or
                       any(type(v) is not int for v in pair) or pair[0] < 0 or pair[1] <= pair[0]
                       for pair in ranges):
                    raise ValueError("Invalid replay capture range")
                required = max(end for _, end in ranges)
                # One extra forced token ensures the final selected token is actually
                # consumed by the cached model path before generation stops.
                replay_count = required + 1
                if replay_count > len(expected):
                    raise ValueError("Replay prefix is too short for the capture window")
            eos_positions = [i for i, token_id in enumerate(expected[:replay_count]) if token_id == tok.eos_token_id]
            if eos_positions and eos_positions != [replay_count - 1]:
                raise ValueError("Replay sequence terminates at EOS before the capture window")
            prompt, input_ids = template(rows[ident]["problem"])
            prompt_ids = input_ids[0].cpu().tolist()
            if item.get("prompt_token_ids") != prompt_ids:
                raise ValueError(f"Saved prompt token IDs do not match the frozen TRAIN prompt for {ident}")
            if max(expected) >= len(tok) or min(expected) < 0:
                raise ValueError("Replay token ID is outside the tokenizer vocabulary")
            if any(token_id == tok.eos_token_id for token_id in context_prefix):
                raise ValueError("Replay context prefix already contains EOS")
            if context_prefix and max(context_prefix + prompt_ids) >= len(tok):
                raise ValueError("Replay context token ID is outside the tokenizer vocabulary")
            if source_attempt is not None:
                if type(source_attempt) is not int or type(source_sample) is not int:
                    raise ValueError("Source attempt and sample must be integers")
                source_path = PASSES / f"attempt_{source_attempt:04d}" / "branch.jsonl"
                if not source_path.is_file():
                    raise ValueError(f"Replay branch source is missing: {source_path}")
                source = next((json.loads(line) for line in source_path.read_text().splitlines()
                               if line.strip() and
                               json.loads(line).get("id") == ident and
                               json.loads(line).get("sample") == source_sample), None)
                if (source is None or source.get("prompt_token_ids") != prompt_ids or
                        source.get("prefix_generated_token_ids") != context_prefix or
                        source.get("suffix_token_ids") != expected):
                    raise ValueError(f"Replay branch token IDs do not match saved source for {ident}/{source_sample}")
            full_context = prompt_ids + context_prefix
            if len(full_context) + replay_count > 65536:
                raise ValueError("Replay context plus suffix exceeds supported sequence length")
            input_ids = torch.tensor([full_context], dtype=torch.long, device="mps")
            generation_start = input_ids.shape[1]
            if bin_size is not None:
                capture = Capture(requested_layers,
                                  mlp_layers=range(len(layers)) if mlp_input_bins else (),
                                  hidden=bool(requested_layers), bin_size=bin_size,
                                  max_generated_tokens=replay_count,
                                  mlp_output_layers=mlp_output_layers)
            else:
                capture = Capture(requested_layers, hidden=bool(requested_layers),
                                  capture_ranges=ranges, mlp_output_layers=mlp_output_layers)
            start = time.time()
            try:
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids, max_new_tokens=replay_count, do_sample=False,
                        logits_processor=[ForceSavedTokens(generation_start, expected[:replay_count])],
                        stopping_criteria=[Progress(generation_start, ident, req["model"])],
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    )
            finally:
                capture.close()
            actual = output[0, generation_start:].cpu().tolist()
            exact = actual == expected[:replay_count]
            raw = {"id": ident, "model": req["model"], "pass": pass_no,
                   "source_attempt": source_attempt, "source_sample": source_sample,
                   "context_prefix_token_count": len(context_prefix),
                   "expected_full_token_count": len(expected), "replayed_token_count": replay_count,
                   "expected_prefix": expected[:replay_count], "actual_prefix": actual,
                   "exact_prefix_match": exact, "capture_ranges_generated_token_positions": ranges,
                   "activation_bin_size": bin_size,
                   "seconds": time.time() - start}
            append(outdir / "replay.jsonl", raw)
            capture.save(outdir / f"replay_{ix:02d}_{ident}_activations.npz",
                         generated_tokens=len(actual),
                         final_token_is_eos=bool(actual and actual[-1] == tok.eos_token_id))
            if not exact:
                mismatch = next((j for j, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
                raise RuntimeError(f"Forced replay did not match saved prefix for {ident}; first mismatch={mismatch}")
            records.append({"id": ident, "replayed_tokens": len(actual),
                            "source_sample": source_sample,
                            "capture_positions": (sum(end - begin for begin, end in ranges)
                                                  if ranges else None),
                            "activation_bin_size": bin_size,
                            "exact_prefix_match": exact, "seconds": raw["seconds"]})
            print(json.dumps(records[-1]), flush=True)
            del output, capture, input_ids
            torch.mps.empty_cache()
        return {"summary": {"sequences": len(records), "replayed_tokens": sum(r["replayed_tokens"] for r in records),
                             "exact_prefix_matches": sum(r["exact_prefix_match"] for r in records)},
                "records": records}

    def branch_request(req, pass_no, outdir):
        if req.get("model") not in {"original", "edited"}:
            raise ValueError("branch model must be original or edited")
        apply(req["model"], req.get("edit_dir", str(RUN / "edited")))
        sequences = req.get("sequences")
        seeds = req.get("seeds")
        cap = int(req.get("max_suffix_tokens", 4096))
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("branch requires a non-empty sequences list")
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError("branch seeds must be unique unsigned 32-bit integers")
        if not 0 < cap <= 4096:
            raise ValueError("max_suffix_tokens must be 1..4096")
        target = outdir / "branch_unscored.jsonl"
        graded = outdir / "branch.jsonl"
        records = []
        for item in sequences:
            ident = item.get("id")
            source_attempt = item.get("source_attempt")
            if ident not in rows:
                raise ValueError(f"Branch ID is not in the editor TRAIN partition: {ident}")
            if type(source_attempt) is not int or source_attempt < 1:
                raise ValueError("Branch requires the source generation attempt number")
            source_path = PASSES / f"attempt_{source_attempt:04d}" / "original.jsonl"
            if not source_path.is_file():
                raise ValueError(f"Branch source attempt is missing: {source_path}")
            source_rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
            source = next((r for r in source_rows if r.get("id") == ident), None)
            if source is None:
                raise ValueError(f"Branch source output does not contain {ident}")
            prompt_ids = item.get("prompt_token_ids")
            base_ids = item.get("base_token_ids")
            branch_position = item.get("branch_position")
            phrase_ids = item.get("phrase_token_ids")
            if (not isinstance(base_ids, list) or not base_ids or
                    any(type(token_id) is not int for token_id in base_ids) or
                    not isinstance(prompt_ids, list) or not prompt_ids or
                    any(type(token_id) is not int for token_id in prompt_ids)):
                raise ValueError("Branch requires exact saved prompt and base token IDs")
            if (type(branch_position) is not int or not 0 <= branch_position < len(base_ids) or
                    branch_position + cap > 65536):
                raise ValueError("Invalid branch position or combined sequence length")
            if source.get("prompt_token_ids") != prompt_ids or source.get("token_ids") != base_ids:
                raise ValueError(f"Branch source token IDs differ from saved attempt for {ident}")
            if (not isinstance(phrase_ids, list) or len(phrase_ids) != 16 or
                    any(type(token_id) is not int for token_id in phrase_ids)):
                raise ValueError("phrase_token_ids must contain the fixed 16-token diagnostic phrase")
            prompt, prompt_tensor = template(rows[ident]["problem"])
            if prompt_tensor[0].tolist() != prompt_ids:
                raise ValueError(f"Saved prompt token IDs do not match frozen TRAIN prompt for {ident}")
            if any(token_id == tok.eos_token_id for token_id in base_ids[:branch_position]):
                raise ValueError("Branch prefix already contains EOS")
            if max(base_ids + prompt_ids + phrase_ids) >= len(tok) or min(base_ids + prompt_ids + phrase_ids) < 0:
                raise ValueError("Branch token ID is outside tokenizer vocabulary")
            full_prefix = prompt_ids + base_ids[:branch_position]
            if len(full_prefix) + cap > 65536:
                raise ValueError("Branch continuation exceeds supported combined context length")
            input_ids = torch.tensor([full_prefix], dtype=torch.long, device="mps")
            prompt_len = len(prompt_ids)
            for sample, seed in enumerate(seeds):
                torch.manual_seed(seed)
                torch.mps.manual_seed(seed)
                started = time.time()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids, max_new_tokens=cap, do_sample=True,
                        temperature=.6, top_p=.95, top_k=20,
                        logits_processor=[Presence(prompt_len)],
                        stopping_criteria=[Progress(prompt_len + branch_position, ident, req["model"])],
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    )
                suffix = output[0, input_ids.shape[1]:].cpu().tolist()
                text = tok.decode(suffix, skip_special_tokens=False)
                combined = base_ids[:branch_position] + suffix
                phrase_positions = [p for p in range(branch_position, len(combined) - 15)
                                    if combined[p:p + 16] == phrase_ids]
                repeat_windows = []
                for start in range(0, max(0, len(suffix) - 255), 256):
                    window = suffix[start:start + 256]
                    grams = [tuple(window[j:j + 8]) for j in range(max(0, len(window) - 7))]
                    fraction = 1 - len(set(grams)) / max(1, len(grams))
                    repeat_windows.append({"start": start, "repeat_8gram_fraction": fraction})
                cap_hit = len(suffix) >= cap and suffix[-1] != tok.eos_token_id
                raw = {"id": ident, "sample": sample, "seed": seed, "branch_position_generated_token": branch_position,
                       "source_attempt": source_attempt,
                       "model": req["model"], "prompt_token_ids": prompt_ids,
                       "prefix_generated_token_ids": base_ids[:branch_position],
                       "phrase_token_ids": phrase_ids, "suffix_token_ids": suffix, "text": text,
                       "tokens": len(suffix), "cutoff_4096": cap_hit,
                       "eos": bool(suffix and suffix[-1] == tok.eos_token_id),
                       "think_closed": "</think>" in text,
                       "boxed_count": text.count("\\boxed"),
                       "phrase_occurrence_positions_in_combined_generation": phrase_positions,
                       "repeat_8gram_windows": repeat_windows,
                       "sampling": {"temperature": .6, "top_p": .95, "top_k": 20,
                                    "presence_penalty": 1.5, "penalty_history_start": prompt_len},
                       "seconds": time.time() - started}
                append(target, raw)
                try:
                    prediction = answer(text)
                    correct = grade(prediction, rows[ident]["answer"])
                except Exception as exc:
                    raise RuntimeError(f"Unscored branch generation preserved in {target}: {type(exc).__name__}: {exc}") from exc
                record = {**raw, "prediction": prediction, "correct": correct,
                          "failure": "correct" if correct else "cutoff" if cap_hit else "no_box" if prediction is None else "wrong_answer"}
                append(graded, record)
                records.append(record)
                print(json.dumps({k: record[k] for k in ("id", "sample", "tokens", "correct", "failure", "seconds")}), flush=True)
        return {"summary": {"n": len(records), "correct": sum(r["correct"] for r in records),
                             "cutoff_4096": sum(r["cutoff_4096"] for r in records),
                             "think_closed": sum(r["think_closed"] for r in records)},
                "branches": [{"id": r["id"], "sample": r["sample"], "prediction": r["prediction"],
                              "correct": r["correct"], "failure": r["failure"],
                              "phrase_occurrences": len(r["phrase_occurrence_positions_in_combined_generation"])}
                             for r in records]}

    def tf_request(req, pass_no, outdir):
        if req.get("model") not in {"original", "edited"}:
            raise ValueError("teacher_forced model must be original or edited")
        apply(req["model"], req.get("edit_dir", str(RUN / "edited")))
        texts = req.get("texts")
        chats = req.get("chats")
        if (texts is None) == (chats is None):
            raise ValueError("Provide exactly one of texts or chats")
        items = texts if texts is not None else chats
        if not 0 < len(items) <= 1024:
            raise ValueError("teacher_forced accepts 1..1024 texts")
        ret = req.get("return", {}) or {}
        topk = int(ret.get("topk", 0) or 0)
        if not 0 <= topk <= 50:
            raise ValueError("teacher_forced topk must be between 0 and 50")
        attn_layers = ret.get("attn_layers", []) or []
        hidden_layers = ret.get("hidden_states", []) or []
        mlp_layers = ret.get("mlp_act", []) or []
        total = 0
        metrics = []
        prepared = []
        for ix, item in enumerate(items):
            if chats is not None:
                messages = item
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                               enable_thinking=req.get("enable_thinking", True))
                enc = tok.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                              add_generation_prompt=False,
                                              enable_thinking=req.get("enable_thinking", True))
                inputs = enc["input_ids"].to("mps")
            else:
                text = item.get("text", "") if isinstance(item, dict) else item
                inputs = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("mps")
            length = inputs.shape[1]
            total += length
            if length > 32768 or total > 524288:
                raise ValueError("Teacher-forced token limit exceeded")
            if attn_layers and (ix >= 2 or length > 2048 or any(i not in {3, 7, 11, 15, 19, 23} for i in attn_layers)):
                raise ValueError("Attention probabilities: max 2 texts, 2048 tokens, full-attention layers only")
            prepared.append((item, text, inputs, length))

        if any(not isinstance(i, int) or i < 0 or i > 24 for i in hidden_layers):
            raise ValueError("hidden_states layers must be integers in 0..24")
        if any(not isinstance(i, int) or i < 0 or i >= len(layers) for i in mlp_layers):
            raise ValueError("mlp_act layers must be integers in 0..23")
        hidden_width = int(model.config.text_config.hidden_size)
        mlp_width = int(model.config.text_config.intermediate_size)
        attention_heads = int(model.config.text_config.num_attention_heads)
        estimated_output_bytes = 0
        for _, _, _, length in prepared:
            estimated_output_bytes += length * (12 + 12 * topk)
            estimated_output_bytes += length * hidden_width * 4 * len(hidden_layers)
            estimated_output_bytes += length * mlp_width * 4 * len(mlp_layers)
            estimated_output_bytes += sum(attention_heads * length * length * 4 for _ in attn_layers)
            estimated_output_bytes += 4096
        if estimated_output_bytes > 6 * 1024**3:
            raise ValueError(
                f"Estimated uncompressed teacher-forced outputs exceed 6 GiB: {estimated_output_bytes} bytes"
            )

        for ix, (item, text, inputs, length) in enumerate(prepared):
            capture = Capture(hidden_layers, mlp_layers=mlp_layers,
                              hidden=bool(hidden_layers), teacher=True)
            with torch.inference_mode():
                out = model(input_ids=inputs, use_cache=False, output_attentions=bool(attn_layers),
                            output_hidden_states=False, return_dict=True)
            logprobs = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
            target = inputs[0, 1:]
            arrays = {"token_ids": inputs[0].cpu().numpy(), "logprobs": logprobs.gather(1, target[:, None]).squeeze(1).cpu().numpy()}
            if topk:
                vals, idxs = torch.topk(logprobs, k=min(topk, logprobs.shape[-1]), dim=-1)
                arrays["topk_ids"] = idxs.cpu().numpy(); arrays["topk_logprobs"] = vals.cpu().numpy()
            capture.save(outdir / f"text_{ix:04d}_activations.npz")
            if attn_layers:
                layer_types = model.config.text_config.layer_types
                full_attention_layers = [
                    layer_index for layer_index, layer_type in enumerate(layer_types)
                    if layer_type == "full_attention"
                ]
                if len(out.attentions) == len(layer_types):
                    attention_by_layer = dict(enumerate(out.attentions))
                elif len(out.attentions) == len(full_attention_layers):
                    attention_by_layer = dict(zip(full_attention_layers, out.attentions))
                else:
                    raise ValueError(
                        "Unexpected attention tuple length: "
                        f"{len(out.attentions)} for full layers {full_attention_layers}"
                    )
                for li in attn_layers:
                    layer_attn = attention_by_layer[li]
                    arrays[f"attention_{li}"] = layer_attn.detach().float().cpu().numpy()
            np.savez_compressed(outdir / f"text_{ix:04d}_scores.npz", **arrays)
            metrics.append({"index": ix, "label": item.get("label", str(ix)) if isinstance(item, dict) else str(ix),
                            "tokens": length, "mean_logprob": float(arrays["logprobs"].mean())})
            del out
            del logprobs
            torch.mps.empty_cache()
        return {"summary": {"texts": len(items), "tokens": total, "metrics": metrics},
                "note": "token logprobs/top-k and requested attention probabilities saved; hidden state/MLP capture availability recorded per file"}

    if not (RUN / "worker_ready.json").exists():
        (RUN / "worker_ready.json").write_text(json.dumps({"pid": os.getpid(), "device": "mps", "dtype": "bfloat16",
                                                           "model_sha256": digest(TENSOR_FILE), "train_rows": len(rows)}))
    print(json.dumps({"status": "ready", "pid": os.getpid(), "device": "mps", "train_rows": len(rows)}), flush=True)
    while True:
        pending = sorted(p for p in REQUESTS.glob("*.json") if not (RESPONSES / p.name).exists())
        if not pending:
            time.sleep(.5)
            continue
        p = pending[0]
        wrapper = json.loads(p.read_text())
        req = wrapper
        expanded_request_path = None
        if "request_file" in wrapper:
            if set(wrapper) != {"type", "request_file"}:
                raise ValueError("request_file wrappers may contain only type and request_file")
            source = wrapper["request_file"]
            if not isinstance(source, str) or not source.startswith("/work/"):
                raise ValueError("request_file must name a file under /work")
            expanded_request_path = (RUN / source.removeprefix("/work/")).resolve()
            if RUN.resolve() not in expanded_request_path.parents or not expanded_request_path.is_file():
                raise ValueError("request_file is missing or outside /work")
            req = json.loads(expanded_request_path.read_text())
            if req.get("type") != wrapper["type"]:
                raise ValueError("wrapper type differs from the JSON request file")
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()] if ledger_path.exists() else []
        consumed = sum(1 for event in ledger if event.get("status") == "success")
        if consumed >= 50:
            raise RuntimeError("Fresh run's 50-pass forward budget exhausted")
        pass_no = consumed + 1
        attempt_no = len(ledger) + 1
        outdir = PASSES / f"attempt_{attempt_no:04d}"
        outdir.mkdir(parents=True, exist_ok=False)
        response = {"pass": pass_no, "request": p.name, "status": "running", "start": time.time(),
                    "edit_dir": req.get("edit_dir", str(RUN / "edited"))}
        try:
            request_hash = digest(p)
            response["request_sha256"] = request_hash
            if expanded_request_path is not None:
                response["expanded_request_file"] = str(expanded_request_path)
                response["expanded_request_sha256"] = digest(expanded_request_path)
            edit_dir = Path(req.get("edit_dir", str(RUN / "edited")))
            response["edit_sha256"] = {q.stem: digest(q) for q in edit_dir.glob("*.npy")}
            if req.get("type") == "generate":
                response.update(gen_request(req, pass_no, outdir))
            elif req.get("type") == "replay":
                response.update(replay_request(req, pass_no, outdir))
            elif req.get("type") == "branch":
                response.update(branch_request(req, pass_no, outdir))
            elif req.get("type") == "teacher_forced":
                response.update(tf_request(req, pass_no, outdir))
            else:
                raise ValueError("Unknown forward request type")
            response.update(status="success", seconds=time.time() - response["start"], output_dir=str(outdir))
        except Exception:
            response.update(status="failed", seconds=time.time() - response["start"], error=traceback.format_exc())
            print(response["error"], flush=True)
        response["request_sha256"] = digest(p)
        append(ledger_path, response)
        (RESPONSES / p.name).write_text(json.dumps(response, ensure_ascii=False, indent=2))
        print(json.dumps({k: response[k] for k in ("pass", "request", "status", "seconds", "summary") if k in response}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
