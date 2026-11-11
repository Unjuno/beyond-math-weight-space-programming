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
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, LogitsProcessor, StoppingCriteria

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
        def __init__(self, requested_layers=(), mlp_layers=None, hidden=False, teacher=False):
            self.layers = set(requested_layers)
            self.mlp_layers = set(mlp_layers or ())
            self.hidden = bool(hidden)
            self.teacher = teacher
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

        def _hook(self, key, input_mode):
            def capture(module, args, output=None):
                x = args[0] if input_mode else output
                if isinstance(x, tuple):
                    x = x[0]
                if x is None or not hasattr(x, "detach"):
                    return
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                if x.ndim != 3:
                    return
                call = self.calls[key]
                self.calls[key] += 1
                d = self.data.setdefault(key, {"prompt": None, "sum": None, "count": 0})
                z = x.detach().float()
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

        def save(self, path):
            arrays = {}
            for name, d in self.data.items():
                if d["prompt"] is not None:
                    arrays[name + "_prompt_last"] = d["prompt"]
                if d["sum"] is not None and d["count"]:
                    arrays[name + ("_teacher_tokens" if self.teacher else "_answer_mean_first8192")] = d["sum"] if self.teacher else d["sum"] / d["count"]
            if arrays:
                np.savez_compressed(path, **arrays)
            for handle in self.handles:
                handle.remove()

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
        if any(i < 0 or i > 24 for i in requested_layers):
            raise ValueError("Activation layers must be 0..24")
        arms = {}
        for variant in variants:
            apply(variant, req.get("edit_dir", str(RUN / "edited")))
            records = []
            target = outdir / (variant + ".jsonl")
            if target.exists():
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
                                      hidden=bool(requested_layers))
                    start = time.time()
                    with torch.inference_mode():
                        output = model.generate(input_ids=input_ids, max_new_tokens=cap, do_sample=True,
                                                temperature=.6, top_p=.95, top_k=20,
                                                logits_processor=[Presence(prompt_len)],
                                                stopping_criteria=[Progress(prompt_len, ident, variant)],
                                                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                                use_cache=True)
                    gen_ids = output[0, prompt_len:].cpu().tolist()
                    text = tok.decode(gen_ids, skip_special_tokens=False)
                    pred = answer(text)
                    correct = grade(pred, row["answer"])
                    cap_hit = len(gen_ids) >= cap and gen_ids[-1] != tok.eos_token_id
                    grams = [tuple(gen_ids[j:j + 8]) for j in range(max(0, len(gen_ids) - 7))]
                    rec = {"id": ident, "source": row["source"], "sample": sample, "seed": seed,
                           "model": variant, "prompt": prompt, "prompt_token_ids": input_ids[0].cpu().tolist(),
                           "text": text, "token_ids": gen_ids, "prediction": pred, "gold": row["answer"],
                           "correct": correct, "tokens": len(gen_ids), "cutoff": cap_hit,
                           "failure": "correct" if correct else "cutoff" if cap_hit else "no_box" if pred is None else "wrong_answer",
                           "repeat_8gram_fraction": 1 - len(set(grams)) / max(1, len(grams)),
                           "seconds": time.time() - start}
                    append(target, rec)
                    capture.save(outdir / f"{variant}_{ident}_{sample}_activations.npz")
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
            for li in hidden_layers:
                # Requested states are captured by explicit model output below if enabled.
                pass
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
            del out, logprobs
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
