#!/usr/bin/env python3
"""Freeze token-only matched-context activation ranges for fresh Astra pass 9.

This script selects saved TRAIN generations before opening any activation
archive. It writes a local-only manifest and replay request; it never edits
weights or reads model activations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/astra_reproduction_20261111_fresh50/raw/editor_work"
PASSES = RUN / "forward/passes"
BASE = Path(os.environ.get("WSP_BASE_DIR", ROOT / "model-cache/Qwen3.5-0.8B"))
OUT = RUN / "analysis"
REQUESTS = RUN / "forward/requests"
END_THINK = 248069
START_THINK = 248068
POSITION_LIMIT = 8192
NGRAM = 8
POSITION_BUDGET_PER_SEQUENCE = 256
MAX_MATCHES_PER_ITEM_PAIR = 4
MAX_REPEAT_PAIRS_PER_FAILURE = 2


def read_rows(attempt: int) -> list[dict]:
    path = PASSES / f"attempt_{attempt:04d}" / "original.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compress_positions(positions: set[int]) -> list[list[int]]:
    ordered = sorted(positions)
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for pos in ordered[1:]:
        if pos == previous + 1:
            previous = pos
        else:
            ranges.append([start, previous + 1])
            start = previous = pos
    ranges.append([start, previous + 1])
    return ranges


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    if tokenizer.convert_tokens_to_ids("<think>") != START_THINK:
        raise RuntimeError("Pinned tokenizer <think> token ID changed")
    if tokenizer.convert_tokens_to_ids("</think>") != END_THINK:
        raise RuntimeError("Pinned tokenizer </think> token ID changed")

    prior = read_rows(1)
    fresh = read_rows(14)
    successes = [row for row in prior if row.get("correct") is True]
    failures = [row for row in fresh if row.get("correct") is False and row.get("cutoff") is True]
    if len(successes) != 5 or len(failures) != 12:
        raise RuntimeError(f"Expected the frozen 5-success/12-failure groups, got {len(successes)}/{len(failures)}")

    streams: dict[str, dict] = {}
    for group, rows in (("success", successes), ("failure", failures)):
        for row in rows:
            key = f"{group}:{row['id']}:{row['sample']}"
            ids = row["token_ids"]
            try:
                think_start = ids.index(START_THINK) + 1
            except ValueError:
                think_start = 0
            try:
                think_end = ids.index(END_THINK, think_start)
            except ValueError:
                think_end = len(ids)
            end = min(think_end, POSITION_LIMIT, len(ids))
            begin = min(think_start, end)
            content = ids[begin:end]
            grams: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for local_pos in range(max(0, len(content) - NGRAM + 1)):
                gram = tuple(content[local_pos:local_pos + NGRAM])
                phrase = tokenizer.decode(list(gram), skip_special_tokens=True)
                if len(re.findall(r"[A-Za-z0-9]+", phrase)) >= 2:
                    grams[gram].append(begin + local_pos)
            streams[key] = {
                "group": group, "row": row, "ids": ids,
                "thinking_start": begin, "thinking_end": end,
                "grams": grams, "positions": set(), "selection_counts": defaultdict(int),
            }

    # Exact shared 8-grams, restricted to the first fixed 8192 generated-token
    # positions and the thinking span. Each success/failure item pair contributes
    # at most four token-selected matches. Hash order, not activation magnitude,
    # resolves all remaining ties.
    cross_candidates = []
    success_keys = sorted(k for k, s in streams.items() if s["group"] == "success")
    failure_keys = sorted(k for k, s in streams.items() if s["group"] == "failure")
    for sk in success_keys:
        for fk in failure_keys:
            sg, fg = streams[sk]["grams"], streams[fk]["grams"]
            for gram in sg.keys() & fg.keys():
                occ_pairs = [(sp, fp) for sp in sg[gram] for fp in fg[gram]
                             if abs(sp - fp) <= 512]
                if not occ_pairs:
                    continue
                sp, fp = min(occ_pairs, key=lambda pair: (abs(pair[0] - pair[1]), pair[0], pair[1]))
                canonical = json.dumps([sk, fk, gram, sp, fp], separators=(",", ":"))
                cross_candidates.append((sha256(canonical), sk, fk, gram, sp, fp))

    selected_matches = []
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for digest, sk, fk, gram, sp, fp in sorted(cross_candidates):
        pair = (sk, fk)
        if pair_counts[pair] >= MAX_MATCHES_PER_ITEM_PAIR:
            continue
        spos = set(range(sp, sp + NGRAM))
        fpos = set(range(fp, fp + NGRAM))
        # Add a fixed same-offset background control when it stays in the
        # thinking span. It is selected without looking at activation values.
        s_bg = set(range(sp + 256, sp + 256 + NGRAM))
        f_bg = set(range(fp + 256, fp + 256 + NGRAM))
        if max(s_bg) >= streams[sk]["thinking_end"]:
            s_bg = set()
        if max(f_bg) >= streams[fk]["thinking_end"]:
            f_bg = set()
        sadd = spos | s_bg
        fadd = fpos | f_bg
        if len(streams[sk]["positions"] | sadd) > POSITION_BUDGET_PER_SEQUENCE:
            continue
        if len(streams[fk]["positions"] | fadd) > POSITION_BUDGET_PER_SEQUENCE:
            continue
        streams[sk]["positions"].update(sadd)
        streams[fk]["positions"].update(fadd)
        pair_counts[pair] += 1
        selected_matches.append({"success_stream": sk, "failure_stream": fk,
                                 "ngram_token_ids": list(gram), "success_position": sp,
                                 "failure_position": fp, "position_delta": fp - sp,
                                 "success_background_position": sp + 256 if s_bg else None,
                                 "failure_background_position": fp + 256 if f_bg else None,
                                 "selection_hash": digest})

    # Within each failed trajectory, compare the first and second occurrence of
    # a repeated thinking 8-gram, again using token-only deterministic ranking.
    repeated_candidates = []
    for fk in failure_keys:
        stream = streams[fk]
        for gram, occ in stream["grams"].items():
            if len(occ) < 2:
                continue
            first = occ[0]
            later = next((pos for pos in occ[1:] if pos > first and pos - first <= 512), None)
            if later is None:
                continue
            canonical = json.dumps([fk, gram, first, later], separators=(",", ":"))
            repeated_candidates.append((sha256(canonical), fk, gram, first, later))
    repeated_selected = []
    repeated_count: dict[str, int] = defaultdict(int)
    for digest, fk, gram, first, later in sorted(repeated_candidates):
        if repeated_count[fk] >= MAX_REPEAT_PAIRS_PER_FAILURE:
            continue
        add = set(range(first, first + NGRAM)) | set(range(later, later + NGRAM))
        if len(streams[fk]["positions"] | add) > POSITION_BUDGET_PER_SEQUENCE:
            continue
        streams[fk]["positions"].update(add)
        repeated_count[fk] += 1
        repeated_selected.append({"failure_stream": fk, "ngram_token_ids": list(gram),
                                  "first_position": first, "later_position": later,
                                  "position_delta": later - first,
                                  "selection_hash": digest})

    manifest_streams = []
    request_sequences = []
    for key in sorted(streams):
        stream = streams[key]
        ranges = compress_positions(stream["positions"])
        row = stream["row"]
        if not ranges:
            continue
        manifest_streams.append({"stream_key": key, "group": stream["group"],
                                 "id": row["id"], "sample": row["sample"],
                                 "source_attempt": 1 if stream["group"] == "success" else 14,
                                 "generated_tokens": len(stream["ids"]),
                                 "thinking_bounds_generated_token_positions":[stream["thinking_start"],stream["thinking_end"]],
                                 "selected_position_count": len(stream["positions"]),
                                 "capture_ranges": ranges})
        request_sequences.append({"id": row["id"], "prompt_token_ids": row["prompt_token_ids"],
                                  "expected_token_ids": row["token_ids"],
                                  "capture_ranges": ranges})

    request = {"type":"replay","model":"original",
               "activations":{"layers":list(range(25)),"mlp_outputs":list(range(24))},
               "sequences":request_sequences}
    request_path = REQUESTS / "0015_matched_thinking_8gram_contexts.json"
    manifest = {"purpose":"Exploratory token-matched activation localization; no causal or edit claim.",
                "selection_rule":{"success_group":"All five correct completed rows from attempt 0001.",
                                  "failure_group":"All 12 cap-hit incorrect rows from pass 0008.",
                                  "thinking_token_ids":{"start":START_THINK,"end":END_THINK},
                                  "position_window":"First 8192 generated-token positions, restricted to thinking span.",
                                  "shared_phrase":"Exact token-ID 8-gram decoded to at least two ASCII alphanumeric word groups.",
                                  "cross_group_match":"Same 8-gram; occurrence starts within 512 generated-token positions.",
                                  "cross_group_order":"SHA-256 order of canonical IDs, token IDs, and occurrence positions; at most four matches per success/failure stream pair.",
                                  "within_failure":"First and next occurrence of same 8-gram within 512 positions; SHA-256 order; at most two pairs per failure stream.",
                                  "background":"For selected cross-group matches, fixed +256-token offset when it stays inside the thinking span.",
                                  "per_stream_capture_budget":POSITION_BUDGET_PER_SEQUENCE,
                                  "feature_selection":"Token IDs and group labels only; activation archives were not read."},
                "selected_cross_group_matches":selected_matches,
                "selected_within_failure_repeats":repeated_selected,
                "streams":manifest_streams,
                "replay_request":"results/astra_reproduction_20261111_fresh50/raw/editor_work/forward/requests/0015_matched_thinking_8gram_contexts.json"}
    OUT.mkdir(parents=True, exist_ok=True)
    REQUESTS.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, ensure_ascii=False)+"\n")
    manifest_path = OUT / "preregistration_0015_matched_thinking_8gram_contexts.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"request":str(request_path),"manifest":str(manifest_path),
                      "streams":len(request_sequences),"shared_matches":len(selected_matches),
                      "within_failure_pairs":len(repeated_selected),
                      "captured_positions":sum(x["selected_position_count"] for x in manifest_streams)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
