#!/usr/bin/env python3
"""Analyze preregistered token-matched activation replays at item level."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/astra_reproduction_20261111_fresh50/raw/editor_work"
PASS = RUN / "forward/passes/attempt_0015"
MANIFEST = RUN / "analysis/preregistration_0015_matched_thinking_8gram_contexts.json"
OUT = RUN / "analysis"


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else None


def summarize_item_vectors(vectors: dict[tuple[str, str], list[np.ndarray]]) -> tuple[dict, np.ndarray | None]:
    item_vectors = {key: np.mean(np.stack(values), axis=0)
                    for key, values in vectors.items() if values}
    if not item_vectors:
        return {"item_pair_count": 0}, None
    keys = sorted(item_vectors)
    matrix = np.stack([item_vectors[key] for key in keys])
    mean_direction = np.mean(matrix, axis=0)
    pairwise = [cosine(matrix[i], matrix[j]) for i in range(len(matrix))
                for j in range(i + 1, len(matrix))]
    pairwise = [v for v in pairwise if v is not None]
    loo_success, loo_failure = [], []
    for side in (0, 1):
        names = sorted({key[side] for key in keys})
        scores = []
        for name in names:
            kept = [item_vectors[key] for key in keys if key[side] != name]
            if kept:
                scores.append(cosine(mean_direction, np.mean(np.stack(kept), axis=0)))
        (loo_success if side == 0 else loo_failure).extend(v for v in scores if v is not None)
    dominant = np.argsort(np.abs(mean_direction))[-16:][::-1]
    summary = {
        "item_pair_count": len(item_vectors),
        "mean_pairwise_cosine": float(np.mean(pairwise)) if pairwise else None,
        "median_pairwise_cosine": float(np.median(pairwise)) if pairwise else None,
        "leave_one_success_item_out_cosine_min": min(loo_success) if loo_success else None,
        "leave_one_failure_item_out_cosine_min": min(loo_failure) if loo_failure else None,
        "dominant_feature_indices_by_abs_mean": [int(i) for i in dominant],
        "dominant_feature_mean_deltas": [float(mean_direction[i]) for i in dominant],
    }
    return summary, mean_direction


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    replay_rows = [json.loads(s) for s in (PASS / "replay.jsonl").read_text().splitlines() if s.strip()]
    streams = manifest["streams"]
    if len(streams) != 17 or len(replay_rows) != len(streams):
        raise RuntimeError(f"Expected 17 manifest/replay streams, got {len(streams)}/{len(replay_rows)}")

    arrays_by_stream = {}
    positions_by_stream = {}
    archive_checks = []
    for index, (stream, record) in enumerate(zip(streams, replay_rows)):
        if record.get("id") != stream["id"] or not record.get("exact_prefix_match"):
            raise RuntimeError(f"Replay provenance or token parity mismatch at sequence {index}")
        path = PASS / f"replay_{index:02d}_{stream['id']}_activations.npz"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing activation archive: {path}")
        arrays = np.load(path)
        if not all(np.isfinite(arrays[name]).all() for name in arrays.files):
            raise RuntimeError(f"Non-finite activation values: {path}")
        selected_positions = [p for a, b in stream["capture_ranges"] for p in range(a, b)]
        if len(selected_positions) != stream["selected_position_count"]:
            raise RuntimeError(f"Selection count mismatch in {stream['stream_key']}")
        arrays_by_stream[stream["stream_key"]] = arrays
        positions_by_stream[stream["stream_key"]] = {pos: slot for slot, pos in enumerate(selected_positions)}
        archive_checks.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes":path.stat().st_size,"array_count":len(arrays.files),
                               "selected_positions":len(selected_positions),"all_finite":True})

    feature_specs = [(f"hidden_{layer}", layer) for layer in range(25)] + \
                    [(f"mlp_out_{layer}", layer) for layer in range(24)]
    cross_vectors: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {
        name: defaultdict(list) for name, _ in feature_specs}
    background_vectors: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {
        name: defaultdict(list) for name, _ in feature_specs}
    within_vectors: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name, _ in feature_specs}

    def span_mean(stream_key: str, feature: str, start: int, width: int = 8) -> np.ndarray:
        pos_map = positions_by_stream[stream_key]
        slots = [pos_map[pos] for pos in range(start, start + width)]
        return np.mean(arrays_by_stream[stream_key][feature + "_selected_values"][slots], axis=0)

    used_cross = 0
    for match in manifest["selected_cross_group_matches"]:
        sk, fk = match["success_stream"], match["failure_stream"]
        success_id = sk.split(":", 2)[1]
        failure_id = fk.split(":", 2)[1]
        key = (success_id, failure_id)
        sp, fp = match["success_position"], match["failure_position"]
        sb, fb = match["success_background_position"], match["failure_background_position"]
        for feature, _layer in feature_specs:
            cross_vectors[feature][key].append(span_mean(fk, feature, fp) - span_mean(sk, feature, sp))
            if sb is not None and fb is not None:
                background_vectors[feature][key].append(span_mean(fk, feature, fb) - span_mean(sk, feature, sb))
        used_cross += 1

    used_within = 0
    for repeat in manifest["selected_within_failure_repeats"]:
        stream_key = repeat["failure_stream"]
        failure_id = stream_key.split(":", 2)[1]
        for feature, _layer in feature_specs:
            within_vectors[feature][failure_id].append(
                span_mean(stream_key, feature, repeat["later_position"]) -
                span_mean(stream_key, feature, repeat["first_position"]))
        used_within += 1

    result = {
        "status":"exploratory_descriptive_analysis",
        "manifest":"preregistration_0015_matched_thinking_8gram_contexts.json",
        "streams":len(streams),
        "exact_replays":sum(bool(r["exact_prefix_match"]) for r in replay_rows),
        "replayed_tokens":sum(r["replayed_token_count"] for r in replay_rows),
        "cross_group_matches_analyzed":used_cross,
        "within_failure_repeat_pairs_analyzed":used_within,
        "activation_archives":archive_checks,
        "cross_group_by_feature":{},
        "background_by_feature":{},
        "within_failure_by_feature":{},
        "caveat":"Observational, outcome-conditioned TRAIN analysis. Token positions and pairwise item contrasts are not independent; no causal or edit effect is established.",
    }
    vectors_to_save = {}
    for feature, layer in feature_specs:
        summary, direction = summarize_item_vectors(cross_vectors[feature])
        result["cross_group_by_feature"][feature] = summary
        if direction is not None:
            vectors_to_save["cross_mean_" + feature] = direction
        bg_summary, bg_direction = summarize_item_vectors(background_vectors[feature])
        result["background_by_feature"][feature] = bg_summary
        if bg_direction is not None:
            vectors_to_save["background_mean_" + feature] = bg_direction
        group_vectors = {key: values for key, values in within_vectors[feature].items()}
        item_vectors = {key: np.mean(np.stack(values), axis=0) for key, values in group_vectors.items() if values}
        if item_vectors:
            matrix = np.stack([item_vectors[key] for key in sorted(item_vectors)])
            mean_direction = np.mean(matrix, axis=0)
            pairwise = [cosine(matrix[i], matrix[j]) for i in range(len(matrix))
                        for j in range(i + 1, len(matrix))]
            pairwise = [x for x in pairwise if x is not None]
            dom = np.argsort(np.abs(mean_direction))[-16:][::-1]
            result["within_failure_by_feature"][feature] = {
                "failure_item_count":len(item_vectors),
                "mean_pairwise_cosine":float(np.mean(pairwise)) if pairwise else None,
                "dominant_feature_indices_by_abs_mean":[int(i) for i in dom],
                "dominant_feature_mean_deltas":[float(mean_direction[i]) for i in dom],
            }
            vectors_to_save["within_mean_" + feature] = mean_direction
        else:
            result["within_failure_by_feature"][feature] = {"failure_item_count":0}

    vector_path = OUT / "pass09_descriptive_direction_vectors.npz"
    np.savez_compressed(vector_path, **vectors_to_save)
    vector_hash = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    result["derived_vector_file"] = vector_path.name
    result["derived_vector_sha256"] = vector_hash
    result_path = OUT / "pass09_matched_activation_analysis.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary_file":str(result_path),"vector_file":str(vector_path),
                      "cross_pairs":used_cross,"within_pairs":used_within,
                      "archives":len(archive_checks),"all_finite":all(x["all_finite"] for x in archive_checks),
                      "vector_sha256":vector_hash},ensure_ascii=False))


if __name__ == "__main__":
    main()
