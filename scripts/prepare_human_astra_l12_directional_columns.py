#!/usr/bin/env python3
"""Prepare, but do not apply, frozen BF16 L12 directional edit columns.

The base checkpoint is read-only. The three small column artifacts are written
to an ignored, new output directory and can later be applied in memory by a
separate human-operated evaluator. This script refuses an existing output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RAW_ROOT = ROOT / "results/human_astra_manual_20261105/raw"
EXPECTED_BASE_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
TENSOR_KEY = "model.language_model.layers.12.mlp.down_proj.weight"
COORDINATE = 1226
INTERVENTION_ID = "human-astra-manual-20261105-l12c1226-direction-v1"
CONTROL_SEED = INTERVENTION_ID + "|signed-control-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bf16_abs_fraction(bits: int) -> Fraction:
    """Exact value of a non-negative finite BF16 bit pattern."""
    if not 0 <= bits <= 0x7F7F:
        raise ValueError(f"Not a non-negative finite BF16 pattern: {bits:#06x}")
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exponent == 0:
        return Fraction(mantissa, 1 << 133)
    significand = 128 + mantissa
    power = exponent - 134
    return Fraction(significand << power, 1) if power >= 0 else Fraction(significand, 1 << -power)


FINITE_ABS = [(bf16_abs_fraction(bits), bits) for bits in range(0x0000, 0x7F80)]
ABS_BITS_BY_VALUE = {value: bits for value, bits in FINITE_ABS}


def nearest_bf16_abs(value: Fraction) -> int:
    """Round an exact non-negative rational to BF16, ties to even."""
    lo, hi = 0, len(FINITE_ABS) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        current = FINITE_ABS[mid][0]
        if current < value:
            lo = mid + 1
        elif current > value:
            hi = mid - 1
        else:
            return FINITE_ABS[mid][1]
    candidates = []
    if hi >= 0:
        candidates.append(FINITE_ABS[hi])
    if lo < len(FINITE_ABS):
        candidates.append(FINITE_ABS[lo])
    distances = [(abs(candidate - value), bits & 1, bits) for candidate, bits in candidates]
    return min(distances)[2]


def signed_fraction(bits: int) -> Fraction:
    magnitude = bf16_abs_fraction(bits & 0x7FFF)
    return -magnitude if bits & 0x8000 else magnitude


def signed_bits(sign: int, abs_bits: int) -> int:
    return (sign & 0x8000) | abs_bits


def vector_norm(values: list[Fraction]) -> float:
    return math.sqrt(math.fsum(float(value) ** 2 for value in values))


def cosine(left: list[Fraction], right: list[Fraction]) -> float:
    left_norm, right_norm = vector_norm(left), vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("STOP: cosine undefined for zero vector")
    dot = math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def build_target(original_bits: list[int]) -> tuple[list[int], list[int], list[Fraction]]:
    target_bits: list[int] = []
    mirror_bits: list[int] = []
    deltas: list[Fraction] = []
    for raw in original_bits:
        bits = raw & 0xFFFF
        magnitude_bits = bits & 0x7FFF
        exponent = (magnitude_bits >> 7) & 0xFF
        if exponent == 0xFF:
            raise ValueError("STOP: NaN/Inf in candidate column")
        if magnitude_bits == 0:
            # Preserve the original signed-zero bit pattern.
            target_bits.append(bits)
            mirror_bits.append(bits)
            deltas.append(Fraction(0))
            continue

        original = bf16_abs_fraction(magnitude_bits)
        first = nearest_bf16_abs(original * Fraction(9, 10))
        selected = None
        for candidate_bits in range(first, magnitude_bits + 1):
            candidate = bf16_abs_fraction(candidate_bits)
            reflected = 2 * original - candidate
            reflected_bits = ABS_BITS_BY_VALUE.get(reflected)
            if reflected_bits is not None:
                selected = candidate_bits, reflected_bits
                break
        if selected is None:
            raise ValueError("STOP: no exact finite BF16 mirror found")
        target_abs, mirror_abs = selected
        sign = bits & 0x8000
        target_value = signed_fraction(signed_bits(sign, target_abs))
        original_value = signed_fraction(bits)
        target_bits.append(signed_bits(sign, target_abs))
        mirror_bits.append(signed_bits(sign, mirror_abs))
        deltas.append(target_value - original_value)
    return target_bits, mirror_bits, deltas


def choose_signs(original: list[Fraction], delta: list[Fraction]) -> tuple[list[int], int, float, float]:
    for trial in range(1024):
        signs = []
        for coordinate in range(len(delta)):
            material = f"{CONTROL_SEED}|trial={trial}|coordinate={coordinate}".encode("utf-8")
            signs.append(1 if hashlib.sha256(material).digest()[0] & 1 == 0 else -1)
        vector = [sign * value for sign, value in zip(signs, delta, strict=True)]
        try:
            weight_cos = abs(cosine(vector, original))
            delta_cos = abs(cosine(vector, delta))
        except ValueError:
            continue
        if weight_cos <= 0.1 and delta_cos <= 0.1:
            return signs, trial, weight_cos, delta_cos
    raise ValueError("STOP: no signed-direction control among frozen 1024 trials")


def to_bf16_tensor(bits: list[int]) -> torch.Tensor:
    unsigned = torch.tensor(bits, dtype=torch.uint16, device="cpu")
    return unsigned.view(torch.bfloat16).contiguous()


def bits_bytes(bits: list[int]) -> bytes:
    return b"".join(value.to_bytes(2, "little") for value in bits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "model-cache/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
    parser.add_argument("--out", type=Path, default=ROOT / "results/human_astra_manual_20261105/raw/directional_control_v1/columns_v1")
    args = parser.parse_args()
    base = args.base.resolve()
    out = args.out.resolve()
    if not out.is_relative_to(LOCAL_RAW_ROOT.resolve()):
        raise ValueError("STOP: output directory must remain under the ignored experiment raw directory")
    if not base.is_file():
        raise FileNotFoundError(base)
    if out.exists():
        raise FileExistsError(f"Refuse to overwrite: {out}")
    base_hash = sha256_file(base)
    if base_hash != EXPECTED_BASE_SHA256:
        raise ValueError(f"STOP: base checkpoint hash mismatch: {base_hash}")
    if torch.__version__ != "2.14.1":
        raise ValueError(f"STOP: expected torch 2.14.1, found {torch.__version__}")
    from importlib.metadata import version
    if version("safetensors") != "0.8.0":
        raise ValueError(f"STOP: expected safetensors 0.8.0, found {version('safetensors')}")

    with safe_open(str(base), framework="pt", device="cpu") as checkpoint:
        if TENSOR_KEY not in checkpoint.keys():
            raise KeyError(f"STOP: tensor key absent: {TENSOR_KEY}")
        weight = checkpoint.get_tensor(TENSOR_KEY)
    if weight.dtype != torch.bfloat16 or list(weight.shape) != [1024, 3584]:
        raise ValueError(f"STOP: unexpected tensor dtype/shape: {weight.dtype}, {list(weight.shape)}")
    column_bits = weight[:, COORDINATE].contiguous().view(torch.uint16).tolist()
    if len(column_bits) != 1024:
        raise ValueError("STOP: candidate column length mismatch")
    original = [signed_fraction(int(bits)) for bits in column_bits]
    target_bits, mirror_bits, delta = build_target([int(bits) for bits in column_bits])
    weight_norm, delta_norm = vector_norm(original), vector_norm(delta)
    relative_norm = delta_norm / weight_norm
    target_direction_cosine = cosine(delta, [-value for value in original])
    if not 0.09 <= relative_norm <= 0.11:
        raise ValueError(f"STOP: target relative change norm outside gate: {relative_norm:.17g}")
    if target_direction_cosine < 0.99:
        raise ValueError(f"STOP: target direction cosine outside gate: {target_direction_cosine:.17g}")

    signs, trial, weight_cos, delta_cos = choose_signs(original, delta)
    cplus_bits, cminus_bits = [], []
    for wbits, tbits, mbits, sign in zip(column_bits, target_bits, mirror_bits, signs, strict=True):
        if (tbits & 0x7FFF) == (mbits & 0x7FFF):
            cplus_bits.append(tbits)
            cminus_bits.append(tbits)
        elif sign > 0:
            cplus_bits.append(tbits)
            cminus_bits.append(mbits)
        else:
            cplus_bits.append(mbits)
            cminus_bits.append(tbits)

    # Verify exact stored BF16 component magnitudes and total change norms.
    target_delta_abs = [abs(signed_fraction(t) - signed_fraction(w)) for t, w in zip(target_bits, column_bits, strict=True)]
    for arm_bits in (cplus_bits, cminus_bits):
        arm_delta_abs = [abs(signed_fraction(a) - signed_fraction(w)) for a, w in zip(arm_bits, column_bits, strict=True)]
        if arm_delta_abs != target_delta_abs:
            raise ValueError("STOP: exact per-component change magnitudes differ")
    target_norm = vector_norm(delta)
    if any(vector_norm([signed_fraction(a) - signed_fraction(w) for a, w in zip(arm, column_bits, strict=True)]) != target_norm for arm in (cplus_bits, cminus_bits)):
        raise ValueError("STOP: change Frobenius norm mismatch")

    arm_bits = {"T": target_bits, "Cplus": cplus_bits, "Cminus": cminus_bits}
    base_col_hash = sha256_bytes(bits_bytes([int(bits) for bits in column_bits]))
    out.mkdir(parents=True)
    arms = {}
    for arm, values in arm_bits.items():
        path = out / f"{arm}.safetensors"
        save_file({"column": to_bf16_tensor(values)}, str(path), metadata={"intervention_id": INTERVENTION_ID, "arm": arm, "base_column_sha256": base_col_hash})
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            roundtrip = saved.get_tensor("column")
        roundtrip_bits = roundtrip.contiguous().view(torch.uint16).tolist()
        if roundtrip.dtype != torch.bfloat16 or roundtrip_bits != values:
            raise ValueError(f"STOP: saved BF16 roundtrip changed bits for arm {arm}")
        roundtrip_delta_abs = [
            abs(signed_fraction(value) - signed_fraction(original_value))
            for value, original_value in zip(roundtrip_bits, column_bits, strict=True)
        ]
        if roundtrip_delta_abs != target_delta_abs:
            raise ValueError(f"STOP: saved BF16 roundtrip changed component deltas for arm {arm}")
        arms[arm] = {
            "file": path.name,
            "file_sha256": sha256_file(path),
            "column_bitpattern_sha256": sha256_bytes(bits_bytes(values)),
            "changed_elements": sum(a != w for a, w in zip(values, column_bits, strict=True)),
        }
    manifest = {
        "intervention_id": INTERVENTION_ID,
        "status": "prepared_columns_only_not_applied_to_model",
        "base_checkpoint_sha256": base_hash,
        "tensor_key": TENSOR_KEY,
        "coordinate": COORDINATE,
        "shape": [1024, 3584],
        "dtype": "bfloat16",
        "base_column_bitpattern_sha256": base_col_hash,
        "alpha_exact": "1/10",
        "target_relative_change_norm": relative_norm,
        "target_cosine_delta_to_negative_original": target_direction_cosine,
        "control_trial": trial,
        "control_abs_cosine_to_original": weight_cos,
        "control_abs_cosine_to_delta": delta_cos,
        "arms": arms,
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "base_checkpoint_modified": False,
        "full_model_variant_written": False,
        "exact_per_component_change_magnitudes_equal": True,
        "exact_change_frobenius_norms_equal": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
