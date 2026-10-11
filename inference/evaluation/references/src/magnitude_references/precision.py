"""Development controls for local precision premises on catalog-rendered fixtures.

This is a CPU reference-family perturbation experiment, not native-kernel qualification.
Noise is applied at its defined variance, without normalizing each random realization.
Run with ``python -m magnitude_references.precision --help``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from magnitude_references import digest, record
from magnitude_references.contract import check_match, read_artifacts, read_logits, read_rendered
from magnitude_references.families import family_of, open_reference
from magnitude_references.sanity import Bounds, Gate


KINDS = ("uniform_weight", "gaussian_weight", "gaussian_output", "row_sign_weight",
         "f16acc", "f16acc_reflect", "channel_offset", "row_scale")


def stable_seed(identity: str, unit: str, seed: int, stream: str) -> int:
    return int(digest.canonical([identity, unit, seed, stream])[:16], 16) % (2**63 - 1)


def q40_steps(package, unit: str, weight: torch.Tensor) -> torch.Tensor:
    """Decode the actual Q4_0 grid and verify it against the reference's weight."""
    if package.info(unit).type.name != "Q4_0":
        raise ValueError(f"{unit}: this controlled experiment currently requires Q4_0")
    n, k = weight.shape
    blocks = np.asarray(package.raw(unit)).reshape(n, k // 32, 18)
    steps = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    packed = blocks[..., 2:]
    codes = np.concatenate([packed & 15, packed >> 4], axis=-1).astype(np.float32) - 8
    decoded = torch.from_numpy((codes * steps[..., None]).reshape(n, k))
    if not torch.equal(decoded, weight):
        raise ValueError(f"{unit}: decoded Q4_0 grid does not match the reference weight")
    return torch.from_numpy(steps).repeat_interleave(32, dim=1)


def _k4_scales(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit (scale, minimum) pairs of a Q4_K / Q5_K super-block from its 12 bytes."""
    s = packed.astype(np.int32)
    low, high = s[..., :4], s[..., 4:8]
    tail = s[..., 8:12]
    scale = np.concatenate([low & 63, (tail & 15) | ((low >> 6) << 4)], axis=-1)
    minimum = np.concatenate([high & 63, (tail >> 4) | ((high >> 6) << 4)], axis=-1)
    return scale, minimum


def storage_grid(package, unit: str, weight: torch.Tensor) -> dict:
    """The stored integer grid of a projection: weight[n, k] = scale[n, g] * codes[n, g, j] - minimum[n, g].

    One storage step of a weight is its group's `scale`. Supports Q4_0, Q8_0, Q4_K, Q5_K and Q6_K and
    verifies the decoded grid against the reference's own dequantized weight.
    """
    kind = package.info(unit).type.name
    n, k = weight.shape
    raw = np.asarray(package.raw(unit)).reshape(n, -1)
    half = lambda b: b.copy().view(np.float16)[..., 0].astype(np.float32)
    if kind == "Q4_0":
        blocks = raw.reshape(n, k // 32, 18)
        scale, group = half(blocks[..., :2]), 32
        codes = np.concatenate([blocks[..., 2:] & 15, blocks[..., 2:] >> 4], axis=-1).astype(np.int32) - 8
        minimum = np.zeros_like(scale)
    elif kind == "Q8_0":
        blocks = raw.reshape(n, k // 32, 34)
        scale, group = half(blocks[..., :2]), 32
        codes = blocks[..., 2:].copy().view(np.int8).astype(np.int32)
        minimum = np.zeros_like(scale)
    elif kind in ("Q4_K", "Q5_K"):
        blocks = raw.reshape(n, k // 256, 144 if kind == "Q4_K" else 176)
        sub_scale, sub_minimum = _k4_scales(blocks[..., 4:16])
        scale = (half(blocks[..., 0:2])[..., None] * sub_scale.astype(np.float32)).reshape(n, -1)
        minimum = (half(blocks[..., 2:4])[..., None] * sub_minimum.astype(np.float32)).reshape(n, -1)
        nibbles = blocks[..., -128:].astype(np.int32).reshape(n, -1, 4, 32)
        low, high = nibbles & 15, nibbles >> 4
        if kind == "Q5_K":
            fifth = blocks[..., 16:48].astype(np.int32)
            for j in range(4):
                low[:, :, j] += ((fifth >> (2 * j)) & 1) << 4
                high[:, :, j] += ((fifth >> (2 * j + 1)) & 1) << 4
        codes, group = np.stack([low, high], axis=3).reshape(n, -1, 32), 32
    elif kind == "Q6_K":
        blocks = raw.reshape(n, k // 256, 210)
        low, high = blocks[..., :128].astype(np.int32), blocks[..., 128:192].astype(np.int32)
        scale = (half(blocks[..., 208:210])[..., None]
                 * blocks[..., 192:208].copy().view(np.int8).astype(np.float32)).reshape(n, -1)
        minimum, group = np.zeros_like(scale), 16
        parts = []
        for part in range(2):
            l, h = low[..., 64 * part:64 * part + 64], high[..., 32 * part:32 * part + 32]
            parts += [(l[..., :32] & 15) | ((h & 3) << 4), (l[..., 32:] & 15) | (((h >> 2) & 3) << 4),
                      (l[..., :32] >> 4) | (((h >> 4) & 3) << 4), (l[..., 32:] >> 4) | (((h >> 6) & 3) << 4)]
        codes = (np.concatenate(parts, axis=-1) - 32).reshape(n, -1, 16)
    else:
        raise ValueError(f"{unit}: no integer storage grid for {kind}")
    codes = codes.reshape(n, k // group, group)
    grid = {"kind": kind, "group": group, "codes": torch.from_numpy(codes.astype(np.int16)),
            "scale": torch.from_numpy(np.ascontiguousarray(scale, dtype=np.float32)),
            "minimum": torch.from_numpy(np.ascontiguousarray(minimum, dtype=np.float32))}
    decoded = (grid["codes"].float() * grid["scale"][..., None] - grid["minimum"][..., None]).reshape(n, k)
    if float((decoded - weight).abs().max()) > 1e-6 * float(weight.abs().max()):
        raise ValueError(f"{unit}: decoded {kind} grid does not match the reference weight")
    return grid


FLOATS = {"f32": (np.float32, np.uint32, 0), "f16": (np.float16, np.uint16, 0), "bf16": (np.float32, np.uint32, 16)}


def cell_width(values: np.ndarray, kind: str) -> np.ndarray:
    """Rounding-cell width of a floating type at each value, in float64.

    The value is first rounded to the type (nearest, ties to even). Its cell is half the gap to the
    adjacent representable value below plus half the gap to the one above, with adjacency by payload
    step as `inference/seismic/lang/src/lattice.rs` defines it: the gaps differ across a binade
    boundary, and either zero is adjacent to the smallest subnormal of each sign.
    """
    stored, payload, shift = FLOATS[kind]
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("cell width is defined on finite values only")
    if kind == "bf16":
        rounded = torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16).float().numpy()
    else:
        rounded = values.astype(stored)
    code = (rounded.view(payload) & payload(np.iinfo(payload).max >> 1)).astype(np.int64) >> shift
    def magnitude(c):
        return (c << shift).astype(payload).view(stored).astype(np.float64)
    above = magnitude(code + 1)
    if not np.isfinite(above).all():
        raise ValueError(f"a value has no finite {kind} neighbour above")
    below = np.where(code > 0, magnitude(np.maximum(code - 1, 0)), -magnitude(np.ones_like(code)))
    return (above - below) / 2


def lattice_ratio(error, canonical, kind: str) -> float:
    """sqrt(sum e^2 / sum(w_T(d)^2 / 12)): candidate error against the rounding noise of type T at
    the canonical output's own values. An absolute scale, independent of weight quantization."""
    e = np.asarray(error, dtype=np.float64)
    power = float(np.square(cell_width(np.asarray(canonical), kind)).sum()) / 12
    return math.sqrt(float(np.square(e).sum()) / power)


def reflect_error(error, weight, y):
    """Reflect in the channel subspace spanned by weight columns and all-ones.

    The orthogonal transformation fixes y = x W^T and channel sums. Consequently
    it preserves error Gram matrices, row/channel mean energies, per-row output
    alignment and global cosine, while changing error orientation. This is a
    development control; it does not describe a deployable candidate kernel.
    """
    n, k = weight.shape
    if n <= k + 1:
        raise ValueError("orientation reflection needs more output channels than inputs plus one")
    spanning = torch.cat([weight.double(), torch.ones(n, 1, dtype=torch.float64)], dim=1)
    basis = torch.linalg.qr(spanning, mode="reduced").Q
    def reflected(value):
        v = value.double()
        return 2 * ((v @ basis) @ basis.T) - v
    changed_y = reflected(y)
    relative = float((changed_y - y).norm() / y.double().norm().clamp_min(1e-300))
    if relative > 100 * torch.finfo(y.dtype).eps:
        raise ValueError(f"orientation reflection changed baseline output by {relative:g}")
    return reflected(error).to(error.dtype)


def make_error(x, y, weight, steps, kind, seed, identity, unit, fixture):
    """Return unscaled error and analytic expected per-output variance.

    Weight streams deliberately omit fixture identity: one weight draw is reused over
    every token and fixture. Output streams include fixture identity. row_sign_weight
    keeps a uniform-weight draw's element powers exactly while randomizing token signs.
    """
    gen = torch.Generator().manual_seed(stable_seed(identity, unit, seed, "weight"))
    variance = x.square() @ (steps.square() / 12).T
    if kind in ("uniform_weight", "row_sign_weight"):
        error = x @ ((torch.rand(weight.shape, generator=gen) - 0.5) * steps).T
        if kind == "row_sign_weight":
            gen.manual_seed(stable_seed(identity, unit, seed, "row-sign:" + fixture))
            error = error * (torch.randint(0, 2, (len(x), 1), generator=gen) * 2 - 1)
    elif kind == "gaussian_weight":
        error = x @ (torch.randn(weight.shape, generator=gen) * steps / math.sqrt(12)).T
    elif kind == "gaussian_output":
        gen.manual_seed(stable_seed(identity, unit, seed, "output:" + fixture))
        error = torch.randn(y.shape, generator=gen) * variance.sqrt()
    elif kind in ("f16acc", "f16acc_reflect"):
        error = torch.zeros_like(y)
        half_x, half_weight = x.half().float(), weight.half().float()
        for start in range(0, x.shape[1], 8):
            error = (error + half_x[:, start:start + 8] @ half_weight[:, start:start + 8].T).half().float()
        error = error - y
        if kind == "f16acc_reflect":
            error = reflect_error(error, weight, y)
    elif kind == "channel_offset":
        error = torch.randn(1, y.shape[1], generator=gen).expand_as(y).clone()
    elif kind == "row_scale":
        gen.manual_seed(stable_seed(identity, unit, seed, "row-scale:" + fixture))
        error = torch.randn(len(y), 1, generator=gen) * y
    else:
        raise ValueError(kind)
    return error, variance


def statistics(error, y, variance) -> dict:
    e, target = error.double(), y.double()
    power = float(e.square().sum())
    expected = float(variance.double().sum())
    if power == 0:
        return {"error_power": 0.0, "resolution_power": expected, "ratio": 0.0}
    m, n = e.shape
    norms = target.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return {
        "error_power": power, "resolution_power": expected,
        "ratio": math.sqrt(power / expected) if expected else None,
        "row_power_min": float(e.square().sum(1).min()),
        "row_power_max": float(e.square().sum(1).max()),
        "channel_coherence": float(e.mean(0).square().sum()) * m * m / power,
        "row_coherence": float(e.sum(1).square().sum()) / power,
        "output_alignment": float((e * (target / norms)).sum(1).square().sum()) * n / power,
        "cosine_output": (float((e * target).sum()) / math.sqrt(power * float(target.square().sum()))
                          if bool(target.any()) else None),
    }


def subspace_correction(error, x, y, steps) -> dict:
    """A coefficient-free candidate rule; its sensitivity-cone premise is unvalidated."""
    e, x, y, steps = (value.double() for value in (error, x, y, steps))
    m, n = e.shape
    variance = x.square() @ (steps.square() / 12).T
    row_norm = y.norm(dim=1, keepdim=True)
    direction = y / row_norm.clamp_min(1e-300)
    powers = {
        "total": (float(e.square().sum()), float(variance.sum())),
        "shared_tokens": (float(e.sum(0).square().sum()) / m,
                          float((x.sum(0).square()[None] * steps.square() / 12).sum()) / m),
        "constant_channels": (float(e.sum(1).square().sum()) / n, float(variance.sum()) / n),
        "output_alignment": (float((e * direction).sum(1).square().sum()),
                             float((variance * direction.square()).sum())),
    }
    ratios = {name: (a / b if b else (0.0 if a == 0 else None)) for name, (a, b) in powers.items()}
    if any(value is None for value in ratios.values()):
        return {"status": "outside_noise_support", "power_ratios": ratios, "shape_factor": None}
    total = ratios["total"]
    maximum = max(ratios.values())
    return {"status": "measured", "power_ratios": ratios,
            "shape_factor": math.sqrt(maximum / total) if total else 1.0,
            "corrected_amplitude_ratio": math.sqrt(maximum),
            "assumption": "downstream sensitivity is a nonnegative combination of these projectors"}


def distribution(values: np.ndarray) -> dict:
    if not values.size or not np.isfinite(values).all():
        raise ValueError("quality values must be nonempty and finite")
    ordered = np.sort(values)
    kept = ordered[:max(1, len(ordered) - math.ceil(len(ordered) * 0.001))]
    return {"count": len(ordered), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "p99": float(np.quantile(values, 0.99)),
            "max": float(ordered[-1]), "trimmed_top_0_1_percent": float(kept.mean()),
            "mean_without_worst_10": float(ordered[:-10].mean()) if len(ordered) > 10 else None,
            "positions_above_half": int((values > 0.5).sum())}


def compare_logits(default, candidate, reference=None) -> dict[str, np.ndarray]:
    """Keep distribution movement and signed added reference loss explicitly separate."""
    a, b = default.double(), candidate.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("mismatched or nonfinite default/candidate logits")
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    result = {"default_kl": (la.exp() * (la - lb)).sum(-1).numpy(),
              "top1_agreement": (a.argmax(-1) == b.argmax(-1)).numpy()}
    if reference is not None:
        ref = torch.as_tensor(np.asarray(reference).copy()).double()
        if ref.shape != a.shape or not torch.isfinite(ref).all():
            raise ValueError("mismatched or nonfinite independent-reference logits")
        lr = ref.log_softmax(-1)
        result["reference_default_kl"] = (lr.exp() * (lr - la)).sum(-1).numpy()
        result["reference_candidate_kl"] = (lr.exp() * (lr - lb)).sum(-1).numpy()
        # Direct log-probability difference avoids subtracting two nearly equal KL totals.
        result["added_reference_kl"] = (lr.exp() * (la - lb)).sum(-1).numpy()
    return result


@contextmanager
def inject(reference, unit, callback):
    family = importlib.import_module("magnitude_references.families." + family_of(reference.weights.package))
    original = family.linear
    weight = reference.weights(unit)
    observed = 0

    def linear(x, w, bias=None):
        nonlocal observed
        y = original(x, w, bias)
        if w is weight:
            observed += 1
            return callback(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])).reshape_as(y)
        return y

    family.linear = linear
    try:
        yield
        if observed != 1:
            raise ValueError(f"{unit}: expected one observed projection call, saw {observed}")
    finally:
        family.linear = original


def checked_reference(path, rendered, vocabulary):
    record_value = json.loads((path / "reference.json").read_text())
    header, values = read_logits(path, "logits")
    if (record_value.get("model") != rendered.model or record_value.get("rendered_digest") != rendered.digest
            or not record_value.get("implementation_digest") or not record_value.get("sanity", {}).get("passed")
            or header.get("model") != rendered.model or header.get("rendered_digest") != rendered.digest
            or header.get("vocabulary") != vocabulary or not header.get("producer", "").startswith("reference:")):
        raise ValueError(f"{path}: independent reference identity/sanity does not match")
    expected = [{"fixture_id": f.fixture_id, "position": p} for f in rendered.fixtures for p in f.positions()]
    if header["rows"] != expected:
        raise ValueError(f"{path}: independent reference rows do not match rendered fixtures")
    offset, result, sanity = 0, {}, Gate()
    for fixture in rendered.fixtures:
        length = len(fixture.positions())
        result[fixture.fixture_id] = values[offset:offset + length]
        for start in range(0, length, 64):
            logits = torch.from_numpy(np.asarray(values[offset + start:offset + min(length, start + 64)]).copy())
            if not torch.isfinite(logits).all():
                raise ValueError(f"{path}: nonfinite independent-reference logits")
            sanity.add(fixture.fixture_id, logits, torch.tensor(fixture.targets()[start:start + 64]))
        offset += length
    measured_sanity = sanity.result(Bounds())
    if not measured_sanity["passed"]:
        raise ValueError(f"{path}: exact independent-reference logits failed their sanity gate")
    return result, {"directory": str(path), "record": record_value,
                    "remeasured_sanity": measured_sanity, "logits_sha256": digest.sha256_file(path / "logits.f32")}


def run(args) -> dict:
    started = record.utc_now()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    artifacts = read_artifacts(args.artifacts)
    if digest.sha256_file(artifacts.target) != artifacts.target_sha256:
        raise ValueError("catalog artifact bytes do not match their identity")
    rendered_sets = [read_rendered(path) for path in args.rendered]
    for rendered in rendered_sets:
        check_match(rendered, artifacts)
    selected = set(args.fixture or [])
    fixtures = [fixture for rendered in rendered_sets for fixture in rendered.fixtures
                if not selected or fixture.fixture_id in selected]
    if not fixtures or (selected and selected != {f.fixture_id for f in fixtures}):
        raise ValueError("selected fixtures must all occur in the rendered inputs")
    if len({f.fixture_id for f in fixtures}) != len(fixtures):
        raise ValueError("duplicate selected fixture identities")
    reference = open_reference(artifacts.target, torch.device("cpu"), cache=True)
    weight = reference.weights(args.unit)
    steps = q40_steps(reference.weights.package, args.unit, weight)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists() or any(args.output.glob("baseline-*.npz")):
        raise ValueError("precision controls require a fresh output directory")
    external, external_records = {}, []
    if args.reference:
        if len(args.reference) != len(rendered_sets):
            raise ValueError("one reference directory is required per rendered file, in matching order")
        for path, rendered in zip(args.reference, rendered_sets):
            values, identity = checked_reference(path, rendered, reference.vocab_size)
            external.update(values)
            external_records.append(identity)
    identity = {
        "artifacts": artifacts.record, "rendered": [{"path": str(r.path), "sha256": r.digest} for r in rendered_sets],
        "fixtures": [f.fixture_id for f in fixtures], "reference_implementation": digest.implementation(),
        "probe_sha256": digest.sha256_file(Path(__file__)), "torch": torch.__version__, "numpy": np.__version__,
        "unit": args.unit, "independent_references": external_records,
    }
    identity_digest = digest.canonical(identity)
    fixture_sets = {r.fixtures[0].fixture_id.split("/", 1)[0]: r.fixture_set_digest for r in rendered_sets}
    result = {"schema": "evaluation-result/1", "tool": "precision-controls", "started_at": started,
              "build": json.loads(args.build_identity.read_text()) if args.build_identity else record.build(),
              "model": artifacts.model, "artifacts": artifacts.record,
              "fixture_set": {"path": None, "sets": fixture_sets, "digest": digest.canonical(fixture_sets)},
              "rendered_digest": rendered_sets[0].digest if len(rendered_sets) == 1 else None,
              "host": record.host(torch.device("cpu")), "command": sys.argv, "identity": identity,
              "identity_digest": identity_digest, "measurements": {}, "traces": {},
              "reference_scope": "same-catalog-artifact" if external else None,
              "limitations": ["CPU-emulated projection errors, not native device candidates",
                              "same-artifact references do not establish transfer to original higher-precision weights"]}
    (args.output / "probe-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    bases, captured, gate = {}, {}, Gate()

    def hidden(fixture):
        values, _ = reference.hidden(torch.tensor(fixture.tokens)[None])
        return values[0, fixture.positions()].clone()

    with torch.inference_mode():
        for fixture in fixtures:
            saved = {}

            def capture(x, y):
                saved.update(x=x.clone(), y=y.clone())
                return y

            with inject(reference, args.unit, capture):
                base = hidden(fixture)
            again = hidden(fixture)
            if not torch.equal(base, again):
                raise ValueError(f"{fixture.fixture_id}: exact baseline replay changed hidden states")
            bases[fixture.fixture_id], captured[fixture.fixture_id] = base, saved
            for start in range(0, len(base), 64):
                logits = reference.logits(base[start:start + 64])
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{fixture.fixture_id}: nonfinite baseline logits")
                gate.add(fixture.fixture_id, logits, torch.tensor(fixture.targets()[start:start + 64]))
            filename = digest.canonical(fixture.fixture_id)[:16]
            np.savez(args.output / f"baseline-{filename}.npz", hidden=base.numpy(), x=saved["x"].numpy(),
                     y=saved["y"].numpy(), positions=np.array(fixture.positions()), fixture=fixture.fixture_id)
            print(f"baseline replayed and captured: {fixture.fixture_id}", flush=True)
        sanity = gate.result(Bounds())
        result["measurements"]["sanity"] = sanity
        result["measurements"]["exact_baseline_replay"] = True
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not sanity["passed"]:
            raise ValueError(f"fresh baseline sanity failed: {Gate.failing(sanity)}")
        rows = []
        for kind in args.kind:
            for seed in args.seed:
                errors, stats = {}, {}
                for fixture in fixtures:
                    values = captured[fixture.fixture_id]
                    error, variance = make_error(values["x"], values["y"], weight, steps, kind,
                                                 seed, artifacts.target_sha256, args.unit, fixture.fixture_id)
                    native = statistics(error, values["y"], variance)
                    # Deterministic technique/structure probes use one fixed normalization per
                    # captured tensor, then reuse that exact tensor at all signed amplitudes.
                    if kind in ("f16acc", "f16acc_reflect", "channel_offset", "row_scale"):
                        power = float(error.double().square().sum())
                        if power == 0:
                            raise ValueError(f"{kind}: cannot normalize zero technique error")
                        error *= math.sqrt(float(variance.double().sum()) / power)
                    errors[fixture.fixture_id] = error
                    stats[fixture.fixture_id] = {
                        "native": native, "applied_at_one": statistics(error, values["y"], variance),
                        "subspace_correction": subspace_correction(error, values["x"], values["y"], steps),
                    }
                    np.save(args.output / f"error-{kind}-{seed}-{digest.canonical(fixture.fixture_id)[:16]}.npy", error.numpy())
                for amplitude in args.amplitude:
                    begun, metrics, fixture_stats = time.monotonic(), {}, {}
                    for fixture in fixtures:
                        error = errors[fixture.fixture_id]

                        def perturb(x, y):
                            original = captured[fixture.fixture_id]
                            if not torch.equal(x, original["x"]) or not torch.equal(y, original["y"]):
                                raise ValueError("single-site baseline inputs changed before perturbation")
                            return y + amplitude * error

                        with inject(reference, args.unit, perturb):
                            candidate = hidden(fixture)
                        parts = {}
                        for start in range(0, len(candidate), 64):
                            default_logits = reference.logits(bases[fixture.fixture_id][start:start + 64])
                            candidate_logits = reference.logits(candidate[start:start + 64])
                            independent = external.get(fixture.fixture_id)
                            measured = compare_logits(default_logits, candidate_logits,
                                                      None if independent is None else independent[start:start + 64])
                            for key, value in measured.items():
                                parts.setdefault(key, []).append(value)
                        combined = {key: np.concatenate(value) for key, value in parts.items()}
                        for key, value in combined.items():
                            metrics.setdefault(key, []).append(value)
                        fixture_stats[fixture.fixture_id] = {key: (float(value.mean()) if key == "top1_agreement"
                                                                  else distribution(value)) for key, value in combined.items()}
                        output = args.output / f"{kind}-{seed}-{amplitude:g}-{digest.canonical(fixture.fixture_id)[:16]}.npz"
                        np.savez(output, **combined, fixture=fixture.fixture_id, positions=np.array(fixture.positions()))
                    totals = {key: np.concatenate(value) for key, value in metrics.items()}
                    measured = {key: (float(value.mean()) if key == "top1_agreement" else distribution(value))
                                for key, value in totals.items()}
                    row = {"kind": kind, "seed": seed, "amplitude": amplitude, "seconds": time.monotonic() - begun,
                           "metrics": measured, "fixtures": fixture_stats, "local": stats}
                    rows.append(row)
                    result["measurements"]["conditions"] = rows
                    result["finished_at"] = record.utc_now()
                    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    print(f"{kind} seed={seed} amplitude={amplitude:g}: default KL={measured['default_kl']['mean']:.6g}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rendered", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, action="append")
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--unit", required=True)
    parser.add_argument("--kind", choices=KINDS, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--amplitude", type=float, action="append", required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--build-identity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or any(not math.isfinite(a) for a in args.amplitude):
        parser.error("threads must be positive and amplitudes finite")
    run(args)


if __name__ == "__main__":
    main()
