"""Explicit-gate, same-artifact prefill/decode comparison, outside production.

Capture `mlx` with the V2 Python environment and `v3` with the V3 environment.
Both consume the exact same fixture, chunk boundaries, and teacher-forced decode
tokens. Compare records only after both processes have released their models.
The V2 continuation envelope is a diagnostic, not full Gate 0 qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def artifact_identity(path: Path) -> str:
    digest = hashlib.sha256((path / "config.json").read_bytes())
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("expected an MLX Safetensors artifact")
    for file in files:
        with file.open("rb") as stream:
            content = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(file.name.encode() + b"\x00" + content.encode())
    return digest.hexdigest()


def capture_mlx(path: Path, chunks: list[list[int]], capacity: int, fp32=False):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    loaded, _ = load_model(path)
    model = loaded.language_model
    if fp32:
        model.set_dtype(mx.float32)
    cache = model.make_cache()
    outputs = []
    for chunk in chunks:
        logits = model(mx.array([chunk]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        outputs.append(np.array(logits))
        print(f"mlx captured {len(chunk)} tokens", flush=True)
    return outputs


def capture_v3(path: Path, chunks: list[list[int]], capacity: int):
    import magnitensor as mt
    from magnitude_engine.data import TokenId
    from magnitude_engine.models.qwen35.formats.mlx import describe
    from magnitude_engine.models.qwen35.inputs import InputPlan
    from magnitude_engine.models.qwen35.runtime import DenseRuntime
    from magnitude_engine.models.sequence import LogitsSelection, ModelRequest
    from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat
    from magnitude_engine.weights.tensor_residency import TensorWeights

    with ExitStack() as cleanup:
        format = MLXFormat(str(path))
        cleanup.callback(format.close)
        device = mt.device("auto", budget_bytes=32 * 1024**3)
        cleanup.callback(device.close)
        weights = TensorWeights(format, device)
        cleanup.callback(weights.close)
        runtime = DenseRuntime(
            describe(format),
            device,
            weights,
            max_sequences=1,
            prefill_rows=max(map(len, chunks)),
            context_capacity=capacity,
        )
        cleanup.callback(runtime.close)
        sequence = runtime.create(InputPlan.text(tuple(TokenId(t) for c in chunks for t in c)))
        cleanup.callback(sequence.close)
        outputs = []
        for chunk in chunks:
            batch = runtime.prepare(
                (
                    ModelRequest(
                        sequence,
                        tuple(TokenId(t) for t in chunk),
                        LogitsSelection.LAST,
                        (0, 0, 0, 0, 0, 0),
                    ),
                )
            )
            try:
                batch.completion.wait()
                advance = batch.advances[0]
                if advance.logits is None or advance.logits.spec.dtype != mt.DType.F32:
                    raise ValueError("expected float32 logits")
                outputs.append(
                    np.frombuffer(advance.forward.read_logits(), dtype=np.float32).copy()
                )
                advance.commit()
            finally:
                batch.close()
            print(f"v3 captured {len(chunk)} tokens", flush=True)
        return outputs


def compare(reference: Path, candidate: Path, output: Path, anchor_path: Path | None = None):
    with (
        np.load(reference, allow_pickle=False) as ref,
        np.load(candidate, allow_pickle=False) as got,
    ):
        expected_meta = json.loads(str(ref["metadata"]))
        actual_meta = json.loads(str(got["metadata"]))
        for key in ("artifact_identity", "chunks", "capacity"):
            if expected_meta[key] != actual_meta[key]:
                raise ValueError(f"incompatible capture {key}")
        if expected_meta["backend"] != "mlx" or actual_meta["backend"] != "v3":
            raise ValueError("comparison requires independent MLX reference and V3 candidate")
        expected, actual = ref["logits"], got["logits"]
        if expected.shape != actual.shape or expected.ndim != 2:
            raise ValueError("logit shape mismatch")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("nonfinite logits")
        anchor = anchor_meta = None
        if anchor_path is not None:
            with np.load(anchor_path, allow_pickle=False) as stored:
                anchor_meta = json.loads(str(stored["metadata"]))
                anchor = stored["logits"].copy()
            if anchor_meta["backend"] != "mlx-f32":
                raise ValueError("paired comparison requires an independent FP32 anchor")
            for key in ("artifact_identity", "chunks", "capacity"):
                if anchor_meta[key] != expected_meta[key]:
                    raise ValueError(f"incompatible anchor {key}")
            if anchor.shape != expected.shape:
                raise ValueError("anchor shape mismatch")
        records = []
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            error = a.astype(np.float64) - b
            maximum = float(np.max(np.abs(error)))
            relative = float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-30))
            matches = int(a.argmax()) == int(b.argmax())
            records.append(
                dict(
                    chunk=index,
                    max_error=maximum,
                    relative_l2=relative,
                    candidate_token=int(a.argmax()),
                    reference_token=int(b.argmax()),
                    within_v2_continuation_envelope=maximum < 0.5 and relative < 0.01 and matches,
                )
            )
            if anchor is not None:
                from performance.model_accuracy import compare as compare_precision

                assessed = compare_precision(
                    anchor[index].astype(np.float64),
                    b.astype(np.float64),
                    a.astype(np.float64)[None, :],
                )
                records[-1]["paired_precision"] = assessed.model_dump(mode="json")
                records[-1]["paired_validation"] = assessed.validation().model_dump(mode="json")
        report = dict(
            reference=expected_meta,
            candidate=actual_meta,
            anchor=anchor_meta,
            records=records,
            gate0_pass=False,
            note="Numerical evidence only; no throughput or full Gate 0 claim",
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(records, indent=2), flush=True)
        if not all(
            row["paired_validation"]["passed"]
            if anchor is not None
            else row["within_v2_continuation_envelope"]
            for row in records
        ):
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("backend", choices=("mlx", "mlx-f32", "v3"))
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--fixture", type=Path, required=True)
    capture.add_argument("--prefill", type=int, default=2048)
    capture.add_argument("--chunk", type=int, default=2048)
    capture.add_argument("--decode", type=int, default=4)
    capture.add_argument("--capacity", type=int, default=65792)
    capture.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--anchor", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite existing qualification evidence")
    if args.command == "compare":
        compare(args.reference, args.candidate, args.output, args.anchor)
        return
    tokens = json.loads(args.fixture.read_text())["prompt"]
    if not (
        args.prefill >= 2
        and args.chunk >= 2
        and args.decode >= 1
        and args.prefill + args.decode <= min(len(tokens), args.capacity)
    ):
        raise ValueError("invalid prefill/decode geometry")
    chunks = [
        tokens[i : min(i + args.chunk, args.prefill)] for i in range(0, args.prefill, args.chunk)
    ]
    chunks.extend([[token] for token in tokens[args.prefill : args.prefill + args.decode]])
    identity = artifact_identity(args.model)
    outputs = (
        capture_v3(args.model, chunks, args.capacity)
        if args.backend == "v3"
        else capture_mlx(args.model, chunks, args.capacity, fp32=args.backend == "mlx-f32")
    )
    metadata = dict(
        artifact_identity=identity,
        backend=args.backend,
        chunks=chunks,
        capacity=args.capacity,
        versions={
            name: importlib.metadata.version(name)
            for name in (("tilelang", "torch") if args.backend == "v3" else ("mlx", "mlx-lm"))
        },
    )
    with args.output.open("xb") as stream:
        np.savez(stream, logits=np.stack(outputs), metadata=json.dumps(metadata))


if __name__ == "__main__":
    main()
