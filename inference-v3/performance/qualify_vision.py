"""Compare the production image encoder to an independent MLX-VLM reference.

Run reference and candidate in separate processes/environments on an idle worker.
The reference is solely an oracle; all candidate execution goes through Ops.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def reference(artifact, output):
    import mlx.core as mx
    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    config = json.loads((artifact / "config.json").read_text())["vision_config"]
    model = VisionModel(VisionConfig.from_dict(config))
    weights = {}
    for path in artifact.glob("*.safetensors"):
        weights.update(
            {
                key.removeprefix("vision_tower."): value
                for key, value in mx.load(str(path)).items()
                if key.startswith("vision_tower.")
            }
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    grid = (1, 4, 6)
    width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
    pixels = np.random.default_rng(20260916).uniform(-1, 1, (24, width)).astype(np.float32)
    result, _ = model(mx.array(pixels, dtype=mx.bfloat16), mx.array([grid], dtype=mx.int32))
    mx.eval(result)
    np.savez(
        output, pixels=pixels, grid=np.asarray(grid), features=np.asarray(result.astype(mx.float32))
    )
    print(json.dumps({"reference": "mlx_vlm", "shape": result.shape, "dtype": str(result.dtype)}))


def candidate(artifact, output, backend):
    import ops
    from engine import DevicePlan
    from engine.inputs.layout import ConditioningIdentity
    from engine.inputs.media import PreparedTensor
    from engine.models.qwen35.formats.vision_mlx import describe
    from engine.models.qwen35.preparation import ImagePatches
    from engine.models.qwen35.vision_runtime import VisionEncoder
    from engine.weights.formats.mlx_safetensors import MLXFormat
    from engine.weights.tensor_residency import TensorWeights

    fixture = np.load(output)
    image = ImagePatches(
        ConditioningIdentity("qualification"),
        tuple(map(int, fixture["grid"])),
        PreparedTensor.from_array("pixel_values", fixture["pixels"]),
    )
    artifact_format = MLXFormat(str(artifact))
    try:
        description = describe(artifact_format)
        with ops.DeviceRuntime.open(
            DevicePlan.discover(backend=backend, maximum_bytes=8 << 30)
        ) as device:
            weights = TensorWeights(artifact_format, device)
            encoder = VisionEncoder(description, device, weights)
            try:
                started = perf_counter()
                encoded = encoder.submit(image)
                try:
                    encoded.completion.wait()
                    actual = (
                        np.frombuffer(device.read(encoded.feature.values), np.float32)
                        .reshape(fixture["features"].shape)
                        .copy()
                    )
                    elapsed = perf_counter() - started
                finally:
                    encoded.close()
                expected = fixture["features"]
                difference = actual - expected
                cosine = float(
                    np.dot(actual.ravel(), expected.ravel())
                    / (np.linalg.norm(actual) * np.linalg.norm(expected))
                )
                report = {
                    "shape": actual.shape,
                    "max_abs": float(np.max(np.abs(difference))),
                    "rms_abs": float(np.sqrt(np.mean(difference**2))),
                    "cosine": cosine,
                    "cold_seconds": elapsed,
                    "artifact_identity": str(artifact_format.identity),
                }
                np.save(output.with_suffix(".actual.npy"), actual)
                output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
                # BF16 graph boundaries differ slightly between independent
                # runtimes; require strong feature-direction agreement and a
                # bounded error relative to each output row's signal.
                relative_rms = np.linalg.norm(difference, axis=-1) / np.maximum(
                    np.linalg.norm(expected, axis=-1), 1e-6
                )
                assert np.isfinite(actual).all() and cosine > 0.999
                assert float(relative_rms.max()) < 0.05, relative_rms
            finally:
                encoder.close()
                weights.close()
    finally:
        artifact_format.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "candidate"))
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("metal", "cuda"), default="metal")
    args = parser.parse_args()
    if args.mode == "reference":
        reference(args.artifact, args.output)
    else:
        candidate(args.artifact, args.output, args.backend)
