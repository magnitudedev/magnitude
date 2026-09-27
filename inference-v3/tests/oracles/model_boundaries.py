"""Failed-gate diagnostic: observe dense prefill boundaries, never production.

Adds observability to already materialized layer outputs. The resulting run is
numerical diagnosis, not a throughput measurement. Process-local hooks are always
restored, and the production model/runtime implementations are not modified.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from model_logits import artifact_identity, capture_mlx, capture_v3


def v3_boundaries(path, tokens, capacity):
    import ops
    from engine.models.qwen35 import equations

    observed = {}
    original_compile, original_block = ops.compile, equations.block

    def compile_observed(function, **kwargs):
        if function.__module__ != "engine.models.qwen35.tensor_program":
            return original_compile(function, **kwargs)
        boundaries = []

        def block(*args, **options):
            result = original_block(*args, **options)
            boundaries.append(result[0])
            return result

        def expanded(*args, **bound):
            boundaries.clear()
            result = function(*args, **bound)
            return (*result, *boundaries)

        equations.block = block
        try:
            compiled = original_compile(expanded, **kwargs)
        finally:
            equations.block = original_block
        submit = compiled.submit
        output_count = len(compiled.graph.outputs) - len(boundaries)

        def submit_observed(*args, **options):
            execution = submit(*args, **options)
            execution.completion.wait()
            n = output_count
            try:
                for index, output in enumerate(execution.outputs[n:]):
                    observed[f"layer.{index}"] = output.native[-8:].float().cpu().numpy().copy()
            finally:
                for output in execution.outputs[n:]:
                    output.close()
            return replace(execution, outputs=execution.outputs[:n])

        compiled.submit = submit_observed
        return compiled

    ops.compile = compile_observed
    try:
        observed["logits"] = capture_v3(path, [tokens], capacity)[0]
    finally:
        ops.compile = original_compile
        equations.block = original_block
    return observed


def mlx_boundaries(path, tokens, capacity, fp32):
    import mlx.core as mx
    from mlx_lm.models import qwen3_5

    observed = {}
    original = qwen3_5.DecoderLayer.__call__

    def call(layer, *args, **kwargs):
        result = original(layer, *args, **kwargs)
        tail = result[0, -8:].astype(mx.float32)
        mx.eval(tail)
        observed[f"layer.{len(observed)}"] = np.array(tail)
        return result

    qwen3_5.DecoderLayer.__call__ = call
    try:
        observed["logits"] = capture_mlx(path, [tokens], capacity, fp32=fp32)[0]
    finally:
        qwen3_5.DecoderLayer.__call__ = original
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("v3", "mlx", "mlx-f32"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnostic evidence")
    tokens = json.loads(args.fixture.read_text())["prompt"][:2048]
    capacity = 65792
    arrays = (
        v3_boundaries(args.model, tokens, capacity)
        if args.backend == "v3"
        else mlx_boundaries(args.model, tokens, capacity, args.backend == "mlx-f32")
    )
    metadata = dict(artifact_identity=artifact_identity(args.model), tokens=tokens,
                    capacity=capacity, backend=args.backend, purpose="failed-gate boundary diagnosis")
    with args.output.open("xb") as stream:
        np.savez(stream, **arrays, metadata=json.dumps(metadata))
    print(f"captured {len(arrays) - 1} layer boundaries", flush=True)


if __name__ == "__main__":
    main()
