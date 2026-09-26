"""Actual Qwen GGUF feed-forward boundary, with explicit synthetic hidden inputs.

The model trace supplies the equations, roles and composition. Physical weights
use production bindings; independent GGUF decoding supplies only the CPU oracle.
No full model is loaded, executed or compiled to measure the selected boundary.
"""

import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

import ops
from engine import DevicePlan
from engine.models.qwen35 import equations
from engine.models.qwen35.formats.gguf import describe
from engine.models.qwen35.tensor_program import define, weight_roles
from engine.qualification import QualificationCase, invocation_specs
from engine.weights.descriptor import WeightTransform
from engine.weights.formats.gguf import Encoding, GGUFFormat
from engine.weights.tensor_residency import describe_binding
from ops.lab import Configuration, Fixture, MeasurementProtocol, show
from ops.tensor.primitive import round_reference


def feedforward(format: GGUFFormat, *, rows: int, plan: DevicePlan, store: Path,
                layer: int = 0, seed: int = 157) -> Configuration:
    """Caller retains the format until every configuration worker has closed."""
    description = describe(format)
    if type(rows) is not int or rows < 1 or type(layer) is not int or not 0 <= layer < len(description.blocks):
        raise ValueError("feed-forward fixture requires positive rows and an existing layer")
    roles = {role.name: role for role, _ in weight_roles(description)}
    bindings = {role.name: describe_binding(format, role, dtype)
                for role, dtype in weight_roles(description)}
    mode = "decode" if rows == 1 else "prefill"
    # These are metadata only. Isolation below discards every mixer/state port;
    # no KV capacity or upstream decoder reference is allocated.
    case = QualificationCase("formula-extraction", mode, rows, 1, rows, True)
    definition = define(description, {name: item.spec for name, item in bindings.items()},
                        mode, invocation_specs(description, case, slots=1))
    model_graph = ops.trace(definition.function, definition.signature)
    formula = (equations.routed_feedforward if description.geometry.experts is not None
               else equations.dense_feedforward)
    # The routed formula contains a dense shared child; choose the architecture's
    # actual FFN occurrences, not every nested dense equation by display name.
    target = ops.FormulaTree(model_graph).occurrences(formula)[layer]
    graph = target.isolate().graph
    physical = {identity: bindings[graph.value(identity).name] for identity in graph.constants}

    def capture(value: ops.Value):
        if value.id in graph.inputs:
            rng = np.random.default_rng(np.random.SeedSequence((seed, value.id)))
            return rng.integers(-16, 17, size=value.spec.shape, dtype=np.int16).astype(np.float32) / 8
        import gguf

        entry = format.directory.tensor(value.name)
        content = format.source.read(format.directory.data_offset + entry.offset, entry.nbytes)
        if entry.encoding in (Encoding.F32, Encoding.F16):
            dtype = np.float32 if entry.encoding == Encoding.F32 else np.float16
            result = np.frombuffer(content, dtype=dtype).astype(np.float32).reshape(entry.shape)
            if roles[value.name].transform == WeightTransform.NEGATIVE_EXP:
                result = -np.exp(result)
            return round_reference(result, value.spec.dtype)
        # gguf's independent reader is development/reference-only. Numerical
        # execution still imports encoded bytes through ops and TileLang.
        result = gguf.dequantize(np.frombuffer(content, dtype=np.uint8),
                                 gguf.GGMLQuantizationType(int(entry.encoding)))
        return result.reshape(entry.shape)

    return Configuration(
        label=f"Qwen GGUF layer {layer} FFN · {rows} synthetic hidden rows · {format.identity}",
        fixture=Fixture.from_inputs(graph, {}, bindings=physical, capture=capture),
        device=partial(ops.DeviceRuntime.open, plan), options=definition.options, store=store,
        protocol=MeasurementProtocol(absolute_tolerance=3e-3, relative_tolerance=2**-7),
        prepared_limit=1, reference_bytes=4 << 30,
    )


@contextmanager
def configuration(model: str, rows: int = 2048, layer: int = 0,
                  maximum_bytes: int = 4 << 30, seed: int = 157):
    """Headless factory retaining the existing artifact fixture's source lifetime."""
    format = GGUFFormat(model)
    try:
        yield feedforward(format, rows=rows, layer=layer, seed=seed,
                          plan=DevicePlan.discover(maximum_bytes=maximum_bytes),
                          store=Path("runs/formula-lab.sqlite"))
    finally:
        format.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--store", type=Path, default=Path("runs/formula-lab.sqlite"))
    args = parser.parse_args()
    format = GGUFFormat(str(args.model))
    try:
        configuration = feedforward(format, rows=args.rows, layer=args.layer,
                                    plan=DevicePlan.discover(maximum_bytes=4 << 30), store=args.store)
        show((configuration,))
    finally:
        format.close()


if __name__ == "__main__":
    main()
