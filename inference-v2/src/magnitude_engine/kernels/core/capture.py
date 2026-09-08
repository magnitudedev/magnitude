"""The sole adapter for MLX's experimental in-memory export callback."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import mlx.core as mx

from .graph import Graph, Node, Tensor, Value
from .primitive import markers

capturing: ContextVar[bool] = ContextVar("magnitude_kernel_capture", default=False)


class UnsupportedExport(RuntimeError):
    """The qualified MLX adapter cannot reconstruct this program."""


lowering_disabled: ContextVar[bool] = ContextVar("magnitude_lowering_disabled", default=False)


def freeze(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(map(freeze, value))
    if isinstance(value, dict):
        return tuple((k, freeze(v)) for k, v in sorted(value.items()))
    return value


def capture(function: Callable[..., tuple[mx.array, ...]], arrays: tuple[mx.array, ...]) -> Graph:
    # Attribute encodings are versioned by upstream, not a stable Python ABI.
    version = getattr(mx, "__version__", "unavailable")
    if tuple(version.split(".")[:2]) != ("0", "32"):
        raise UnsupportedExport(f"MLX {version} has no qualified graph export adapter")
    records: list[dict[str, Any]] = []
    token = capturing.set(True)
    owned = {}
    marker_token = markers.set(owned)
    try:
        try:
            mx.export_function(records.append, function, *arrays)
        except ValueError as error:
            if str(error).startswith("[export_function] Unable to get state for primitive "):
                raise UnsupportedExport(str(error)) from error
            raise
    finally:
        capturing.reset(token)
        markers.reset(marker_token)
    values: dict[str, Value] = {}

    def value(record) -> Value:
        name, shape, dtype = record
        result = Value(name, Tensor(tuple(shape), dtype))
        if name in values and values[name] != result:
            raise ValueError(f"MLX export changed the type of {name}")
        values[name] = result
        return result

    inputs, outputs, constants, nodes = [], [], [], []
    seen: set[str] = set()
    for record in records:
        kind = record["type"]
        if kind == "primitive":
            if set(record) != {"type", "name", "inputs", "outputs", "arguments"}:
                raise UnsupportedExport("unrecognized MLX primitive export schema")
            operation = record["name"]
            attrs = freeze(record["arguments"])
            operands = tuple(map(value, record["inputs"]))
            results = tuple(map(value, record["outputs"]))
            if operation == "CustomKernel":
                # Only our exact capture marker is interpreted. Full numerical
                # CustomKernel source is never parsed or assigned a made-up contract.
                declaration = next(
                    (
                        decl
                        for name, decl in owned.items()
                        if attrs[0] == name or attrs[0].startswith(name + "_")
                    ),
                    None,
                )
                if declaration is not None:
                    operation, attrs = declaration, ()
                    if operation.infer(tuple(v.tensor for v in operands)) != tuple(
                        v.tensor for v in results
                    ):
                        raise ValueError("owned operation's inferred result changed during capture")
            nodes.append(Node(operation, operands, results, attrs))
            continue
        if kind in seen:
            raise ValueError(f"duplicate MLX export section: {kind}")
        seen.add(kind)
        if kind == "inputs":
            inputs = list(map(value, record["inputs"]))
        elif kind == "outputs":
            outputs = list(map(value, record["outputs"]))
        elif kind == "constants":
            constants = [(value((name, a.shape, a.dtype)), a) for name, a in record["constants"]]
        elif kind == "keyword_inputs":
            if record["keywords"]:
                raise UnsupportedExport("internal capture must flatten keyword inputs")
        else:
            raise UnsupportedExport(f"unrecognized MLX export record: {kind}")
    if seen != {"inputs", "outputs", "constants", "keyword_inputs"}:
        raise UnsupportedExport("incomplete MLX export")
    return Graph(tuple(inputs), tuple(outputs), tuple(constants), tuple(nodes))
