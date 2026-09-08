"""A specialization-only graph pass inside MLX's native compilation boundary."""

from functools import wraps
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_map_with_path

from . import _graph
from .context import capturing
from .native import snapshot
from .primitive import markers
from .scheduling import Automatic


def arrays(tree):
    return tuple(a for _, a in tree_flatten(tree) if isinstance(a, mx.array))


def replace(tree, values):
    return tree_map(lambda a: next(values) if isinstance(a, mx.array) else a, tree)


def compile(fun, inputs=None, outputs=None, shapeless=False):
    reports = []

    @wraps(fun)
    def trace(*args, **kwargs):
        if capturing.get() or not _graph.in_tracing():
            return fun(*args, **kwargs)
        # Keep MLX's own symbolic-shape behavior; fixed launches are not symbolic.
        if shapeless:
            reports[:] = [
                {"description": "Native MLX shapeless compilation; fixed-shape fusion disabled"}
            ]
            return fun(*args, **kwargs)
        boundaries = arrays((args, kwargs, inputs))
        owned = {}
        capture_token = capturing.set(True)
        marker_token = markers.set(owned)
        try:
            result = fun(*args, **kwargs)
        finally:
            markers.reset(marker_token)
            capturing.reset(capture_token)
        if not owned:
            reports[:] = [{"description": "Native MLX graph; no owned kernels"}]
            return result
        roots = arrays((result, outputs))
        graph, operands = snapshot(roots, boundaries, owned)
        plan = Automatic().lower(graph)
        transformed = iter(plan(*operands))
        result = replace(result, transformed)
        if outputs is not None:
            # MLX will collect this state after the trace returns. Preserve its containers.

            def update(path, item):
                return next(transformed) if isinstance(item, mx.array) else item

            replacement = tree_map_with_path(update, outputs)
            _update(outputs, replacement)
        reports[:] = [plan.artifact()]
        return result

    trace.__dict__["__kernel_reports__"] = reports
    return mx.compile(trace, inputs=inputs, outputs=outputs, shapeless=shapeless)


def _update(target, source):
    if isinstance(target, dict):
        for key in target:
            if isinstance(target[key], (dict, list)):
                _update(target[key], source[key])
            else:
                target[key] = source[key]
    elif isinstance(target, list):
        for i in range(len(target)):
            if isinstance(target[i], (dict, list)):
                _update(target[i], source[i])
            else:
                target[i] = source[i]


def artifact(fun) -> dict[str, Any]:
    reports = getattr(fun, "__kernel_reports__", None)
    if reports is None:
        raise TypeError("expected a callable returned by kernels.compile")
    return reports[-1] if reports else {"description": "Not yet compiled"}


def explain(fun):
    record = artifact(fun)
    if "description" in record:
        return record["description"]
    return "\n".join(
        f"{i + 1}. {r['backend']}: {r['description']}" for i, r in enumerate(record["regions"])
    )
