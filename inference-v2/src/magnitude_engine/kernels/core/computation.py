"""Composable MLX functions with specialization-time capture and lowering."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import update_wrapper
from threading import RLock
from typing import Any, Protocol

import mlx.core as mx

from .capture import UnsupportedExport, capture, capturing, lowering_disabled
from .graph import Graph, Node, Tensor, Value
from .runtime import execution_context
from .trees import Tree, flatten


class Schedule(Protocol):
    def lower(self, graph: Graph) -> Callable[..., tuple[mx.array, ...]]: ...


@dataclass(frozen=True)
class Executable:
    graph: Graph
    inputs: Tree
    outputs: Tree
    call: Callable[..., tuple[mx.array, ...]]
    stream: mx.Stream
    definition: Callable[..., tuple[mx.array, ...]]

    def artifact(self):
        return (
            self.call.artifact()
            if hasattr(self.call, "artifact")
            else {"description": self.explain()}
        )

    def explain(self):
        return self.call.explain() if hasattr(self.call, "explain") else self.graph.describe()

    def __call__(self, *args, **kwargs):
        tree, arrays = flatten((args, kwargs))
        if mx.default_stream(mx.default_device()) != self.stream:
            raise ValueError("bound computation belongs to a different execution stream")
        if tree != self.inputs or tuple(Tensor(a.shape, a.dtype) for a in arrays) != tuple(
            v.tensor for v in self.graph.inputs
        ):
            raise ValueError("bound computation called with a different specialization")
        call = self.definition if capturing.get() else self.call
        return self.outputs.rebuild(tuple(call(*arrays)))


class Computation:
    def __init__(self, function: Callable[..., Any], schedule: Schedule | None = None):
        self.function = function
        self.schedule = schedule
        self._cache: OrderedDict[Any, Executable] = OrderedDict()
        self._lock = RLock()
        update_wrapper(self, function)

    def with_schedule(self, schedule: Schedule) -> "Computation":
        return Computation(self.function, schedule)

    def specialize(self, *args, **kwargs) -> Executable:
        tree, arrays = flatten((args, kwargs))
        stream = mx.default_stream(mx.default_device())
        key = (tree, tuple(Tensor(a.shape, a.dtype) for a in arrays), execution_context())
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            outputs: Tree | None = None

            def definition(*inputs):
                nonlocal outputs
                positional, keywords = tree.rebuild(inputs)
                result = self.function(*positional, **keywords)
                outputs, flat = flatten(result)
                return flat

            try:
                graph = capture(definition, arrays)
                if outputs is None:
                    raise RuntimeError("capture did not execute the computation")
                if self.schedule is None:
                    from .scheduling import Automatic

                    schedule = Automatic()
                else:
                    schedule = self.schedule
                call = schedule.lower(graph)
            except UnsupportedExport as error:
                if self.schedule is not None:
                    raise

                def original(*inputs):
                    token = lowering_disabled.set(True)
                    try:
                        return definition(*inputs)
                    finally:
                        lowering_disabled.reset(token)

                # MLX traces this original function with numerical standalone calls;
                # no capture marker or recursive attempt at our lowering can escape.
                compiled = mx.compile(original)
                result = tuple(compiled(*arrays))
                ins = tuple(Value(f"a{i}", Tensor(a.shape, a.dtype)) for i, a in enumerate(arrays))
                outs = tuple(Value(f"o{i}", Tensor(a.shape, a.dtype)) for i, a in enumerate(result))
                graph = Graph(ins, outs, (), (Node("RetainedComputation", ins, outs),))
                call = Retained(compiled, str(error))
            if outputs is None:
                raise RuntimeError("specialization did not determine output structure")
            executable = Executable(graph, tree, outputs, call, stream, definition)
            self._cache[key] = executable
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
            return executable

    def __call__(self, *args, **kwargs):
        if capturing.get() or lowering_disabled.get():
            return self.function(*args, **kwargs)
        return self.specialize(*args, **kwargs)(*args, **kwargs)


def computation(function: Callable[..., Any]) -> Computation:
    return Computation(function)


@dataclass(frozen=True)
class Retained:
    call: Callable
    reason: str

    def __call__(self, *arrays):
        return tuple(self.call(*arrays))

    def explain(self):
        return "Original MLX computation retained: " + self.reason
