"""Declare numerical operations once; bind handwritten implementations to that object."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import update_wrapper
from pathlib import Path
from threading import RLock

import mlx.core as mx

from .graph import Tensor
from .metal import UNSUPPORTED, Binding, TensorSpec
from .plan import Source
from .primitive import Primitive
from .runtime import execution_context
from .trees import Tree, flatten


@dataclass(frozen=True)
class Implementation:
    source: Source
    function: str
    interface: object


class Operation:
    def __init__(self, function):
        self.function = function
        self._implementations = ()
        self._sealed = False
        self._infer = None
        self._inference = OrderedDict()
        self._executables = OrderedDict()
        self._lock = RLock()
        update_wrapper(self, function)

    @property
    def implementations(self):
        return self._implementations

    def infer(self, function):
        """Attach output Tensor/tree inference when the MLX reference is expensive."""
        with self._lock:
            if self._sealed or self._infer is not None:
                raise RuntimeError("register output inference once, before specialization")
            self._infer = function
        return function

    def metal(self, *, source, function):
        def register(binding):
            with self._lock:
                if self._sealed:
                    raise RuntimeError("register implementations before specializing the operation")
                if isinstance(source, Source):
                    snapshot = source
                else:
                    path = Path(source)
                    if not path.is_absolute():
                        path = Path(binding.__code__.co_filename).resolve().parent / path
                    snapshot = Source(str(path))
                implementation = Implementation(snapshot, function, binding)
                if any(
                    i.source.path == implementation.source.path and i.function == function
                    for i in self.implementations
                ):
                    raise ValueError("duplicate Metal implementation")
                self._implementations = (*self._implementations, implementation)
            return implementation

        return register

    def __call__(self, *args, **kwargs):
        tree, arrays = flatten((args, kwargs))
        with self._lock:
            self._sealed = True
        invocation = Invocation(self, tree)
        outputs, _ = invocation.result(tuple(Tensor(a.shape, a.dtype) for a in arrays))
        return outputs.rebuild(invocation(*arrays))


@dataclass(frozen=True)
class Invocation(Primitive):
    declaration: Operation
    inputs: Tree

    def result(self, inputs):
        key = self.inputs, inputs, execution_context()
        cache = self.declaration._inference
        with self.declaration._lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = self._result(inputs)
            cache[key] = result
            if len(cache) > 64:
                cache.popitem(last=False)
            return result

    def specialize(self, inputs):
        key = self.inputs, inputs, execution_context()
        cache = self.declaration._executables
        with self.declaration._lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = self.lower(inputs)
            cache[key] = result
            if len(cache) > 64:
                cache.popitem(last=False)
            return result

    def _result(self, inputs):
        if self.declaration._infer is not None:
            from .graph import signature

            values = signature(tuple(f"a{i}" for i in range(len(inputs))), inputs)
            args, kwargs = self.inputs.rebuild(tuple(TensorSpec(v) for v in values))
            return flatten(self.declaration._infer(*args, **kwargs), leaf_type=Tensor)
        # MLX shape inference builds lazy graphs; it does not realize these arrays.
        args, kwargs = self.inputs.rebuild(tuple(mx.zeros(t.shape, t.dtype) for t in inputs))
        tree, arrays = flatten(self.declaration.function(*args, **kwargs))
        return tree, tuple(Tensor(a.shape, a.dtype) for a in arrays)

    def infer(self, inputs):
        return self.result(inputs)[1]

    def bindings(self, values) -> tuple[Binding, ...]:
        args, kwargs = self.inputs.rebuild(tuple(TensorSpec(v) for v in values))
        result = []
        for implementation in self.declaration.implementations:
            interface = implementation.interface(*args, **kwargs)
            if interface is not UNSUPPORTED:
                result.append(Binding(implementation.source, implementation.function, interface))
        return tuple(result)

    def lower(self, inputs):
        from .graph import Graph, Node, signature
        from .metal import OrderedReduction
        from .scheduling import Automatic, select_binding

        values = signature(tuple(f"a{i}" for i in range(len(inputs))), inputs)
        outputs = signature(
            tuple(f"o{i}" for i in range(len(self.infer(inputs)))), self.infer(inputs)
        )
        node = Node(self, values, outputs)
        binding = select_binding(node)
        # A reduction hook needs a producer to fuse with. Alone, its declared MLX
        # definition is the implementation; other interfaces use the same planner.
        if binding is not None and not isinstance(binding.interface, OrderedReduction):
            return Automatic().lower(Graph(values, outputs, (), (node,)))

        def reference(*arrays):
            args, kwargs = self.inputs.rebuild(arrays)
            return flatten(self.declaration.function(*args, **kwargs))[1]

        return mx.compile(reference)


def operation(function):
    return Operation(function)
