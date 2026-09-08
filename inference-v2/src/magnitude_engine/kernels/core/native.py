"""Preserve native MLX graph nodes; inspect only the attributes needed for fusion."""

from . import _graph
from .graph import Graph, Node, Tensor, Value


def snapshot(outputs, inputs, owned):
    inputs = tuple({_graph.identity(a): a for a in inputs}.values())
    nodes, leaves = _graph.inspect(list(outputs), list(inputs))

    names = {}

    def value(a):
        identity = _graph.identity(a)
        if identity not in names:
            names[identity] = f"a{len(names)}"
        return Value(names[identity], Tensor(a.shape, a.dtype))

    for a in (*inputs, *leaves):
        value(a)

    records = []
    for native in nodes:
        results = tuple(value(a) for a in native.outputs)
        declaration = owned.get(_graph.identity(native.outputs[0]))
        attrs = tuple(tuple(a) if isinstance(a, list) else a for a in native.arguments)
        records.append(
            Node(
                declaration if declaration is not None else native.name,
                tuple(value(a) for a in native.inputs),
                results,
                () if declaration is not None else attrs,
                native,
            )
        )
    return Graph(
        tuple(value(a) for a in inputs),
        tuple(value(a) for a in outputs),
        tuple((value(a), a) for a in leaves),
        tuple(records),
    ), inputs
