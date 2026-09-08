"""Automatic, bounded planning from execution interfaces and actual dependencies."""

from dataclasses import dataclass

import mlx.core as mx

from .capture import UnsupportedExport
from .elementwise import Elementwise, supported
from .execution import ExecutionPlan, OperandBinding, Region
from .graph import Graph, Node
from .primitive import Primitive


def replay(node: Node, values: list[mx.array]) -> tuple[mx.array, ...]:
    op, attrs = node.operation, node.attributes
    if isinstance(op, Primitive):
        return op(*values)
    binary = {
        "Add": mx.add,
        "Subtract": mx.subtract,
        "Multiply": mx.multiply,
        "Divide": mx.divide,
        "Maximum": mx.maximum,
        "Minimum": mx.minimum,
        "Equal": mx.equal,
        "NotEqual": mx.not_equal,
        "Greater": mx.greater,
        "GreaterEqual": mx.greater_equal,
        "Less": mx.less,
        "LessEqual": mx.less_equal,
    }
    unary = {
        "Sigmoid": mx.sigmoid,
        "Abs": mx.abs,
        "Exp": mx.exp,
        "Log": mx.log,
        "Sin": mx.sin,
        "Cos": mx.cos,
        "Tanh": mx.tanh,
        "Negative": mx.negative,
        "Square": mx.square,
        "StopGradient": mx.stop_gradient,
    }
    if op in binary and not attrs:
        return (binary[op](*values),)
    if op in unary and not attrs:
        return (unary[op](*values),)
    if op == "Sqrt" and len(attrs) == 1:
        return ((mx.rsqrt if attrs[0] else mx.sqrt)(values[0]),)
    if op == "AsType" and len(attrs) == 1:
        return (values[0].astype(attrs[0]),)
    if op in ("Broadcast", "Full"):
        return (mx.broadcast_to(values[0], node.outputs[0].tensor.shape),)
    if op in ("Reshape", "Flatten", "Unflatten", "Squeeze", "ExpandDims"):
        return (values[0].reshape(node.outputs[0].tensor.shape),)
    if op == "Transpose" and len(attrs) == 1:
        return (mx.transpose(values[0], attrs[0]),)
    if op == "Select" and not attrs:
        return (mx.where(*values),)
    if op == "Matmul" and not attrs:
        return (mx.matmul(*values),)
    if op == "QuantizedMatmul" and len(attrs) == 4 and attrs[2] == 0:
        return (
            mx.quantized_matmul(
                *values, group_size=attrs[0], bits=attrs[1], mode="affine", transpose=attrs[3]
            ),
        )
    if op == "RMSNorm" and len(attrs) == 1:
        x, weight = values
        if weight.ndim == 0:
            weight = mx.broadcast_to(weight, (x.shape[-1],))
        return (mx.fast.rms_norm(x, weight, eps=attrs[0]),)
    raise UnsupportedExport(f"no MLX execution adapter for {op}{attrs!r}")


@dataclass(frozen=True)
class MLX:
    """Retain explicit MLX execution; the graph is assembled only during compilation."""

    def lower(self, graph: Graph):
        def run(*inputs):
            values = {v.name: a for v, a in zip(graph.inputs, inputs, strict=True)}
            values.update((v.name, a) for v, a in graph.constants)
            for node in graph.nodes:
                outputs = replay(node, [values[v.name] for v in node.inputs])
                values.update((v.name, a) for v, a in zip(node.outputs, outputs, strict=True))
            return tuple(values[v.name] for v in graph.outputs)

        # Validate reconstruction while still in specialization, so an unfamiliar
        # node can retain the original enclosing callable instead of failing at eval.
        run(*(mx.zeros(v.tensor.shape, v.tensor.dtype) for v in graph.inputs))
        return mx.compile(run)


@dataclass(frozen=True)
class Automatic:
    """Bounded DAG partitioning by tile coverage and data dependencies."""

    max_region_nodes: int = 64
    max_connections: int = 4096
    max_local_values: int = 128

    def __post_init__(self):
        if min(self.max_region_nodes, self.max_connections, self.max_local_values) < 1:
            raise ValueError("planner limits must be positive")

    def lower(self, graph: Graph):
        from .execution import order_regions, select
        from .fragments import lower_fragments, lower_ordered
        from .hooks import row_transform
        from .metal import FragmentCall, FragmentFold, OrderedReduction, ReadOnly, RowTransform
        from .tiles import compatible, connected, lower_tiles

        bindings = {}
        for node in graph.nodes:
            binding = select_binding(node)
            if binding is not None:
                bindings[node] = binding
        groups = [(node,) for node in graph.nodes]

        def eligible(nodes):
            tiles = {n: bindings[n] for n in nodes if n in bindings}
            scalar = [n for n in nodes if n not in tiles]
            local_values = sum(
                b.interface.layout.items if isinstance(b.interface, FragmentCall) else 1
                for b in tiles.values()
            )
            if local_values > self.max_local_values:
                return False
            if not tiles:
                return (
                    all(supported(n) for n in scalar)
                    and len({n.outputs[0].tensor.shape for n in scalar}) == 1
                )
            reductions = [
                (n, b) for n, b in tiles.items() if isinstance(b.interface, OrderedReduction)
            ]
            if reductions:
                if len(tiles) != 2 or scalar or len(reductions) != 1:
                    return False
                reduction, fold = reductions[0]
                producer, binding = next((n, b) for n, b in tiles.items() if n != reduction)
                region = select(graph, nodes)
                return (
                    isinstance(binding.interface, FragmentCall)
                    and not isinstance(binding.interface, FragmentFold)
                    and binding.interface.layout.tile_rows == 1
                    and fold.interface.input == producer.outputs[0]
                    and region.outputs == reduction.outputs
                )
            fragments = [(n, b) for n, b in tiles.items() if isinstance(b.interface, FragmentCall)]
            if fragments:
                if len(fragments) != len(tiles):
                    return False
                interface = fragments[0][1].interface
                produced = {v for n in nodes for v in n.outputs}
                region = select(graph, nodes)
                return (
                    all(
                        b.interface.layout == interface.layout
                        and b.interface.output == interface.output
                        for _, b in fragments
                    )
                    and all(
                        supported(n) and n.outputs[0].tensor.shape == interface.output.shape
                        for n in scalar
                    )
                    and all(v.tensor.shape == interface.output.shape for v in region.outputs)
                    and not any(v in produced for n in tiles for v in n.inputs)
                )
            rows = [(n, b) for n, b in tiles.items() if isinstance(b.interface, RowTransform)]
            if rows:
                if len(tiles) != 1:
                    return False
                node, binding = rows[0]
                interface = binding.interface
                region = select(graph, nodes)
                produced = {v for n in nodes for v in n.outputs}
                return (
                    not any(
                        isinstance(a, ReadOnly) and a.view.tensor.value in produced
                        for a in interface.arguments.values()
                    )
                    and all(
                        supported(n) and n.outputs[0].tensor.shape == interface.output.shape
                        for n in scalar
                    )
                    and all(v.tensor.shape == interface.output.shape for v in region.outputs)
                )
            domain, scope, result_type = compatible(next(iter(tiles.values())))
            produced = {v for n in nodes for v in n.outputs}
            region = select(graph, nodes)
            return (
                all(compatible(b) == (domain, scope, result_type) for b in tiles.values())
                and all(supported(n) and n.outputs[0].tensor.shape == domain.shape for n in scalar)
                and all(v.tensor.shape == domain.shape for v in region.outputs)
                and connected(nodes, tiles)
            )

        # Merge legal compatible regions, including siblings separated in export order.
        # Contraction of the whole DAG proves that no hidden dependency is crossed.
        changed = True
        attempts = 0
        while changed and attempts < self.max_connections:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    attempts += 1
                    if attempts > self.max_connections:
                        break
                    nodes = (*groups[i], *groups[j])
                    if len(nodes) > self.max_region_nodes or not eligible(nodes):
                        continue
                    proposal = [g for k, g in enumerate(groups) if k not in (i, j)] + [nodes]
                    try:
                        ordered = order_regions(graph, proposal)
                    except ValueError:
                        continue
                    groups = ordered
                    changed = True
                    break
                if changed or attempts > self.max_connections:
                    break
        regions = []
        for group in order_regions(graph, groups):
            region = select(graph, group)
            tiles = {n: bindings[n] for n in region.nodes if n in bindings}
            if tiles:
                reductions = [
                    (n, b) for n, b in tiles.items() if isinstance(b.interface, OrderedReduction)
                ]
                if reductions and len(tiles) == 1:
                    call, backend = bind_primitive(group[0], region), "METAL"
                    regions.append(Region(region, backend, call))
                    continue
                if reductions:
                    reduction, fold = reductions[0]
                    producer, binding = next((n, b) for n, b in tiles.items() if n != reduction)
                    call = lower_ordered(region, producer, binding, reduction, fold)
                    regions.append(Region(region, "METAL", call))
                    continue
                row = next(
                    ((n, b) for n, b in tiles.items() if isinstance(b.interface, RowTransform)),
                    None,
                )
                if row:
                    call = row_transform(region, *row)
                elif isinstance(next(iter(tiles.values())).interface, FragmentCall):
                    call = lower_fragments(region, tiles)
                else:
                    call = lower_tiles(region, tiles)
                backend = "METAL"
            elif all(supported(n) for n in region.nodes):
                call, backend = Elementwise().lower(region), "METAL"
            elif len(group) == 1 and isinstance(group[0].operation, Primitive):
                call, backend = bind_primitive(group[0], region), "METAL"
            else:
                call, backend = MLX().lower(region), "MLX"
            regions.append(Region(region, backend, call))
        return ExecutionPlan(graph, tuple(regions))


def select_binding(node):
    from .metal import OrderedFold, TileCall

    if not isinstance(node.operation, Primitive):
        return None
    options = node.operation.bindings(node.inputs)
    for binding in options:
        interface = binding.interface
        if isinstance(interface, (OrderedFold, TileCall)):
            from .graph import Tensor

            output = Tensor(interface.domain.shape, interface.result.dtype)
        else:
            output = interface.output
        if len(node.outputs) != 1 or node.outputs[0].tensor != output:
            raise ValueError("Metal interface output differs from its operation declaration")
    # Qualified shared traversal is preferred to independent complete calls. This
    # is a deterministic policy, not a runtime performance measurement.
    return min(
        options,
        key=lambda b: (not hasattr(b.interface, "iteration"), b.source.path, b.function),
        default=None,
    )


def bind_primitive(node: Node, region: Graph):
    assert isinstance(node.operation, Primitive)
    implementation = node.operation.lower(tuple(v.tensor for v in node.inputs))
    operands = tuple(region.inputs.index(v) for v in node.inputs)
    results = tuple(node.outputs.index(v) for v in region.outputs)

    if operands == tuple(range(len(region.inputs))) and results == tuple(range(len(node.outputs))):
        return implementation

    return OperandBinding(implementation, operands, results)
