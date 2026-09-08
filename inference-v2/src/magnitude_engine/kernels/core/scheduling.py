"""Select dependency-safe regions; mechanisms own validation and emission plans."""

from dataclasses import dataclass
from functools import partial

import mlx.core as mx

from .elementwise import plan_scalar
from .execution import ExecutionPlan, OperandBinding, Region, order_regions, select
from .fragments import plan_fragments, plan_ordered
from .graph import Graph, Node, Tensor
from .hooks import plan_rows
from .metal import TileCall
from .primitive import Primitive
from .proposal import Proposal
from .tiles import plan_tiles

MECHANISMS = (plan_ordered, plan_fragments, plan_rows, plan_tiles, plan_scalar)


@dataclass(frozen=True)
class Automatic:
    max_region_nodes: int = 64
    max_connections: int = 4096
    max_local_values: int = 128

    def __post_init__(self):
        if min(self.max_region_nodes, self.max_connections, self.max_local_values) < 1:
            raise ValueError("planner limits must be positive")

    def lower(self, graph: Graph):
        bindings = {n: b for n in graph.nodes if (b := select_binding(n)) is not None}

        def propose(nodes):
            region = select(graph, nodes)
            selected = {n: bindings[n] for n in region.nodes if n in bindings}
            local = sum(
                getattr(getattr(b.interface, "layout", None), "items", 1) for b in selected.values()
            )
            if local > self.max_local_values:
                return None
            streams = {str(n.native.stream) for n in region.nodes if n.native is not None}
            if len(streams) > 1:
                return None
            mechanisms = MECHANISMS
            if any(
                n.native is not None and n.native.stream.device.type != mx.gpu for n in region.nodes
            ):
                mechanisms = ()
            for mechanism in mechanisms:
                proposal = mechanism(region, selected)
                if proposal is not None:
                    return proposal
            if len(region.nodes) == 1 and not selected:
                node = region.nodes[0]
                if isinstance(node.operation, Primitive):
                    return Proposal(region, partial(bind_primitive, node, region))
                if node.native is not None:
                    return Proposal(region, partial(bind_native, node, region), "MLX")
            return None

        groups = [(n,) for n in graph.nodes]
        plans = {frozenset(group): propose(group) for group in groups}
        attempts, changed = 0, True
        while changed and attempts < self.max_connections:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    attempts += 1
                    if attempts > self.max_connections:
                        break
                    nodes = (*groups[i], *groups[j])
                    if len(nodes) > self.max_region_nodes:
                        continue
                    plan = propose(nodes)
                    if plan is None:
                        continue
                    proposal = [g for k, g in enumerate(groups) if k not in (i, j)] + [nodes]
                    try:
                        ordered = order_regions(graph, proposal)
                    except ValueError:
                        continue
                    plans[frozenset(plan.graph.nodes)] = plan
                    groups, changed = ordered, True
                    break
                if changed or attempts > self.max_connections:
                    break
        regions = []
        for group in order_regions(graph, groups):
            plan = plans.get(frozenset(group))
            if plan is None:
                raise ValueError("no legal execution for declared kernel region")
            regions.append(Region(plan.graph, plan.backend, plan.emit()))
        return ExecutionPlan(graph, tuple(regions))


def select_binding(node):

    if not isinstance(node.operation, Primitive):
        return None
    options = node.operation.bindings(node.inputs)
    for binding in options:
        interface = binding.interface
        if isinstance(interface, (TileCall)):
            output = Tensor(interface.domain.shape, interface.result.dtype)
        else:
            output = interface.output
        if len(node.outputs) != 1 or node.outputs[0].tensor != output:
            raise ValueError("Metal interface output differs from its operation declaration")
    if len(options) > 1:
        raise ValueError("one kernel declaration must describe one implementation")
    return next(iter(options), None)


def bind_primitive(node: Node, region: Graph):
    assert isinstance(node.operation, Primitive)
    implementation = node.operation.lower(tuple(v.tensor for v in node.inputs))
    operands = tuple(region.inputs.index(v) for v in node.inputs)
    results = tuple(node.outputs.index(v) for v in region.outputs)

    if operands == tuple(range(len(region.inputs))) and results == tuple(range(len(node.outputs))):
        return implementation

    return OperandBinding(implementation, operands, results)


def bind_native(node, region):
    operands = tuple(region.inputs.index(v) for v in node.inputs)
    results = tuple(node.outputs.index(v) for v in region.outputs)

    def call(*arrays):
        outputs = node.native.apply([arrays[i] for i in operands])
        return tuple(outputs[i] for i in results)

    return call
