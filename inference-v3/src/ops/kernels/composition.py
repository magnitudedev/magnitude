"""Physical bodies for common composed formulas, without implementation search.

The engine associates these bodies with its Formula objects. These definitions
consume actual traced boundaries; they do not redeclare model mathematics.
"""

from dataclasses import replace

from ..binding import Residency
from ..compiler.dependencies import code_dependencies
from ..compiler.schedules import CollectSchedules, select_composed_schedule
from ..operation import OperationContext
from .attention import AttentionOutputRule
from .attention_fusion import AttentionPrepareAppendRule
from .experts import DenseSwiGLURule, RoutedSharedExpertsRule
from .fusion import pointwise
from .grouped_experts import GroupedExpertsRule
from .matrix import ParallelPackedMatrixRule
from .normalization import ResidualRMSRule
from .persistent_attention import PersistentAttentionGateRule
from .persistent_component import PersistentAttentionComponentRule, PersistentAttentionMixerRule
from .recurrent import RecurrentOutputRule
from .routing import RouterTopKRule


def residual_normalization(context: OperationContext):
    # The residual addition can follow an explicit widening publication. Start
    # from the addition; its physical body owns any legal widening absorption.
    authored = []
    for node in sorted(context.nodes):
        if context.graph.node(node).operation == "add":
            authored.extend(_build(context, ResidualRMSRule(), node))
    return context.compose(*authored)


def parallel_projections(context: OperationContext):
    return context.compose(*_build(context, ParallelPackedMatrixRule(), min(context.nodes)))


def _build(context, body, root, **arguments):
    if root is None or root not in context.nodes:
        return ()
    collected = CollectSchedules() if context.lowering.schedules is not None else None
    lowering = (replace(context.lowering, schedules=collected) if collected is not None else context.lowering)
    built = body.build(context.graph, root, lowering, **arguments)
    if any(not item.nodes <= context.nodes for item in built):
        return ()
    for item in built:
        visible = {value for node in item.nodes for value in context.graph.node(node).outputs
                   if value in context.graph.outputs or
                   any(user not in item.nodes for user in context.graph.users[value])}
        if not visible <= set(item.outputs):
            # A fixture capture can expose an intermediate that this fused body
            # normally keeps private. Compose the children for that capture.
            return ()
    # A resident parent pipeline must not hide a streamed full-weight import.
    # Its composed projections own bounded physical source execution instead.
    if any(value in context.lowering.bindings and
           context.lowering.bindings[value].residency == Residency.STREAMED
           for item in built for value in item.inputs):
        return ()
    if any(item.workspace_bytes > context.lowering.workspace_limit for item in built):
        return ()
    if collected is not None and collected.requests and built:
        graph = context.graph
        workload = tuple((tuple(graph.value(value).spec for value in (*item.inputs, *item.outputs)),
                          tuple((item.outputs.index(output), item.inputs.index(source))
                                for output, source in item.aliases), item.kernel_count) for item in built)
        expected = tuple((item.nodes, item.inputs, item.outputs, item.aliases) for item in built)
        built = select_composed_schedule(
            context.lowering, collected, name="+".join(item.name.split("@", 1)[0] for item in built),
            template=type(body), workload=(workload, arguments),
            build=lambda selected: body.build(graph, root, selected, **arguments),
        )
        if tuple((item.nodes, item.inputs, item.outputs, item.aliases) for item in built) != expected:
            raise ValueError("selected schedule changed the composed operation boundary")
        if any(item.workspace_bytes > context.lowering.workspace_limit for item in built):
            raise ValueError("selected composed schedule exceeds the workspace budget")
    dependencies = code_dependencies(body)
    return tuple(replace(item, dependencies=dependencies) for item in built)


def dense_feedforward(context: OperationContext):
    return context.compose(*_build(context, DenseSwiGLURule(), min(context.nodes)))


def routed_feedforward(context: OperationContext):
    graph = context.graph
    authored = []
    for node_id in sorted(context.nodes):
        node = graph.node(node_id)
        if node.operation == "route_topk":
            authored.extend(_build(context, RouterTopKRule(), graph.value(node.inputs[0]).producer))
        elif node.operation == "routed_experts":
            if context.lowering.mode == "decode":
                authored.extend(_build(context, RoutedSharedExpertsRule(), node_id))
            else:
                authored.extend(_build(context, GroupedExpertsRule(), node_id, shared=True))
    return context.compose(*authored)


def _attention_prefix(context, *, persistent_gate=False):
    authored = []
    graph = context.graph
    for node_id in sorted(context.nodes):
        node = graph.node(node_id)
        if node.operation == "attention_prepare":
            authored.extend(_build(context, ParallelPackedMatrixRule(), graph.value(node.inputs[0]).producer))
            component = (_build(context, PersistentAttentionMixerRule(), node_id) if persistent_gate else ())
            if not component:
                component = _build(context, PersistentAttentionComponentRule(gated=persistent_gate), node_id)
            if not component and persistent_gate:
                component = _build(context, PersistentAttentionComponentRule(), node_id)
            authored.extend(component or _build(context, AttentionPrepareAppendRule(), node_id))
    return authored


def attention_state(context: OperationContext):
    return context.compose(*_attention_prefix(context))


def attention_mixer(context: OperationContext):
    authored = _attention_prefix(context, persistent_gate=True)
    for node_id in sorted(context.nodes):
        if context.graph.node(node_id).operation == "causal_attention":
            authored.extend(_build(context, AttentionOutputRule(), node_id))
        elif context.graph.node(node_id).operation == "persistent_attention":
            authored.extend(_build(context, PersistentAttentionGateRule(), node_id))
    return context.compose(*authored)


def _recurrent_prefix(context):
    authored = []
    for node_id in sorted(context.nodes):
        node = context.graph.node(node_id)
        if node.operation == "recurrent_prepare":
            authored.extend(_build(context, ParallelPackedMatrixRule(), context.graph.value(node.inputs[0]).producer))
    return authored


def recurrent_state(context: OperationContext):
    return context.compose(*_recurrent_prefix(context))


def recurrent_mixer(context: OperationContext):
    authored = _recurrent_prefix(context)
    for node_id in sorted(context.nodes):
        if context.graph.node(node_id).operation == "rms_norm":
            authored.extend(_build(context, RecurrentOutputRule(), node_id))
    return context.compose(*authored)


def block(context: OperationContext):
    # A block owns its final residual. Realize the whole child FFN together with
    # that publication; the child measured alone still returns its own dtype.
    children = context.graph.formulas.children(context.call.occurrence)
    child_nodes = {node for child in children for node in child.nodes}
    authored = []
    for node_id in sorted(context.nodes - child_nodes):
        if context.graph.node(node_id).operation == "add":
            authored.extend(_build(context, ResidualRMSRule(), node_id))
    for child in children:
        if not child.nodes:
            continue
        routed = next((node for node in child.nodes
                       if context.graph.node(node).operation == "routed_experts"), None)
        if routed is None:
            authored.extend(_build(context, DenseSwiGLURule(), min(child.nodes), residual=True))
            continue
        if context.lowering.mode == "decode":
            extended = _build(context, RoutedSharedExpertsRule(), routed, residual=True)
        else:
            extended = _build(context, GroupedExpertsRule(), routed, shared=True, residual=True)
        if extended:
            authored.extend(extended)
            for node_id in child.nodes:
                node = context.graph.node(node_id)
                if node.operation == "route_topk":
                    authored.extend(_build(context, RouterTopKRule(), context.graph.value(node.inputs[0]).producer))
    return context.compose(*authored)
