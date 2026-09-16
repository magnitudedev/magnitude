"""Portable physical bodies. Each primitive has one explicit implementation."""

from dataclasses import replace

from ..binding import Residency
from ..compiler.dependencies import code_dependencies
from ..compiler.lowering import BoundOperation
from .attention import CausalAttentionRule
from .attention_fusion import AttentionPrepareRule, KVAppendRule
from .chunked_recurrent import ChunkedDeltaRule
from .experts import SelectedExpertsRule
from .grouped_experts import GroupedExpertsRule
from .indexing import PackedEmbeddingRule
from .layer_norm import LayerNormRule
from .matrix import DenseMatrixRule, PackedMatrixRule
from .normalization import RMSRule, RowDotRule
from .packed import packet_format
from .persistent_attention import PersistentAttentionRule
from .portable import PrimitiveLoweringRule
from .recurrent import GatedDeltaRule, RecurrentPrepareRule
from .reductions import AffineScanRule, SoftmaxRule
from .routing import RoutingRule
from .sampling import SamplingRule
from .transfer import ByteCopyRule


def build_primitive(graph, root, context, *, remaining):
    node = graph.node(root)
    if node.operation == "embedding":
        binding = context.bindings.get(node.inputs[1])
        if binding is not None and binding.residency == Residency.STREAMED:
            from ..compiler.streaming import gather_loop

            return (gather_loop(graph, root, context, binding),)
    if node.operation == "reshape":
        source, output = node.inputs[0], node.outputs[0]
        if graph.value(source).spec.storage_nbytes != graph.value(output).spec.storage_nbytes:
            raise ValueError("reshape changes physical storage extent")
        return (
            BoundOperation(
                f"reshape.view@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                lambda operands: None,
                aliases=((output, source),),
                kernel_count=0,
            ),
        )
    if node.operation in {"linear", "matmul"}:
        binding = context.bindings.get(node.inputs[1])
        if binding is not None and binding.residency == Residency.STREAMED:
            from ..compiler.streaming import projection_loop

            return (projection_loop(graph, root, context, binding),)
        body = (
            PackedMatrixRule()
            if packet_format(graph.value(node.inputs[1]).spec) is not None
            else DenseMatrixRule()
        )
    elif node.operation == "causal_attention":
        body = CausalAttentionRule()
    elif node.operation == "persistent_attention":
        body = PersistentAttentionRule()
    elif node.operation == "attention_prepare":
        body = AttentionPrepareRule()
    elif node.operation == "kv_append":
        body = KVAppendRule()
    elif node.operation == "gated_delta_recurrence":
        if context.mode == "prefill":
            chunked = ChunkedDeltaRule()
            result = chunked.build(graph, root, context)
            if result:
                return _authored(result, chunked)
        body = GatedDeltaRule()
    elif node.operation == "delta_recurrence":
        body = AffineScanRule()
    elif node.operation == "softmax":
        body = SoftmaxRule()
    elif node.operation == "routed_experts":
        bindings = tuple(context.bindings.get(value) for value in node.inputs[3:6])
        if any(
            binding is not None and binding.residency == Residency.STREAMED for binding in bindings
        ):
            from ..compiler.streaming import expert_loop

            if any(binding is None for binding in bindings):
                raise ValueError("streamed experts require all three source bank bindings")
            return (expert_loop(graph, root, context, bindings),)
        routes = graph.value(node.inputs[1]).spec
        experts = graph.value(node.inputs[3]).spec.shape[0]
        if context.mode == "decode" or routes.elements < experts:
            selected = SelectedExpertsRule()
            result = selected.build(graph, root, context)
            if result:
                return _authored(result, selected)
        grouped = GroupedExpertsRule()
        result = grouped.build(graph, root, context)
        if result:
            return _authored(result, grouped)
        body = SelectedExpertsRule()
    elif (
        node.operation == "embedding"
        and packet_format(graph.value(node.inputs[1]).spec) is not None
    ):
        body = PackedEmbeddingRule()
    elif node.operation == "recurrent_prepare":
        body = RecurrentPrepareRule()
    elif node.operation == "rms_norm":
        body = RMSRule()
    elif node.operation == "layer_norm":
        body = LayerNormRule()
    elif node.operation == "row_dot":
        body = RowDotRule()
    elif node.operation == "route_topk":
        body = RoutingRule()
    elif node.operation in {"sample", "sample_constrained"}:
        body = SamplingRule()
    elif node.operation == "byte_copy":
        body = ByteCopyRule()
    else:
        body = PrimitiveLoweringRule()
    result = body.build(graph, root, context)
    if not result:
        raise ValueError(
            f"{node.operation} has no legal realization "
            "for these shapes, precision and compiler_target"
        )
    return _authored(result, body)


def _authored(result, body):
    dispatch = tuple(
        item
        for item in code_dependencies(build_primitive)
        if item.module == build_primitive.__module__ and item.symbol == build_primitive.__qualname__
    )
    dependencies = (*code_dependencies(body), *dispatch)
    return tuple(replace(item, dependencies=dependencies) for item in result)
