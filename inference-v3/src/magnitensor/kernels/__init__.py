"""Portable TileLang schedules and capability-selected lowering rules."""

from .attention import AttentionOutputRule, CausalAttentionRule
from .attention_fusion import AttentionPrepareAppendRule
from .chunked_recurrent import ChunkedDeltaRule
from .experts import DenseSwiGLURule, RoutedSharedExpertsRule, SelectedExpertsRule
from .fusion import PointwiseFusionRule
from .grouped_experts import GroupedExpertsRule
from .indexing import PackedEmbeddingRule
from .matrix import (
    DenseMatrixRule,
    PackedMatrixRule,
    ParallelPackedMatrixRule,
)
from .normalization import ResidualRMSRule, RMSRule, RowDotRule
from .portable import PrimitiveLoweringRule
from .recurrent import GatedDeltaRule, RecurrentOutputRule, RecurrentPrepareRule
from .routing import RouterTopKRule, RoutingRule


def register_builtin_lowerings(registry) -> None:
    names = {rule.name for rule in registry.rules}
    for rule in (
        DenseMatrixRule(),
        ParallelPackedMatrixRule(),
        PackedMatrixRule(),
        PackedEmbeddingRule(),
        DenseSwiGLURule(),
        SelectedExpertsRule(),
        RoutedSharedExpertsRule(),
        GroupedExpertsRule(),
        CausalAttentionRule(),
        AttentionOutputRule(),
        AttentionPrepareAppendRule(),
        ResidualRMSRule(),
        RMSRule(),
        RowDotRule(),
        PointwiseFusionRule(),
        RecurrentPrepareRule(),
        GatedDeltaRule(),
        ChunkedDeltaRule(),
        RecurrentOutputRule(),
        RoutingRule(),
        RouterTopKRule(),
        PrimitiveLoweringRule(),
    ):
        if rule.name not in names:
            registry.register(rule)


__all__ = [
    "DenseMatrixRule",
    "PackedMatrixRule",
    "DenseSwiGLURule",
    "SelectedExpertsRule",
    "RoutedSharedExpertsRule",
    "GroupedExpertsRule",
    "GatedDeltaRule",
    "ChunkedDeltaRule",
    "PackedEmbeddingRule",
    "ParallelPackedMatrixRule",
    "CausalAttentionRule",
    "AttentionOutputRule",
    "AttentionPrepareAppendRule",
    "PointwiseFusionRule",
    "PrimitiveLoweringRule",
    "RecurrentPrepareRule",
    "RecurrentOutputRule",
    "RoutingRule",
    "RouterTopKRule",
    "ResidualRMSRule",
    "RMSRule",
    "RowDotRule",
    "register_builtin_lowerings",
]
