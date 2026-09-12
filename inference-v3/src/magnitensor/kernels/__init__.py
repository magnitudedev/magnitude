"""Portable TileLang schedules and capability-selected lowering rules."""

from .attention import OnlineAttentionRule
from .experts import DenseSwiGLURule, DirectExpertsRule, MatrixDenseSwiGLURule
from .fusion import PointwiseFusionRule
from .grouped_experts import GroupedExpertsRule
from .matrix import (
    DenseMatrixRule,
    DirectDenseMatrixRule,
    DirectEncodedMatrixRule,
    EncodedMatrixRule,
    PacketAffineMatrixRule,
    ParallelDirectMatrixRule,
)
from .normalization import ResidualRMSRule
from .portable import PrimitiveLoweringRule
from .recurrent import RecurrentPrepareRule


def register_builtin_lowerings(registry) -> None:
    names = {rule.name for rule in registry.rules}
    for rule in (
        DenseMatrixRule(),
        ParallelDirectMatrixRule(),
        PacketAffineMatrixRule(),
        DirectDenseMatrixRule(),
        DirectEncodedMatrixRule(),
        EncodedMatrixRule(),
        DenseSwiGLURule(),
        MatrixDenseSwiGLURule(),
        DirectExpertsRule(),
        GroupedExpertsRule(),
        OnlineAttentionRule(),
        ResidualRMSRule(),
        PointwiseFusionRule(),
        RecurrentPrepareRule(),
        PrimitiveLoweringRule(),
    ):
        if rule.name not in names:
            registry.register(rule)


__all__ = [
    "DenseMatrixRule",
    "DirectDenseMatrixRule",
    "DirectEncodedMatrixRule",
    "EncodedMatrixRule",
    "DenseSwiGLURule",
    "DirectExpertsRule",
    "GroupedExpertsRule",
    "MatrixDenseSwiGLURule",
    "ParallelDirectMatrixRule",
    "PacketAffineMatrixRule",
    "OnlineAttentionRule",
    "PointwiseFusionRule",
    "PrimitiveLoweringRule",
    "RecurrentPrepareRule",
    "ResidualRMSRule",
    "register_builtin_lowerings",
]
