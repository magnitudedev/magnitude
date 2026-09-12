"""Portable TileLang schedules and capability-selected lowering rules."""

from .fusion import PointwiseFusionRule
from .matrix import DenseMatrixRule
from .portable import PrimitiveLoweringRule


def register_builtin_lowerings(registry) -> None:
    names = {rule.name for rule in registry.rules}
    for rule in (DenseMatrixRule(), PointwiseFusionRule(), PrimitiveLoweringRule()):
        if rule.name not in names:
            registry.register(rule)


__all__ = [
    "DenseMatrixRule",
    "PointwiseFusionRule",
    "PrimitiveLoweringRule",
    "register_builtin_lowerings",
]
