"""How a kernel body reaches its iteration space, given only a capability.

A schedule is written once and asks this module what launch shape the endpoint
has. Where ``capability.threads_per_group`` is one there are no threadgroups, so
the same iteration space is a ``T.Parallel`` loop over flattened work items
rather than a ``T.Kernel`` grid.

``T.Kernel`` and ``T.Parallel`` are TIR-script syntax: the parser rewrites the
``with``/``for`` statements themselves, so the choice cannot be hidden behind a
Python context manager. It is therefore a trace-time branch in the kernel body
on ``serial(capability)``, with the shared body factored into a ``T.macro``.
Lowering ``T.Kernel(..., threads=n)`` on the host to the same parallel loop is a
fork change that would remove the branch; see design/inference/engine/kernels.md.
"""

from __future__ import annotations

from magnitude_engine.kernels.capabilities import Capability


def serial(capability: Capability) -> bool:
    """The endpoint has no threadgroups; a grid is realized as a parallel loop."""
    _check(capability)
    return capability.threads_per_group == 1


def group_threads(capability: Capability, requested: int) -> int:
    """The largest usable group width at or below a schedule's request."""
    _check(capability)
    if requested <= 0:
        raise ValueError("requested group width must be positive")
    return min(requested, capability.threads_per_group)


def subgroup(capability: Capability) -> bool:
    """Lanes within a group can exchange values through ``T.warp_reduce_*``."""
    _check(capability)
    return capability.subgroup_width > 1


def _check(capability: Capability) -> None:
    if not isinstance(capability, Capability):
        raise TypeError("a kernel launch shape requires a driver-reported capability")
