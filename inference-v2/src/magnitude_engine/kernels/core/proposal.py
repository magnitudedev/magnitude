"""A validated region carries its emission decision through graph selection."""

from collections.abc import Callable
from dataclasses import dataclass

from .graph import Graph


@dataclass(frozen=True)
class Proposal:
    graph: Graph
    emit: Callable
    backend: str = "METAL"
