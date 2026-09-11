"""Bind a container to device operations the way a bound program does.

Residency, scratch and the operation binding are three owners with one lifetime;
a test that wants operations over a container wants all three.
"""

from contextlib import ExitStack
from dataclasses import dataclass

from magnitude_engine.kernels.precision import NATIVE_BF16, Precision
from magnitude_engine.models.qwen35.arena import Arena
from magnitude_engine.operations.binding import Operations
from magnitude_engine.platform.execution import DeviceContext
from magnitude_engine.weights.residency import Weights


@dataclass
class Binding:
    weights: Weights
    arena: Arena
    operations: Operations

    def close(self) -> None:
        self.operations.close()
        self.arena.close()
        self.weights.close()


def bind(format, context: DeviceContext, precision: Precision = NATIVE_BF16) -> Binding:
    with ExitStack() as cleanup:
        weights = Weights(format, context)
        cleanup.callback(weights.close)
        arena = Arena(context, precision)
        cleanup.callback(arena.close)
        operations = Operations(weights, precision, arena)
        cleanup.callback(operations.close)
        result = Binding(weights, arena, operations)
        cleanup.pop_all()
        return result
