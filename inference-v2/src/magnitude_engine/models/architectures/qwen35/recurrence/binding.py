"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from dataclasses import dataclass

from magnitude_engine.models.normalization import GatedRMSNorm
from magnitude_engine.models.projections import ParallelProjections
from magnitude_engine.models.recurrence.contracts import DeltaRecurrence
from magnitude_engine.models.recurrence.gated_delta import GatedDelta
from magnitude_engine.models.recurrence.graph import DeltaGraph

from ..contracts import (
    RecurrentFactory,
)
from .operation import RecurrentMixer


@dataclass(eq=False)
class Mixer(RecurrentFactory):
    update: DeltaRecurrence

    def bind(self, layer, slot: int, inputs: ParallelProjections) -> RecurrentMixer:
        return RecurrentMixer(
            slot,
            GatedDelta(
                DeltaGraph(
                    inputs,
                    layer.conv1d,
                    layer.A_log,
                    layer.dt_bias,
                    GatedRMSNorm(layer.norm.weight, layer.norm.eps),
                    layer.out_proj,
                    layer.num_k_heads,
                    layer.num_v_heads,
                    layer.head_k_dim,
                    layer.head_v_dim,
                    layer.conv_kernel_size - 1,
                    self.update,
                )
            ),
        )
