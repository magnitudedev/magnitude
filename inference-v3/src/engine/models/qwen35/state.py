"""Qwen component geometry; shared state owns storage and continuation lifecycle."""

from contextlib import ExitStack

import ops
from engine.models.qwen35.description import Geometry, MixerKind
from engine.state.sequence import StateStore


def make_state_store(
    device: ops.DeviceRuntime,
    geometry: Geometry,
    max_sequences: int,
    context_capacity: int | None = None,
) -> StateStore:
    capacity = geometry.context_limit if context_capacity is None else context_capacity
    if (
        type(max_sequences) is not int
        or max_sequences <= 0
        or not 0 < capacity <= geometry.context_limit
    ):
        raise ValueError("invalid Qwen state capacity")
    representation = ops.default_kv_representation(
        geometry.attention_width, geometry.attention_width
    )
    specs = tuple(
        ops.kv_state_spec(
            max_sequences * capacity, geometry.kv_heads, geometry.activation_dtype, representation
        )
        for kind in geometry.layers
        if kind == MixerKind.ATTENTION
    )

    def initial_values():
        g = geometry
        with ExitStack() as cleanup:
            result = []
            for kind in g.layers:
                if kind != MixerKind.RECURRENT:
                    continue
                convolution = ops.TensorSpec(
                    (1, g.recurrent_channels, g.convolution_width - 1),
                    g.activation_dtype,
                )
                delta = ops.TensorSpec(
                    (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width),
                    ops.DType.F32,
                )
                convolution_resource = device.upload(convolution, bytes(convolution.storage_nbytes))
                cleanup.callback(convolution_resource.close)
                delta_resource = device.upload(delta, bytes(delta.storage_nbytes))
                cleanup.callback(delta_resource.close)
                result.extend((convolution_resource, delta_resource))
            cleanup.pop_all()
            return tuple(result)

    return StateStore(
        device,
        context_capacity=capacity,
        history_capacity=max_sequences * capacity,
        history_specs=specs,
        initial_values=initial_values,
    )
