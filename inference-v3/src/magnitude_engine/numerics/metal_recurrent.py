"""Independent SIMD groups own complete recurrent state channels.

The state lives in lane registers across the entire sequence. Only the two
dot-product reductions communicate, within the owning SIMD group; different
channels never synchronize or exchange state through threadgroup memory.
"""

import tilelang.language as T

from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.platform.execution import DType


def delta_sequence(
    batch: int,
    steps: int,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    mapping: HeadMapping,
    *,
    subgroup_width: int,
    channels_per_group: int = 4,
    dtype: DType = DType.F32,
):
    if min(batch, steps, key_heads, value_heads, key_width, value_width, channels_per_group) <= 0:
        raise ValueError("delta geometry must be positive")
    if subgroup_width <= 0 or subgroup_width & (subgroup_width - 1):
        raise ValueError("delta requires a queried power-of-two SIMD width")
    if value_heads % key_heads or not isinstance(mapping, HeadMapping):
        raise ValueError("invalid delta head mapping")
    elements = (key_width + subgroup_width - 1) // subgroup_width

    @T.prim_func
    def main(
        Q: T.Tensor((batch * steps, key_heads, key_width), dtype.value),
        K: T.Tensor((batch * steps, key_heads, key_width), dtype.value),
        V: T.Tensor((batch * steps, value_heads, value_width), dtype.value),
        Decay: T.Tensor((batch * steps, value_heads), "float32"),
        Beta: T.Tensor((batch * steps, value_heads), dtype.value),
        Previous: T.Tensor((batch, value_heads, value_width, key_width), "float32"),
        Next: T.Tensor((batch, value_heads, value_width, key_width), "float32"),
        Output: T.Tensor((batch * steps, value_heads, value_width), dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(value_width, channels_per_group),
            value_heads,
            batch,
            threads=subgroup_width * channels_per_group,
        ) as (block, head, sequence):
            lane = T.get_thread_binding() % subgroup_width
            channel = block * channels_per_group + T.get_thread_binding() // subgroup_width
            key_head = (
                head % key_heads
                if mapping == HeadMapping.TILED
                else head // (value_heads // key_heads)
            )
            state = T.alloc_local((elements,), "float32")
            key = T.alloc_local((elements,), "float32")
            query = T.alloc_local((elements,), "float32")
            partial = T.alloc_local((1,), "float32")
            residual = T.alloc_local((1,), "float32")
            for j in T.unroll(elements, explicit=True):
                k = lane * elements + j if dtype == DType.BF16 else j * subgroup_width + lane
                state[j] = 0
                if channel < value_width and k < key_width:
                    state[j] = Previous[sequence, head, channel, k]
            for step in T.serial(steps):
                row = sequence * steps + step
                partial[0] = 0
                for j in T.unroll(elements, explicit=True):
                    k = lane * elements + j if dtype == DType.BF16 else j * subgroup_width + lane
                    key[j] = 0
                    query[j] = 0
                    if k < key_width:
                        key[j] = K[row, key_head, k]
                        query[j] = Q[row, key_head, k]
                    state[j] *= Decay[row, head]
                    partial[0] += state[j] * key[j]
                remembered = T.call_extern("float32", "simd_sum", partial[0])
                residual[0] = 0
                if channel < value_width:
                    residual[0] = (V[row, head, channel] - remembered) * Beta[row, head]
                partial[0] = 0
                for j in T.unroll(elements, explicit=True):
                    state[j] += residual[0] * key[j]
                    partial[0] += state[j] * query[j]
                answer = T.call_extern("float32", "simd_sum", partial[0])
                if lane == 0 and channel < value_width:
                    Output[row, head, channel] = answer
            for j in T.unroll(elements, explicit=True):
                k = lane * elements + j if dtype == DType.BF16 else j * subgroup_width + lane
                if channel < value_width and k < key_width:
                    Next[sequence, head, channel, k] = state[j]

    return main
