"""Gated delta recurrence over explicit previous and next state operands."""

import math
import struct

import tilelang.language as T

from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.platform.execution import DType


def prepare_sequence(
    batch: int,
    steps: int,
    key_heads: int,
    value_heads: int,
    width: int,
    convolution_width: int,
    epsilon: float,
    *,
    cpu: bool,
    threads: int = 128,
    dtype: DType = DType.F32,
    native_rounding: bool = False,
    simd_width: int = 0,
):
    """Convolution, SiLU, normalized Q/K, and delta gates for uniformly packed sequences.

    The decay parameter is the negative exponential rate, resolved at binding.
    Head order is preserved; the delta consumer owns the Q/K mapping.
    """
    if min(batch, steps, key_heads, value_heads, width, threads) <= 0 or convolution_width < 2:
        raise ValueError("invalid recurrent preparation geometry")
    if threads & (threads - 1) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid recurrent normalization parameters")
    heads = 2 * key_heads + value_heads
    channels = heads * width
    history = convolution_width - 1
    activation = DType.BF16 if native_rounding else DType.F32
    simd = native_rounding and simd_width > 0 and width % simd_width == 0
    from magnitude_engine.numerics.native_bf16 import native_sigmoid

    gain_bits = struct.unpack("I", struct.pack("f", 1 / math.sqrt(width)))[0]
    gain_bits = (gain_bits + 0x7FFF + ((gain_bits >> 16) & 1)) & 0xFFFF0000

    @T.macro
    def convolve(X, Weight, Previous, Next, row, h, d):
        channel = h * width + d
        value = T.alloc_local((1,), "float32")
        sequence, step = row // steps, row % steps
        value[0] = 0
        for time in T.serial(history):
            source = step - history + time
            if source < 0:
                value[0] += (
                    Previous[sequence, channel, source + history].astype("float32")
                    * Weight[channel, time]
                )
            else:
                value[0] += (
                    X[sequence * steps + source, channel].astype("float32") * Weight[channel, time]
                )
            if step == steps - 1:
                final_source = steps - history + time
                if final_source < 0:
                    Next[sequence, channel, time] = Previous[
                        sequence, channel, final_source + history
                    ]
                else:
                    Next[sequence, channel, time] = X[sequence * steps + final_source, channel]
        value[0] += X[row, channel].astype("float32") * Weight[channel, history]
        if native_rounding:
            x = value[0].astype("bfloat16").astype("float32")
            return (
                (x * native_sigmoid(x, not simd).astype("float32"))
                .astype("bfloat16")
                .astype("float32")
            )
        return value[0] / (1 + T.exp(-value[0]))

    @T.macro
    def gates(Alpha, BetaInput, A, Bias, Beta, Decay, row, h):
        if native_rounding:
            value = (Alpha[row, h].astype("float32") + Bias[h]).astype("bfloat16").astype("float32")
            exponential = T.exp(-T.abs(value)).astype("bfloat16").astype("float32")
            logged = (
                T.if_then_else(exponential < 1e-4, exponential, T.log(1 + exponential))
                .astype("bfloat16")
                .astype("float32")
            )
            softplus = (T.max(value, 0) + logged).astype("bfloat16").astype("float32")
            Beta[row, h] = native_sigmoid(BetaInput[row, h].astype("float32"))
            Decay[row, h] = T.exp(A[h] * softplus)
        else:
            value = Alpha[row, h].astype("float32") + Bias[h]
            softplus = T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))
            Beta[row, h] = 1 / (1 + T.exp(-BetaInput[row, h].astype("float32")))
            Decay[row, h] = T.exp(A[h] * softplus)

    @T.macro
    def store(Q, K, V, row, h, d, value, inverse):
        if h < key_heads:
            Q[row, h, d] = (
                (value * inverse).astype("bfloat16").astype("float32") * (1 / width)
                if native_rounding
                else value * inverse * (1 / math.sqrt(width))
            )
        elif h < 2 * key_heads:
            K[row, h - key_heads, d] = (
                (value * inverse).astype("bfloat16").astype("float32")
                * T.reinterpret(T.uint32(gain_bits), "float32")
                if native_rounding
                else value * inverse
            )
        else:
            V[row, h - 2 * key_heads, d] = value

    @T.prim_func
    def main(
        X: T.Tensor((batch * steps, channels), dtype.value),
        Weight: T.Tensor((channels, convolution_width), "float32"),
        Previous: T.Tensor((batch, channels, history), dtype.value),
        Next: T.Tensor((batch, channels, history), dtype.value),
        Alpha: T.Tensor((batch * steps, value_heads), dtype.value),
        BetaInput: T.Tensor((batch * steps, value_heads), dtype.value),
        A: T.Tensor((value_heads,), "float32"),
        Bias: T.Tensor((value_heads,), "float32"),
        Q: T.Tensor((batch * steps, key_heads, width), activation.value),
        K: T.Tensor((batch * steps, key_heads, width), activation.value),
        V: T.Tensor((batch * steps, value_heads, width), activation.value),
        Beta: T.Tensor((batch * steps, value_heads), activation.value),
        Decay: T.Tensor((batch * steps, value_heads), "float32"),
    ):
        if cpu:
            for row, h in T.Parallel(batch * steps, heads):
                values = T.alloc_local((width,), "float32")
                squares = T.alloc_local((1,), "float32")
                squares[0] = 0
                for d in T.serial(width):
                    values[d] = convolve(X, Weight, Previous, Next, row, h, d)
                    squares[0] += values[d] * values[d]
                inverse = (
                    T.rsqrt(squares[0] / width + epsilon)
                    if native_rounding
                    else T.rsqrt(squares[0] + epsilon)
                )
                for d in T.serial(width):
                    store(Q, K, V, row, h, d, values[d], inverse)
                if h >= 2 * key_heads:
                    gates(Alpha, BetaInput, A, Bias, Beta, Decay, row, h - 2 * key_heads)
        elif simd:
            with T.Kernel(heads, batch * steps, threads=simd_width) as (h, row):
                lane = T.get_thread_binding()
                values = T.alloc_local((width // simd_width,), "float32")
                squares = T.alloc_local((1,), "float32")
                squares[0] = 0
                for j in T.unroll(width // simd_width, explicit=True):
                    d = lane * (width // simd_width) + j
                    values[j] = convolve(X, Weight, Previous, Next, row, h, d)
                    if h < 2 * key_heads:
                        squares[0] += values[j] * values[j]
                total = T.call_extern("float32", "simd_sum", squares[0])
                inverse = T.call_pure_extern(
                    "float32", "metal::precise::rsqrt", total / width + epsilon
                )
                for j in T.unroll(width // simd_width, explicit=True):
                    d = lane * (width // simd_width) + j
                    store(Q, K, V, row, h, d, values[j], inverse)
                if h == 0:
                    for j in T.serial(T.ceildiv(value_heads, simd_width)):
                        head = j * simd_width + lane
                        if head < value_heads:
                            gates(Alpha, BetaInput, A, Bias, Beta, Decay, row, head)
        else:
            with T.Kernel(heads, batch * steps, threads=threads) as (h, row):
                lane = T.get_thread_binding(0)
                values = T.alloc_local((T.ceildiv(width, threads),), "float32")
                squares = T.alloc_local((1,), "float32")
                shared = T.alloc_shared((threads,), "float32")
                squares[0] = 0
                for chunk in T.serial(T.ceildiv(width, threads)):
                    d = chunk * threads + lane
                    if d < width:
                        values[chunk] = convolve(X, Weight, Previous, Next, row, h, d)
                        squares[0] += values[chunk] * values[chunk]
                shared[lane] = squares[0]
                T.sync_threads()
                for step in T.unroll(int(math.log2(threads))):
                    if lane < (threads >> (step + 1)):
                        shared[lane] += shared[lane + (threads >> (step + 1))]
                    T.sync_threads()
                inverse = (
                    T.rsqrt(shared[0] / width + epsilon)
                    if native_rounding
                    else T.rsqrt(shared[0] + epsilon)
                )
                for chunk in T.serial(T.ceildiv(width, threads)):
                    d = chunk * threads + lane
                    if d < width:
                        store(Q, K, V, row, h, d, values[chunk], inverse)
                if lane == 0 and h >= 2 * key_heads:
                    gates(Alpha, BetaInput, A, Bias, Beta, Decay, row, h - 2 * key_heads)

    return main


def delta_sequence(
    batch: int,
    steps: int,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    mapping: HeadMapping,
    *,
    cpu: bool,
    lanes: int = 32,
    output_tile: int = 4,
    dtype: DType = DType.F32,
):
    """Scan uniformly packed sequences, retaining private state across their steps.

    This baseline uses the exact sequential equations with one final state write.
    It avoids host submission/state materialization per token. Matrix-based chunk
    realizations can replace this scan beneath the same operation contract.
    """
    if min(batch, steps, key_heads, value_heads, key_width, value_width, lanes, output_tile) <= 0:
        raise ValueError("delta geometry must be positive")
    if value_heads % key_heads or lanes & (lanes - 1) or not isinstance(mapping, HeadMapping):
        raise ValueError("invalid delta head mapping or reduction tile")

    @T.macro
    def key_head(h):
        return h % key_heads if mapping == HeadMapping.TILED else h // (value_heads // key_heads)

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
        if cpu:
            for sequence, h, channel in T.Parallel(batch, value_heads, value_width):
                state = T.alloc_local((key_width,), "float32")
                remembered = T.alloc_local((1,), "float32")
                answer = T.alloc_local((1,), "float32")
                for k in T.serial(key_width):
                    state[k] = Previous[sequence, h, channel, k]
                for step in T.serial(steps):
                    row = sequence * steps + step
                    remembered[0] = 0
                    answer[0] = 0
                    for k in T.serial(key_width):
                        state[k] *= Decay[row, h]
                        remembered[0] += state[k] * K[row, key_head(h), k]
                    residual = (V[row, h, channel] - remembered[0]) * Beta[row, h]
                    for k in T.serial(key_width):
                        state[k] += residual * K[row, key_head(h), k]
                        answer[0] += state[k] * Q[row, key_head(h), k]
                    Output[row, h, channel] = answer[0]
                for k in T.serial(key_width):
                    Next[sequence, h, channel, k] = state[k]
        else:
            with T.Kernel(
                T.ceildiv(value_width, output_tile), value_heads, batch, threads=lanes * output_tile
            ) as (block, h, sequence):
                tid = T.get_thread_binding(0)
                lane = tid % lanes
                slot = tid // lanes
                channel = block * output_tile + slot
                state = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
                partial = T.alloc_local((1,), "float32")
                residual = T.alloc_local((1,), "float32")
                sums = T.alloc_shared((output_tile, lanes), "float32")
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    k = chunk * lanes + lane
                    if channel < value_width and k < key_width:
                        state[chunk] = Previous[sequence, h, channel, k]
                for step_index in T.serial(steps):
                    row = sequence * steps + step_index
                    partial[0] = 0
                    for chunk in T.serial(T.ceildiv(key_width, lanes)):
                        k = chunk * lanes + lane
                        if channel < value_width and k < key_width:
                            state[chunk] *= Decay[row, h]
                            partial[0] += state[chunk] * K[row, key_head(h), k]
                    sums[slot, lane] = partial[0]
                    T.sync_threads()
                    for reduction in T.unroll(int(math.log2(lanes))):
                        if lane < (lanes >> (reduction + 1)):
                            sums[slot, lane] += sums[slot, lane + (lanes >> (reduction + 1))]
                        T.sync_threads()
                    residual[0] = 0
                    if channel < value_width:
                        residual[0] = (V[row, h, channel] - sums[slot, 0]) * Beta[row, h]
                    T.sync_threads()
                    partial[0] = 0
                    for chunk in T.serial(T.ceildiv(key_width, lanes)):
                        k = chunk * lanes + lane
                        if channel < value_width and k < key_width:
                            state[chunk] += residual[0] * K[row, key_head(h), k]
                            partial[0] += state[chunk] * Q[row, key_head(h), k]
                    sums[slot, lane] = partial[0]
                    T.sync_threads()
                    for reduction in T.unroll(int(math.log2(lanes))):
                        if lane < (lanes >> (reduction + 1)):
                            sums[slot, lane] += sums[slot, lane + (lanes >> (reduction + 1))]
                        T.sync_threads()
                    if lane == 0 and channel < value_width:
                        Output[row, h, channel] = sums[slot, 0]
                    T.sync_threads()
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    k = chunk * lanes + lane
                    if channel < value_width and k < key_width:
                        Next[sequence, h, channel, k] = state[chunk]

    return main
