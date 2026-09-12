"""Portable semantic baselines authored directly with TileLang Python."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, Capabilities, LoweringContext, LoweringRegistry
from ..representations import Affine, Dense, DirectCoefficients
from ..tensor.graph import Graph, Node
from ..tensor.operation import operations
from ..tensor.types import TensorSpec, dense_strides
from .quantization import represented_load


def _indices(flat, shape: tuple[int, ...]):
    if not shape:
        return ()
    return tuple(
        (flat // stride) % extent
        for stride, extent in zip(dense_strides(shape), shape, strict=True)
    )


def _concatenate_value(sources, source_shapes, destination, coordinate, axis, dtype):
    value = T.cast(0, dtype)
    offset = 0
    for source, shape in zip(sources, source_shapes, strict=True):
        origin = list(destination)
        origin[axis] = coordinate - offset
        value = T.if_then_else(
            (coordinate >= offset) & (coordinate < offset + shape[axis]),
            T.cast(source[tuple(origin)], dtype),
            value,
        )
        offset += shape[axis]
    return value


def _load(buffer, spec: TensorSpec, flat, broadcast: TensorSpec | None = None):
    shape = cast(tuple[int, ...], spec.shape)
    if not shape:
        return buffer[()]
    target = spec if broadcast is None else broadcast
    target_shape = cast(tuple[int, ...], target.shape)
    coordinates = _indices(flat, target_shape)
    pad = len(target_shape) - len(shape)
    indices = tuple(
        0 if extent == 1 else coordinates[pad + axis] for axis, extent in enumerate(shape)
    )
    return buffer[indices]


@T.macro
def _scalar_kernel(output, value):
    with T.Kernel(1, threads=1):
        output[()] = value


@T.macro
def _pointwise_kernel(left, right, output, left_spec, right_spec, output_spec, op, threads):
    elements = output_spec.elements
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                lhs = _load(left, left_spec, flat, output_spec)
                rhs = _load(right, right_spec, flat, output_spec)
                if op == "add":
                    value = lhs + rhs
                elif op == "subtract":
                    value = lhs - rhs
                elif op == "multiply":
                    value = lhs * rhs
                else:
                    value = lhs / rhs
                output[_indices(flat, output_spec.shape)] = value


@T.macro
def _unary_kernel(source, output, source_spec, output_spec, op, threads):
    elements = output_spec.elements
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                value = T.alloc_local((1,), output_spec.dtype.value)
                source_value = _load(source, source_spec, flat)
                if op == "reshape":
                    value[0] = source_value
                elif op == "decode_bfloat16":
                    bits = T.cast(source_value, "uint32") << 16
                    value[0] = T.cast(T.reinterpret(bits, "float32"), output_spec.dtype.value)
                elif op == "cast":
                    value[0] = T.cast(source_value, output_spec.dtype.value)
                elif op == "exp":
                    value[0] = T.exp(source_value)
                elif op == "sigmoid":
                    value[0] = T.sigmoid(source_value)
                elif op == "silu":
                    value[0] = source_value * T.sigmoid(source_value)
                elif op == "tanh":
                    value[0] = T.tanh(source_value)
                output[_indices(flat, output_spec.shape)] = value[0]


@T.macro
def _transpose_kernel(source, output, source_shape, output_shape, axes, threads):
    elements = _elements(output_shape)
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                destination = _indices(flat, output_shape)
                origin = _transpose_indices(destination, axes)
                output[destination] = source[origin]


@T.macro
def _concatenate_kernel(
    sources, output, source_shapes, output_shape, axis, output_dtype, threads
):
    elements = _elements(output_shape)
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                destination = _indices(flat, output_shape)
                coordinate = destination[axis]
                output[destination] = _concatenate_value(
                    sources,
                    source_shapes,
                    destination,
                    coordinate,
                    axis,
                    output_dtype,
                )


@T.macro
def _take_rows_kernel(source, indices, output, source_spec, output_spec, row_elements, threads):
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                row = flat // row_elements
                destination = _indices(flat, output_spec.shape)
                origin = (T.cast(indices[row], "int32"), *destination[1:])
                output[destination] = source[origin]


@T.macro
def _overlay_rows_kernel(
    source, replacement, indices, output, output_spec, replacement_spec, rows, threads
):
    row_elements = output_spec.elements // output_spec.shape[0]
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                row = flat // row_elements
                replacement_row = T.alloc_local((1,), "int32")
                replacement_row[0] = -1
                for candidate in T.serial(rows):
                    if indices[candidate] == row:
                        replacement_row[0] = candidate
                value = T.alloc_local((1,), output_spec.dtype.value)
                value[0] = source[_indices(flat, output_spec.shape)]
                if replacement_row[0] >= 0:
                    value[0] = replacement[
                        _indices(
                            replacement_row[0] * row_elements + flat % row_elements,
                            replacement_spec.shape,
                        )
                    ]
                output[_indices(flat, output_spec.shape)] = value[0]


@T.macro
def _quantized_import_kernel(source, target, extent, target_spec, codec, staged_tiles, threads):
    representation = target_spec.representation
    elements = target_spec.elements
    low_bits = (
        representation.code.low_bits
        if isinstance(representation, Affine)
        else representation.code_bits
    )
    high_bits = representation.code.high_bits if isinstance(representation, Affine) else 0
    low_bytes = math.ceil(elements * low_bits / 8)
    high_bytes = math.ceil(elements * high_bits / 8)
    code_bytes = low_bytes + high_bytes
    groups = math.ceil(elements / representation.group)
    coefficients = representation.coefficients
    tile_elements = codec.block_elements
    tile_groups = tile_elements // representation.group

    with T.Kernel(T.ceildiv(staged_tiles, threads), threads=threads) as block:
        tile = block * threads + T.get_thread_binding()
        if tile < extent[0]:
            source_base = tile * codec.block_bytes
            target_tile = extent[1] + tile
            target_element = target_tile * tile_elements
            for index in T.serial(tile_elements):
                value = codec.code(source, source_base, index)
                bit = (target_element + index) * low_bits
                byte = bit // 8
                lane = bit % 8
                if lane == 0:
                    target[byte] = 0
                target[byte] = T.cast(
                    target[byte] | ((value & ((1 << low_bits) - 1)) << lane),
                    "uint8",
                )
                if high_bits:
                    high_bit = (target_element + index) * high_bits
                    high_byte = high_bit // 8
                    high_lane = high_bit % 8
                    if high_lane == 0:
                        target[low_bytes + high_byte] = 0
                    target[low_bytes + high_byte] = T.cast(
                        target[low_bytes + high_byte]
                        | (((value >> low_bits) & ((1 << high_bits) - 1)) << high_lane),
                        "uint8",
                    )
            if isinstance(coefficients, DirectCoefficients):
                if tile_groups == 1:
                    scale_base = code_bytes + target_tile * coefficients.scale_dtype.itemsize
                    for byte in T.unroll(coefficients.scale_dtype.itemsize):
                        target[scale_base + byte] = codec.scale_byte(source, source_base, byte)
                    if coefficients.bias_dtype is not None:
                        bias_base = (
                            code_bytes
                            + groups * coefficients.scale_dtype.itemsize
                            + target_tile * coefficients.bias_dtype.itemsize
                        )
                        for byte in T.unroll(coefficients.bias_dtype.itemsize):
                            target[bias_base + byte] = codec.bias_byte(source, source_base, byte)
                else:
                    for group in T.serial(tile_groups):
                        target_group = target_tile * tile_groups + group
                        scale = codec.direct_scale(
                            source, source_base, group, T.if_then_else, T.reinterpret
                        )
                        scale_bits = T.reinterpret(T.cast(scale, "float32"), "uint32")
                        scale_base = code_bytes + target_group * coefficients.scale_dtype.itemsize
                        for byte in T.unroll(coefficients.scale_dtype.itemsize):
                            target[scale_base + byte] = T.cast(scale_bits >> (8 * byte), "uint8")
                        if coefficients.bias_dtype is not None:
                            bias = codec.direct_bias(
                                source, source_base, group, T.if_then_else, T.reinterpret
                            )
                            bias_bits = T.reinterpret(T.cast(bias, "float32"), "uint32")
                            bias_base = (
                                code_bytes
                                + groups * coefficients.scale_dtype.itemsize
                                + target_group * coefficients.bias_dtype.itemsize
                            )
                            for byte in T.unroll(coefficients.bias_dtype.itemsize):
                                target[bias_base + byte] = T.cast(bias_bits >> (8 * byte), "uint8")
            else:
                local_scale_bytes = math.ceil(groups * coefficients.local_scale_bits / 8)
                scale_base = code_bytes
                tile_scale_bytes = math.ceil(tile_groups * coefficients.local_scale_bits / 8)
                tile_scale_base = scale_base + target_tile * tile_scale_bytes
                for byte in T.serial(tile_scale_bytes):
                    target[tile_scale_base + byte] = 0
                for group in T.serial(tile_groups):
                    target_group = target_tile * tile_groups + group
                    value = codec.local_scale(source, source_base, group, T.if_then_else)
                    bit = target_group * coefficients.local_scale_bits
                    target[scale_base + bit // 8] |= T.cast(value << (bit % 8), "uint8")
                    if bit % 8 + coefficients.local_scale_bits > 8:
                        target[scale_base + bit // 8 + 1] |= T.cast(value >> (8 - bit % 8), "uint8")
                cursor = scale_base + local_scale_bytes
                if coefficients.local_bias_bits is not None:
                    local_bias_bytes = math.ceil(groups * coefficients.local_bias_bits / 8)
                    tile_bias_bytes = math.ceil(tile_groups * coefficients.local_bias_bits / 8)
                    tile_bias_base = cursor + target_tile * tile_bias_bytes
                    for byte in T.serial(tile_bias_bytes):
                        target[tile_bias_base + byte] = 0
                    for group in T.serial(tile_groups):
                        target_group = target_tile * tile_groups + group
                        value = codec.local_bias(source, source_base, group, T.if_then_else)
                        bit = target_group * coefficients.local_bias_bits
                        target[cursor + bit // 8] |= T.cast(value << (bit % 8), "uint8")
                        if bit % 8 + coefficients.local_bias_bits > 8:
                            target[cursor + bit // 8 + 1] |= T.cast(value >> (8 - bit % 8), "uint8")
                    cursor += local_bias_bytes
                supergroups = math.ceil(elements / coefficients.supergroup)
                supergroup = target_element // coefficients.supergroup
                for byte in T.unroll(coefficients.super_scale_dtype.itemsize):
                    target[cursor + supergroup * coefficients.super_scale_dtype.itemsize + byte] = (
                        codec.scale_byte(source, source_base, byte)
                    )
                cursor += supergroups * coefficients.super_scale_dtype.itemsize
                if coefficients.super_bias_dtype is not None:
                    for byte in T.unroll(coefficients.super_bias_dtype.itemsize):
                        target[
                            cursor + supergroup * coefficients.super_bias_dtype.itemsize + byte
                        ] = codec.bias_byte(source, source_base, byte)


@T.macro
def _mulhilo(a, b):
    mask = T.cast(65535, "uint32")
    p0 = (a & mask) * (b & mask)
    p1 = (a >> 16) * (b & mask)
    p2 = (a & mask) * (b >> 16)
    p3 = (a >> 16) * (b >> 16)
    middle = (p0 >> 16) + (p1 & mask) + (p2 & mask)
    return p3 + (p1 >> 16) + (p2 >> 16) + (middle >> 16), (middle << 16) | (p0 & mask)


@T.macro
def _philox(c0, c1, c2, c3, k0, k1):
    counter = T.alloc_local((4,), "uint32")
    key = T.alloc_local((2,), "uint32")
    following = T.alloc_local((4,), "uint32")
    counter[0], counter[1], counter[2], counter[3] = c0, c1, c2, c3
    key[0], key[1] = k0, k1
    for _ in T.serial(10):
        hi0, lo0 = _mulhilo(T.cast(0xD2511F53, "uint32"), counter[0])
        hi1, lo1 = _mulhilo(T.cast(0xCD9E8D57, "uint32"), counter[2])
        following[0] = hi1 ^ counter[1] ^ key[0]
        following[1] = lo1
        following[2] = hi0 ^ counter[3] ^ key[1]
        following[3] = lo0
        for index in T.unroll(4):
            counter[index] = following[index]
        key[0] += T.cast(0x9E3779B9, "uint32")
        key[1] += T.cast(0xBB67AE85, "uint32")
    return counter[0]


@T.macro
def _sample_kernel(logits, draws, output, rows, vocabulary, threads):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding(0)
        scores = T.alloc_shared((threads,), "float32")
        indices = T.alloc_shared((threads,), "int32")
        invalid = T.alloc_shared((threads,), "int32")
        score = T.alloc_local((1,), "float32")
        index = T.alloc_local((1,), "int32")
        flag = T.alloc_local((1,), "int32")
        score[0] = T.reinterpret(T.cast(0xFF800000, "uint32"), "float32")
        index[0], flag[0] = 0x7FFFFFFF, 0
        for chunk in T.serial(T.ceildiv(vocabulary, threads)):
            token = chunk * threads + lane
            if token < vocabulary:
                raw = logits[row, token]
                bits = T.reinterpret(raw, "uint32")
                if (bits & T.cast(0x7FFFFFFF, "uint32")) > T.cast(
                    0x7F800000, "uint32"
                ) or bits == T.cast(0x7F800000, "uint32"):
                    flag[0] = 1
                elif bits != T.cast(0xFF800000, "uint32"):
                    value = T.alloc_local((1,), "float32")
                    value[0] = raw
                    if draws[row, 0] == 1:
                        word = _philox(
                            T.cast(token, "uint32"),
                            draws[row, 3],
                            draws[row, 4],
                            draws[row, 5],
                            draws[row, 1],
                            draws[row, 2],
                        )
                        uniform = (T.cast(word >> 9, "float32") + 0.5) * (2**-23)
                        value[0] -= T.log(-T.log(uniform))
                    if value[0] > score[0] or (value[0] == score[0] and token < index[0]):
                        score[0], index[0] = value[0], token
        scores[lane], indices[lane], invalid[lane] = score[0], index[0], flag[0]
        T.sync_threads()
        for step in T.unroll(int(math.log2(threads))):
            distance = threads >> (step + 1)
            if lane < distance:
                if scores[lane + distance] > scores[lane] or (
                    scores[lane + distance] == scores[lane]
                    and indices[lane + distance] < indices[lane]
                ):
                    scores[lane] = scores[lane + distance]
                    indices[lane] = indices[lane + distance]
                invalid[lane] |= invalid[lane + distance]
            T.sync_threads()
        if lane == 0:
            status = T.if_then_else(
                invalid[0] != 0,
                2,
                T.if_then_else(indices[0] == 0x7FFFFFFF, 1, 0),
            )
            output[row, 0] = T.if_then_else(status == 0, indices[0], -1)
            output[row, 1] = status


@T.macro
def _softmax_kernel(source, output, rows, width, dtype, threads):
    with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            row = block * threads + lane
            if row < rows:
                maximum = T.alloc_local((1,), "float32")
                total = T.alloc_local((1,), "float32")
                maximum[0] = -3.402823466e38
                for column in T.serial(width):
                    maximum[0] = T.max(maximum[0], T.cast(source[row, column], "float32"))
                total[0] = 0.0
                for column in T.serial(width):
                    total[0] += T.exp(T.cast(source[row, column], "float32") - maximum[0])
                for column in T.serial(width):
                    output[row, column] = T.cast(
                        T.exp(T.cast(source[row, column], "float32") - maximum[0]) / total[0],
                        dtype,
                    )


@T.macro
def _rms_kernel(
    source, weight, output, source_spec, rows, width, epsilon, dtype, weighted, threads
):
    with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            row = block * threads + lane
            if row < rows:
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for column in T.serial(width):
                    index = _indices(row * width + column, source_spec.shape)
                    value = T.cast(source[index], "float32")
                    total[0] += value * value
                inverse = T.rsqrt(total[0] / width + epsilon)
                for column in T.serial(width):
                    index = _indices(row * width + column, source_spec.shape)
                    value = T.alloc_local((1,), "float32")
                    value[0] = T.cast(source[index], "float32") * inverse
                    if weighted:
                        value[0] *= T.cast(weight[column], "float32")
                    output[index] = T.cast(value[0], dtype)


@T.macro
def _contraction_kernel(
    left,
    right,
    bias,
    output,
    left_spec,
    right_spec,
    output_spec,
    operation,
    has_bias,
    threads,
):
    width = left_spec.shape[-1]
    columns = output_spec.shape[-1]
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                row = flat // columns
                column = flat % columns
                accumulator = T.alloc_local((1,), "float32")
                accumulator[0] = 0.0
                for k in T.serial(width):
                    logical = _weight_index(operation, column, k, width, columns)
                    if right_spec.representation is None or isinstance(
                        right_spec.representation, Dense
                    ):
                        weight = _dense_weight(right, operation, column, k)
                    else:
                        weight = represented_load(right, right_spec, logical)
                    accumulator[0] += T.cast(left[row, k], "float32") * T.cast(weight, "float32")
                if has_bias:
                    accumulator[0] += T.cast(bias[column], "float32")
                output[row, column] = T.cast(accumulator[0], output_spec.dtype.value)


@T.macro
def _embedding_kernel(tokens, table, output, token_spec, table_spec, output_spec, threads):
    width = output_spec.shape[-1]
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                row = T.cast(_load(tokens, token_spec, flat // width), "int32")
                if table_spec.representation is None or isinstance(
                    table_spec.representation, Dense
                ):
                    value = table[row, flat % width]
                else:
                    value = represented_load(table, table_spec, row * width + flat % width)
                output[_indices(flat, output_spec.shape)] = T.cast(value, output_spec.dtype.value)


@T.macro
def _rotary_kernel(
    query,
    key,
    positions,
    query_out,
    key_out,
    spec,
    position_spec,
    dimensions,
    base,
    explicit,
    threads,
):
    width = spec.shape[-1]
    heads = spec.shape[-2]
    with T.Kernel(T.ceildiv(spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < spec.elements:
                channel = flat % width
                if explicit:
                    position = T.cast(
                        _load(positions, position_spec, flat // (heads * width)), "float32"
                    )
                else:
                    position = T.cast(flat // (heads * width), "float32")
                pair = (channel + dimensions // 2) % dimensions
                coordinates = _indices(flat, spec.shape)
                paired_coordinates = (*coordinates[:-1], pair)
                angle = position / T.pow(
                    base,
                    T.cast((channel % (dimensions // 2)) * 2, "float32") / dimensions,
                )
                sign = T.if_then_else(channel < dimensions // 2, -1.0, 1.0)
                query_value = T.if_then_else(
                    channel < dimensions,
                    T.cast(query[coordinates], "float32") * T.cos(angle)
                    + sign * T.cast(query[paired_coordinates], "float32") * T.sin(angle),
                    T.cast(query[coordinates], "float32"),
                )
                key_value = T.if_then_else(
                    channel < dimensions,
                    T.cast(key[coordinates], "float32") * T.cos(angle)
                    + sign * T.cast(key[paired_coordinates], "float32") * T.sin(angle),
                    T.cast(key[coordinates], "float32"),
                )
                query_out[coordinates] = T.cast(query_value, spec.dtype.value)
                key_out[coordinates] = T.cast(key_value, spec.dtype.value)


@T.macro
def _attention_prepare_kernel(
    query_gate,
    keys,
    query_norm,
    key_norm,
    coordinates,
    query_out,
    key_out,
    gate_out,
    rows,
    query_heads,
    kv_heads,
    width,
    rotary_width,
    base,
    sections,
    epsilon,
    dtype,
    threads,
):
    half = rotary_width // 2
    with T.Kernel(query_heads + kv_heads, rows, threads=threads) as (head, row):
        lane = T.get_thread_binding(0)
        square = T.alloc_local((1,), "float32")
        shared = T.alloc_shared((threads,), "float32")
        square[0] = 0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                raw = T.if_then_else(
                    head < query_heads,
                    query_gate[row, head * 2 * width + channel],
                    keys[row, (head - query_heads) * width + channel],
                ).astype("float32")
                square[0] += raw * raw
        shared[lane] = square[0]
        T.sync_threads()
        for step in T.unroll(int(math.log2(threads))):
            if lane < (threads >> (step + 1)):
                shared[lane] += shared[lane + (threads >> (step + 1))]
            T.sync_threads()
        inverse = T.rsqrt(shared[0] / width + epsilon)
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                source = T.if_then_else(
                    head < query_heads,
                    query_gate[row, head * 2 * width + channel],
                    keys[row, (head - query_heads) * width + channel],
                ).astype("float32")
                weight = T.if_then_else(
                    head < query_heads, query_norm[channel], key_norm[channel]
                ).astype("float32")
                value = T.alloc_local((1,), "float32")
                value[0] = source * inverse * weight
                if channel < rotary_width:
                    index = channel % half
                    axis = T.if_then_else(
                        index % 3 == 1 and index < sections[1] * 3,
                        1,
                        T.if_then_else(index % 3 == 2 and index < sections[2] * 3, 2, 0),
                    )
                    pair = (channel + half) % rotary_width
                    paired = (
                        T.if_then_else(
                            head < query_heads,
                            query_gate[row, head * 2 * width + pair],
                            keys[row, (head - query_heads) * width + pair],
                        ).astype("float32")
                        * inverse
                        * T.if_then_else(
                            head < query_heads, query_norm[pair], key_norm[pair]
                        ).astype("float32")
                    )
                    angle = coordinates[row, axis].astype("float32") / T.pow(
                        base,
                        T.cast((index * 2), "float32") / rotary_width,
                    )
                    value[0] = value[0] * T.cos(angle) + T.if_then_else(
                        channel < half, -paired, paired
                    ) * T.sin(angle)
                if head < query_heads:
                    query_out[row, head, channel] = T.cast(value[0], dtype)
                    gate_out[row, head, channel] = query_gate[
                        row, head * 2 * width + width + channel
                    ]
                else:
                    key_out[row, head - query_heads, channel] = T.cast(value[0], dtype)


@T.macro
def _kv_append_kernel(cache, keys, values, positions, key_spec, position_spec, threads):
    heads, width = key_spec.shape[-2:]
    with T.Kernel(T.ceildiv(key_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < key_spec.elements:
                token = flat // (heads * width)
                head = (flat // width) % heads
                channel = flat % width
                destination = T.cast(_load(positions, position_spec, token), "int32")
                if destination >= 0:
                    cache[0, destination, head, channel] = _load(keys, key_spec, flat)
                    cache[1, destination, head, channel] = _load(values, key_spec, flat)


@T.macro
def _kv_copy_kernel(cache, ranges, cache_spec, range_spec, max_count, threads):
    heads, width = cache_spec.shape[2:]
    copies = range_spec.shape[0]
    elements = copies * 2 * max_count * heads * width
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                copy = flat // (2 * max_count * heads * width)
                rest = flat % (2 * max_count * heads * width)
                plane = rest // (max_count * heads * width)
                rest %= max_count * heads * width
                position = rest // (heads * width)
                head = rest // width % heads
                channel = rest % width
                if position < ranges[copy, 2]:
                    cache[plane, ranges[copy, 1] + position, head, channel] = cache[
                        plane, ranges[copy, 0] + position, head, channel
                    ]


@T.macro
def _attention_kernel(
    query,
    history,
    visible,
    output,
    query_spec,
    history_spec,
    visible_spec,
    output_dtype,
    scale,
    explicit,
    threads,
):
    tokens, heads, width = query_spec.shape
    capacity = history_spec.shape[1]
    kv_heads = history_spec.shape[2]
    group = heads // kv_heads
    rows = tokens * heads
    with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            row = block * threads + lane
            if row < rows:
                head = row % heads
                kv_head = head // group
                start = 0
                limit = capacity
                if explicit:
                    if visible_spec.rank == 2:
                        start = T.cast(visible[row // heads, 0], "int32")
                        limit = T.cast(visible[row // heads, 1], "int32")
                    else:
                        limit = T.cast(visible[row // heads], "int32")
                maximum = T.alloc_local((1,), "float32")
                total = T.alloc_local((1,), "float32")
                mixed = T.alloc_local((width,), "float32")
                maximum[0] = -3.402823466e38
                for channel in T.serial(width):
                    mixed[channel] = 0.0
                for relative in T.serial(limit):
                    position = start + relative
                    score = T.alloc_local((1,), "float32")
                    score[0] = 0.0
                    for channel in T.serial(width):
                        score[0] += T.cast(query[row // heads, head, channel], "float32") * T.cast(
                            history[0, position, kv_head, channel], "float32"
                        )
                    score[0] *= scale
                    next_maximum = T.max(maximum[0], score[0])
                    correction = T.exp(maximum[0] - next_maximum)
                    probability = T.exp(score[0] - next_maximum)
                    total[0] = total[0] * correction + probability
                    for channel in T.serial(width):
                        mixed[channel] = mixed[channel] * correction + probability * T.cast(
                            history[1, position, kv_head, channel], "float32"
                        )
                    maximum[0] = next_maximum
                for channel in T.serial(width):
                    output[row // heads, head, channel] = T.cast(
                        mixed[channel] / total[0], output_dtype
                    )


@T.macro
def _recurrence_kernel(
    values, state, decay, output, value_spec, state_spec, decay_spec, dtype, has_decay, threads
):
    tokens, width = value_spec.shape
    with T.Kernel(T.ceildiv(width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            channel = block * threads + lane
            if channel < width:
                recurrent = T.alloc_local((1,), "float32")
                recurrent[0] = T.cast(_state_load(state, state_spec, channel), "float32")
                for token in T.serial(tokens):
                    value = T.cast(values[token, channel], "float32")
                    if has_decay:
                        value += (
                            T.cast(_decay_load(decay, decay_spec, token, channel), "float32")
                            * recurrent[0]
                        )
                    recurrent[0] = value
                    output[token, channel] = T.cast(value, dtype)
                if state_spec.rank == 1:
                    state[channel] = T.cast(recurrent[0], state_spec.dtype.value)
                else:
                    state[0, channel] = T.cast(recurrent[0], state_spec.dtype.value)


@T.macro
def _gated_delta_kernel(
    query,
    key,
    value,
    decay,
    beta,
    previous,
    offsets,
    output,
    next_state,
    batch,
    rows,
    key_heads,
    value_heads,
    key_width,
    value_width,
    mapping,
    lanes,
    output_tile,
    dtype,
):
    with T.Kernel(
        T.ceildiv(value_width, output_tile),
        value_heads,
        batch,
        threads=lanes * output_tile,
    ) as (block, head, sequence):
        tid = T.get_thread_binding(0)
        lane = tid % lanes
        slot = tid // lanes
        channel = block * output_tile + slot
        state = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
        partial = T.alloc_local((1,), "float32")
        residual = T.alloc_local((1,), "float32")
        sums = T.alloc_shared((output_tile, lanes), "float32")
        key_head = T.if_then_else(
            mapping == "tiled",
            head % key_heads,
            head // (value_heads // key_heads),
        )
        for chunk in T.serial(T.ceildiv(key_width, lanes)):
            reduction = chunk * lanes + lane
            if channel < value_width and reduction < key_width:
                state[chunk] = previous[sequence, head, channel, reduction]
        count = offsets[sequence + 1] - offsets[sequence]
        for step_index in T.serial(rows):
            if step_index < count:
                row = offsets[sequence] + step_index
                partial[0] = 0
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    reduction = chunk * lanes + lane
                    if channel < value_width and reduction < key_width:
                        state[chunk] *= decay[row, head]
                        partial[0] += state[chunk] * key[row, key_head, reduction]
                sums[slot, lane] = partial[0]
                T.sync_threads()
                for reduction_step in T.unroll(int(math.log2(lanes))):
                    if lane < (lanes >> (reduction_step + 1)):
                        sums[slot, lane] += sums[slot, lane + (lanes >> (reduction_step + 1))]
                    T.sync_threads()
                residual[0] = 0
                if channel < value_width:
                    residual[0] = (value[row, head, channel] - sums[slot, 0]) * beta[row, head]
                T.sync_threads()
                partial[0] = 0
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    reduction = chunk * lanes + lane
                    if channel < value_width and reduction < key_width:
                        state[chunk] += residual[0] * key[row, key_head, reduction]
                        partial[0] += state[chunk] * query[row, key_head, reduction]
                sums[slot, lane] = partial[0]
                T.sync_threads()
                for reduction_step in T.unroll(int(math.log2(lanes))):
                    if lane < (lanes >> (reduction_step + 1)):
                        sums[slot, lane] += sums[slot, lane + (lanes >> (reduction_step + 1))]
                    T.sync_threads()
                if lane == 0 and channel < value_width:
                    output[row, head, channel] = T.cast(sums[slot, 0], dtype)
                T.sync_threads()
        for chunk in T.serial(T.ceildiv(key_width, lanes)):
            reduction = chunk * lanes + lane
            if channel < value_width and reduction < key_width:
                next_state[sequence, head, channel, reduction] = state[chunk]


@T.macro
def _recurrent_prepare_kernel(
    projected,
    convolution,
    previous,
    offsets,
    alpha,
    beta_input,
    rate,
    bias,
    query,
    key,
    value,
    beta,
    decay,
    next_state,
    batch,
    rows,
    key_heads,
    value_heads,
    width,
    history,
    epsilon,
    dtype,
):
    heads = 2 * key_heads + value_heads
    with T.Kernel(heads, batch, rows, threads=1) as (head, sequence, step):
        active = step < offsets[sequence + 1] - offsets[sequence]
        row = offsets[sequence] + step
        convolved = T.alloc_local((width,), "float32")
        squares = T.alloc_local((1,), "float32")
        squares[0] = 0
        if active:
            for channel in T.serial(width):
                packed_channel = head * width + channel
                convolved[channel] = 0
                for time in T.serial(history):
                    source_step = step - history + time
                    if source_step < 0:
                        convolved[channel] += (
                            previous[sequence, packed_channel, source_step + history]
                            * convolution[packed_channel, time]
                        )
                    else:
                        convolved[channel] += (
                            projected[offsets[sequence] + source_step, packed_channel]
                            * convolution[packed_channel, time]
                        )
                    if step == offsets[sequence + 1] - offsets[sequence] - 1:
                        final_step = offsets[sequence + 1] - offsets[sequence] - history + time
                        next_state[sequence, packed_channel, time] = T.if_then_else(
                            final_step < 0,
                            previous[sequence, packed_channel, final_step + history],
                            projected[offsets[sequence] + final_step, packed_channel],
                        )
                convolved[channel] += (
                    projected[row, packed_channel] * convolution[packed_channel, history]
                )
                convolved[channel] *= T.sigmoid(convolved[channel])
                if head < 2 * key_heads:
                    squares[0] += convolved[channel] * convolved[channel]
            inverse = T.rsqrt(squares[0] + epsilon)
            for channel in T.serial(width):
                if head < key_heads:
                    query[row, head, channel] = T.cast(
                        convolved[channel] * inverse / T.sqrt(T.cast(width, "float32")), dtype
                    )
                elif head < 2 * key_heads:
                    key[row, head - key_heads, channel] = T.cast(
                        convolved[channel] * inverse, dtype
                    )
                else:
                    value[row, head - 2 * key_heads, channel] = T.cast(convolved[channel], dtype)
            if head >= 2 * key_heads:
                value_head = head - 2 * key_heads
                shifted = T.cast(alpha[row, value_head], "float32") + bias[value_head]
                softplus = T.max(shifted, 0) + T.log(1 + T.exp(-T.abs(shifted)))
                beta[row, value_head] = T.cast(
                    T.sigmoid(T.cast(beta_input[row, value_head], "float32")), dtype
                )
                decay[row, value_head] = T.exp(rate[value_head] * softplus)


@T.macro
def _routing_kernel(source, indices, weights, tokens, experts, k, scoring, normalize, threads):
    with T.Kernel(T.ceildiv(tokens, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            token = block * threads + lane
            if token < tokens:
                scores = T.alloc_local((experts,), "float32")
                best = T.alloc_local((k,), "float32")
                selected = T.alloc_local((k,), "int32")
                denominator = T.alloc_local((1,), "float32")
                if scoring == "softmax":
                    peak = T.alloc_local((1,), "float32")
                    peak[0] = -3.402823466e38
                    for expert in T.serial(experts):
                        peak[0] = T.max(peak[0], T.cast(source[token, expert], "float32"))
                    denominator[0] = 0.0
                    for expert in T.serial(experts):
                        scores[expert] = T.exp(T.cast(source[token, expert], "float32") - peak[0])
                        denominator[0] += scores[expert]
                    for expert in T.serial(experts):
                        scores[expert] /= denominator[0]
                else:
                    for expert in T.serial(experts):
                        scores[expert] = T.sigmoid(T.cast(source[token, expert], "float32"))
                for choice in T.serial(k):
                    best[choice] = -3.402823466e38
                    selected[choice] = -1
                for choice in T.serial(k):
                    for expert in T.serial(experts):
                        available = T.alloc_local((1,), "int32")
                        available[0] = 1
                        for prior in T.serial(choice):
                            if selected[prior] == expert:
                                available[0] = 0
                        if available[0] != 0 and (
                            scores[expert] > best[choice]
                            or (scores[expert] == best[choice] and expert > selected[choice])
                        ):
                            best[choice] = scores[expert]
                            selected[choice] = expert
                if normalize:
                    denominator[0] = 0.0
                    for choice in T.serial(k):
                        denominator[0] += best[choice]
                for choice in T.serial(k):
                    slot = k - 1 - choice
                    indices[token, slot] = selected[choice]
                    if normalize:
                        weights[token, slot] = best[choice] / denominator[0]
                    else:
                        weights[token, slot] = best[choice]


@T.macro
def _row_dot_kernel(source, weight, output, source_spec, output_dtype, threads):
    rows = source_spec.elements // source_spec.shape[-1]
    width = source_spec.shape[-1]
    with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            row = block * threads + lane
            if row < rows:
                total = T.alloc_local((1,), "float32")
                total[0] = 0
                for channel in T.serial(width):
                    total[0] += T.cast(source[row, channel], "float32") * T.cast(
                        weight[channel], "float32"
                    )
                output[row, 0] = T.cast(total[0], output_dtype)


@T.macro
def _experts_kernel(
    hidden, routes, scores, gate, up, down, output, specs, activation, output_dtype, threads
):
    hidden_spec, route_spec, _, gate_spec, up_spec, down_spec = specs[:6]
    tokens, width = hidden_spec.shape
    choices = route_spec.shape[1]
    intermediate = gate_spec.shape[1]
    with T.Kernel(T.ceildiv(tokens * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < tokens * width:
                token = flat // width
                channel = flat % width
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for choice in T.serial(choices):
                    expert = T.cast(routes[token, choice], "int32")
                    for inner in T.serial(intermediate):
                        gate_value = T.alloc_local((1,), "float32")
                        up_value = T.alloc_local((1,), "float32")
                        gate_value[0] = 0.0
                        up_value[0] = 0.0
                        for source_channel in T.serial(width):
                            gate_index = (expert * intermediate + inner) * width + source_channel
                            gate_value[0] += T.cast(
                                hidden[token, source_channel], "float32"
                            ) * T.cast(_weight_load(gate, gate_spec, gate_index), "float32")
                            up_value[0] += T.cast(
                                hidden[token, source_channel], "float32"
                            ) * T.cast(_weight_load(up, up_spec, gate_index), "float32")
                        if activation == "silu":
                            activated = gate_value[0] * T.sigmoid(gate_value[0])
                        else:
                            activated = T.tanh(gate_value[0])
                        down_index = (expert * width + channel) * intermediate + inner
                        total[0] += (
                            T.cast(scores[token, choice], "float32")
                            * activated
                            * up_value[0]
                            * T.cast(_weight_load(down, down_spec, down_index), "float32")
                        )
                output[token, channel] = T.cast(total[0], output_dtype)


class PrimitiveEmitter:
    def __init__(self, node: Node, graph: Graph, capabilities: Capabilities):
        self.node = node
        self.inputs: tuple[TensorSpec, ...] = tuple(
            graph.values[value].spec for value in node.inputs
        )
        self.outputs: tuple[TensorSpec, ...] = tuple(
            graph.values[value].spec for value in node.outputs
        )
        self.capabilities = capabilities
        self.threads = min(256, capabilities.threads_per_group)

    def __call__(self, operands: tuple[Any, ...]) -> None:
        split = len(self.node.inputs)
        inputs = list(operands[:split])
        outputs = list(operands[split : split + len(self.node.outputs)])
        operation = self.node.operation
        if operation == "scalar":
            _scalar_kernel(outputs[0], self.node.attributes["value"])
        elif operation in {"add", "subtract", "multiply", "divide"}:
            _pointwise_kernel(
                inputs[0],
                inputs[1],
                outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.outputs[0],
                operation,
                self.threads,
            )
        elif operation in {
            "cast",
            "decode_bfloat16",
            "reshape",
            "exp",
            "sigmoid",
            "silu",
            "tanh",
        }:
            _unary_kernel(
                inputs[0], outputs[0], self.inputs[0], self.outputs[0], operation, self.threads
            )
        elif operation == "transpose":
            _transpose_kernel(
                inputs[0],
                outputs[0],
                self.inputs[0].shape,
                self.outputs[0].shape,
                self.node.attributes["axes"],
                self.threads,
            )
        elif operation == "concatenate":
            axis = self.node.attributes["axis"] % self.outputs[0].rank
            _concatenate_kernel(
                tuple(inputs),
                outputs[0],
                tuple(spec.shape for spec in self.inputs),
                self.outputs[0].shape,
                axis,
                self.outputs[0].dtype.value,
                self.threads,
            )
        elif operation == "take_rows":
            row_elements = self.inputs[0].elements // cast(int, self.inputs[0].shape[0])
            _take_rows_kernel(
                inputs[0],
                inputs[1],
                outputs[0],
                self.inputs[0],
                self.outputs[0],
                row_elements,
                self.threads,
            )
        elif operation == "overlay_rows":
            _overlay_rows_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                outputs[0],
                self.outputs[0],
                self.inputs[1],
                cast(int, self.inputs[2].shape[0]),
                self.threads,
            )
        elif operation == "quantized_import":
            _quantized_import_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                self.inputs[1],
                self.node.attributes["codec"],
                self.node.attributes["staged_tiles"],
                min(128, self.threads),
            )
        elif operation == "sample":
            rows, vocabulary = cast(tuple[int, int], self.inputs[0].shape)
            threads = min(128, self.threads)
            _sample_kernel(inputs[0], inputs[1], outputs[0], rows, vocabulary, threads)
        elif operation == "softmax":
            width = cast(int, self.inputs[0].shape[-1])
            _softmax_kernel(
                inputs[0],
                outputs[0],
                self.inputs[0].elements // width,
                width,
                self.outputs[0].dtype.value,
                self.threads,
            )
        elif operation == "rms_norm":
            width = cast(int, self.inputs[0].shape[-1])
            weight = inputs[1] if len(inputs) == 2 else inputs[0]
            _rms_kernel(
                inputs[0],
                weight,
                outputs[0],
                self.inputs[0],
                self.inputs[0].elements // width,
                width,
                self.node.attributes["epsilon"],
                self.outputs[0].dtype.value,
                len(inputs) == 2,
                self.threads,
            )
        elif operation in {"matmul", "linear"}:
            bias = inputs[2] if len(inputs) == 3 else inputs[0]
            _contraction_kernel(
                inputs[0],
                inputs[1],
                bias,
                outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.outputs[0],
                operation,
                len(inputs) == 3,
                self.threads,
            )
        elif operation == "embedding":
            _embedding_kernel(
                inputs[0],
                inputs[1],
                outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.outputs[0],
                self.threads,
            )
        elif operation == "rotary":
            position = inputs[2] if len(inputs) == 3 else inputs[0]
            position_spec = self.inputs[2] if len(inputs) == 3 else self.inputs[0]
            _rotary_kernel(
                inputs[0],
                inputs[1],
                position,
                outputs[0],
                outputs[1],
                self.inputs[0],
                position_spec,
                self.node.attributes["dimensions"],
                self.node.attributes["base"],
                len(inputs) == 3,
                self.threads,
            )
        elif operation == "attention_prepare":
            attrs = self.node.attributes
            rows = cast(int, self.inputs[0].shape[0])
            _attention_prepare_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                outputs[0],
                outputs[1],
                outputs[2],
                rows,
                attrs["query_heads"],
                attrs["kv_heads"],
                attrs["width"],
                attrs["rotary_width"],
                attrs["base"],
                attrs["sections"],
                attrs["epsilon"],
                self.outputs[0].dtype.value,
                min(self.threads, 128),
            )
        elif operation == "kv_append":
            _kv_append_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                self.inputs[1],
                self.inputs[3],
                self.threads,
            )
        elif operation == "kv_copy":
            _kv_copy_kernel(
                inputs[0],
                inputs[1],
                self.inputs[0],
                self.inputs[1],
                self.node.attributes["max_count"],
                self.threads,
            )
        elif operation == "causal_attention":
            visible = inputs[2] if len(inputs) == 3 else inputs[0]
            _attention_kernel(
                inputs[0],
                inputs[1],
                visible,
                outputs[0],
                self.inputs[0],
                self.inputs[1],
                self.inputs[2] if len(inputs) == 3 else self.inputs[0],
                self.outputs[0].dtype.value,
                self.node.attributes["scale"],
                len(inputs) == 3,
                self.threads,
            )
        elif operation == "delta_recurrence":
            input_specs = list(self.inputs)
            decay = inputs[2] if len(inputs) > 2 else inputs[0]
            decay_spec = input_specs[2] if len(input_specs) > 2 else input_specs[0]
            _recurrence_kernel(
                inputs[0],
                inputs[1],
                decay,
                outputs[0],
                input_specs[0],
                input_specs[1],
                decay_spec,
                self.outputs[0].dtype.value,
                len(inputs) > 2,
                self.threads,
            )
        elif operation == "gated_delta_recurrence":
            batch, value_heads, value_width, key_width = cast(
                tuple[int, int, int, int], self.inputs[5].shape
            )
            rows, key_heads, _ = cast(tuple[int, int, int], self.inputs[0].shape)
            lanes = min(32, self.capabilities.subgroup_width)
            output_tile = min(4, self.capabilities.threads_per_group // lanes)
            _gated_delta_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                outputs[0],
                outputs[1],
                batch,
                rows,
                key_heads,
                value_heads,
                key_width,
                value_width,
                self.node.attributes["mapping"],
                lanes,
                output_tile,
                self.outputs[0].dtype.value,
            )
        elif operation == "recurrent_prepare":
            batch, _, history = cast(tuple[int, int, int], self.inputs[2].shape)
            rows = cast(int, self.inputs[0].shape[0])
            attrs = self.node.attributes
            _recurrent_prepare_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[7],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3],
                outputs[4],
                outputs[5],
                batch,
                rows,
                attrs["key_heads"],
                attrs["value_heads"],
                attrs["width"],
                history,
                attrs["epsilon"],
                self.outputs[0].dtype.value,
            )
        elif operation == "route_topk":
            tokens, experts = cast(tuple[int, int], self.inputs[0].shape)
            _routing_kernel(
                inputs[0],
                outputs[0],
                outputs[1],
                tokens,
                experts,
                self.node.attributes["k"],
                self.node.attributes["scoring"],
                self.node.attributes["normalize"],
                self.threads,
            )
        elif operation == "row_dot":
            _row_dot_kernel(
                inputs[0],
                inputs[1],
                outputs[0],
                self.inputs[0],
                self.outputs[0].dtype.value,
                self.threads,
            )
        elif operation == "routed_experts":
            _experts_kernel(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                outputs[0],
                self.inputs,
                self.node.attributes["activation"],
                self.outputs[0].dtype.value,
                self.threads,
            )
        else:
            raise NotImplementedError(f"no portable lowering for {operation}")


class PrimitiveLoweringRule:
    name = "portable-primitive"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        definition = operations.get(node.operation)
        inputs = tuple(value for value in node.inputs if graph.values[value].producer != root)
        outputs = node.outputs
        aliases = tuple(
            (node.outputs[output], node.inputs[source]) for output, source in definition.aliases
        )
        moved = sum(spec.storage_nbytes for spec in specs)
        kernel_count = 1
        estimated = 1e-6 + moved / 100e9
        if node.operation in {"linear", "matmul"}:
            left = graph.values[node.inputs[0]].spec
            output = graph.values[node.outputs[0]].spec
            reduction = cast(int, left.shape[-1])
            # The portable contraction assigns one output element to one lane and
            # performs its reduction serially.  Charging it as a pure memory copy
            # made it look cheaper than matrix-instruction schedules and selected
            # scalar dot products for almost the entire decoder.
            flops = 2 * output.elements * reduction
            estimated += flops / 25e9
        elif node.operation == "routed_experts":
            hidden = graph.values[node.inputs[0]].spec
            routes = graph.values[node.inputs[1]].spec
            gate = graph.values[node.inputs[3]].spec
            rows, width = cast(tuple[int, int], hidden.shape)
            selected = cast(int, routes.shape[1])
            intermediate = cast(int, gate.shape[1])
            # The portable baseline performs gate, up, and down reductions
            # serially for every selected route. It is compute, not a table copy.
            flops = 6 * rows * selected * width * intermediate
            estimated += flops / 25e9
        elif node.operation == "causal_attention":
            query = graph.values[node.inputs[0]].spec
            history = graph.values[node.inputs[1]].spec
            rows, heads, width = cast(tuple[int, int, int], query.shape)
            capacity = cast(int, history.shape[1])
            # Both primitive attention emitters synchronize a channel reduction
            # for every visible history row. Charge the worst-case arithmetic so
            # tiled matrix or partitioned schedules win when they are available.
            flops = 4 * rows * heads * capacity * width
            estimated += flops / 25e9
        return (
            Candidate(
                f"{node.operation}.portable@{root}",
                frozenset({root}),
                inputs,
                outputs,
                PrimitiveEmitter(node, graph, context.capabilities),
                estimated,
                aliases=aliases,
                kernel_count=kernel_count,
            ),
        )


def register_builtin_lowerings(registry: LoweringRegistry) -> None:
    if not any(rule.name == PrimitiveLoweringRule.name for rule in registry.rules):
        registry.register(PrimitiveLoweringRule())


def _elements(shape) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def _transpose_indices(destination, axes):
    origin = [0] * len(axes)
    for output_axis, source_axis in enumerate(axes):
        origin[source_axis] = destination[output_axis]
    return tuple(origin)


def _weight_index(operation, column, k, width, columns):
    return column * width + k if operation == "linear" else k * columns + column


def _dense_weight(weight, operation, column, k):
    return weight[column, k] if operation == "linear" else weight[k, column]


def _weight_load(weight, spec, index):
    if spec.representation is None or isinstance(spec.representation, Dense):
        return _load(weight, spec, index)
    return represented_load(weight, spec, index)


def _state_load(state, spec, channel):
    return state[channel] if spec.rank == 1 else state[0, channel]


def _decay_load(decay, spec, token, channel):
    return decay[token, channel] if spec.rank == 2 else decay[channel]
