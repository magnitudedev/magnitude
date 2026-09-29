"""Portable semantic baselines authored directly with TileLang Python."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, CompilerTarget, LoweringContext
from ..kv import KVRepresentation
from ..representations import (
    Affine,
    Dense,
    DirectCoefficients,
    HierarchicalCoefficients,
    canonical_layout,
)
from ..tensor.graph import Graph, Node
from ..tensor.primitive import primitives
from ..tensor.types import DType, TensorSpec, dense_strides
from .kv_packed import copy_bundle


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
def _unpack_field(source, output, spec, flat, offset):
    if offset <= flat and flat < offset + spec.elements:
        output[_indices(flat - offset, spec.shape)] = T.reinterpret(source[flat], spec.dtype.value)


def _unpack_fields(source, outputs, specs, flat):
    offset = 0
    for output, spec in zip(outputs, specs, strict=True):
        _unpack_field(source, output, spec, flat, offset)
        offset += spec.elements


@T.macro
def _unpack_words_kernel(source, outputs, specs, words, threads):
    with T.Kernel(T.ceildiv(words, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            _unpack_fields(source, outputs, specs, flat)


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
                elif op == "less":
                    value = lhs < rhs
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
                if op == "gelu":
                    x = T.cast(source_value, "float32")
                    value[0] = 0.5 * x * (1.0 + T.erf(x * 0.7071067811865476))
                if op == "gelu_tanh":
                    x = T.cast(source_value, "float32")
                    value[0] = (
                        0.5 * x * (1.0 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                    )
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
def _concatenate_kernel(sources, output, source_shapes, output_shape, axis, output_dtype, threads):
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
    # Represented resources use a word-oriented compute ABI.  Residency owns
    # construction of the byte-exact canonical format, so alias that storage
    # as bytes only for this import kernel.
    target_bytes = T.decl_buffer((target_spec.storage_nbytes,), "uint8", data=target.data)
    representation = target_spec.representation
    resident = canonical_layout(representation, target_spec.elements)
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
            target_base = target_tile * resident.tile_bytes if resident.hierarchical else 0
            for index in T.serial(tile_elements):
                value = codec.code(source, source_base, index)
                logical = index if resident.hierarchical else target_element + index
                bit = logical * low_bits
                byte = bit // 8
                lane = bit % 8
                if lane == 0:
                    target_bytes[target_base + resident.low + byte] = 0
                target_bytes[target_base + resident.low + byte] = T.cast(
                    target_bytes[target_base + resident.low + byte]
                    | ((value & ((1 << low_bits) - 1)) << lane),
                    "uint8",
                )
                if high_bits:
                    high_bit = logical * high_bits
                    high_byte = high_bit // 8
                    high_lane = high_bit % 8
                    if high_lane == 0:
                        target_bytes[target_base + resident.high + high_byte] = 0
                    target_bytes[target_base + resident.high + high_byte] = T.cast(
                        target_bytes[target_base + resident.high + high_byte]
                        | (((value >> low_bits) & ((1 << high_bits) - 1)) << high_lane),
                        "uint8",
                    )
            if isinstance(coefficients, DirectCoefficients):
                if tile_groups == 1:
                    scale_base = code_bytes + target_tile * coefficients.scale_dtype.itemsize
                    for byte in T.unroll(coefficients.scale_dtype.itemsize):
                        target_bytes[scale_base + byte] = codec.scale_byte(
                            source, source_base, byte
                        )
                    if coefficients.bias_dtype is not None:
                        bias_base = (
                            code_bytes
                            + groups * coefficients.scale_dtype.itemsize
                            + target_tile * coefficients.bias_dtype.itemsize
                        )
                        for byte in T.unroll(coefficients.bias_dtype.itemsize):
                            target_bytes[bias_base + byte] = codec.bias_byte(
                                source, source_base, byte
                            )
                else:
                    for group in T.serial(tile_groups):
                        target_group = target_tile * tile_groups + group
                        scale = codec.direct_scale(
                            source, source_base, group, T.if_then_else, T.reinterpret
                        )
                        scale_bits = T.reinterpret(T.cast(scale, "float32"), "uint32")
                        scale_base = code_bytes + target_group * coefficients.scale_dtype.itemsize
                        for byte in T.unroll(coefficients.scale_dtype.itemsize):
                            target_bytes[scale_base + byte] = T.cast(
                                scale_bits >> (8 * byte), "uint8"
                            )
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
                                target_bytes[bias_base + byte] = T.cast(
                                    bias_bits >> (8 * byte), "uint8"
                                )
            else:
                assert isinstance(coefficients, HierarchicalCoefficients)
                scale_base = target_base + resident.scales
                tile_scale_bytes = math.ceil(tile_groups * coefficients.local_scale_bits / 8)
                tile_scale_base = scale_base
                for byte in T.serial(tile_scale_bytes):
                    target_bytes[tile_scale_base + byte] = 0
                for group in T.serial(tile_groups):
                    value = codec.local_scale(source, source_base, group, T.if_then_else)
                    bit = group * coefficients.local_scale_bits
                    target_bytes[scale_base + bit // 8] |= T.cast(value << (bit % 8), "uint8")
                    if bit % 8 + coefficients.local_scale_bits > 8:
                        target_bytes[scale_base + bit // 8 + 1] |= T.cast(
                            value >> (8 - bit % 8), "uint8"
                        )
                cursor = scale_base + tile_scale_bytes
                if coefficients.local_bias_bits is not None:
                    tile_bias_bytes = math.ceil(tile_groups * coefficients.local_bias_bits / 8)
                    tile_bias_base = cursor
                    for byte in T.serial(tile_bias_bytes):
                        target_bytes[tile_bias_base + byte] = 0
                    for group in T.serial(tile_groups):
                        value = codec.local_bias(source, source_base, group, T.if_then_else)
                        bit = group * coefficients.local_bias_bits
                        target_bytes[cursor + bit // 8] |= T.cast(value << (bit % 8), "uint8")
                        if bit % 8 + coefficients.local_bias_bits > 8:
                            target_bytes[cursor + bit // 8 + 1] |= T.cast(
                                value >> (8 - bit % 8), "uint8"
                            )
                    cursor += tile_bias_bytes
                if coefficients.super_scale_dtype == DType.F32:
                    bits = T.reinterpret(codec.super_scale(source, source_base, T.reinterpret), "uint32")
                    for byte in T.unroll(4):
                        target_bytes[cursor + byte] = T.cast(bits >> (byte * 8), "uint8")
                else:
                    for byte in T.unroll(coefficients.super_scale_dtype.itemsize):
                        target_bytes[cursor + byte] = codec.scale_byte(source, source_base, byte)
                cursor += coefficients.super_scale_dtype.itemsize
                if coefficients.super_bias_dtype is not None:
                    if coefficients.super_bias_dtype == DType.F32:
                        bits = T.reinterpret(codec.super_bias(source, source_base, T.reinterpret), "uint32")
                        for byte in T.unroll(4):
                            target_bytes[cursor + byte] = T.cast(bits >> (byte * 8), "uint8")
                    else:
                        for byte in T.unroll(coefficients.super_bias_dtype.itemsize):
                            target_bytes[cursor + byte] = codec.bias_byte(source, source_base, byte)


@T.macro
def _embedding_kernel(tokens, table, output, token_spec, table_spec, output_spec, threads):
    width = output_spec.shape[-1]
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                row = T.cast(_load(tokens, token_spec, flat // width), "int32")
                value = table[row, flat % width]
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


class PrimitiveEmitter:
    def __init__(self, node: Node, graph: Graph, compiler_target: CompilerTarget):
        self.node = node
        self.inputs: tuple[TensorSpec, ...] = tuple(
            graph.values[value].spec for value in node.inputs
        )
        self.outputs: tuple[TensorSpec, ...] = tuple(
            graph.values[value].spec for value in node.outputs
        )
        self.compiler_target = compiler_target
        self.threads = min(256, compiler_target.threads_per_group)

    def specialization_key(self):
        """Identify generated code without graph-local value or node ids."""
        return (
            self.node.operation,
            dict(self.node.attributes),
            self.inputs,
            self.outputs,
            self.threads,
        )

    def __call__(self, operands: tuple[Any, ...]) -> None:
        split = len(self.node.inputs)
        inputs = list(operands[:split])
        outputs = list(operands[split : split + len(self.node.outputs)])
        operation = self.node.operation
        if operation == "scalar":
            _scalar_kernel(outputs[0], self.node.attributes["value"])
        elif operation in {"add", "subtract", "multiply", "divide", "less"}:
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
            "gelu",
            "gelu_tanh",
        }:
            _unary_kernel(
                inputs[0], outputs[0], self.inputs[0], self.outputs[0], operation, self.threads
            )
        elif operation == "unpack_words":
            _unpack_words_kernel(
                inputs[0], tuple(outputs), self.outputs, self.inputs[0].elements, self.threads
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
        elif operation == "kv_copy":
            copier = (
                copy_bundle
                if isinstance(self.inputs[0].representation, KVRepresentation)
                else _kv_copy_kernel
            )
            copier(
                inputs[0],
                inputs[1],
                self.inputs[0],
                self.inputs[1],
                self.node.attributes["max_count"],
                self.threads,
            )
        else:
            raise NotImplementedError(f"no portable lowering for {operation}")


_PRODUCTION_PRIMITIVES = frozenset(
    {
        "scalar",
        "add",
        "subtract",
        "multiply",
        "divide",
        "less",
        "cast",
        "decode_bfloat16",
        "unpack_words",
        "concatenate",
        "take_rows",
        "overlay_rows",
        "quantized_import",
        "exp",
        "sigmoid",
        "silu",
        "tanh",
        "gelu",
        "gelu_tanh",
        "rotary",
        "kv_copy",
        "embedding",
        "transpose",
    }
)

_REFERENCE_PRIMITIVES = frozenset({"reshape"})


class PrimitiveLoweringRule:
    name = "portable-primitive"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        # This is an allowlist, not a catch-all fallback. New transformer or
        # contraction primitives must acquire an explicit optimized schedule
        # before production graphs can lower them.
        reference = (
            context.precision == "reference"
            or context.compiler_target.reference_schedules
        )
        if node.operation not in _PRODUCTION_PRIMITIVES and not (
            reference and node.operation in _REFERENCE_PRIMITIVES
        ):
            return ()
        if context.mode in {"decode", "prefill"}:
            if node.operation == "embedding":
                table = graph.values[node.inputs[1]].spec
                if table.representation is not None and not isinstance(table.representation, Dense):
                    return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        definition = primitives.get(node.operation)
        inputs = tuple(value for value in node.inputs if graph.values[value].producer != root)
        outputs = node.outputs
        aliases = tuple(
            (node.outputs[output], node.inputs[source]) for output, source in definition.aliases
        )
        moved = sum(spec.storage_nbytes for spec in specs)
        kernel_count = 1
        if node.operation == "kv_copy" and isinstance(specs[0].representation, KVRepresentation):
            kernel_count = len(specs[0].representation.planes(specs[0].shape[0] * specs[0].shape[1]))
        return (
            BoundOperation(
                f"{node.operation}.portable@{root}",
                frozenset({root}),
                inputs,
                outputs,
                PrimitiveEmitter(node, graph, context.compiler_target),
                aliases=aliases,
                kernel_count=kernel_count,
            ),
        )


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
