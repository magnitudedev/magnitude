"""Portable semantic baselines authored directly with TileLang Python."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, Capabilities, LoweringContext, LoweringRegistry
from ..representations import Dense
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
                value = _load(source, source_spec, flat)
                if op == "cast":
                    value = T.cast(value, output_spec.dtype.value)
                elif op == "exp":
                    value = T.exp(value)
                elif op == "sigmoid":
                    value = T.sigmoid(value)
                elif op == "silu":
                    value *= T.sigmoid(value)
                elif op == "tanh":
                    value = T.tanh(value)
                output[_indices(flat, output_spec.shape)] = value


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
def _concatenate_piece(source, output, source_shape, output_shape, axis, offset, threads):
    elements = _elements(source_shape)
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                origin = _indices(flat, source_shape)
                destination = _offset_index(origin, axis, offset)
                output[destination] = source[origin]


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
def _rms_kernel(source, weight, output, rows, width, epsilon, dtype, weighted, threads):
    with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            row = block * threads + lane
            if row < rows:
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for column in T.serial(width):
                    value = T.cast(source[row, column], "float32")
                    total[0] += value * value
                inverse = T.rsqrt(total[0] / width + epsilon)
                for column in T.serial(width):
                    value = T.cast(source[row, column], "float32") * inverse
                    if weighted:
                        value *= T.cast(weight[column], "float32")
                    output[row, column] = T.cast(value, dtype)


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
def _embedding_kernel(tokens, table, output, token_spec, output_spec, threads):
    width = output_spec.shape[-1]
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                output[_indices(flat, output_spec.shape)] = table[
                    T.cast(_load(tokens, token_spec, flat // width), "int32"),
                    flat % width,
                ]


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
                cache[0, destination, head, channel] = _load(keys, key_spec, flat)
                cache[1, destination, head, channel] = _load(values, key_spec, flat)


@T.macro
def _attention_kernel(
    query,
    history,
    visible,
    output,
    query_spec,
    history_spec,
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
                limit = capacity
                if explicit:
                    limit = T.cast(visible[row // heads], "int32")
                maximum = T.alloc_local((1,), "float32")
                total = T.alloc_local((1,), "float32")
                mixed = T.alloc_local((width,), "float32")
                maximum[0] = -3.402823466e38
                for channel in T.serial(width):
                    mixed[channel] = 0.0
                for position in T.serial(limit):
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
                        if available[0] != 0 and scores[expert] > best[choice]:
                            best[choice] = scores[expert]
                            selected[choice] = expert
                if normalize:
                    denominator[0] = 0.0
                    for choice in T.serial(k):
                        denominator[0] += best[choice]
                for choice in T.serial(k):
                    indices[token, choice] = selected[choice]
                    if normalize:
                        weights[token, choice] = best[choice] / denominator[0]
                    else:
                        weights[token, choice] = best[choice]


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
        elif operation in {"cast", "reshape", "exp", "sigmoid", "silu", "tanh"}:
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
            offset = 0
            for source, spec in zip(inputs, self.inputs, strict=True):
                _concatenate_piece(
                    source,
                    outputs[0],
                    spec.shape,
                    self.outputs[0].shape,
                    axis,
                    offset,
                    self.threads,
                )
                offset += spec.shape[axis]
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
                inputs[0], inputs[1], outputs[0], self.inputs[0], self.outputs[0], self.threads
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
        elif operation == "causal_attention":
            visible = inputs[2] if len(inputs) == 3 else inputs[0]
            _attention_kernel(
                inputs[0],
                inputs[1],
                visible,
                outputs[0],
                self.inputs[0],
                self.inputs[1],
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
        outputs = tuple(
            value
            for value in node.outputs
            if value in graph.outputs or any(user != root for user in graph.users[value])
        )
        aliases = tuple(
            (node.outputs[output], node.inputs[source]) for output, source in definition.aliases
        )
        moved = sum(spec.storage_nbytes for spec in specs)
        kernel_count = len(node.inputs) if node.operation == "concatenate" else 1
        return (
            Candidate(
                f"{node.operation}.portable@{root}",
                frozenset({root}),
                inputs,
                outputs,
                PrimitiveEmitter(node, graph, context.capabilities),
                1e-6 + moved / 100e9,
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


def _offset_index(origin, axis, offset):
    result = list(origin)
    result[axis] += offset
    return tuple(result)


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
