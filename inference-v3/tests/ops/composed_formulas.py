"""Explicit test formula boundaries using the same physical bodies as production."""

import ops


@ops.formula
def parallel_projections(x, *weights):
    return tuple(ops.linear(x, weight) for weight in weights)


@ops.formula(id="test.compiler.activation_residual")
def activation_residual(value):
    return value + ops.silu(value)


@ops.formula
def attention_output(q, history, visible, gate, w):
    attended = ops.causal_attention(q, history, visible, sequence_count=1)
    rows, heads, width = q.shape
    return ops.linear(ops.reshape(attended * ops.sigmoid(gate), (rows, heads * width)), w)


@ops.formula
def recurrent_output(mixed, norm, gate, w):
    rows, heads, width = mixed.shape
    normalized = ops.rms_norm(mixed, norm, epsilon=1e-6)
    return ops.linear(ops.reshape(normalized, (rows, heads * width)) * ops.silu(gate), w)


@ops.formula
def dense_feedforward(value, gate, up, down):
    return ops.linear(ops.silu(ops.linear(value, gate)) * ops.linear(value, up), down)


@ops.formula
def residual_normalization(left, update, gain):
    residual = left + ops.cast(update, ops.DType.F32)
    return residual, ops.rms_norm(residual, gain, output_dtype=ops.DType.BF16)


@ops.formula
def routed_feedforward(x, ids, probabilities, eg, eu, ed, sg, su, sd, sr):
    routed = ops.routed_experts(x, ids, probabilities, eg, eu, ed)
    shared = ops.linear(ops.silu(ops.linear(x, sg)) * ops.linear(x, su), sd)
    coefficient = ops.cast(ops.sigmoid(ops.row_dot(x, sr, output_dtype=ops.DType.F32)), x.dtype)
    return routed + shared * coefficient


@ops.formula
def dense_residual(x, skip, gate, up, down):
    return skip + ops.cast(dense_feedforward(x, gate, up, down), ops.DType.F32)


@ops.formula
def routed_residual(x, skip, ids, probabilities, eg, eu, ed, sg, su, sd, sr):
    return skip + ops.cast(routed_feedforward(x, ids, probabilities, eg, eu, ed, sg, su, sd, sr), ops.DType.F32)


ops.operation(attention_output)(ops.operation_bodies.attention_mixer)
ops.operation(parallel_projections)(ops.operation_bodies.parallel_projections)
ops.operation(activation_residual)(ops.operation_bodies.pointwise)
ops.operation(recurrent_output)(ops.operation_bodies.recurrent_mixer)
ops.operation(dense_feedforward)(ops.operation_bodies.dense_feedforward)
ops.operation(residual_normalization)(ops.operation_bodies.residual_normalization)
ops.operation(routed_feedforward)(ops.operation_bodies.routed_feedforward)
ops.operation(dense_residual)(ops.operation_bodies.block)
ops.operation(routed_residual)(ops.operation_bodies.block)
