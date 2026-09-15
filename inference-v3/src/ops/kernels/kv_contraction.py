"""Vector coefficients applied at attention contractions, not per coordinate."""
from __future__ import annotations

import tilelang.language as T

from ..kv import AffineKVCodec, RotatedLloydMax
from .kv_packed import plane_view


@T.macro
def query_sums(query, output, tokens, heads, width):
    with T.Kernel(heads, tokens, threads=32) as (head, token):
        lane = T.get_thread_binding()
        total = T.alloc_local((1,), "float32")
        total[0] = 0.0
        for item in T.unroll(width // 32):
            total[0] += T.cast(query[token, head, lane * (width // 32) + item], "float32")
        result = T.warp_reduce_sum(total[0])
        if lane == 0:
            output[token, head] = result


@T.macro
def _key_coefficients(scores, sums, coefficients, zeros, coefficient_base, zero_base,
                       base, first, count, kv_head, kv_heads, rows, keys, query_tile,
                       first_row, first_head, tokens, rotated, width):
    for row, item in T.Parallel(rows, keys):
        token = first_row + row % query_tile
        head = first_head + row // query_tile
        vector = (base + first + item) * kv_heads + kv_head
        factor = T.if_then_else(first + item < count,
                                T.cast(coefficients[coefficient_base + vector], "float32"), 0.0)
        if rotated:
            scores[row, item] *= factor * width ** -0.5
        else:
            zero = T.if_then_else(first + item < count,
                                  T.cast(zeros[zero_base + vector], "float32"), 0.0)
            query_sum = T.if_then_else(token < tokens, sums[token, head], 0.0)
            scores[row, item] = scores[row, item] * factor + query_sum * zero


def correct_key_scores(scores, sums, storage, spec, base, first, count, kv_head,
                        rows, keys, query_tile, first_row, first_head, tokens):
    codec = spec.representation.key
    if not isinstance(codec, (AffineKVCodec, RotatedLloydMax)):
        return
    rotated = isinstance(codec, RotatedLloydMax)
    coefficients, offset, _ = plane_view(storage, spec, "key.norm" if rotated else "key.scale")
    zeros, zero_offset, _ = (coefficients, offset, 1) if rotated else plane_view(storage, spec, "key.zero")
    _key_coefficients(scores, sums, coefficients, zeros, offset, zero_offset,
                       base, first, count, kv_head, spec.shape[1], rows, keys, query_tile,
                       first_row, first_head, tokens, rotated, spec.representation.key_width)


@T.macro
def _value_coefficients(scores, probabilities, scales, zeros, scale_base, zero_base,
                         base, first, count, kv_head, kv_heads, rows, keys):
    # Preserve the unscaled probabilities in the denominator before calling.
    # Reuse the score registers for the common value-offset contraction.
    for row, item in T.Parallel(rows, keys):
        vector = (base + first + item) * kv_heads + kv_head
        scale = T.if_then_else(first + item < count,
                               T.cast(scales[scale_base + vector], "float32"), 0.0)
        zero = T.if_then_else(first + item < count,
                              T.cast(zeros[zero_base + vector], "float32"), 0.0)
        probabilities[row, item] = scores[row, item] * scale
        scores[row, item] *= zero


def value_coefficients(scores, probabilities, storage, spec, base, first, count,
                         kv_head, rows, keys):
    scales, scale_base, _ = plane_view(storage, spec, "value.scale")
    zeros, zero_base, _ = plane_view(storage, spec, "value.zero")
    _value_coefficients(scores, probabilities, scales, zeros, scale_base, zero_base,
                         base, first, count, kv_head, spec.shape[1], rows, keys)


@T.macro
def update_bias(scores, bias, local_bias, alpha, rows):
    T.reduce_sum(scores, local_bias, dim=1)
    for row in T.Parallel(rows):
        bias[row] = bias[row] * alpha[row] + local_bias[row]


@T.macro
def _add_bias(output, bias, rows, columns):
    for row, column in T.Parallel(rows, columns):
        output[row, column] += bias[row]


def finish_bias(outputs, bias, rows, columns):
    for output in outputs:
        _add_bias(output, bias, rows, columns)
