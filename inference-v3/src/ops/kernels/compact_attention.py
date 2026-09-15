"""Cooperative packed tiles feeding native matrix register operands."""
from __future__ import annotations

import tilelang.language as T

from ..kv import RotatedLloydMax
from .kv_packed import plane_view, word_offset, _packet_layout, lookup_centroid


def _compact_layout(words, keys):
    # Both token-major and four-word-block producers cover distinct banks.
    # Matrix readers of one word broadcast the same location across its codes.
    return T.Layout((words, keys), lambda word, key:
                     (word, key ^ ((word % 4) * 8 + (word // 4) % 8)))


def allocate_compact(spec, keys):
    planes = spec.representation.planes(spec.shape[0] * spec.shape[1])
    words = max(p.row_elements for p in planes if p.name.endswith(".codes"))
    storage = T.alloc_shared((words, keys), "uint32")
    T.annotate_layout({storage: _compact_layout(words, keys)})
    return storage


def _row_packet_layout(keys, words, threads):
    return T.Fragment((keys, words),
                       forward_thread_fn=lambda key, word: (key * words + word) % threads,
                       forward_index_fn=lambda key, word: (key * words + word) // threads)


@T.macro
def _stage_words(source, shared, offset, words, capacity, heads, head, base, first, count, keys, blocked):
    if blocked:
        packet = T.alloc_fragment((words // 4, keys, 4), "uint32")
        T.annotate_layout({packet: _packet_layout(words // 4, keys, T.get_thread_extent())})
        for group, key, word in T.Parallel(words // 4, keys, 4):
            vector = (base + first + key) * heads + head
            packet[group, key, word] = T.if_then_else(first + key < count,
                source[offset + word_offset(vector, group * 4 + word, words, capacity, heads, True)], T.uint32(0))
        for group, key, word in T.Parallel(words // 4, keys, 4):
            shared[group * 4 + word, key] = packet[group, key, word]
    else:
        packet = T.alloc_fragment((keys, words), "uint32")
        T.annotate_layout({packet: _row_packet_layout(keys, words, T.get_thread_extent())})
        for key, word in T.Parallel(keys, words):
            vector = (base + first + key) * heads + head
            packet[key, word] = T.if_then_else(first + key < count,
                                               source[offset + vector * words + word], T.uint32(0))
        for key, word in T.Parallel(keys, words):
            shared[word, key] = packet[key, word]


def stage_compact(history, shared, spec, prefix, head, base, first, count, keys):
    source, offset, words = plane_view(history, spec, prefix + ".codes")
    _stage_words(source, shared, offset, words, spec.shape[0], spec.shape[1], head,
                  base, first, count, keys, spec.representation.packing_version == 2)


@T.macro
def _key_operand(shared, operand, table, block, contraction, keys, bits, rotated):
    per_word = 32 // bits
    for channel, key in T.Parallel(contraction, keys):
        coordinate = block * contraction + channel
        packed = shared[coordinate // per_word, key]
        code = (packed >> ((coordinate % per_word) * bits)) & ((1 << bits) - 1)
        if rotated:
            operand[channel, key] = lookup_centroid(code, table)
        else:
            operand[channel, key] = T.cast(code, "float32")


@T.macro
def compact_decode_scores(query, shared, table, scores, token, first_head,
                           rows, width, keys, contraction, bits, rotated):
    q = T.alloc_fragment((rows, contraction), "float32")
    k = T.alloc_fragment((contraction, keys), "float32")
    T.clear(scores)
    for block in T.serial(width // contraction):
        for row, channel in T.Parallel(rows, contraction):
            q[row, channel] = T.cast(query[token, first_head + row, block * contraction + channel], "float32")
        _key_operand(shared, k, table, block, contraction, keys, bits, rotated)
        T.gemm(q, k, scores, policy=T.GemmWarpPolicy.FullRow)


@T.macro
def compact_prefill_scores(query, shared, table, scores, rows, width, keys,
                            contraction, bits, rotated, dtype):
    q = T.alloc_fragment((rows, contraction), dtype)
    k = T.alloc_fragment((contraction, keys), dtype)
    T.clear(scores)
    for block in T.unroll(width // contraction):
        for row, channel in T.Parallel(rows, contraction):
            q[row, channel] = query[row, block * contraction + channel]
        _key_operand(shared, k, table, block, contraction, keys, bits, rotated)
        T.gemm(q, k, scores, policy=T.GemmWarpPolicy.FullRow)


@T.macro
def _value_column(probability, shared, output, first_key, first_column, columns, reduction, bits):
    operand = T.alloc_fragment((reduction, columns), "float32")
    per_word = 32 // bits
    for key, column in T.Parallel(reduction, columns):
        coordinate = first_column + column
        packed = shared[coordinate // per_word, first_key + key]
        operand[key, column] = T.cast((packed >> ((coordinate % per_word) * bits)) & ((1 << bits) - 1), "float32")
    with T.attr(0, "pragma_auto_unroll_max_step", 4096):
        with T.attr(0, "pragma_unroll_explicit", 1):
            T.gemm(probability, operand, output, policy=T.GemmWarpPolicy.FullRow)


def _value_columns(probability, shared, outputs, first_key, columns, reduction, bits):
    for index, output in enumerate(outputs):
        _value_column(probability, shared, output, first_key, index * columns, columns, reduction, bits)


@T.macro
def compact_values(scores, shared, outputs, rows, keys, columns, reduction, bits, probability_start=0):
    probability = T.alloc_fragment((rows, reduction), "float32")
    for block in T.serial(keys // reduction):
        for row, key in T.Parallel(rows, reduction):
            probability[row, key] = scores[row, probability_start + block * reduction + key]
        _value_columns(probability, shared, outputs, block * reduction, columns, reduction, bits)
