"""Position-addressed sampling with an exact partition-summary reduction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec


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
def _reduce_summary(scores, indices, invalid, lane, threads):
    T.sync_threads()
    for step in T.unroll(int(math.log2(threads))):
        distance = threads >> (step + 1)
        # Every thread participates in every reduction step. Inactive lanes
        # select themselves, avoiding a conditional region around shared-memory
        # dependencies and keeping the inter-step barrier meaningful on every
        # backend.
        partner = T.if_then_else(lane < distance, lane + distance, lane)
        candidate_score = scores[partner]
        candidate_index = indices[partner]
        choose = lane < distance and (
            candidate_score > scores[lane]
            or (candidate_score == scores[lane] and candidate_index < indices[lane])
        )
        scores[lane] = T.if_then_else(choose, candidate_score, scores[lane])
        indices[lane] = T.if_then_else(choose, candidate_index, indices[lane])
        invalid[lane] |= invalid[partner]
        T.sync_threads()


@T.macro
def _publish_sample(output, row, index, invalid):
    status = T.if_then_else(invalid != 0, 2, T.if_then_else(index == 0x7FFFFFFF, 1, 0))
    output[row, 0] = T.if_then_else(status == 0, index, -1)
    output[row, 1] = status


@T.macro
def _score_partitions(logits, draws, output, partials, rows, vocabulary, threads, partitions):
    with T.Kernel(rows, partitions, threads=threads) as (row, partition):
        lane = T.get_thread_binding(0)
        scores = T.alloc_shared((threads,), "float32")
        indices = T.alloc_shared((threads,), "int32")
        invalid = T.alloc_shared((threads,), "int32")
        score = T.alloc_local((1,), "float32")
        index = T.alloc_local((1,), "int32")
        flag = T.alloc_local((1,), "int32")
        score[0] = T.reinterpret(T.cast(0xFF800000, "uint32"), "float32")
        index[0], flag[0] = 0x7FFFFFFF, 0
        kind = draws[row, 0]
        seed_low, seed_high = draws[row, 1], draws[row, 2]
        position_low, position_high, domain = draws[row, 3], draws[row, 4], draws[row, 5]
        chunks = T.ceildiv(vocabulary, threads)
        for chunk in T.serial(chunks * partition // partitions, chunks * (partition + 1) // partitions):
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
                    if kind == 1:
                        word = _philox(T.cast(token, "uint32"), position_low,
                                       position_high, domain, seed_low, seed_high)
                        uniform = (T.cast(word >> 9, "float32") + 0.5) * (2**-23)
                        value[0] -= T.log(-T.log(uniform))
                    if value[0] > score[0] or (value[0] == score[0] and token < index[0]):
                        score[0], index[0] = value[0], token
        scores[lane], indices[lane], invalid[lane] = score[0], index[0], flag[0]
        _reduce_summary(scores, indices, invalid, lane, threads)
        if lane == 0:
            if partitions == 1:
                _publish_sample(output, row, indices[0], invalid[0])
            else:
                # Preserve the selected score's exact bits through the merge.
                partials[row, partition, 0] = T.reinterpret(scores[0], "uint32")
                partials[row, partition, 1] = T.cast(indices[0], "uint32")
                partials[row, partition, 2] = T.cast(invalid[0], "uint32")


@T.macro
def _merge_partitions(partials, output, rows, partitions, threads):
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
        for chunk in T.serial(T.ceildiv(partitions, threads)):
            partition = chunk * threads + lane
            if partition < partitions:
                candidate = T.reinterpret(partials[row, partition, 0], "float32")
                token = T.cast(partials[row, partition, 1], "int32")
                flag[0] |= T.cast(partials[row, partition, 2], "int32")
                if candidate > score[0] or (candidate == score[0] and token < index[0]):
                    score[0], index[0] = candidate, token
        scores[lane], indices[lane], invalid[lane] = score[0], index[0], flag[0]
        _reduce_summary(scores, indices, invalid, lane, threads)
        if lane == 0:
            _publish_sample(output, row, indices[0], invalid[0])


@dataclass(frozen=True)
class _SamplingEmitter:
    rows: int
    vocabulary: int
    threads: int
    partitions: int

    def __call__(self, operands):
        logits, draws, output = operands[:3]
        partials = operands[3] if self.partitions > 1 else output
        _score_partitions(logits, draws, output, partials, self.rows, self.vocabulary,
                          self.threads, self.partitions)
        if self.partitions > 1:
            _merge_partitions(partials, output, self.rows, self.partitions, self.threads)


class SamplingRule:
    name = "partitioned-sampling"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.node(root)
        if node.operation != "sample":
            return ()
        logits = graph.value(node.inputs[0]).spec
        if not logits.static or context.compiler_target.shared_memory_bytes <= 0:
            return ()
        rows, vocabulary = logits.shape
        available = min(128, context.compiler_target.threads_per_group,
                        context.compiler_target.shared_memory_bytes // 12)
        if available < 1 or vocabulary > 0x7FFFFFFF:
            return ()
        threads = 1 << (available.bit_length() - 1)
        desired = max(1, math.ceil(vocabulary / (threads * 8)))
        affordable = context.workspace_limit // (rows * 12) if rows else 1
        partitions = (max(1, min(desired, affordable))
)
        workspace = ((TensorSpec((rows, partitions, 3), DType.U32),)
                     if partitions > 1 else ())
        return (BoundOperation(
            f"sample.{'partitioned' if partitions > 1 else 'single'}@{root}",
            frozenset({root}), node.inputs, node.outputs,
            _SamplingEmitter(rows, vocabulary, threads, partitions),
            workspace=workspace, kernel_count=2 if partitions > 1 else 1,
        ),)
