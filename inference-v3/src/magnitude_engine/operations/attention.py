"""Causal attention and KV writes over the state's physical run views."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.operations.candidates import ScratchArena, Selection, realize
from magnitude_engine.operations.kv_binding import ReadBinding, WriteBinding
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.binding import BoundSequence
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Executable,
    ExecutionOrder,
    Prepared,
    Tensor,
    TensorSpec,
)

if TYPE_CHECKING:
    from magnitude_engine.kernels.attention.select import AttentionPlan, HistoryShape

SCORES = "attention.scores"
PARTIALS = "attention.partials"
STATISTICS = "attention.statistics"


class CausalAttention:
    def __init__(
        self,
        context: DeviceContext,
        heads: int,
        kv_heads: int,
        width: int,
        precision: Precision,
        arena: ScratchArena,
    ):
        self.context, self.heads, self.kv_heads, self.width = context, heads, kv_heads, width
        self.precision, self.arena = precision, arena
        self._plans: dict[tuple, AttentionPlan] = {}

    def plan(self, rows: int, dtype: DType, groups: tuple[HistoryShape, ...]) -> AttentionPlan:
        key = rows, dtype, groups
        if key not in self._plans:
            from magnitude_engine.kernels.attention.select import TABLE, AttentionShape

            shape = AttentionShape(rows, self.heads, self.kv_heads, self.width, dtype, groups)
            self._plans[key] = realize(
                "attention",
                TABLE,
                self.context,
                Selection(shape, self.precision, self.context.capability),
                available_bytes=self.arena.available((SCORES, PARTIALS, STATISTICS)),
            )
        return self._plans[key]

    def reserve(self, rows: int, dtype: DType, groups: tuple[HistoryShape, ...]) -> None:
        self.arena.reserve(self.plan(rows, dtype, groups).scratch)

    def prepare(
        self,
        queries: Tensor,
        positions: Tensor,
        history: tuple[ReadBinding, ...],
        output: Tensor,
    ) -> tuple[Prepared, ...]:
        rows = queries.spec.shape[0]
        dtype = queries.spec.dtype
        floating(dtype)
        expected = TensorSpec((rows, self.heads, self.width), dtype)
        if (
            queries.spec != expected
            or output.spec != expected
            or positions.spec != TensorSpec((rows,), DType.I32)
        ):
            raise ValueError("attention operands differ from bound geometry")
        if not history:
            raise ValueError("attention requires visible history")
        from magnitude_engine.kernels.attention.select import HistoryShape

        groups = tuple(
            HistoryShape(
                read.geometry.capacity, read.geometry.segment_capacity, read.geometry.segments
            )
            for read in history
        )
        plan = self.plan(rows, dtype, groups)
        regions = {region.name: region for region in plan.scratch}
        stats_spec = TensorSpec((rows, self.heads, 2), DType.F32)
        with Preparation(self.context) as p, ExitStack() as range_ownership:
            statistics = self.arena.region(p, STATISTICS, regions[STATISTICS].spec)
            if plan.partial_count == 1:
                partials = output
            else:
                partials = self.arena.region(p, PARTIALS, regions[PARTIALS].spec)
            partial_spec = TensorSpec(expected.shape, plan.partial_dtype)
            scores = (
                self.arena.region(p, SCORES, regions[SCORES].spec) if SCORES in regions else None
            )
            range_commands: list[Prepared] = []
            offset = 0
            for index, (group, read) in enumerate(zip(groups, history, strict=True)):
                count = group.segments * plan.partitions[index]
                stage = plan.executables[index * plan.stages : (index + 1) * plan.stages]
                result = p.view(
                    partials,
                    TensorSpec((count, *expected.shape), plan.partial_dtype),
                    offset * partial_spec.nbytes,
                )
                summary = p.view(
                    statistics,
                    TensorSpec((count, *stats_spec.shape), DType.F32),
                    offset * stats_spec.nbytes,
                )
                if plan.stages == 3:
                    contract_scores, normalize, contract = stage
                    assert scores is not None
                    scratch = p.view(scores, contract_scores.signature[-1])
                    # Three dependent stages execute consecutively; their scratch
                    # is reusable by the next group or layer on the same queue.
                    p.add(
                        Prepared(
                            self.context,
                            contract_scores,
                            [queries, read.keys, positions, read.metadata, scratch],
                        )
                    )
                    p.add(
                        Prepared(
                            self.context,
                            normalize,
                            [scratch, positions, read.metadata, summary],
                        )
                    )
                    p.add(
                        Prepared(
                            self.context, contract, [scratch, read.values, read.metadata, result]
                        )
                    )
                else:
                    (run,) = stage
                    command = Prepared(
                        self.context,
                        run,
                        [
                            queries,
                            read.keys,
                            read.values,
                            positions,
                            read.metadata,
                            result,
                            summary,
                        ],
                    )
                    range_ownership.callback(command.close)
                    range_commands.append(command)
                offset += count
            if len(range_commands) > 1:
                # Each range reads independent KV and writes disjoint partial
                # output/statistics slices. Their merge joins the region below.
                operands = tuple(
                    dict.fromkeys(
                        (
                            queries,
                            positions,
                            partials,
                            statistics,
                            *(operand for read in history for operand in read.operands),
                        )
                    )
                )
                binding = BoundSequence(
                    self.context,
                    tuple(range_commands),
                    operands,
                    order=ExecutionOrder.INDEPENDENT,
                )
                try:
                    p.add(binding.prepare(operands))
                finally:
                    binding.close()
            else:
                p.add(*range_commands)
            range_ownership.pop_all()
            if plan.merge is not None:
                p.add(Prepared(self.context, plan.merge, [partials, statistics, output]))
            return p.finish()

    def close(self) -> None:
        self._plans.clear()


class KVAppend:
    def __init__(self, context: DeviceContext, heads: int, width: int):
        self.context, self.heads, self.width = context, heads, width
        self._plans: dict[tuple[int, int, DType], Executable] = {}

    def prepare(
        self, keys: Tensor, values: Tensor, writes: tuple[WriteBinding, ...]
    ) -> tuple[Prepared, ...]:
        if keys.spec != values.spec or keys.spec.shape[1:] != (self.heads, self.width):
            raise ValueError("KV append operands differ from bound geometry")
        floating(keys.spec.dtype)
        with Preparation(self.context) as p:
            for write in writes:
                if write.geometry.source + write.geometry.length > keys.spec.shape[0]:
                    raise ValueError("KV write exceeds provided input rows")
                key = (write.geometry.length, write.geometry.capacity, keys.spec.dtype)
                if key not in self._plans:
                    from magnitude_engine.kernels.kv.append import append_kv

                    self._plans[key] = self.context.specialize(
                        append_kv,
                        write.geometry.length,
                        self.heads,
                        self.width,
                        write.geometry.capacity,
                        capability=self.context.capability,
                        dtype=keys.spec.dtype,
                    )
                spec = TensorSpec((write.geometry.length, self.heads, self.width), keys.spec.dtype)
                byte_offset = (
                    write.geometry.source * self.heads * self.width * keys.spec.dtype.itemsize
                )
                p.add(
                    Prepared(
                        self.context,
                        self._plans[key],
                        [
                            p.view(keys, spec, byte_offset),
                            p.view(values, spec, byte_offset),
                            write.offset,
                            write.keys,
                            write.values,
                        ],
                    )
                )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
