"""Causal attention and KV writes over the state's physical run views."""

from contextlib import ExitStack
from enum import StrEnum

from magnitude_engine.numerics.policy import floating
from magnitude_engine.operations.kv_binding import ReadBinding, WriteBinding
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.binding import BoundSequence
from magnitude_engine.platform.execution import (
    CapacityError,
    DeviceContext,
    DType,
    Executable,
    ExecutionOrder,
    Prepared,
    Tensor,
    TensorSpec,
)


class WorkspaceSlot(StrEnum):
    SCORES = "scores"
    PARTIALS = "partials"
    STATISTICS = "statistics"


class CausalAttention:
    def __init__(self, context: DeviceContext, heads: int, kv_heads: int, width: int):
        self.context, self.heads, self.kv_heads, self.width = context, heads, kv_heads, width
        self._runs: dict[tuple[int, int, int, int, int, DType, DType, bool], Executable] = {}
        self._mergers: dict[tuple[int, int, DType], Executable] = {}
        self._matrices: dict[
            tuple[int, int, int, int, DType, DType, DType], tuple[Executable, ...]
        ] = {}
        self._workspace: dict[WorkspaceSlot, Tensor] = {}

    def _reserve(self, slot: WorkspaceSlot, spec: TensorSpec) -> Tensor:
        # An invocation consumes these intermediates before returning its logical
        # output. Ordered invocations can share backing even while commands are
        # pending; prepared operand leases protect old backing when it grows.
        if spec.dtype not in (DType.F32, DType.BF16):
            raise ValueError("unsupported attention workspace dtype")
        current = self._workspace.get(slot)
        if current is None or current.spec.nbytes < spec.nbytes:
            replacement = self.context.allocate(TensorSpec((spec.nbytes // 4,), DType.F32))
            self._workspace[slot] = replacement
            if current is not None:
                current.close()
        return self._workspace[slot]

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
        groups = history
        cpu = self.context.backend == Backend.LLVM
        streaming = (
            self.context.backend == Backend.METAL
            and self.context.subgroup_width == 32
            and rows >= 8
            and self.width <= 256
            and self.width % 8 == 0
        )
        decode_matrix = (
            self.context.backend == Backend.METAL
            and self.context.subgroup_width == 32
            and rows < 8
            and self.width <= 256
            and self.width % 8 == 0
        )
        total_capacity = sum(
            group.geometry.segments * group.geometry.segment_capacity for group in groups
        )
        native_matrix = (
            streaming
            and rows >= 256
            and dtype == DType.BF16
            and self.width == 256
        )
        score_dtype = DType.BF16 if native_matrix else DType.F32
        if native_matrix:
            elements = max(
                g.geometry.segments * self.heads * rows * g.geometry.segment_capacity
                for g in groups
            )
            try:
                # Keep merge/statistic storage feasible before choosing the
                # optional materialized-score strategy.
                count = sum(g.geometry.segments for g in groups)
                self._reserve(
                    WorkspaceSlot.STATISTICS, TensorSpec((count, rows, self.heads, 2), DType.F32)
                )
                if count > 1:
                    self._reserve(
                        WorkspaceSlot.PARTIALS,
                        TensorSpec((count, rows, self.heads, self.width), DType.F32),
                    )
                self._reserve(WorkspaceSlot.SCORES, TensorSpec((elements,), score_dtype))
            except CapacityError:
                native_matrix = False
                score_dtype = DType.F32
        streaming = streaming and not native_matrix
        decode_online = (
            decode_matrix
            and dtype == DType.BF16
            and self.width == 256
            and total_capacity > 8192
            and self.heads // self.kv_heads <= 15
        )
        online_span = max(32, ((total_capacity + 64 * 32 - 1) // (64 * 32)) * 32)
        matrix = native_matrix or (
            self.context.backend == Backend.METAL and rows >= 8 and not streaming
        )
        head_tile = self.heads // self.kv_heads if not cpu and rows < 8 else 1
        # Small query batches need parallelism along history as well as heads.
        # This baseline is an execution choice; physical run ownership is unchanged.
        partitions = tuple(
            max(1, (group.geometry.segment_capacity + online_span - 1) // online_span)
            if decode_online
            else max(1, (group.geometry.segment_capacity + 127) // 128)
            if decode_matrix
            else max(1, (group.geometry.segment_capacity + 4095) // 4096)
            if streaming
            else min(
                (group.geometry.segment_capacity + 31) // 32,
                max(1, (group.geometry.segment_capacity * head_tile + 511) // 512),
            )
            if not cpu and rows < 8
            else 1
            for group in groups
        )
        partial_count = sum(
            group.geometry.segments * count for group, count in zip(groups, partitions, strict=True)
        )
        partial_dtype = dtype if partial_count == 1 else DType.F32
        partial_spec = TensorSpec(expected.shape, partial_dtype)
        with Preparation(self.context) as p, ExitStack() as range_ownership:
            range_commands = []
            if matrix:
                elements = max(
                    group.geometry.segments * self.heads * rows * group.geometry.segment_capacity
                    for group in groups
                )
                self._reserve(WorkspaceSlot.SCORES, TensorSpec((elements,), score_dtype))
            stats_spec = TensorSpec((rows, self.heads, 2), DType.F32)
            if partial_count == 1:
                partials = output
                statistics_spec = stats_spec
            else:
                outputs_spec = TensorSpec((partial_count, *expected.shape), partial_dtype)
                partials = p.view(self._reserve(WorkspaceSlot.PARTIALS, outputs_spec), outputs_spec)
                statistics_spec = TensorSpec((partial_count, *stats_spec.shape), DType.F32)
            statistics = p.view(
                self._reserve(WorkspaceSlot.STATISTICS, statistics_spec), statistics_spec
            )
            offset = 0
            for group, partitions_per_segment in zip(groups, partitions, strict=True):
                segments = group.geometry.segments
                count = segments * partitions_per_segment
                key = (
                    rows,
                    group.geometry.capacity,
                    group.geometry.segment_capacity,
                    segments,
                    partitions_per_segment,
                    dtype,
                    partial_dtype,
                    decode_online,
                )
                if not matrix and key not in self._runs:
                    from magnitude_engine.numerics.attention import run_attention, run_attention_cpu
                    from magnitude_engine.numerics.metal_attention import (
                        run_attention as stream_attention,
                    )
                    from magnitude_engine.numerics.metal_decode import (
                        run_attention as decode_attention,
                    )

                    if decode_online:
                        from magnitude_engine.numerics.metal_online_decode import (
                            run_attention as decode_attention,
                        )

                    args = (rows, self.heads, self.kv_heads, self.width, group.geometry.capacity)
                    executable = (
                        self.context.specialize(
                            run_attention_cpu,
                            *args,
                            segments=segments,
                            segment_capacity=group.geometry.segment_capacity,
                            dtype=dtype,
                            output_dtype=partial_dtype,
                        )
                        if cpu
                        else self.context.specialize(
                            stream_attention,
                            *args,
                            segments=segments,
                            segment_capacity=group.geometry.segment_capacity,
                            partitions=partitions_per_segment,
                            dtype=dtype,
                            output_dtype=partial_dtype,
                        )
                        if streaming
                        else self.context.specialize(
                            decode_attention,
                            *args,
                            segments=segments,
                            segment_capacity=group.geometry.segment_capacity,
                            partitions=partitions_per_segment,
                            dtype=dtype,
                            output_dtype=partial_dtype,
                        )
                        if decode_matrix
                        else self.context.specialize(
                            run_attention,
                            *args,
                            segments=segments,
                            segment_capacity=group.geometry.segment_capacity,
                            dtype=dtype,
                            output_dtype=partial_dtype,
                            partitions=partitions_per_segment,
                            head_tile=head_tile,
                        )
                    )
                    self._runs[key] = executable
                keys, values = group.keys, group.values
                result = p.view(
                    partials,
                    TensorSpec((count, *expected.shape), partial_dtype),
                    offset * partial_spec.nbytes,
                )
                stats = p.view(
                    statistics,
                    TensorSpec((count, *stats_spec.shape), DType.F32),
                    offset * stats_spec.nbytes,
                )
                span = group.geometry.segment_capacity
                metadata = group.metadata
                if matrix:
                    from magnitude_engine.numerics import matrix_attention

                    matrix_key = (
                        rows,
                        group.geometry.capacity,
                        count,
                        span,
                        dtype,
                        partial_dtype,
                        score_dtype,
                    )
                    if matrix_key not in self._matrices:
                        args = (
                            rows,
                            self.heads,
                            self.kv_heads,
                            self.width,
                            group.geometry.capacity,
                            count,
                            span,
                        )
                        self._matrices[matrix_key] = (
                            self.context.specialize(
                                matrix_attention.scores, *args, dtype=dtype, score_dtype=score_dtype
                            ),
                            self.context.specialize(
                                matrix_attention.normalize_bf16
                                if native_matrix
                                else matrix_attention.normalize,
                                rows,
                                self.heads,
                                count,
                                span,
                            ),
                            self.context.specialize(
                                matrix_attention.values,
                                *args,
                                dtype=dtype,
                                output_dtype=partial_dtype,
                                score_dtype=score_dtype,
                            ),
                        )
                    scores, normalize, contract = self._matrices[matrix_key]
                    scratch = p.view(self._workspace[WorkspaceSlot.SCORES], scores.signature[-1])
                    # These three dependent stages execute consecutively. Scratch
                    # can be reused by the next group/layer on the same owner queue.
                    p.add(
                        Prepared(
                            self.context, scores, [queries, keys, positions, metadata, scratch]
                        )
                    )
                    p.add(Prepared(self.context, normalize, [scratch, positions, metadata, stats]))
                    p.add(Prepared(self.context, contract, [scratch, values, metadata, result]))
                else:
                    command = Prepared(
                        self.context,
                        self._runs[key],
                        [queries, keys, values, positions, metadata, result, stats],
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
                            *(operand for group in groups for operand in group.operands),
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
            if partial_count > 1:
                merge_key = (rows, partial_count, dtype)
                if merge_key not in self._mergers:
                    from magnitude_engine.numerics.attention import merge_runs

                    self._mergers[merge_key] = self.context.specialize(
                        merge_runs,
                        rows,
                        self.heads,
                        self.width,
                        partial_count,
                        cpu=cpu,
                        output_dtype=dtype,
                    )
                p.add(
                    Prepared(self.context, self._mergers[merge_key], [partials, statistics, output])
                )
            return p.finish()

    def release_workspace(self) -> None:
        """Release cached scratch; prepared/submitted consumers retain their leases."""
        for tensor in self._workspace.values():
            tensor.close()
        self._workspace.clear()

    def close(self) -> None:
        self._runs.clear()
        self._mergers.clear()
        self._matrices.clear()
        self.release_workspace()


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
                    from magnitude_engine.numerics.state import append_kv

                    self._plans[key] = self.context.specialize(
                        append_kv,
                        write.geometry.length,
                        self.heads,
                        self.width,
                        write.geometry.capacity,
                        cpu=self.context.backend == Backend.LLVM,
                        dtype=keys.spec.dtype,
                    )
                spec = TensorSpec((write.geometry.length, self.heads, self.width), keys.spec.dtype)
                byte_offset = (
                    write.geometry.source * self.heads * self.width * keys.spec.dtype.itemsize
                )
                key_store, value_store = write.keys, write.values
                offset = write.offset
                p.add(
                    Prepared(
                        self.context,
                        self._plans[key],
                        [
                            p.view(keys, spec, byte_offset),
                            p.view(values, spec, byte_offset),
                            offset,
                            key_store,
                            value_store,
                        ],
                    )
                )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
