"""Encoded representation and kernel selection terminate at this boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.numerics.encoded_layout import EncodedLayout
from magnitude_engine.numerics.policy import floating
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.weights import ResidentWeight
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    DType,
    Executable,
    Prepared,
    Tensor,
)


@dataclass(frozen=True)
class LinearParameters:
    input_width: int
    output_width: int


class Linear(ABC):
    @property
    @abstractmethod
    def parameters(self) -> LinearParameters: ...

    @abstractmethod
    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]: ...

    @abstractmethod
    def close(self) -> None: ...


@dataclass(frozen=True)
class EncodedParameters:
    encoding: Encoding
    resident_bytes: int
    artifact_identity: ArtifactIdentity
    tensor_name: str
    layout: EncodedLayout


class ProjectionSchedule(StrEnum):
    BASELINE = "baseline"
    PORTABLE = "portable"
    METAL_SUBGROUP = "metal_subgroup"
    METAL_PACKED_K = "metal_packed_k"


@dataclass(frozen=True)
class LinearPlan:
    contraction: Executable
    reduction: Executable | None = None


class EncodedLinear(Linear):
    def __init__(
        self,
        weight: ResidentWeight,
        output_tile: int = 4,
        reduction_lanes: int = 32,
        schedule: ProjectionSchedule = ProjectionSchedule.PORTABLE,
        pack: int = 8,
        vector_rows: int = 1,
    ):
        descriptor = weight.descriptor
        context = weight.context
        if len(descriptor.shape) != 2:
            raise ValueError("linear weights must be a matrix")
        if output_tile <= 0 or reduction_lanes <= 0 or reduction_lanes & (reduction_lanes - 1):
            raise ValueError("invalid encoded projection tile")
        if not isinstance(schedule, ProjectionSchedule) or pack <= 0 or vector_rows <= 0:
            raise ValueError("invalid encoded projection schedule")
        if (
            schedule not in (ProjectionSchedule.PORTABLE, ProjectionSchedule.BASELINE)
            and context.backend != Backend.METAL
        ):
            raise ValueError("selected projection schedule requires a Metal endpoint")
        self.context = context
        self.weight = weight
        self._parameters = LinearParameters(descriptor.shape[1], descriptor.shape[0])
        self.encoded = EncodedParameters(
            descriptor.encoding,
            weight.nbytes,
            weight.artifact.identity,
            descriptor.name,
            weight.layout,
        )
        self.output_tile, self.reduction_lanes = output_tile, reduction_lanes
        self.schedule, self.pack = schedule, pack
        self.vector_rows = vector_rows
        self._plans: dict[tuple[int, DType, DType], LinearPlan] = {}

    @property
    def parameters(self) -> LinearParameters:
        return self._parameters

    def plan(
        self, rows: int, dtype: DType = DType.F32, output_dtype: DType = DType.F32
    ) -> LinearPlan:
        floating(dtype)
        floating(output_dtype)
        key = rows, dtype, output_dtype
        if key not in self._plans:
            from magnitude_engine.numerics.encoded import projection, projection_cpu

            parameters = self.parameters
            reduction = None
            row_tile = min(rows, self.vector_rows)
            selected = self.schedule
            if selected == ProjectionSchedule.BASELINE:
                selected = ProjectionSchedule.PORTABLE
                if self.context.backend == Backend.METAL:
                    selected = ProjectionSchedule.METAL_SUBGROUP
                    if (
                        self.encoded.encoding in (Encoding.Q4_K, Encoding.Q5_K)
                        and self.context.subgroup_width == 32
                    ):
                        selected = ProjectionSchedule.METAL_PACKED_K
            if self.encoded.layout == EncodedLayout.AFFINE_PLANES:
                selected = ProjectionSchedule.PORTABLE
            if (
                self.encoded.layout == EncodedLayout.AFFINE_PLANES
                and self.context.backend == Backend.METAL
                and parameters.input_width % 512 == 0
                and (rows < 8 or self.schedule != ProjectionSchedule.BASELINE)
            ):
                from magnitude_engine.numerics.planar_affine import vector

                executable = self.context.specialize(
                    vector,
                    rows,
                    parameters.output_width,
                    parameters.input_width,
                    self.encoded.encoding,
                    dtype=dtype,
                    output_dtype=output_dtype,
                )
            elif self.context.backend == Backend.LLVM:
                executable = self.context.specialize(
                    projection_cpu,
                    rows,
                    parameters.output_width,
                    parameters.input_width,
                    self.encoded.encoding,
                    row_tile=row_tile,
                    dtype=dtype,
                    output_dtype=output_dtype,
                    layout=self.encoded.layout,
                )
            elif self.schedule == ProjectionSchedule.BASELINE and rows >= 8:
                from magnitude_engine.numerics.matrix import merge_partitions
                from magnitude_engine.numerics.matrix import projection as matrix_projection

                # Narrow projections cannot fill the device from M/N tiles alone.
                # Split their contraction axis, then reconcile the partial sums.
                tiles = ((rows + 63) // 64) * ((parameters.output_width + 63) // 64)
                partitions = min(
                    (parameters.input_width + 31) // 32, max(1, (256 + tiles - 1) // tiles)
                )

                executable = self.context.specialize(
                    matrix_projection,
                    rows,
                    parameters.output_width,
                    parameters.input_width,
                    self.encoded.encoding,
                    partitions=partitions,
                    dtype=dtype,
                    output_dtype=output_dtype if partitions == 1 else DType.F32,
                    layout=self.encoded.layout,
                )
                if partitions > 1:
                    reduction = self.context.specialize(
                        merge_partitions,
                        rows,
                        parameters.output_width,
                        partitions,
                        output_dtype=output_dtype,
                    )
            elif selected in (
                ProjectionSchedule.METAL_SUBGROUP,
                ProjectionSchedule.METAL_PACKED_K,
            ):
                from magnitude_engine.numerics.metal_encoded import (
                    k_projection,
                )
                from magnitude_engine.numerics.metal_encoded import (
                    projection as metal_projection,
                )

                width = self.context.subgroup_width
                if self.context.backend != Backend.METAL or width is None:
                    raise ValueError("Metal subgroup plan requires a queried Metal execution width")
                if selected == ProjectionSchedule.METAL_PACKED_K:
                    executable = self.context.specialize(
                        k_projection,
                        rows,
                        parameters.output_width,
                        parameters.input_width,
                        self.encoded.encoding,
                        subgroup_width=width,
                        output_tile=self.output_tile,
                        row_tile=row_tile,
                        dtype=dtype,
                        output_dtype=output_dtype,
                    )
                elif width == 32 and self.encoded.encoding in (
                    Encoding.Q6_K,
                    Encoding.IQ4_XS,
                    Encoding.Q8_0,
                    Encoding.F16,
                ):
                    from magnitude_engine.numerics.metal_blocks import (
                        projection as block_projection,
                    )

                    executable = self.context.specialize(
                        block_projection,
                        rows,
                        parameters.output_width,
                        parameters.input_width,
                        self.encoded.encoding,
                        row_tile=row_tile,
                        dtype=dtype,
                        output_dtype=output_dtype,
                        output_tile=self.output_tile,
                    )
                else:
                    executable = self.context.specialize(
                        metal_projection,
                        rows,
                        parameters.output_width,
                        parameters.input_width,
                        self.encoded.encoding,
                        subgroup_width=width,
                        output_tile=self.output_tile,
                        pack=self.pack,
                        row_tile=row_tile,
                        dtype=dtype,
                        output_dtype=output_dtype,
                    )
            else:
                executable = self.context.specialize(
                    projection,
                    rows,
                    parameters.output_width,
                    parameters.input_width,
                    self.encoded.encoding,
                    output_tile=self.output_tile,
                    reduction_lanes=self.reduction_lanes,
                    row_tile=row_tile,
                    dtype=dtype,
                    output_dtype=output_dtype,
                    layout=self.encoded.layout,
                )
            self._plans[key] = LinearPlan(executable, reduction)
        return self._plans[key]

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        if len(inputs.spec.shape) != 2 or inputs.spec.shape[1] != self.parameters.input_width:
            raise ValueError("input tensor differs from linear geometry")
        plan = self.plan(inputs.spec.shape[0], inputs.spec.dtype, outputs.spec.dtype)
        weight = self.weight.acquire(plan.contraction.signature[1])
        try:
            with Preparation(self.context) as p:
                target = (
                    outputs
                    if plan.reduction is None
                    else p.allocate(plan.contraction.signature[-1])
                )
                p.add(Prepared(self.context, plan.contraction, (inputs, weight, target)))
                if plan.reduction is not None:
                    p.add(Prepared(self.context, plan.reduction, (target, outputs)))
                return p.finish()
        finally:
            weight.close()

    def close(self) -> None:
        self._plans.clear()
