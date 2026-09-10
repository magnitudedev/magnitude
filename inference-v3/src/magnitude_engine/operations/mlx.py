"""Resident MLX affine weights behind logical operations, executed by TileLang."""

import math
from contextlib import ExitStack
from dataclasses import dataclass

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.mlx import MLXArtifact
from magnitude_engine.artifacts.weights import WeightDescriptor, WeightTransform
from magnitude_engine.operations.embedding import Embedding, EmbeddingParameters
from magnitude_engine.operations.factory import WeightOperations
from magnitude_engine.operations.gated import GatedLinear, ProjectionWorkspace
from magnitude_engine.operations.linear import Linear, LinearParameters
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.projections import Projections
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DeviceContext, DType, Prepared, Tensor, TensorSpec
from magnitude_engine.platform.storage import ConcatenatedSource


@dataclass(frozen=True)
class AffineWeight:
    codes: Tensor
    scales: Tensor
    biases: Tensor

    @property
    def operands(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.codes, self.scales, self.biases

    def close(self) -> None:
        for tensor in self.operands:
            tensor.close()


class AffineProjections(Projections):
    def __init__(self, context: DeviceContext, weight: AffineWeight, widths: tuple[int, ...]):
        self.context, self.weight, self._widths = context, weight, widths

    @property
    def widths(self) -> tuple[int, ...]:
        return self._widths

    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        from magnitude_engine.numerics import mlx_affine as k

        rows, width = inputs.spec.shape
        if inputs.spec.dtype != DType.BF16 or width != self.weight.codes.spec.shape[1] * 8:
            raise ValueError("affine projection requires BF16 input of its logical width")
        if output.spec.shape != (rows * sum(self.widths),) or output.spec.dtype not in (
            DType.BF16,
            DType.F32,
        ):
            raise ValueError("affine projection output differs from its logical segments")
        parts = 1
        reduction = None
        if self.context.backend == Backend.LLVM:
            plan = self.context.specialize(
                k.portable, rows, self.widths, width, True, output.spec.dtype
            )
        elif self.context.backend == Backend.METAL and rows < 8:
            plan = self.context.specialize(k.vector, rows, self.widths, width, output.spec.dtype)
        else:
            n = sum(self.widths)
            parts = k.partitions(rows, n, width)
            large = rows >= 256 and min(n, width) >= 512
            bm = 64 if large or rows >= 64 and n >= 8192 else 8 if rows == 1 else 32
            plan = self.context.specialize(
                k.matrix,
                rows,
                self.widths,
                width,
                parts,
                bm,
                64 if n >= 512 else 32,
                32,
                8 if large else 0,
                DType.BF16 if parts > 1 else output.spec.dtype,
            )
            if parts > 1:
                reduction = self.context.specialize(
                    k.finish, rows, self.widths, parts, output.spec.dtype
                )
        with Preparation(self.context) as p:
            target = (
                p.allocate(plan.signature[-1])
                if reduction is not None
                else p.view(output, plan.signature[-1])
            )
            p.add(Prepared(self.context, plan, (inputs, *self.weight.operands, target)))
            if reduction is not None:
                p.add(Prepared(self.context, reduction, (target, output)))
            return p.finish()


class AffineLinear(Linear):
    def __init__(self, projection: AffineProjections):
        self.projection = projection

    @property
    def parameters(self) -> LinearParameters:
        return LinearParameters(
            self.projection.weight.codes.spec.shape[1] * 8, sum(self.projection.widths)
        )

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        if outputs.spec.shape != (inputs.spec.shape[0], self.parameters.output_width):
            raise ValueError("affine linear output geometry differs")
        with Preparation(self.projection.context) as p:
            flat = p.view(outputs, TensorSpec((math.prod(outputs.spec.shape),), outputs.spec.dtype))
            p.add(*self.projection.prepare(inputs, flat))
            return p.finish()

    def close(self) -> None:
        pass


class AffineGatedLinear(GatedLinear):
    def __init__(
        self,
        projections: AffineProjections,
        workspace: ProjectionWorkspace,
        *,
        native_rounding: bool,
    ):
        super().__init__(
            projections,
            projections.weight.codes.spec.shape[1] * 8,
            workspace,
            native_rounding=native_rounding,
        )
        self.affine = projections
        self.native_rounding = native_rounding

    def reserve_workspace(self, rows: int, dtype: DType) -> None:
        if self.native_rounding and self.context.backend == Backend.METAL and rows < 8:
            return
        super().reserve_workspace(rows, dtype)

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        if (
            self.native_rounding
            and self.context.backend == Backend.METAL
            and inputs.spec.shape[0] < 8
        ):
            from magnitude_engine.numerics.mlx_affine import gated_vector

            plan = self.context.specialize(
                gated_vector,
                inputs.spec.shape[0],
                self.parameters.output_width,
                self.parameters.input_width,
            )
            return (Prepared(self.context, plan, (inputs, *self.affine.weight.operands, outputs)),)
        return super().prepare(inputs, outputs)


class AffineEmbedding(Embedding):
    def __init__(self, context: DeviceContext, weight: AffineWeight):
        self.context, self.weight = context, weight

    @property
    def parameters(self) -> EmbeddingParameters:
        n, packed = self.weight.codes.spec.shape
        return EmbeddingParameters(n, packed * 8)

    def prepare(self, tokens: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        from magnitude_engine.numerics.mlx_affine import embedding

        p = self.parameters
        plan = self.context.specialize(
            embedding,
            p.vocabulary,
            p.width,
            tokens.spec.shape[0],
            self.context.backend == Backend.LLVM,
        )
        return (Prepared(self.context, plan, (*self.weight.operands, tokens, output)),)

    def close(self) -> None:
        pass


class FloatingParameter(Parameter):
    def __init__(self, value: Tensor):
        self.value = value

    @property
    def spec(self) -> TensorSpec:
        return self.value.spec

    def acquire(self) -> Tensor:
        return self.value.view(self.spec)

    def close(self) -> None:
        self.value.close()


class MLXOperations(WeightOperations):
    def __init__(self, artifact: MLXArtifact, context: DeviceContext):
        self.artifact, self._context = artifact, context
        self._cleanup = ExitStack()
        self._workspace = ProjectionWorkspace(context)
        self._cleanup.callback(self._workspace.close)
        self._gated: dict[tuple[WeightDescriptor, WeightDescriptor, bool], GatedLinear] = {}
        self._weights: dict[WeightDescriptor, AffineWeight] = {}
        self._groups: dict[tuple[WeightDescriptor, ...], AffineProjections] = {}
        self._linears: dict[WeightDescriptor, AffineLinear] = {}
        self._embeddings: dict[WeightDescriptor, AffineEmbedding] = {}
        self._parameters: dict[WeightDescriptor, FloatingParameter] = {}
        self._closed = False

    @property
    def context(self) -> DeviceContext:
        return self._context

    @property
    def artifact_identity(self) -> ArtifactIdentity:
        return self.artifact.identity

    def _check(self, weight: WeightDescriptor) -> None:
        self.context.check()
        if self._closed:
            raise RuntimeError("operation binding is closed")
        self.artifact.descriptor(weight.name, weight.shape)

    def _pack(self, weights: tuple[WeightDescriptor, ...]) -> AffineWeight:
        if not weights or any(
            len(w.shape) != 2 or w.transform != WeightTransform.IDENTITY for w in weights
        ):
            raise ValueError("affine packing requires logical matrices")
        if len({w.shape[1] for w in weights}) != 1 or len(set(weights)) != len(weights):
            raise ValueError("affine packing requires distinct matrices of one input width")
        for weight in weights:
            self._check(weight)
        if len(weights) == 1 and weights[0] in self._weights:
            return self._weights[weights[0]]
        if any(weight in self._weights for weight in weights):
            raise ValueError("projection groups must be declared before individual residency")
        n, k = sum(w.shape[0] for w in weights), weights[0].shape[1]
        with ExitStack() as cleanup:
            planes = []
            for suffix, spec in (
                ("weight", TensorSpec((n, k // 8), DType.U32)),
                ("scales", TensorSpec((n, k // 64), DType.BF16)),
                ("biases", TensorSpec((n, k // 64), DType.BF16)),
            ):
                stored = tuple(
                    self.artifact.tensors[w.name.removesuffix("weight") + suffix] for w in weights
                )
                source = ConcatenatedSource(
                    tuple((s.source, s.offset, s.spec.nbytes) for s in stored)
                )
                tensor = self.context.upload_source(spec, source, 0)
                cleanup.callback(tensor.close)
                planes.append(tensor)
            packed = AffineWeight(*planes)
            views = {}
            start = 0
            for weight in weights:
                values = []
                for tensor in packed.operands:
                    spec = TensorSpec((weight.shape[0], tensor.spec.shape[1]), tensor.spec.dtype)
                    value = tensor.view(
                        spec, start * tensor.spec.shape[1] * tensor.spec.dtype.itemsize
                    )
                    cleanup.callback(value.close)
                    values.append(value)
                views[weight] = AffineWeight(*values)
                start += weight.shape[0]
            ownership = cleanup.pop_all()
        self._cleanup.callback(ownership.close)
        self._weights.update(views)
        return packed

    def projections(self, weights: tuple[WeightDescriptor, ...]) -> AffineProjections:
        if weights not in self._groups:
            self._groups[weights] = AffineProjections(
                self.context, self._pack(weights), tuple(w.shape[0] for w in weights)
            )
        return self._groups[weights]

    def linear(self, weight: WeightDescriptor) -> Linear:
        if weight not in self._linears:
            self._linears[weight] = AffineLinear(self.projections((weight,)))
        return self._linears[weight]

    def gated_linear(
        self, gate: WeightDescriptor, up: WeightDescriptor, *, native_rounding: bool
    ) -> Linear:
        key = gate, up, native_rounding
        if key not in self._gated:
            operation = AffineGatedLinear(
                self.projections((gate, up)), self._workspace, native_rounding=native_rounding
            )
            self._cleanup.callback(operation.close)
            self._gated[key] = operation
        return self._gated[key]

    def embedding(self, weight: WeightDescriptor) -> Embedding:
        if weight not in self._embeddings:
            self._embeddings[weight] = AffineEmbedding(self.context, self._pack((weight,)))
        return self._embeddings[weight]

    def parameter(self, weight: WeightDescriptor) -> Parameter:
        from magnitude_engine.numerics.mlx_affine import parameter

        self._check(weight)
        if weight not in self._parameters:
            stored = self.artifact.tensors[weight.name]
            if stored.spec.dtype not in (DType.BF16, DType.F32):
                raise ValueError("small MLX parameter must be floating")
            count = math.prod(weight.shape)
            with Preparation(self.context) as p:
                source = self.context.upload_source(
                    TensorSpec((count,), stored.spec.dtype), stored.source, stored.offset
                )
                try:
                    plan = self.context.specialize(
                        parameter,
                        count,
                        stored.spec.dtype,
                        weight.transform,
                        self.context.backend == Backend.LLVM,
                    )
                    target = p.allocate(TensorSpec((count,), DType.F32))
                    p.add(Prepared(self.context, plan, (source, target)))
                    ticket = self.context.submit(p.finish())
                    ticket.wait()
                    value = FloatingParameter(target.view(TensorSpec(weight.shape, DType.F32)))
                finally:
                    source.close()
            self._cleanup.callback(value.close)
            self._parameters[weight] = value
        return self._parameters[weight]

    def reserve_workspace(self, rows: int, dtype: DType) -> None:
        for operation in self._gated.values():
            operation.reserve_workspace(rows, dtype)

    def release_workspace(self) -> None:
        self._workspace.close()

    def close(self) -> None:
        if not self._closed:
            self._cleanup.close()
            self._closed = True
