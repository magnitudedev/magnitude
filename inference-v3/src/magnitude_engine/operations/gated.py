"""A gated pair of linear maps; implementations own contraction/activation fusion."""

from magnitude_engine.numerics.semantics import Pointwise
from magnitude_engine.operations.activation import Elementwise
from magnitude_engine.operations.linear import Linear, LinearParameters
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.projections import Projections
from magnitude_engine.platform.execution import DeviceContext, DType, Prepared, Tensor, TensorSpec


class ProjectionWorkspace:
    """Scratch shared by ordered projection operations of one model owner."""

    def __init__(self, context: DeviceContext):
        self.context = context
        self.value: Tensor | None = None

    def reserve(self, spec: TensorSpec) -> None:
        if self.value is None or self.value.spec.nbytes < spec.nbytes:
            value = self.context.allocate(spec)
            if self.value is not None:
                self.value.close()
            self.value = value

    def acquire(self, spec: TensorSpec) -> Tensor:
        self.reserve(spec)
        assert self.value is not None
        return self.value.view(spec)

    def close(self) -> None:
        if self.value is not None:
            self.value.close()
            self.value = None


class GatedLinear(Linear):
    def __init__(
        self,
        projections: Projections,
        input_width: int,
        workspace: ProjectionWorkspace,
        *,
        native_rounding: bool,
    ):
        if len(projections.widths) != 2 or projections.widths[0] != projections.widths[1]:
            raise ValueError("gated projection requires two equal output widths")
        self.projections, self.workspace = projections, workspace
        self.context = workspace.context
        self._parameters = LinearParameters(input_width, projections.widths[0])
        self.activation = Elementwise(
            self.context, Pointwise.SILU_PRODUCT, native_rounding=native_rounding
        )

    @property
    def parameters(self) -> LinearParameters:
        return self._parameters

    def reserve_workspace(self, rows: int, dtype: DType) -> None:
        self.workspace.reserve(TensorSpec((rows * sum(self.projections.widths),), dtype))

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        rows = inputs.spec.shape[0]
        if outputs.spec.shape != (rows, self.parameters.output_width):
            raise ValueError("gated projection output geometry differs")
        with Preparation(self.context) as p:
            packed = p.own(
                self.workspace.acquire(
                    TensorSpec((rows * sum(self.projections.widths),), outputs.spec.dtype)
                )
            )
            gate, up = self.projections.outputs(p, packed, rows)
            p.add(*self.projections.prepare(inputs, packed))
            p.add(*self.activation.prepare(gate, up, outputs))
            return p.finish()

    def close(self) -> None:
        self.activation.close()
