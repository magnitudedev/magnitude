"""Query/key preparation keeps its physical schedule out of model equations."""

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import DeviceContext, DType, Executable, Prepared, Tensor


class AttentionPreparation:
    def __init__(
        self,
        context: DeviceContext,
        query_norm: Parameter,
        key_norm: Parameter,
        query_heads: int,
        kv_heads: int,
        width: int,
        rotary_width: int,
        base: float,
        sections: tuple[int, int, int, int],
        epsilon: float,
        precision: Precision,
    ):
        self.precision = precision
        self.context, self.query_norm, self.key_norm = context, query_norm, key_norm
        self.query_heads, self.kv_heads, self.width = query_heads, kv_heads, width
        self.rotary_width, self.base, self.sections, self.epsilon = (
            rotary_width,
            base,
            sections,
            epsilon,
        )
        self._plans: dict[tuple[int, DType], Executable] = {}

    def prepare(
        self,
        query_gate: Tensor,
        keys: Tensor,
        coordinates: Tensor,
        query_output: Tensor,
        key_output: Tensor,
        gate: Tensor,
    ) -> tuple[Prepared, ...]:
        floating(query_gate.spec.dtype)
        rows = query_gate.spec.shape[0]
        key = rows, query_gate.spec.dtype
        if key not in self._plans:
            from magnitude_engine.kernels.rotary.portable import prepare_attention

            self._plans[key] = self.context.specialize(
                prepare_attention,
                rows,
                self.query_heads,
                self.kv_heads,
                self.width,
                self.rotary_width,
                self.base,
                self.sections,
                self.epsilon,
                capability=self.context.capability,
                precision=self.precision,
                dtype=query_gate.spec.dtype,
            )
        with Preparation(self.context) as p:
            p.add(
                Prepared(
                    self.context,
                    self._plans[key],
                    [
                        query_gate,
                        keys,
                        p.parameter(self.query_norm),
                        p.parameter(self.key_norm),
                        coordinates,
                        query_output,
                        key_output,
                        gate,
                    ],
                )
            )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
