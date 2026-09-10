"""Recurrent preparation and state update behind portable operation contracts."""

from magnitude_engine.numerics.policy import floating
from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DeviceContext, DType, Executable, Prepared, Tensor


class RecurrentPreparation:
    def __init__(
        self,
        context: DeviceContext,
        convolution: Parameter,
        decay: Parameter,
        time_bias: Parameter,
        key_heads: int,
        value_heads: int,
        width: int,
        convolution_width: int,
        epsilon: float,
        *,
        native_rounding: bool = False,
    ):
        self.native_rounding = native_rounding
        self.context = context
        self.convolution, self.decay, self.time_bias = convolution, decay, time_bias
        self.key_heads, self.value_heads, self.width = key_heads, value_heads, width
        self.convolution_width, self.epsilon = convolution_width, epsilon
        self._plans: dict[tuple[int, int, DType], Executable] = {}

    def prepare(
        self,
        projected: Tensor,
        alpha: Tensor,
        beta_input: Tensor,
        previous: Tensor,
        next_state: Tensor,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        beta: Tensor,
        decay: Tensor,
    ) -> tuple[Prepared, ...]:
        if len(projected.spec.shape) != 2 or len(previous.spec.shape) != 3:
            raise ValueError("recurrent preparation requires packed tokens and sequence histories")
        batch = previous.spec.shape[0]
        if projected.spec.shape[0] % batch:
            raise ValueError("recurrent inputs must contain equally sized packed sequences")
        steps = projected.spec.shape[0] // batch
        floating(projected.spec.dtype)
        key = batch, steps, projected.spec.dtype
        if key not in self._plans:
            from magnitude_engine.numerics.recurrent import prepare_sequence

            self._plans[key] = self.context.specialize(
                prepare_sequence,
                batch,
                steps,
                self.key_heads,
                self.value_heads,
                self.width,
                self.convolution_width,
                self.epsilon,
                cpu=self.context.backend == Backend.LLVM,
                dtype=projected.spec.dtype,
                native_rounding=self.native_rounding,
                simd_width=(self.context.subgroup_width or 0)
                if self.context.backend == Backend.METAL
                else 0,
            )
        with Preparation(self.context) as p:
            p.add(
                Prepared(
                    self.context,
                    self._plans[key],
                    [
                        projected,
                        p.parameter(self.convolution),
                        previous,
                        next_state,
                        alpha,
                        beta_input,
                        p.parameter(self.decay),
                        p.parameter(self.time_bias),
                        queries,
                        keys,
                        values,
                        beta,
                        decay,
                    ],
                )
            )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()


class DeltaRecurrence:
    def __init__(
        self,
        context: DeviceContext,
        key_heads: int,
        value_heads: int,
        width: int,
        mapping: HeadMapping,
    ):
        self.context = context
        self.key_heads, self.value_heads, self.width, self.mapping = (
            key_heads,
            value_heads,
            width,
            mapping,
        )
        self._plans: dict[tuple[int, int, DType], Executable] = {}

    def prepare(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        decay: Tensor,
        beta: Tensor,
        previous: Tensor,
        next_state: Tensor,
        output: Tensor,
    ) -> tuple[Prepared, ...]:
        if len(queries.spec.shape) != 3 or len(previous.spec.shape) != 4:
            raise ValueError("delta recurrence requires packed queries and sequence states")
        batch = previous.spec.shape[0]
        if queries.spec.shape[0] % batch:
            raise ValueError("delta inputs must contain equally sized packed sequences")
        steps = queries.spec.shape[0] // batch
        key = batch, steps, queries.spec.dtype
        if key not in self._plans:
            from magnitude_engine.numerics.recurrent import delta_sequence

            if self.context.backend == Backend.METAL:
                from magnitude_engine.numerics.metal_recurrent import delta_sequence as metal_delta

                width = self.context.subgroup_width
                if width is None:
                    raise ValueError("Metal recurrence requires a queried SIMD width")
                executable = self.context.specialize(
                    metal_delta,
                    batch,
                    steps,
                    self.key_heads,
                    self.value_heads,
                    self.width,
                    self.width,
                    self.mapping,
                    subgroup_width=width,
                    dtype=queries.spec.dtype,
                )
            else:
                executable = self.context.specialize(
                    delta_sequence,
                    batch,
                    steps,
                    self.key_heads,
                    self.value_heads,
                    self.width,
                    self.width,
                    self.mapping,
                    cpu=self.context.backend == Backend.LLVM,
                    dtype=queries.spec.dtype,
                )
            self._plans[key] = executable
        return (
            Prepared(
                self.context,
                self._plans[key],
                [queries, keys, values, decay, beta, previous, next_state, output],
            ),
        )

    def close(self) -> None:
        self._plans.clear()
