"""Recurrent preparation and state update behind portable operation contracts."""

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.operations.candidates import Plan, Selection, realize
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
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
        precision: Precision,
    ):
        self.precision = precision
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
            from magnitude_engine.kernels.recurrence.prepare import prepare_sequence

            self._plans[key] = self.context.specialize(
                prepare_sequence,
                batch,
                steps,
                self.key_heads,
                self.value_heads,
                self.width,
                self.convolution_width,
                self.epsilon,
                capability=self.context.capability,
                precision=self.precision,
                dtype=projected.spec.dtype,
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
        precision: Precision,
    ):
        self.context, self.precision = context, precision
        self.key_heads, self.value_heads, self.width, self.mapping = (
            key_heads,
            value_heads,
            width,
            mapping,
        )
        self._plans: dict[tuple[int, int, DType], Plan] = {}

    def plan(self, batch: int, steps: int, dtype: DType) -> Plan:
        key = batch, steps, dtype
        if key not in self._plans:
            from magnitude_engine.kernels.recurrence.select import TABLE, RecurrenceShape

            shape = RecurrenceShape(
                batch,
                steps,
                self.key_heads,
                self.value_heads,
                self.width,
                self.width,
                self.mapping,
                dtype,
            )
            self._plans[key] = realize(
                "recurrence",
                TABLE,
                self.context,
                Selection(shape, self.precision, self.context.capability),
            )
        return self._plans[key]

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
        plan = self.plan(batch, steps, queries.spec.dtype)
        return (
            Prepared(
                self.context,
                plan.executables[0],
                [queries, keys, values, decay, beta, previous, next_state, output],
            ),
        )

    def close(self) -> None:
        self._plans.clear()
