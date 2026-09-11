"""The encoded-linear benchmark owns independent interpretation and reset rules."""

from contextlib import ExitStack

import gguf
import numpy as np
from pydantic import Field

from magnitude_engine.operations.linear import ResidentLinear
from magnitude_engine.platform.execution import DType, TensorSpec, Ticket
from magnitude_engine.weights.representation import Blocked, resident_bytes
from performance.metrics import (
    Latency,
    LinearMetrics,
    ReadTraffic,
    Record,
    Sample,
    TimingBoundary,
    TimingPass,
    TrafficInterface,
    Validation,
)
from performance.precision import decode, encode, rounded
from performance.selection import realized


class LinearWorkload(Record):
    rows: int = Field(gt=0)
    seed: int = Field(default=765, ge=0)
    dtype: DType = DType.F32
    output_dtype: DType = DType.F32


class LinearCase:
    @property
    def observation_passes(self) -> tuple[TimingPass, ...]:
        return (
            (TimingPass.COMPLETED,)
            if self.component.context.capability.threads_per_group == 1
            else (TimingPass.COMPLETED, TimingPass.DEVICE_EVENTS)
        )

    def __init__(self, component: ResidentLinear, workload: LinearWorkload):
        self.component, self.workload = component, workload
        parameters = component.parameters
        # Read the container independently of residency: the resident weight
        # kept what it was made from, so the oracle never guesses a layout.
        stored = component.weight.stored
        elements = parameters.output_width * parameters.input_width
        raw = stored.source.read(
            stored.offset, resident_bytes(Blocked(stored.encoding), elements)
        )
        encoded = np.frombuffer(raw, np.uint8).reshape(parameters.output_width, -1)
        weights = gguf.dequantize(encoded, gguf.GGMLQuantizationType(stored.encoding))
        inputs = (
            np.random.default_rng(workload.seed)
            .normal(size=(workload.rows, parameters.input_width))
            .astype(np.float32)
        )
        inputs = rounded(inputs, workload.dtype)
        self.expected = inputs.astype(np.float64) @ weights.astype(np.float64).T
        self.error_bound = (
            2e-6 * (np.abs(inputs.astype(np.float64)) @ np.abs(weights.astype(np.float64)).T) + 1e-6
        )
        self.method = "gguf-0.19 independent decode and FP64 contraction; FP32 accumulation bound"
        if workload.dtype == DType.BF16:
            # A BF16 matrix tile may round interpreted weights. The vector path
            # can be more precise; both share this declared operation domain.
            rounded_weights = rounded(weights, DType.BF16)
            self.error_bound += (
                np.abs(inputs.astype(np.float64))
                @ np.abs(rounded_weights.astype(np.float64) - weights.astype(np.float64)).T
            )
            self.method += "; plus exact sum(abs(input) * abs(weight-rounding error))"
        if workload.output_dtype == DType.BF16:
            unit_roundoff = 2.0**-8
            self.error_bound = (1 + unit_roundoff) * self.error_bound + unit_roundoff * np.abs(
                self.expected
            )
            self.method += "; plus BF16 output rounding (unit roundoff 2^-8)"
        component.plan(workload.rows, workload.dtype, workload.output_dtype)
        with ExitStack() as cleanup:
            self.inputs = component.context.upload(
                TensorSpec(inputs.shape, workload.dtype), encode(inputs, workload.dtype)
            )
            cleanup.callback(self.inputs.close)
            output = np.full((workload.rows, parameters.output_width), np.nan, np.float32)
            self.outputs = component.context.upload(
                TensorSpec(output.shape, workload.output_dtype),
                encode(output, workload.output_dtype),
            )
            cleanup.callback(self.outputs.close)
            self._cleanup = cleanup.pop_all()

    def reset(self) -> None:
        # Linear writes its entire output and never mutates input/weight state.
        pass

    def invoke(self, *, timing: bool) -> Ticket:
        return self.component.context.submit(
            self.component.prepare(self.inputs, self.outputs), timing=timing
        )

    def validate(self, ticket: Ticket) -> Validation:
        output = decode(
            self.component.context.read(self.outputs, after=ticket), self.workload.output_dtype
        ).reshape(self.expected.shape)
        error = np.abs(output - self.expected)
        finite = bool(np.all(np.isfinite(error)))
        return Validation(
            passed=finite and bool(np.all(error <= self.error_bound)),
            method=self.method + "; accumulation error <= 2e-6 * sum(abs(products)) + 1e-6",
            maximum_absolute_error=float(np.max(error)) if finite else None,
            maximum_bound_fraction=float(np.max(error / self.error_bound)) if finite else None,
            failure=None if finite else "non-finite output or reference error",
        )

    def assess(self, samples: tuple[Sample, ...]) -> LinearMetrics:
        device = tuple(
            sample.device_seconds for sample in samples if sample.device_seconds is not None
        )
        return LinearMetrics(
            selection=realized(self.component),
            completed_latency=Latency(
                samples_seconds=tuple(
                    s.completed_seconds for s in samples if s.pass_kind == TimingPass.COMPLETED
                ),
                boundary=TimingBoundary.PREPARE_THROUGH_COMPLETION,
                missing_model_evidence=(
                    "no authoritative matching compute/traffic time floor established",
                ),
            ),
            device_latency=Latency(
                samples_seconds=device,
                boundary=TimingBoundary.DEVICE_EXECUTION,
                unavailable_observations=()
                if device
                else ("separate device timestamps unavailable",),
            ),
            reads=ReadTraffic(
                minimum_bytes=self.component.encoded.resident_bytes + self.inputs.spec.nbytes,
                interface=TrafficInterface.LOGICAL_OPERANDS,
                formula="encoded weight payload bytes + rows * input_width * input_item_bytes",
                assumptions=(
                    "arbitrary dense input and encoded weight values",
                    "does not imply traffic crosses the DRAM interface on every reuse",
                ),
                unavailable=("no compatible physical traffic counter observation collected",),
            ),
        )

    def close(self):
        self._cleanup.close()


class LinearProcedure:
    def prepare(self, component: ResidentLinear, workload: LinearWorkload) -> LinearCase:
        return LinearCase(component, workload)
