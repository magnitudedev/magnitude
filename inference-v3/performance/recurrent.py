"""Recurrent operation observations against independent FP64 sequence equations."""

from contextlib import ExitStack

import numpy as np
from pydantic import Field

from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.operations.recurrent import DeltaRecurrence
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec, Ticket
from performance.metrics import Latency, Record, Sample, TimingBoundary, TimingPass, Validation


class RecurrentWorkload(Record):
    batch: int = Field(default=1, gt=0)
    steps: int = Field(gt=0)
    seed: int = Field(default=438, ge=0)


class RecurrentMetrics(Record):
    completed_latency: Latency
    device_latency: Latency


class RecurrentCase:
    def __init__(self, component: DeltaRecurrence, workload: RecurrentWorkload):
        self.component = component
        batch, steps = workload.batch, workload.steps
        kh, vh, width = component.key_heads, component.value_heads, component.width
        rng = np.random.default_rng(workload.seed)
        q = rng.normal(size=(batch * steps, kh, width))
        k = rng.normal(size=q.shape)
        q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(width)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        arrays = [
            a.astype(np.float32)
            for a in (
                q,
                k,
                rng.normal(size=(batch * steps, vh, width)),
                rng.uniform(0, 1, (batch * steps, vh)),
                rng.uniform(0, 1, (batch * steps, vh)),
                rng.normal(size=(batch, vh, width, width)),
            )
        ]
        q, k, v, decay, beta, previous = (a.astype(np.float64) for a in arrays)
        heads = (
            np.arange(vh) % kh
            if component.mapping == HeadMapping.TILED
            else np.arange(vh) // (vh // kh)
        )
        state = previous.copy()
        output = np.empty_like(v)
        for step in range(steps):
            rows = np.arange(batch) * steps + step
            state *= decay[rows, :, None, None]
            keys = k[rows][:, heads]
            residual = (v[rows] - np.einsum("bhvk,bhk->bhv", state, keys)) * beta[rows, :, None]
            state += residual[..., None] * keys[:, :, None, :]
            output[rows] = np.einsum("bhvk,bhk->bhv", state, q[rows][:, heads])
        self.expected = (state, output)
        with ExitStack() as cleanup:
            self.inputs = []
            self.outputs = []
            for array in arrays:
                tensor = component.context.upload(
                    TensorSpec(array.shape, DType.F32), array.tobytes()
                )
                cleanup.callback(tensor.close)
                self.inputs.append(tensor)
            for array in self.expected:
                tensor = component.context.allocate(TensorSpec(array.shape, DType.F32))
                cleanup.callback(tensor.close)
                self.outputs.append(tensor)
            self.cleanup = cleanup.pop_all()

    @property
    def observation_passes(self) -> tuple[TimingPass, ...]:
        return (
            (TimingPass.COMPLETED,)
            if self.component.context.backend == Backend.LLVM
            else (
                TimingPass.COMPLETED,
                TimingPass.DEVICE_EVENTS,
            )
        )

    def reset(self) -> None:
        # Every invocation reads the same immutable prior state and overwrites
        # private outputs; repeated samples never advance the reference sequence.
        pass

    def invoke(self, *, timing: bool) -> Ticket:
        return self.component.context.submit(
            self.component.prepare(*self.inputs, *self.outputs), timing=timing
        )

    def validate(self, ticket: Ticket) -> Validation:
        errors, fractions = [], []
        for tensor, expected in zip(self.outputs, self.expected, strict=True):
            actual = np.frombuffer(self.component.context.read(tensor, after=ticket), np.float32)
            error = np.abs(actual.reshape(expected.shape) - expected)
            errors.append(float(np.max(error)))
            fractions.append(float(np.max(error / (2e-6 + 3e-6 * np.abs(expected)))))
        finite = bool(np.isfinite(errors).all() and np.isfinite(fractions).all())
        return Validation(
            passed=finite and max(fractions) <= 1,
            method=(
                "Independent FP64 gated-delta sequence; "
                "state/output abs error <= 2e-6 + 3e-6*abs(reference)"
            ),
            maximum_absolute_error=max(errors) if finite else None,
            maximum_bound_fraction=max(fractions) if finite else None,
            failure=None if finite and max(fractions) <= 1 else "recurrent state/output mismatch",
        )

    def assess(self, samples: tuple[Sample, ...]) -> RecurrentMetrics:
        return RecurrentMetrics(
            completed_latency=Latency(
                samples_seconds=tuple(
                    s.completed_seconds for s in samples if s.pass_kind == TimingPass.COMPLETED
                ),
                boundary=TimingBoundary.PREPARE_THROUGH_COMPLETION,
            ),
            device_latency=Latency(
                samples_seconds=tuple(
                    s.device_seconds for s in samples if s.device_seconds is not None
                ),
                boundary=TimingBoundary.DEVICE_EXECUTION,
                missing_model_evidence=(
                    "No matched recurrent compute/traffic lower bound measured",
                ),
            ),
        )

    def close(self) -> None:
        self.cleanup.close()


class RecurrentProcedure:
    def prepare(self, component: DeltaRecurrence, workload: RecurrentWorkload) -> RecurrentCase:
        return RecurrentCase(component, workload)
