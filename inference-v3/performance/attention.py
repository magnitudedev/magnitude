"""Attention observations over explicit KV bindings and independent causal equations."""

from contextlib import ExitStack

import numpy as np
from pydantic import Field

from magnitude_engine.operations.attention import CausalAttention
from magnitude_engine.operations.kv_binding import ReadBinding, ReadGeometry
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec, Ticket
from performance.metrics import Latency, Record, Sample, TimingBoundary, TimingPass, Validation
from performance.precision import decode, encode, rounded


class AttentionWorkload(Record):
    rows: int = Field(gt=0)
    history_tokens: int = Field(gt=0)
    seed: int = Field(default=482, ge=0)
    physical_groups: int = Field(default=1, gt=0)
    dtype: DType = DType.F32


def attention_error_bound(
    expected: np.ndarray, magnitude: np.ndarray, dtype: DType, rows: int
) -> np.ndarray:
    """Bound storage rounding separately from the unchanged FP32 arithmetic guard.

    BF16 RNE has unit roundoff u=2^-8. Rounding a locally scaled probability
    contributes at most u*sum(p*abs(v)) after exact normalization and merging.
    Output rounding adds u*abs(output), including the preceding error. Decode
    retains FP32 probabilities, so only its final store contributes BF16 error.
    This bound assumes normal finite operands; generated references satisfy it.
    """
    bound = 2e-6 + 2e-5 * np.abs(expected)
    if dtype == DType.BF16:
        u = 2.0**-8
        if rows >= 8:
            bound = bound + u * magnitude
        bound = (1 + u) * bound + u * np.abs(expected)
    return bound


class AttentionMetrics(Record):
    completed_latency: Latency
    device_latency: Latency


class AttentionCase:
    def __init__(self, component: CausalAttention, workload: AttentionWorkload):
        if workload.physical_groups > workload.history_tokens:
            raise ValueError("physical groups must each contain at least one history token")
        if workload.rows > workload.history_tokens:
            raise ValueError("query rows exceed the supplied history")
        if workload.dtype not in (DType.F32, DType.BF16):
            raise ValueError("attention observations require FP32 or BF16 storage")
        self.dtype = workload.dtype
        self.component = component
        rows, length = workload.rows, workload.history_tokens
        heads, kh, width = component.heads, component.kv_heads, component.width
        rng = np.random.default_rng(workload.seed)
        q = rng.normal(size=(rows, heads, width)).astype(np.float32)
        k = rng.normal(size=(length, kh, width)).astype(np.float32)
        v = rng.normal(size=k.shape).astype(np.float32)
        q, k, v = (rounded(a, self.dtype) for a in (q, k, v))
        magnitude = np.empty(q.shape, np.float64)
        positions = np.arange(length - rows, length, dtype=np.int32)
        expected = np.empty(q.shape, np.float64)
        for head in range(heads):
            index = head // (heads // kh)
            score = (
                q[:, head].astype(np.float64) @ k[:, index].astype(np.float64).T / np.sqrt(width)
            )
            score = np.where(np.arange(length)[None, :] <= positions[:, None], score, -np.inf)
            probability = np.exp(score - score.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            expected[:, head] = probability @ v[:, index].astype(np.float64)
            magnitude[:, head] = probability @ np.abs(v[:, index].astype(np.float64))
        self.expected = expected
        self.bound = attention_error_bound(expected, magnitude, self.dtype, rows)
        with ExitStack() as cleanup:

            def upload(a, dtype=None):
                dtype = self.dtype if dtype is None else dtype
                content = a.tobytes() if dtype == DType.I32 else encode(a, dtype)
                tensor = component.context.upload(TensorSpec(a.shape, dtype), content)
                cleanup.callback(tensor.close)
                return tensor

            self.queries = upload(q)
            self.positions = upload(positions, DType.I32)
            history = []
            start = 0
            for group in range(workload.physical_groups):
                end = (group + 1) * length // workload.physical_groups
                count = end - start
                keys, values = upload(k[start:end]), upload(v[start:end])
                metadata = upload(np.array([[0, start, count]], np.int32), DType.I32)
                history.append(ReadBinding(ReadGeometry(count, count, 1), keys, values, metadata))
                start = end
            self.history = tuple(history)
            self.output = component.context.allocate(TensorSpec(q.shape, self.dtype))
            cleanup.callback(self.output.close)
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
        # Fixed visible history and queries; all private outputs are overwritten.
        pass

    def invoke(self, *, timing: bool) -> Ticket:
        return self.component.context.submit(
            self.component.prepare(self.queries, self.positions, self.history, self.output),
            timing=timing,
        )

    def validate(self, ticket: Ticket) -> Validation:
        actual = decode(self.component.context.read(self.output, after=ticket), self.dtype)
        error = np.abs(actual.reshape(self.expected.shape) - self.expected)
        finite = bool(np.isfinite(error).all())
        fraction = float(np.max(error / self.bound)) if finite else None
        passed = finite and fraction is not None and fraction <= 1
        return Validation(
            passed=passed,
            method=(
                "Independent FP64 causal attention on stored operands; "
                "FP32 bound 2e-6 + 2e-5*abs(reference); BF16 adds derived "
                "probability-operand and final-output rounding bounds"
            ),
            maximum_absolute_error=float(np.max(error)) if finite else None,
            maximum_bound_fraction=fraction,
            failure=None if passed else "attention output differs from independent equations",
        )

    def assess(self, samples: tuple[Sample, ...]) -> AttentionMetrics:
        return AttentionMetrics(
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
                    "No matching attention throughput/traffic lower bound measured",
                ),
            ),
        )

    def close(self) -> None:
        self.cleanup.close()


class AttentionProcedure:
    def prepare(self, component: CausalAttention, workload: AttentionWorkload) -> AttentionCase:
        return AttentionCase(component, workload)
