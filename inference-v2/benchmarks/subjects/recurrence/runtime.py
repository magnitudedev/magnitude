"""Completed recurrence work at fixed inputs and state, with a library oracle."""

import hashlib
from typing import cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_kernel

from benchmarks.contracts import Observation
from magnitude_engine.models.recurrence.contracts import DeltaRecurrence
from magnitude_engine.models.recurrence.inputs import DeltaInputs


class DeltaTrace:
    def __init__(
        self,
        *,
        update: DeltaRecurrence,
        tokens: int,
        key_heads: int,
        value_heads: int,
        key_width: int,
        value_width: int,
        dtype: str,
    ):
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("invalid recurrence dtype")
        self.operation = update
        mx.random.seed(73)
        element_type = getattr(mx, dtype)
        q = mx.random.normal((1, tokens, key_heads, key_width)).astype(element_type) / key_width
        k = mx.random.normal(q.shape).astype(element_type) / key_width**0.5
        v = mx.random.normal((1, tokens, value_heads, value_width)).astype(element_type)
        decay = mx.random.uniform(shape=(1, tokens, value_heads))
        beta = mx.random.uniform(shape=decay.shape).astype(element_type)
        self.inputs = DeltaInputs(q, k, v, decay, beta)
        self.state = mx.random.normal((1, value_heads, value_width, key_width)) * 0.1
        self.oracle = gated_delta_kernel(q, k, v, decay, beta, self.state)
        mx.eval(q, k, v, decay, beta, self.state, *self.oracle)
        self.output: tuple[mx.array, mx.array] | None = None

    def reset(self) -> None:
        self.output = None

    def invoke(self) -> None:
        self.output = self.operation.advance(self.inputs, self.state)

    def complete(self) -> None:
        assert self.output is not None
        mx.eval(*self.output)

    def observe(self) -> Observation:
        assert self.output is not None
        errors = [
            cast(float, mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
            for a, b in zip(self.output, self.oracle, strict=True)
        ]
        if not all(
            mx.allclose(a, b, atol=1e-5).item()
            for a, b in zip(self.output, self.oracle, strict=True)
        ):
            raise ValueError(f"recurrence differs from library oracle: {errors}")
        digest = hashlib.sha256()
        for array in self.output:
            digest.update(np.asarray(array.astype(mx.float32)).tobytes())
        x = self.inputs
        input_bytes = sum(a.nbytes for a in (x.queries, x.keys, x.values, x.decay, x.beta))
        return Observation(
            digest.hexdigest(),
            {
                "state_bytes": self.state.nbytes,
                "minimum_state_transfer_bytes": 2 * self.state.nbytes,
                "prepared_input_bytes": input_bytes,
                "output_bytes": self.output[0].nbytes,
                "tokens": x.length,
                "max_output_error": errors[0],
                "max_state_error": errors[1],
            },
        )

    def close(self) -> None:
        self.output = None


class DeltaShapeTrace:
    """First owned calls at distinct lengths, with prepared library oracles.

    Run once without warmup in a fresh benchmark process. Per-length durations
    include invocation and completion; their differences are not a pure compiler
    timer. No engine, model loading or scheduler work is included.
    """

    def __init__(self, *, update: DeltaRecurrence, first_tokens: int, shapes: int):
        if first_tokens <= 8 or not 2 <= shapes <= 32:
            raise ValueError("shape sweep needs 2–32 distinct prefill lengths above eight")
        self.work = tuple(
            DeltaTrace(
                update=update,
                tokens=first_tokens + i,
                key_heads=16,
                value_heads=32,
                key_width=128,
                value_width=128,
                dtype="bfloat16",
            )
            for i in range(shapes)
        )
        self.durations = []

    def reset(self) -> None:
        self.durations = []
        for work in self.work:
            work.reset()

    def invoke(self) -> None:
        from time import perf_counter_ns

        for work in self.work:
            start = perf_counter_ns()
            work.invoke()
            work.complete()
            self.durations.append((perf_counter_ns() - start) / 1e6)

    def complete(self) -> None:
        mx.synchronize()

    def observe(self) -> Observation:
        import statistics

        observations = [work.observe() for work in self.work]
        digest = hashlib.sha256("".join(o.output_digest for o in observations).encode()).hexdigest()
        return Observation(
            digest,
            {
                "distinct_lengths": len(self.work),
                "first_length_ms": self.durations[0],
                "subsequent_median_ms": statistics.median(self.durations[1:]),
                "max_output_error": max(o.counters["max_output_error"] for o in observations),
                "max_state_error": max(o.counters["max_state_error"] for o in observations),
            },
            {"lengths": [work.inputs.length for work in self.work], "completed_ms": self.durations},
        )

    def close(self) -> None:
        for work in self.work:
            work.close()
        self.work = ()
