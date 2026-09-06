"""Completed causal prefill from a fixed checkpoint, independent of architecture."""

import hashlib
from typing import Literal

import mlx.core as mx

from benchmarks.contracts import Observation
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest


class PrefillTrace:
    """Restore and continuation validation are outside the measured forward.

    A one-token continuation probes the resulting state without requiring a model
    to expose internal residuals. Repeatability is a diagnostic invariant; independent
    numerical and semantic tests remain required for performance qualification.
    """

    def __init__(
        self, *, engine: EngineInstance, prefix_tokens: int, input_tokens: int,
        rows: int = 1, execution: Literal["shared", "independent"] = "shared",
    ):
        if prefix_tokens < 0 or input_tokens < 1 or not 1 <= rows <= 64:
            raise ValueError("invalid prefill trace geometry")
        if execution not in ("shared", "independent"):
            raise ValueError("unknown prefill execution control")
        self.model = engine.engine.generation.model
        self.budget = engine.budget
        self.sequences = []
        self.rows, self.execution = rows, execution
        self.checkpoint = None
        self.expected = None
        self.prefix_tokens = prefix_tokens
        self.inputs = tuple(1 + i % 16 for i in range(input_tokens))
        original = self.model.create()
        try:
            for start in range(0, prefix_tokens, 512):
                self.model.prefill(
                    original,
                    tuple(1 + i % 16 for i in range(start, min(prefix_tokens, start + 512))),
                )
            self.checkpoint = original.checkpoint()
        finally:
            original.close()
        try:
            self.reset()
            self.invoke()
            self.complete()
            self.expected = self._probe()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        for sequence in self.sequences:
            sequence.close()
        self.sequences = []
        for _ in range(self.rows):
            sequence = self.model.create(self.checkpoint)
            self.sequences.append(sequence)
            self.model.reserve(sequence, len(self.inputs) + 1)
        mx.reset_peak_memory()

    def invoke(self) -> None:
        if self.rows == 1 or self.execution == "independent":
            for sequence in self.sequences:
                self.model.prefill(sequence, self.inputs)
            return
        sequences = tuple(self.sequences)
        if not self.model.can_batch(sequences):
            raise ValueError("declared shared prefill requires a compatible model/state batch")
        advances = self.model.forward_batch(
            sequences, (ModelInputs.from_tokens(self.inputs),) * self.rows,
            ForwardRequest(False, committed_inputs=len(self.inputs)),
        )
        for advance in advances:
            advance.accept(len(self.inputs))

    def complete(self) -> None:
        self.model.owner.backend.drain()

    def _probe(self) -> mx.array:
        values = []
        for sequence in self.sequences:
            checkpoint = sequence.checkpoint()
            try:
                if checkpoint.length != self.prefix_tokens + len(self.inputs):
                    raise ValueError("prefill did not commit its full input")
            finally:
                checkpoint.close()
            advance = self.model.forward(sequence, (1,), ForwardRequest(committed_inputs=1))
            if advance.output.logits is None:
                raise ValueError("model omitted continuation logits")
            advance.accept(1)
            values.append(advance.output.logits)
        logits = values[0] if len(values) == 1 else mx.concatenate(values)
        if not mx.all(mx.isfinite(logits)).item():
            raise ValueError("prefill continuation produced nonfinite logits")
        return logits

    def observe(self) -> Observation:
        peak_bytes = mx.get_peak_memory()
        reserved_bytes = self.budget.snapshot().reserved
        actual = self._probe()
        assert self.expected is not None
        if not mx.array_equal(actual, self.expected).item():
            raise ValueError("fixed-shape prefill continuation is not repeatable")
        return Observation(
            hashlib.sha256(bytes(memoryview(actual.astype(mx.float32)))).hexdigest(),
            {
                "input_tokens": len(self.inputs) * self.rows,
                "input_tokens_per_request": len(self.inputs),
                "requests": self.rows,
                "prefix_tokens": self.prefix_tokens,
                "committed_tokens": self.prefix_tokens + len(self.inputs),
                "reserved_bytes": reserved_bytes,
                "mlx_peak_bytes": peak_bytes,
            },
        )

    def close(self) -> None:
        for sequence in self.sequences:
            sequence.close()
        self.sequences.clear()
        if self.checkpoint is not None:
            self.checkpoint.close()
            self.checkpoint = None
        self.expected = None
