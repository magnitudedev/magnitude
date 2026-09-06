"""Ordinary generation from a fixed context, with prefill outside measurement."""

import hashlib
import json

import mlx.core as mx

from benchmarks.contracts import Observation
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.generation.sampling_policy import SamplingPolicy


class DecodeTrace:
    def __init__(
        self, *, engine: EngineInstance, prompt_tokens: int, output_tokens: int,
        token_allowance: int = 1,
    ):
        if min(prompt_tokens, output_tokens, token_allowance) < 1:
            raise ValueError("decode trace requires positive input and output counts")
        self.generation = engine.engine.generation
        if self.generation.method.identity != "plain":
            raise ValueError("ordinary decode requires a plain generation composition")
        self.prompt = tuple(1 + i % 16 for i in range(prompt_tokens))
        self.output_tokens = output_tokens
        self.token_allowance = token_allowance
        self.sampling = SamplingPolicy(temperature=0)
        self.budget = engine.budget
        self.checkpoint = None
        self.sequence = None
        self.outputs = []
        self.expected = ()
        original = self.generation.create(
            self.prompt,
            self.sampling,
            output_tokens,
            chunk_size=engine.engine.scheduler.prefill_tokens,
        )
        try:
            self.checkpoint = original.checkpoint()
        finally:
            original.close()
        try:
            self.reset()
            self.invoke()
            self.complete()
            self.expected = tuple(self.outputs)
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        if self.sequence is not None:
            self.sequence.close()
        self.sequence = self.generation.create(
            self.prompt,
            self.sampling,
            self.output_tokens,
            checkpoint=self.checkpoint,
        )
        if self.sequence.prefill_remaining:
            raise ValueError("decode checkpoint left prefill inside the timed interval")
        self.outputs = []
        mx.reset_peak_memory()

    def invoke(self) -> None:
        assert self.sequence is not None
        while not self.sequence.finished:
            result = self.sequence.step(self.token_allowance)
            if (
                result.proposed or result.accepted or result.forced
                or not 1 <= len(result.tokens) <= self.token_allowance
                or result.evaluated_inputs != len(result.tokens)
            ):
                raise ValueError("plain decode did not perform the bounded causal advancement")
            self.outputs.extend(result.tokens)

    def complete(self) -> None:
        self.generation.model.owner.backend.drain()

    def observe(self) -> Observation:
        assert self.sequence is not None
        if not self.sequence.finished or tuple(self.outputs) != self.expected:
            raise ValueError("ordinary decode did not reproduce its fixed-context continuation")
        return Observation(
            hashlib.sha256(json.dumps(self.outputs).encode()).hexdigest(),
            {
                "prompt_tokens": len(self.prompt),
                "output_tokens": len(self.outputs),
                "token_allowance": self.token_allowance,
                "reserved_bytes": self.budget.snapshot().reserved,
                "mlx_peak_bytes": mx.get_peak_memory(),
            },
            {"output_tokens": self.outputs},
        )

    def close(self) -> None:
        if self.sequence is not None:
            self.sequence.close()
            self.sequence = None
        if self.checkpoint is not None:
            self.checkpoint.close()
            self.checkpoint = None
        self.outputs = []
