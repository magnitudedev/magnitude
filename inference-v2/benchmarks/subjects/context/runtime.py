"""Fixed-token replay and autoregressive continuation share preparation and state restoration."""

import asyncio
import hashlib
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from benchmark_fixtures.preparation import Fixture, Tokenization, prepare
from benchmarks.contracts import Observation
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest


class ContextTrace:
    def __init__(
        self,
        *,
        engine: EngineInstance,
        artifact: str,
        fixture: Literal["prose.moby-dick", "tools.bfcl"],
        context_tokens: int,
        mode: Literal["prefill", "replay", "generate"],
        measured_tokens: int,
        offset: int,
    ):
        tokenizer = Tokenization(Path(artifact))
        self.prepared = asyncio.run(
            prepare(
                Fixture(
                    identity=fixture,
                    context_tokens=context_tokens,
                    continuation_tokens=measured_tokens,
                    offset=offset,
                ),
                tokenizer,
            )
        )
        if mode not in ("prefill", "replay", "generate"):
            raise ValueError("unknown context execution mode")
        self.mode, self.limit, self.eos = mode, measured_tokens, tokenizer.eos_tokens
        self.model, self.budget = engine.engine.generation.model, engine.budget
        self.sequence = self.checkpoint = self.last = None
        self.generated: list[int] = []
        prompt = self.prepared.prompt
        self.prefix = prompt[:-1] if mode == "generate" else prompt
        self.first = prompt[-1]
        self.replay = self.prepared.continuation
        if mode != "generate" and len(self.replay) < measured_tokens:
            raise ValueError(
                f"fixture has {len(self.replay)} continuation tokens; {measured_tokens} requested"
            )
        self.consumed = 0
        original = self.model.create()
        try:
            for start in range(0, len(self.prefix), 512):
                self.model.prefill(original, self.prefix[start : start + 512])
            original.complete_committed()
            self.checkpoint = original.checkpoint()
        finally:
            original.close()
        try:
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        if self.sequence is not None:
            self.sequence.close()
        self.sequence = self.model.create(self.checkpoint)
        self.model.reserve(self.sequence, self.limit + 1)
        self.last = None
        self.generated = []
        self.consumed = 0
        mx.reset_peak_memory()

    def invoke(self) -> None:
        assert self.sequence is not None
        if self.mode == "prefill":
            self.model.prefill(self.sequence, self.replay)
            self.consumed = len(self.replay)
            return
        next_input = ModelInputs.from_tokens((self.first,)) if self.mode == "generate" else None
        for index in range(self.limit):
            inputs = next_input if next_input is not None else (self.replay[index],)
            advance = self.model.forward(self.sequence, inputs, ForwardRequest(committed_inputs=1))
            logits = advance.output.logits
            if logits is None:
                raise ValueError("model omitted requested logits")
            advance.accept(1)
            advance.complete()
            self.last = logits
            self.consumed += 1
            if self.mode == "generate":
                # This is synchronous autoregressive model service, including greedy
                # selection. Session-bench measures the serving/scheduling boundary.
                token = cast(int, mx.argmax(logits[0, -1]).item())
                self.generated.append(token)
                if token in self.eos:
                    break
                next_input = ModelInputs.from_tokens((token,))

    def complete(self) -> None:
        self.model.owner.backend.drain()

    def observe(self) -> Observation:
        assert self.sequence is not None
        checkpoint = self.sequence.checkpoint()
        try:
            if checkpoint.length != len(self.prefix) + self.consumed:
                raise ValueError("measured input count differs from committed state advancement")
        finally:
            checkpoint.close()
        value = bytes(memoryview(self.last.astype(mx.float32))) if self.last is not None else b""
        return Observation(
            hashlib.sha256(value).hexdigest(),
            {
                "input_tokens": self.consumed,
                "output_tokens": len(self.generated),
                "prefix_tokens": len(self.prefix),
                "mlx_peak_bytes": mx.get_peak_memory(),
                "reserved_bytes": self.budget.snapshot().reserved,
            },
            {
                "fixture": self.prepared.provenance,
                "mode": self.mode,
                "generated_tokens": self.generated,
                "boundary": (
                    "model runtime through completion; transactions included; no HTTP/scheduler"
                ),
            },
        )

    def close(self) -> None:
        if self.sequence is not None:
            self.sequence.close()
            self.sequence = None
        if self.checkpoint is not None:
            self.checkpoint.close()
            self.checkpoint = None
        self.last = None
