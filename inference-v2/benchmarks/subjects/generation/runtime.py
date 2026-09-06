"""Isolate generation execution from prompt service, HTTP, and prefix lookup."""

import hashlib
import json
from typing import Literal

import mlx.core as mx

from benchmarks.contracts import Observation
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy


class GenerationTrace:
    def __init__(
        self, *, engine: EngineInstance, prompt_tokens: int, output_tokens: int,
        rows: int, token_allowance: int = 4,
        execution: Literal['shared', 'independent'] = 'shared',
    ):
        if min(prompt_tokens, output_tokens, rows, token_allowance) < 1 or rows > 64:
            raise ValueError('invalid generation trace geometry')
        if execution not in ('shared', 'independent'):
            raise ValueError('unknown generation benchmark control')
        self.generation = engine.engine.generation
        self.owner = self.generation.model.owner
        self.budget = engine.budget
        self.prompts = [tuple(1 + (i + row) % 16 for i in range(prompt_tokens + row))
                        for row in range(rows)]
        self.output_tokens, self.allowance = output_tokens, token_allowance
        self.execution = execution
        self.sampling = SamplingPolicy(temperature=0)
        self.checkpoints, self.sequences, self.outputs, self.expected = [], [], [], []
        self.proposed = self.accepted = self.calls = self.max_batch = 0
        try:
            reference = GenerationRuntime(self.generation.model, PlainMethod())
            for prompt in self.prompts:
                oracle = reference.create(prompt, self.sampling, output_tokens)
                try:
                    tokens = []
                    while not oracle.finished:
                        tokens.extend(oracle.step().tokens)
                    self.expected.append(tokens)
                finally:
                    oracle.close()
                seed = self.generation.create(prompt, self.sampling, output_tokens)
                try:
                    self.checkpoints.append(seed.checkpoint())
                finally:
                    seed.close()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        for row in self.sequences:
            row.close()
        self.sequences = []
        self.outputs = [[] for _ in self.prompts]
        self.proposed = self.accepted = self.calls = self.max_batch = 0
        for prompt, checkpoint in zip(self.prompts, self.checkpoints, strict=True):
            self.sequences.append(self.generation.create(
                prompt, self.sampling, self.output_tokens, checkpoint=checkpoint,
            ))
        mx.reset_peak_memory()

    def invoke(self) -> None:
        while any(not row.finished for row in self.sequences):
            indices = [i for i, row in enumerate(self.sequences) if not row.finished]
            if self.execution == 'shared':
                services = self.generation.step_many(
                    tuple(self.sequences[i] for i in indices), (self.allowance,) * len(indices),
                )
                results = [service.outcome for service in services]
                self.max_batch = max(self.max_batch, *(service.batch_size for service in services))
            else:
                results = [self.sequences[i].step(self.allowance) for i in indices]
                self.max_batch = 1
            self.calls += 1
            for i, result in zip(indices, results, strict=True):
                if result is None:
                    continue
                if isinstance(result, BaseException):
                    raise RuntimeError(
                        'generation benchmark did not complete its round'
                    ) from result
                self.outputs[i].extend(result.tokens)
                self.proposed += result.proposed
                self.accepted += result.accepted

    def complete(self) -> None:
        self.owner.backend.drain()

    def observe(self) -> Observation:
        if self.outputs != self.expected or not all(row.finished for row in self.sequences):
            raise ValueError(
                'generation differs from independent plain target execution: '
                + json.dumps({'expected': self.expected, 'actual': self.outputs})
            )
        return Observation(
            hashlib.sha256(json.dumps(self.outputs).encode()).hexdigest(),
            {
                'requests': len(self.sequences),
                'output_tokens': sum(map(len, self.outputs)),
                'proposed_tokens': self.proposed,
                'accepted_tokens': self.accepted,
                'max_physical_batch': self.max_batch,
                'generation_services': self.calls,
                'reserved_bytes': self.budget.snapshot().reserved,
                'mlx_peak_bytes': mx.get_peak_memory(),
            },
            {'outputs': self.outputs, 'prompt_lengths': list(map(len, self.prompts))},
        )

    def close(self) -> None:
        for row in self.sequences:
            row.close()
        self.sequences.clear()
        for checkpoint in self.checkpoints:
            checkpoint.close()
        self.checkpoints.clear()
