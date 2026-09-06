"""Real target/head request traces through the production engine orchestration."""

import hashlib
import json
from dataclasses import asdict

import mlx.core as mx
from transformers import AutoTokenizer

from benchmarks.contracts import Observation
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.engine.delivery import Finished, Tokens
from magnitude_engine.engine.requests import GenerationRequest
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy


class EngineTrace:
    """Two waves: concurrent cold requests, then compatible retained-prefix requests.

    Construction and a plain-generation correctness oracle are outside timing.
    Every timed wave runs actual prefill/decode and drains bounded delivery. This
    does not include HTTP, tokenizer/template costs or BFCL tool-call validation.
    """

    def __init__(
        self,
        *,
        engine: EngineInstance,
        prompt_text: str,
        prompt_tokens: int,
        output_tokens: int,
        rows: int,
        prefix_reuse: bool = True,
    ):
        if min(prompt_tokens, output_tokens, rows) < 1 or rows > 64:
            raise ValueError("invalid engine trace geometry")
        self.outputs = []
        self.finishes = []
        self.services = []
        self.phase_services = []
        self.rows, self.output_tokens = rows, output_tokens
        self.prefix_reuse = prefix_reuse
        self.output_capacity = engine.output_capacity
        try:
            self.engine = engine.engine
            self.budget = engine.budget
            self.generation = self.engine.generation
            self.owner = self.generation.model.owner
            target = self.generation.model
            tokenizer = AutoTokenizer.from_pretrained(
                engine.properties["target_path"], local_files_only=True
            )
            unit = tokenizer.encode(prompt_text)
            if not unit:
                raise ValueError("benchmark prompt text tokenizes to nothing")
            self.prompt = tuple(
                (unit * ((prompt_tokens + len(unit) - 1) // len(unit)))[:prompt_tokens]
            )
            self.sampling = SamplingPolicy(temperature=0)
            oracle = GenerationRuntime(target, PlainMethod()).create(
                self.prompt,
                self.sampling,
                output_tokens,
                chunk_size=engine.engine.scheduler.prefill_tokens,
            )
            try:
                expected = []
                while not oracle.finished:
                    expected.extend(oracle.step().tokens)
                self.expected = tuple(expected)
            finally:
                oracle.close()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        assert self.engine is not None
        self.owner.backend.drain()
        self.engine.prefixes.close()
        self.engine.scheduler.reset()
        self.outputs, self.finishes, self.services = [], [], []
        self.phase_services = []
        mx.reset_peak_memory()

    def invoke(self) -> None:
        assert self.engine is not None
        # Retention is supplied by engine composition. Validate the declared cache
        # behavior instead of overriding policy inside a measured request trace.
        for wave in range(2):
            handles = [
                self.engine.submit(
                    GenerationRequest(self.prompt, self.sampling, self.output_tokens),
                    identity=f"{wave}:{row}",
                    output_capacity=self.output_capacity,
                )
                for row in range(self.rows)
            ]
            outputs = [[] for _ in handles]
            finishes: list[Finished | None] = [None for _ in handles]
            bound = self.rows * (len(self.prompt) + self.output_tokens + 2)
            for _ in range(bound):
                self.services.extend(self.engine.tick())
                if self.engine.last_service is not None:
                    self.phase_services.append(self.engine.last_service)
                for index, handle in enumerate(handles):
                    if finishes[index] is not None:
                        continue
                    while True:
                        try:
                            event = handle.delivery.take(0)
                        except TimeoutError:
                            break
                        if isinstance(event, Tokens):
                            outputs[index].extend(event.values)
                        elif isinstance(event, Finished):
                            finishes[index] = event
                            break
                if all(finish is not None for finish in finishes):
                    break
            else:
                raise RuntimeError("engine trace exceeded its declared work bound")
            self.outputs.extend(tuple(values) for values in outputs)
            self.finishes.extend(finishes)

    def complete(self) -> None:
        self.owner.backend.drain()

    def observe(self) -> Observation:
        if len(self.outputs) != 2 * self.rows or any(
            not isinstance(finish, Finished)
            or finish.reason != "length"
            or finish.generated_tokens != self.output_tokens
            for finish in self.finishes
        ):
            raise ValueError(
                "engine trace did not complete every requested output: "
                + json.dumps([asdict(f) if f is not None else None for f in self.finishes])
            )
        if any(output != self.expected for output in self.outputs):
            raise ValueError(
                "engine continuation differs from the plain-generation oracle: "
                + json.dumps({"expected": self.expected, "actual": self.outputs})
            )
        if any(f.cached_tokens != 0 for f in self.finishes[: self.rows]) or any(
            f.cached_tokens != (len(self.prompt) - 1 if self.prefix_reuse else 0)
            for f in self.finishes[self.rows :]
        ):
            raise ValueError("engine trace did not establish its declared cold/warm prefix state")
        prefill = [service for service in self.services if service.phase == "prefill"]
        decode = [service for service in self.services if service.phase == "decode"]
        digest = hashlib.sha256(json.dumps(self.outputs).encode()).hexdigest()
        return Observation(
            digest,
            {
                "requests": len(self.finishes),
                "actual_prompt_tokens_per_request": len(self.prompt),
                "output_tokens": sum(len(values) for values in self.outputs),
                "cached_tokens": sum(f.cached_tokens for f in self.finishes),
                "prefill_tokens": sum(s.input_tokens for s in prefill),
                "verification_tokens": sum(s.input_tokens for s in decode),
                "proposed_tokens": sum(f.proposed_tokens for f in self.finishes),
                "accepted_tokens": sum(f.accepted_tokens for f in self.finishes),
                "prefill_ns": sum(s.elapsed_ns for s in prefill),
                "decode_request_ns": sum(s.elapsed_ns for s in decode),
                "decode_service_ns": sum(
                    s.elapsed_ns for s in self.phase_services if s.phase == "decode"
                ),
                "prefill_service_ns": sum(
                    s.elapsed_ns for s in self.phase_services if s.phase == "prefill"
                ),
                "max_ttft_ns": max(f.first_token_ns for f in self.finishes),
                "max_request_ns": max(f.finished_ns for f in self.finishes),
                "reserved_bytes": self.budget.snapshot().reserved,
                "mlx_peak_bytes": mx.get_peak_memory(),
            },
            {
                "requests": [
                    {
                        "wave": index // self.rows,
                        "row": index % self.rows,
                        **asdict(finish),
                        "tokens": list(output),
                    }
                    for index, (finish, output) in enumerate(
                        zip(self.finishes, self.outputs, strict=True)
                    )
                ],
                "services": [asdict(service) for service in self.services],
                "phase_services": [asdict(service) for service in self.phase_services],
            },
        )

    def close(self) -> None:
        # The construction scope owns the injected engine and retires it after this subject.
        self.outputs, self.finishes, self.services = [], [], []
        self.phase_services = []
