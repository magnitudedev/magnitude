"""Execution-owner adaptation from rendered text to model-independent service."""

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.data import Record, TokenId
from magnitude_engine.generation.plain import Options, OutputToken
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import Ticket
from magnitude_engine.service.engine import Engine, Snapshot, Submission
from magnitude_engine.service.policy import RequestId


class Config(Record):
    target: str
    model: str = "magnitude"
    memory_bytes: int = Field(gt=0)
    context_tokens: int | None = Field(default=None, gt=0)
    parallel_sequences: int = Field(default=8, gt=0)
    max_queued: int = Field(default=128, ge=0)
    prefill_tokens: int = Field(default=512, gt=0)
    output_capacity: int = Field(default=16, gt=0)
    retained_prefixes: Literal[0] = 0
    backend: Backend | None = None
    ordinal: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class Publication:
    tokens: tuple[OutputToken, ...]
    state: Snapshot


class ServerProperties(Record):
    status: Literal["ready"] = "ready"
    model: str
    context_tokens: int
    parallel_sequences: int
    speculative_backend: Literal["none"] = "none"
    retained_prefixes: Literal[0] = 0
    prefill_tokens: int
    output_capacity: int
    memory_bytes: int
    backend: Backend
    composition_json: str
    composition_digest: str
    artifact_identity: ArtifactIdentity


@dataclass(frozen=True)
class Ready:
    properties: ServerProperties
    tokenizer: TokenizerArtifact


class Runtime:
    def __init__(self, engine: Engine, ready: Ready):
        self.engine, self.ready = engine, ready
        # Architecture selection belongs to this composition adapter. The
        # scheduler and transport never interpret a Qwen input plan.
        if not isinstance(engine.model, DenseRuntime):
            raise ValueError("this serving composition requires the dense Qwen adapter")
        self.model = engine.model
        self.receivers: dict[RequestId, Future[Publication]] = {}
        self.detached: set[RequestId] = set()

    def admit(self, tokens: tuple[TokenId, ...], options: Options) -> RequestId:
        source = self.model.input(InputPlan.text(tokens))
        try:
            return self.engine.admit(source, options)
        except BaseException:
            source.close()
            raise

    def receive(self, identity: RequestId) -> Future[Publication]:
        if identity in self.receivers:
            raise RuntimeError("request already has an output receiver")
        result: Future[Publication] = Future()
        self.receivers[identity] = result
        return result

    def stop(self, identity: RequestId) -> Snapshot:
        self.engine.cancel(identity)
        return self.engine.snapshot(identity)

    def release(self, identity: RequestId) -> None:
        receiver = self.receivers.pop(identity, None)
        if receiver is not None:
            receiver.cancel()
        if identity in self.engine.requests:
            self.engine.cancel(identity)
            # Accepted terminal output still has an explicit discard owner when
            # its transport leaves; publication is otherwise drained normally.
            self.engine.take(identity, max(1, self.engine.snapshot(identity).queued_output))
            self.detached.add(identity)

    def failed(self, error: Exception) -> None:
        self.engine.fail(error)

    def advance(self) -> Ticket | None:
        while True:
            action = self.engine.step()
            delivered = False
            for identity, receiver in tuple(self.receivers.items()):
                if receiver.cancelled():
                    del self.receivers[identity]
                    continue
                state = self.engine.snapshot(identity)
                if state.queued_output or state.finish is not None:
                    # Claim delivery before consuming publication credit.
                    if receiver.set_running_or_notify_cancel():
                        tokens = self.engine.take(identity, 1)
                        state = self.engine.snapshot(identity)
                        receiver.set_result(Publication(tokens, state))
                    del self.receivers[identity]
                    delivered = True
            for identity in tuple(self.detached):
                if (
                    self.engine.pending is None
                    or identity not in self.engine.pending.submission.requests
                ):
                    self.engine.remove(identity)
                    self.detached.remove(identity)
            if isinstance(action, Submission):
                return action.completion
            if not delivered:
                return None

    def close(self) -> None:
        for receiver in self.receivers.values():
            if receiver.set_running_or_notify_cancel():
                receiver.set_exception(RuntimeError("serving execution stopped"))
        self.receivers.clear()


@contextmanager
def open_runtime(config: Config) -> Iterator[Runtime]:
    from magnitude_engine.blueprints import (
        artifacts,
        execution,
        models,
        operations,
        service,
        serving,
    )
    from magnitude_engine.composition import build, digest, dumps
    from magnitude_engine.platform.machine import choose_endpoint

    endpoint = choose_endpoint(config.backend, config.ordinal)
    if endpoint.ordinal is None:
        raise RuntimeError("selected endpoint has no process execution ordinal")
    context = execution.Device(
        backend=endpoint.backend, budget_bytes=config.memory_bytes, ordinal=endpoint.ordinal
    )
    if Path(config.target).is_dir():
        mlx = artifacts.MLX(path=config.target)
        description = models.Qwen35MLXDescription(artifact=mlx)
        weights = operations.MLXOperations(artifact=mlx, context=context)
        metadata = serving.MLXChatMetadata(artifact=mlx)
    else:
        gguf = artifacts.GGUF(path=config.target)
        description = models.Qwen35DenseDescription(artifact=gguf)
        weights = operations.ResidentOperations(artifact=gguf, context=context)
        metadata = serving.ChatMetadata(artifact=gguf)
    model = models.Qwen35Dense(description=description, operations=weights)
    recipe = serving.ChatComponents(
        engine=service.Continuous(
            model=model,
            selector=operations.SampleSelector(context=context),
            limits=service.ServiceLimits(
                max_requests=config.max_queued + config.parallel_sequences,
                max_batch=config.parallel_sequences,
                prefill_tokens=config.prefill_tokens,
            ),
        ),
        tokenizer=metadata,
    )
    with build(recipe) as bound:
        assert isinstance(bound.engine.model, DenseRuntime)
        maximum = bound.engine.model.geometry.context_limit
        context_tokens = maximum if config.context_tokens is None else config.context_tokens
        if context_tokens > maximum:
            raise ValueError("configured context exceeds the model's declared limit")
        ready = Ready(
            ServerProperties(
                status="ready",
                model=config.model,
                context_tokens=context_tokens,
                parallel_sequences=config.parallel_sequences,
                speculative_backend="none",
                retained_prefixes=config.retained_prefixes,
                prefill_tokens=config.prefill_tokens,
                output_capacity=config.output_capacity,
                memory_bytes=config.memory_bytes,
                backend=endpoint.backend,
                composition_json=dumps(recipe),
                composition_digest=digest(recipe),
                artifact_identity=bound.tokenizer.config.artifact_identity,
            ),
            bound.tokenizer,
        )
        runtime = Runtime(bound.engine, ready)
        try:
            yield runtime
        finally:
            runtime.close()
