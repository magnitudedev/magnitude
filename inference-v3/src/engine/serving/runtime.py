"""Execution-owner adaptation from rendered text to model-independent service."""

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

import ops
from engine.blueprints.execution import ScheduleProfile
from engine.data import Record, TokenId
from engine.generation.constraints import ConstraintPlan, ConstraintVocabulary
from engine.generation.plain import Options, OutputToken
from engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from engine.inputs.media import PreparedMedia
from engine.models.qwen35.inputs import InputPlan
from engine.models.qwen35.runtime import DenseRuntime
from engine.platform.backend import Backend
from engine.service.engine import Engine, Snapshot, Submission
from engine.service.policy import RequestId
from engine.weights.identity import ArtifactIdentity
from templates.bundle import Variant


class Config(Record):
    target: str
    model: str = "magnitude"
    memory_bytes: int = Field(gt=0)
    context_tokens: int | None = Field(default=None, gt=0)
    parallel_sequences: int = Field(default=8, gt=0)
    max_queued: int = Field(default=128, ge=0)
    prefill_tokens: int = Field(default=512, gt=0)
    output_capacity: int = Field(default=16, gt=0)
    forced_quantum: int = Field(default=32, ge=0, le=256)
    retained_prefixes: Literal[0] = 0
    backend: Backend | None = None
    ordinal: int = Field(default=0, ge=0)
    schedules: ScheduleProfile | None = None
    template_variant: str | None = None
    template_override: Variant | None = None

    @model_validator(mode="after")
    def template_selection(self):
        if self.template_variant is not None and self.template_override is not None:
            raise ValueError("configure a template source override or variant, not both")
        return self


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
    forced_quantum: int
    memory_bytes: int
    backend: Backend
    composition_json: str
    composition_digest: str
    artifact_identity: ArtifactIdentity


@dataclass(frozen=True)
class Ready:
    properties: ServerProperties
    tokenizer: TokenizerArtifact
    image_directory: Path | None = None


class Runtime:
    def __init__(self, engine: Engine, ready: Ready):
        self.engine, self.ready = engine, ready
        # Architecture selection belongs to this composition adapter. The
        # scheduler and transport never interpret a Qwen input plan.
        if not isinstance(engine.model, DenseRuntime):
            raise ValueError("this serving composition requires the dense Qwen adapter")
        self.model = engine.model
        self._constraint_vocabulary: ConstraintVocabulary | None = None
        self._vision = None
        self._image_processor_identity: str | None = None
        self.receivers: dict[RequestId, Future[Publication]] = {}
        self.detached: set[RequestId] = set()

    def admit(
        self,
        tokens: tuple[TokenId, ...],
        options: Options,
        constraint: ConstraintPlan | None = None,
        media: PreparedMedia | None = None,
    ) -> RequestId:
        state = None
        if constraint is not None:
            # Compile and initialize before creating input state or queueing work.
            if constraint.artifact_identity != self.ready.properties.artifact_identity:
                raise ValueError("constraint plan belongs to another served artifact")
            if self._constraint_vocabulary is None:
                self._constraint_vocabulary = ConstraintVocabulary(
                    self.ready.tokenizer.config,
                    projection_vocabulary=self.model.geometry.vocabulary,
                )
            state = self._constraint_vocabulary.bind(constraint)
        if media is None:
            source = self.model.input(InputPlan.text(tokens))
        else:
            from engine.models.qwen35.formats.vision_mlx import describe
            from engine.models.qwen35.preparation import interpret, preparation_identity
            from engine.models.qwen35.vision_runtime import ImageSource, VisionEncoder
            from engine.weights.formats.mlx_safetensors import MLXFormat
            from engine.weights.tensor_residency import TensorWeights

            if self.ready.image_directory is None:
                raise ValueError("the served artifact does not provide an image encoder")
            if self._vision is None:
                weights = self.model.weights
                if not isinstance(weights, TensorWeights) or not isinstance(
                    weights.format, MLXFormat
                ):
                    raise ValueError("image encoding requires the bound MLX image artifact")
                self._vision = VisionEncoder(describe(weights.format), self.model.device, weights)
                self._image_processor_identity = preparation_identity(self.ready.image_directory)
            assert self._image_processor_identity is not None
            prepared = interpret(
                tokens,
                media,
                processor=self._image_processor_identity,
                geometry=self._vision.description.geometry.image,
            )
            source = ImageSource(self.model, self._vision, prepared)
        try:
            return self.engine.admit(source, options, constraint=state)
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

    def advance(self) -> ops.Completion | None:
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
        if self._vision is not None:
            self.engine.close()
            self._vision.close()


@contextmanager
def open_runtime(config: Config) -> Iterator[Runtime]:
    from engine.blueprints import execution, service, serving
    from engine.loading import model_recipe
    from engine.composition import build, digest, dumps
    from engine.devices import DevicePlan

    plan = DevicePlan.discover(
        backend=config.backend, maximum_bytes=config.memory_bytes, ordinal=config.ordinal
    )
    endpoint = plan.selected_endpoints[0]
    context = execution.DeviceRuntime(plan=plan, schedules=config.schedules)
    model, metadata = model_recipe(
        path=config.target, device=context, max_sequences=config.parallel_sequences,
        batch_tokens=config.prefill_tokens, context_tokens=config.context_tokens,
    )
    recipe = serving.ChatComponents(
        engine=service.Continuous(
            model=model,
            limits=service.ServiceLimits(
                max_requests=config.max_queued + config.parallel_sequences,
                max_batch=config.parallel_sequences,
                prefill_tokens=config.prefill_tokens,
                decode_tokens=min(config.prefill_tokens, max(1, config.forced_quantum)),
            ),
        ),
        tokenizer=metadata,
    )
    with build(recipe) as bound:
        assert isinstance(bound.engine.model, DenseRuntime)
        context_tokens = bound.engine.model.context_capacity
        bound.engine.model.prime(min(config.prefill_tokens, context_tokens), context_tokens)
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
                forced_quantum=config.forced_quantum,
                memory_bytes=config.memory_bytes,
                backend=endpoint.backend,
                composition_json=dumps(recipe),
                composition_digest=digest(recipe),
                artifact_identity=bound.tokenizer.config.artifact_identity,
            ),
            bound.tokenizer,
            Path(config.target).expanduser().resolve() if Path(config.target).is_dir() else None,
        )
        runtime = Runtime(bound.engine, ready)
        try:
            yield runtime
        finally:
            runtime.close()
