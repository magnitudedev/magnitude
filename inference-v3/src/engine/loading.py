"""Reusable model construction, independent of service admission and transport."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.data import TokenId
    from engine.inputs.tokenizer import Tokenizer
    from engine.models.sequence import ModelExecutor, ModelInput
    from engine.platform.backend import Backend


@dataclass(frozen=True)
class LoadedModel:
    """Model and tokenizer borrowed for the lifetime of a load_model context."""

    executor: ModelExecutor
    tokenizer: Tokenizer

    def input(self, tokens: tuple[TokenId, ...]) -> ModelInput:
        return self.executor.text_input(tokens)


def model_recipe(*, path, device, max_sequences, batch_tokens, context_tokens):
    """Shared artifact interpretation used by both serving and library loading."""
    from engine.blueprints import models, serving
    from engine.blueprints import weights as containers

    if Path(path).is_dir():
        container = containers.MLX(path=str(path))
        description = models.Qwen35MLXDescription(format=container)
        metadata = serving.MLXChatMetadata(artifact=container)
    else:
        container = containers.GGUF(path=str(path))
        description = models.Qwen35DenseDescription(format=container)
        metadata = serving.ChatMetadata(artifact=container)
    residency = containers.Weights(format=container, context=device)
    executor = models.Qwen35Dense(
        description=description,
        device=device,
        weights=residency,
        max_sequences=max_sequences,
        prefill_rows=batch_tokens,
        context_capacity=context_tokens,
    )
    return executor, metadata


@contextmanager
def load_model(
    path: str | Path,
    *,
    memory_bytes: int,
    backend: Backend | str | None = None,
    ordinal: int = 0,
    context_tokens: int | None = None,
    max_sequences: int = 8,
    batch_tokens: int = 512,
) -> Iterator[LoadedModel]:
    """Load a local model on this thread and close all owned resources on exit.

    Qwen3.5 GGUF and supported MLX-format directories use the existing model
    interpreters. No download, HTTP server, scheduler, or worker is started.
    Compilation occurs when an execution geometry is first requested.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    for name, value, minimum in (
        ("memory_bytes", memory_bytes, 1),
        ("ordinal", ordinal, 0),
        ("max_sequences", max_sequences, 1),
        ("batch_tokens", batch_tokens, 2),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if context_tokens is not None and (type(context_tokens) is not int or context_tokens < 1):
        raise ValueError("context_tokens must be a positive integer")

    from engine.blueprints.execution import DeviceRuntime
    from engine.blueprints.models import LoadedComponents
    from engine.composition import build
    from engine.devices import DevicePlan
    from engine.platform.backend import Backend

    plan = DevicePlan.discover(
        backend=None if backend is None else Backend(backend),
        maximum_bytes=memory_bytes,
        ordinal=ordinal,
    )
    executor, metadata = model_recipe(
        path=path,
        device=DeviceRuntime(plan=plan),
        max_sequences=max_sequences,
        batch_tokens=batch_tokens,
        context_tokens=context_tokens,
    )
    with build(LoadedComponents(executor=executor, metadata=metadata)) as loaded:
        yield loaded
