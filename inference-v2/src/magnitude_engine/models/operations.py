"""Typed cooperative model work and explicit device observation boundaries."""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .execution import PendingExecution
from .inputs import ModelInputs
from .runtime import ForwardRequest, ModelAdvance, ModelSequence


@dataclass(frozen=True)
class Forward:
    sequence: ModelSequence[Any, Any]
    inputs: ModelInputs
    request: ForwardRequest


@dataclass(frozen=True)
class Observe:
    arrays: tuple[mx.array, ...]


@dataclass(frozen=True)
class Submit:
    execution: PendingExecution
    consumers: tuple[mx.array, ...]


@dataclass(frozen=True)
class Complete:
    execution: PendingExecution


@dataclass(frozen=True)
class Repair:
    advance: ModelAdvance[Any, Any]
    inputs: ModelInputs


@dataclass(frozen=True)
class ProjectVocabulary:
    project: Callable[[mx.array], mx.array]
    hidden: mx.array


type Operation = Forward | Observe | Submit | Complete | Repair | ProjectVocabulary
type Response = ModelAdvance[Any, Any] | mx.array | None
type Task[T] = Generator[Operation, Response, T]


def forward(
    sequence: ModelSequence[Any, Any], inputs: ModelInputs, request: ForwardRequest
) -> Task[ModelAdvance[Any, Any]]:
    advance = yield Forward(sequence, inputs, request)
    if not isinstance(advance, ModelAdvance):
        raise RuntimeError("forward operation omitted its state advance")
    return advance


def observe(*arrays: mx.array) -> Task[None]:
    yield Observe(arrays)


def submit(advance: ModelAdvance, *consumers: mx.array) -> Task[None]:
    yield Submit(advance.execution, consumers)


def complete(advance: ModelAdvance) -> Task[None]:
    """Finish owned device work before a host-side state consumer runs."""
    yield Complete(advance.execution)


def accept(advance: ModelAdvance, count: int) -> Task[None]:
    repair = advance.prepare_accept(count)
    if repair is not None:
        yield Repair(advance, repair)
    advance.finish_accept(count)


def project_vocabulary(project: Callable[[mx.array], mx.array], hidden: mx.array) -> Task[mx.array]:
    logits = yield ProjectVocabulary(project, hidden)
    if not isinstance(logits, mx.array):
        raise RuntimeError("vocabulary projection omitted logits")
    return logits
