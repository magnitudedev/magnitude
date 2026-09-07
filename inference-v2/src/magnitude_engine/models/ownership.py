"""Allocation ownership and explicit loans of shared model operations."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from magnitude_engine.artifacts.materialization import MaterializedResource

from .embeddings.contracts import EmbeddingLookup
from .execution import ExecutionScope
from .inputs import ModelInputs
from .runtime import BatchForward, ForwardRequest, ModelOutput, ModelProgram


class ProgramLease:
    def __init__(self, owner: OwnedProgram):
        self.owner = owner
        self.closed = False
        owner._users += 1

    def close(self) -> None:
        if not self.closed:
            self.owner._users -= 1
            self.closed = True


class VocabularyLoan:
    """An attached head borrows the target's vocabulary operations and their lifetime."""

    def __init__(
        self,
        owner: OwnedProgram,
        embedding: EmbeddingLookup,
        project: Callable[[mx.array], mx.array],
        identity: str,
        size: int,
    ):
        self.owner = owner
        self._embedding: EmbeddingLookup | None = embedding
        self._project: Callable[[mx.array], mx.array] | None = project
        self.identity, self.size = identity, size
        self.closed = False
        owner._loans += 1

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        if self.closed or self._embedding is None:
            raise RuntimeError("vocabulary loan is closed")
        scope.acquire(self.owner.acquire)
        return self._embedding.lookup(rows, scope)

    def project(self, hidden: mx.array) -> mx.array:
        if self.closed or self._project is None:
            raise RuntimeError("vocabulary loan is closed")
        self.owner.check()
        return self._project(hidden)

    def bindings(self) -> EmbeddingLookup:
        if self.closed or self._embedding is None:
            raise RuntimeError("vocabulary loan is closed")
        return self._embedding

    def close(self) -> None:
        if not self.closed:
            self._embedding = None
            self._project = None
            self.owner._loans -= 1
            self.closed = True


class OwnedProgram[S]:
    def __init__(
        self,
        program: ModelProgram[S],
        resources: tuple[MaterializedResource, ...],
        vocabulary: tuple[str, int, EmbeddingLookup, Callable[[mx.array], mx.array]] | None = None,
    ):
        self._program: ModelProgram[S] | None = program
        self._vocabulary = vocabulary
        self.resources = resources
        self.features = program.features
        self.conditioning = program.conditioning
        self._users = 0
        self._loans = 0

    def check(self) -> None:
        if self._program is None:
            raise RuntimeError("model program is closed")

    def acquire(self) -> ProgramLease:
        self.check()
        return ProgramLease(self)

    def borrow_vocabulary(self) -> VocabularyLoan:
        self.check()
        if self._vocabulary is None:
            raise ValueError("model does not expose borrowed vocabulary operations")
        identity, size, embedding, project = self._vocabulary
        return VocabularyLoan(self, embedding, project, identity, size)

    def forward(
        self, inputs: ModelInputs, state: S, request: ForwardRequest, scope: ExecutionScope
    ) -> ModelOutput:
        scope.acquire(self.acquire)
        assert self._program is not None
        return self._program.forward(inputs, state, request, scope)

    @property
    def forward_batch(self) -> BatchForward[S] | None:
        self.check()
        assert self._program is not None
        return None if self._program.forward_batch is None else self._forward_batch

    def _forward_batch(
        self,
        inputs: tuple[ModelInputs, ...],
        states: tuple[S, ...],
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        scope.acquire(self.acquire)
        assert self._program is not None and self._program.forward_batch is not None
        return self._program.forward_batch(inputs, states, request, scope)

    def bindings(self) -> ModelProgram[S]:
        self.check()
        assert self._program is not None
        return self._program

    def close(self) -> None:
        if self._users or self._loans:
            raise RuntimeError(
                "model allocations still have active sequences/executions or borrowed operations"
            )
        self._program = None
        self._vocabulary = None
        failures = []
        for resource in reversed(self.resources):
            try:
                resource.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("model allocation disposal failed", failures)
        self.resources = ()
