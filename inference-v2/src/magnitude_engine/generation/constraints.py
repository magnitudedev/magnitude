"""Request-local token languages, independent of target and proposal implementations."""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx

from .constraint_spec import ConstraintSpec


class TokenConstraint(Protocol):
    def fork(self) -> TokenConstraint: ...

    def apply(self, logits: mx.array) -> mx.array:
        """Mask the next-token distribution without consuming a token."""
        ...

    def consume(self, token: int) -> bool: ...

    def forced(self) -> tuple[int, ...]: ...

    def close(self) -> None: ...


class ConstraintCompiler(Protocol):
    def create(self, spec: ConstraintSpec) -> TokenConstraint: ...
