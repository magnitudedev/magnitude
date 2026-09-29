"""The library forward is one program implementation, not an engine policy."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.components import component
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.native import LibraryState

from .definition import DEFINITION


@dataclass(frozen=True)
class LibraryForward:
    """Bind the upstream language model to our token/cache calling convention."""

    model: nn.Module

    def __call__(self, tokens: mx.array, cache: list) -> mx.array:
        offset = next((c.offset for c in cache if hasattr(c, "offset")), 0)
        if isinstance(offset, mx.array) and offset.ndim == 1:
            offset = offset[:, None]
        positions = mx.arange(tokens.shape[1], dtype=mx.int32)[None, :] + offset
        output = self.model(tokens, cache=cache, position_ids=positions)
        return output if isinstance(output, mx.array) else output.logits


@component("MODEL:FORWARD:VLM:STANDARD", model=DEFINITION)
class LibraryProgram:
    """A bound library call; family adapters provide richer named features separately.

    The callable owns its normal input/cache argument conventions. Keeping the
    binding explicit avoids inspecting signatures or rewriting model classes at run time.
    Unrequested logits are left lazy and excluded from completion roots.
    """

    features: frozenset[str] = frozenset()
    conditioning: frozenset[str] = frozenset()

    def __init__(
        self,
        call: Callable[[mx.array, list], mx.array],
        *,
        input_forward: Callable[[tuple[ModelInputs, ...], list, tuple[int, ...]], mx.array]
        | None = None,
    ):
        self.call = call
        self.input_forward = input_forward

    def _call(self, inputs: tuple[ModelInputs, ...], caches: list, positions: tuple[int, ...]):
        if self.input_forward is not None:
            return self.input_forward(inputs, caches, positions)
        if any(row.data is not None for row in inputs):
            raise ValueError("this library forward has no adapter for model input data")
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([row.tokens for row in inputs])
        )
        return self.call(tokens, caches)

    def forward(
        self,
        inputs: ModelInputs,
        state: LibraryState,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        caches = state.caches if state.batch is None else state.store.batch_caches((state,))
        logits = self._call((inputs,), caches, (state.position,))
        state.store.publish_batch((state,))
        return ModelOutput(logits if request.logits else None)

    def forward_batch(self, inputs, states, request, scope) -> ModelOutput:
        logits = self._call(
            inputs, states[0].store.batch_caches(states), tuple(state.position for state in states)
        )
        states[0].store.publish_batch(states)
        return ModelOutput(logits if request.logits else None)
