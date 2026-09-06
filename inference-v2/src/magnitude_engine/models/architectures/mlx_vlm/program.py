"""The library forward is one program implementation, not an engine policy."""

from collections.abc import Callable

import mlx.core as mx

from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.native import LibraryState


class LibraryProgram:
    """A bound library call; family adapters provide richer named features separately.

    The callable owns its normal input/cache argument conventions. Keeping the
    binding explicit avoids inspecting signatures or rewriting model classes at run time.
    Unrequested logits are left lazy and excluded from completion roots.
    """

    features: frozenset[str] = frozenset()
    conditioning: frozenset[str] = frozenset()

    def __init__(self, call: Callable[[mx.array, list], mx.array]):
        self.call = call

    def forward(
        self,
        inputs: ModelInputs,
        state: LibraryState,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        caches = state.caches if state.batch is None else state.store.batch_caches((state,))
        logits = self.call(inputs.tokens, caches)
        state.store.publish_batch((state,))
        return ModelOutput(logits if request.logits else None)

    def forward_batch(self, inputs, states, request, scope) -> ModelOutput:
        logits = self.call(mx.concatenate([row.tokens for row in inputs]),
                           states[0].store.batch_caches(states))
        states[0].store.publish_batch(states)
        return ModelOutput(logits if request.logits else None)
