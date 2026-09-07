"""Resident Qwen tensor transitions; state authority remains outside compilation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.models.experts.computation import ResidentExperts
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.decode import prepare_decode_append
from magnitude_engine.models.state.hybrid import HybridState
from magnitude_engine.models.state.recurrent import RecurrentBoundaries, read_batch, write_batch

from .attention.operation import GatedAttention
from .feedforward.operation import DenseFeedForward, RoutedFeedForward
from .recurrence.operation import RecurrentMixer

if TYPE_CHECKING:
    from .program import Qwen35Program


class ResidentDecode:
    def __init__(self, program: Qwen35Program):
        self.embedding = program.embedding
        self.blocks = program.blocks
        self.norm, self.output = program.norm, program.output
        self.functions: OrderedDict[tuple, Callable[..., Any]] = OrderedDict()

    @staticmethod
    def supports(program: Qwen35Program) -> bool:
        if not isinstance(program.embedding, (ResidentEmbedding, ResidentAffineEmbedding)):
            return False
        for block in program.blocks:
            mixer, feedforward = block.mixer, block.feedforward
            if isinstance(mixer, GatedAttention):
                if not isinstance(mixer.attention, MetalPagedAttention) or mixer.head_width not in (
                    32,
                    64,
                    128,
                    256,
                    512,
                ):
                    return False
            elif not isinstance(mixer, RecurrentMixer):
                return False
            if isinstance(feedforward, RoutedFeedForward):
                if not isinstance(feedforward.experts, ResidentExperts):
                    return False
            elif not isinstance(feedforward, DenseFeedForward):
                return False
        return True

    def _function(
        self,
        page_size: int,
        table_width: int,
        capacity: int,
        batch: int,
        request: ForwardRequest,
    ):
        feature_names = tuple(sorted(request.features))
        key = (page_size, table_width, capacity, batch, request.logits, feature_names)
        if key in self.functions:
            self.functions.move_to_end(key)
            return self.functions[key]
        embedding, blocks = self.embedding, self.blocks
        norm, output_projection = self.norm, self.output
        assert isinstance(embedding, (ResidentEmbedding, ResidentAffineEmbedding))
        # The complete mapped horizon bounds the launch until the next page grows.
        # Actual causal positions and physical mappings stay dynamic tensors.
        covered = page_size * table_width

        def step(tokens, positions, destinations, pages, keys, values, recurrent):
            hidden = embedding(tokens)
            next_keys, next_values = list(keys), list(values)
            next_recurrent = list(recurrent)
            features = {}
            for index, block in enumerate(blocks):
                name = f"residual:{index}"
                if name in feature_names:
                    features[name] = hidden
                mixer = block.mixer
                x = block.mixer_norm(hidden)
                if isinstance(mixer, GatedAttention):
                    q, k, v, gate = mixer.project(x, positions)
                    layer = mixer.index
                    updated_k, updated_v = keys[layer], values[layer]
                    for row in range(tokens.shape[0]):
                        destination = destinations[row : row + 1]
                        updated_k = mx.slice_update(updated_k, k[row], destination, axes=[1])
                        updated_v = mx.slice_update(updated_v, v[row], destination, axes=[1])
                    attention = mixer.attention
                    assert isinstance(attention, MetalPagedAttention)
                    attended = attention.apply(
                        q,
                        updated_k,
                        updated_v,
                        pages,
                        positions,
                        page_size=page_size,
                        table_width=table_width,
                        covered=covered,
                        scale=mixer.head_width**-0.5,
                    )
                    value = mixer.finish(attended, gate)
                    next_keys[layer], next_values[layer] = updated_k, updated_v
                else:
                    assert isinstance(mixer, RecurrentMixer)
                    value, conv, memory = mixer.operation.graph.advance(x, *recurrent[mixer.index])
                    next_recurrent[mixer.index] = (conv, memory)
                hidden = hidden + value
                feedforward = block.feedforward
                x = block.feedforward_norm(hidden)
                if isinstance(feedforward, RoutedFeedForward):
                    experts = feedforward.experts
                    assert isinstance(experts, ResidentExperts)
                    value = feedforward.apply(x, experts)
                else:
                    assert isinstance(feedforward, DenseFeedForward)
                    value = feedforward.call(x)
                hidden = hidden + value
            name = f"residual:{len(blocks)}"
            if name in feature_names:
                features[name] = hidden
            logits = (output_projection(norm(hidden)),) if request.logits else ()
            return logits, features, tuple(next_keys), tuple(next_values), tuple(next_recurrent)

        compiled = mx.compile(step)
        self.functions[key] = compiled
        # Old geometry must not retain an unbounded collection of compiled graphs.
        if len(self.functions) > 4:
            self.functions.popitem(last=False)
        return compiled

    def forward(
        self,
        tokens: mx.array,
        states: tuple[HybridState, ...],
        request: ForwardRequest,
    ) -> ModelOutput:
        append = prepare_decode_append(tuple(state.pages for state in states))
        # Page-map shape follows the attention partition, rather than recompiling
        # for each smaller storage page. Padding is invisible to causal positions.
        pages_per_partition = max(1, MetalPagedAttention.partition_tokens // append.page_size)
        width = (
            (append.table.width + pages_per_partition - 1) // pages_per_partition
        ) * pages_per_partition
        pages = append.table.device
        if width != append.table.width:
            pages = mx.pad(pages, [(0, 0), (0, width - append.table.width)], constant_values=-1)
        slots = tuple(
            tuple(state.slots[index] for state in states) for index in range(len(states[0].slots))
        )
        recurrent = tuple(read_batch(group) for group in slots)
        initial = tuple(tuple(slot.values for slot in group) for group in slots)
        fn = self._function(
            append.page_size, width, append.capacity, len(states), request
        )
        logits, features, keys, values, final = fn(
            tokens,
            append.positions,
            append.destinations,
            pages,
            append.keys,
            append.values,
            recurrent,
        )
        append.install(keys, values)
        for group, starts, output in zip(slots, initial, final, strict=True):
            ends = write_batch(group, output)
            for slot, start, end in zip(group, starts, ends, strict=True):
                slot.stage(RecurrentBoundaries(start, end, 1))
        return ModelOutput(logits[0] if logits else None, features)
