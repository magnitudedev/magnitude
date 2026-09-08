"""Resident Qwen tensor transitions; state authority remains outside compilation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.attention.contracts import DecodeAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.experts.computation import ResidentExperts
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.decode import DecodeKV, prepare_decode_append
from magnitude_engine.models.state.hybrid import HybridState
from magnitude_engine.models.state.recurrent import read_batch, stage_boundaries

from .attention.operation import GatedAttention
from .definition import DEFINITION
from .feedforward.operation import DenseFeedForward, RoutedFeedForward
from .recurrence.operation import RecurrentMixer

if TYPE_CHECKING:
    from .program import Qwen35Program


@component("MODEL:QWEN35:MAG:RESIDENT_COMPILED", model=DEFINITION)
class ResidentDecode:
    def __init__(self, program: Qwen35Program):
        self.embedding, self.blocks = program.embedding, program.blocks
        self.norm, self.output = program.norm, program.output
        self.functions: OrderedDict[tuple, Callable[..., Any]] = OrderedDict()

    def matches(self, program: Qwen35Program) -> bool:
        return all(
            a is b
            for a, b in zip(
                (self.embedding, self.blocks, self.norm, self.output),
                (program.embedding, program.blocks, program.norm, program.output),
                strict=True,
            )
        )

    @staticmethod
    def supports(program: Qwen35Program) -> bool:
        if not isinstance(program.embedding, (ResidentEmbedding, ResidentAffineEmbedding)):
            return False
        for block in program.blocks:
            mixer, feedforward = block.mixer, block.feedforward
            if isinstance(mixer, GatedAttention):
                if not isinstance(mixer.attention, DecodeAttention) or mixer.head_width not in (
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
        from .program import evaluate

        embedding, blocks, norm, output = self.embedding, self.blocks, self.norm, self.output
        assert isinstance(embedding, (ResidentEmbedding, ResidentAffineEmbedding))

        def step(
            tokens,
            positions,
            offsets,
            pages,
            keys,
            values,
            tails,
            starts,
            recurrent,
            rotary_positions=None,
        ):
            kv = DecodeKV(
                page_size, table_width, positions, offsets, pages, keys, values, list(tails), starts
            )
            next_recurrent = list(recurrent)

            def mix(mixer, hidden):
                if isinstance(mixer, GatedAttention):
                    return mixer.decode(hidden, kv, rotary_positions)
                assert isinstance(mixer, RecurrentMixer)
                value, conv, memory = mixer.operation.graph.advance(
                    hidden, *next_recurrent[mixer.index]
                )
                next_recurrent[mixer.index] = (conv, memory)
                return value

            def feed(feedforward, hidden):
                if isinstance(feedforward, RoutedFeedForward):
                    assert isinstance(feedforward.experts, ResidentExperts)
                    return feedforward.apply(hidden, feedforward.experts)
                assert isinstance(feedforward, DenseFeedForward)
                return feedforward(hidden)

            logits, features = evaluate(
                embedding(tokens),
                blocks=blocks,
                norm=norm,
                output=output,
                mix=mix,
                feed=feed,
                request=request,
            )
            return logits, features, tuple(kv.tails), tuple(next_recurrent)

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
        scope: ExecutionScope,
        positions: mx.array | None = None,
    ) -> ModelOutput:
        append = prepare_decode_append(tuple(state.pages for state in states), scope)
        width, pages = append.padded_table()
        slots = tuple(
            tuple(state.slots[index] for state in states) for index in range(len(states[0].slots))
        )
        recurrent = tuple(read_batch(group) for group in slots)
        fn = self._function(append.page_size, width, append.capacity, len(states), request)
        logits, features, tails, final = fn(
            tokens,
            append.positions,
            append.offsets,
            pages,
            append.keys,
            append.values,
            append.tails,
            append.tail_starts,
            recurrent,
            *((positions,) if positions is not None else ()),
        )
        append.install(tails)
        for group, output in zip(slots, final, strict=True):
            stage_boundaries(group, output, 1)
        return ModelOutput(logits[0] if logits else None, features)
