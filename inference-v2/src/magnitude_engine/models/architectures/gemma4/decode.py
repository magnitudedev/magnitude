"""Gemma's resident tensor step over shared producers and bounded append outputs."""

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
from magnitude_engine.models.state.pages import SequencePages

from .definition import DEFINITION

if TYPE_CHECKING:
    from .program import Gemma4Program


@component("MODEL:GEMMA4:MAG:RESIDENT_COMPILED", model=DEFINITION)
class ResidentDecode:
    def __init__(self, program: Gemma4Program):
        self.embedding, self.embedding_scale = program.embedding, program.embedding_scale
        self.blocks, self.per_layer = program.blocks, program.per_layer
        self.norm, self.output, self.softcap = program.norm, program.output, program.softcap
        self.functions: OrderedDict[tuple, Callable[..., Any]] = OrderedDict()

    def matches(self, program: Gemma4Program) -> bool:
        return all(
            a is b
            for a, b in zip(
                (
                    self.embedding,
                    self.blocks,
                    self.per_layer,
                    self.norm,
                    self.output,
                    self.embedding_scale,
                    self.softcap,
                ),
                (
                    program.embedding,
                    program.blocks,
                    program.per_layer,
                    program.norm,
                    program.output,
                    program.embedding_scale,
                    program.softcap,
                ),
                strict=True,
            )
        )

    @staticmethod
    def supports(program: Gemma4Program) -> bool:
        resident = (ResidentEmbedding, ResidentAffineEmbedding)
        if not isinstance(program.embedding, resident):
            return False
        if program.per_layer is not None and not isinstance(program.per_layer.embedding, resident):
            return False
        for block in program.blocks:
            if not isinstance(block.attention.operation, DecodeAttention):
                return False
            experts = block.feedforward.experts
            if experts is not None and not isinstance(experts.operation, ResidentExperts):
                return False
        return True

    def _function(self, page_size: int, table_width: int, batch: int, request: ForwardRequest):
        feature_names = tuple(sorted(request.features))
        key = (page_size, table_width, batch, request.logits, feature_names)
        if key in self.functions:
            self.functions.move_to_end(key)
            return self.functions[key]
        from .program import evaluate

        embedding, per_layer, blocks = self.embedding, self.per_layer, self.blocks
        norm, output, softcap, scale = self.norm, self.output, self.softcap, self.embedding_scale
        assert isinstance(embedding, (ResidentEmbedding, ResidentAffineEmbedding))

        def step(tokens, positions, offsets, pages, keys, values, tails, starts):
            kv = DecodeKV(
                page_size, table_width, positions, offsets, pages, keys, values, list(tails), starts
            )
            lookup = None
            if per_layer is not None:
                assert isinstance(per_layer.embedding, (ResidentEmbedding, ResidentAffineEmbedding))
                lookup = per_layer.embedding(tokens)

            def feed(feedforward, hidden):
                branch = feedforward.experts
                experts = branch.operation if branch is not None else None
                assert experts is None or isinstance(experts, ResidentExperts)
                return feedforward.apply(hidden, experts)

            logits, features = evaluate(
                embedding(tokens),
                lookup,
                blocks=blocks,
                embedding_scale=scale,
                per_layer=per_layer,
                norm=norm,
                output=output,
                softcap=softcap,
                attend=lambda attention, hidden: attention.decode(hidden, kv),
                feed=feed,
                request=request,
            )
            return logits, features, tuple(kv.tails)

        compiled = mx.compile(step)
        self.functions[key] = compiled
        if len(self.functions) > 4:
            self.functions.popitem(last=False)
        return compiled

    def forward(
        self,
        tokens: mx.array,
        states: tuple[SequencePages, ...],
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        append = prepare_decode_append(states, scope)
        width, pages = append.padded_table()
        function = self._function(append.page_size, width, len(states), request)
        logits, features, tails = function(
            tokens,
            append.positions,
            append.offsets,
            pages,
            append.keys,
            append.values,
            append.tails,
            append.tail_starts,
        )
        append.install(tails)
        return ModelOutput(logits[0] if logits else None, features)
