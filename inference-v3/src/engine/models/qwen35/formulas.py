"""Typed navigation over Qwen's actual traced formula occurrences."""

from __future__ import annotations

from dataclasses import dataclass

import ops

from . import equations


@dataclass(frozen=True, slots=True)
class BlockFormulas:
    whole: ops.FormulaHandle
    mixer: ops.FormulaHandle
    feed_forward: ops.FormulaHandle


@dataclass(frozen=True, slots=True)
class DecoderFormulas:
    whole: ops.FormulaHandle
    blocks: tuple[BlockFormulas, ...]
    # State-only final layers really omit the output/FFN formula. Do not invent
    # missing occurrences just to make the display resemble a full decode.
    state_only: tuple[ops.FormulaHandle, ...]

    @classmethod
    def from_trace(cls, formulas: ops.FormulaTree) -> DecoderFormulas:
        roots = formulas.occurrences(equations.decoder)
        if len(roots) != 1:
            raise ValueError("Qwen navigation requires exactly one traced decoder")
        root, = roots
        blocks = []
        for block in root.occurrences(equations.block):
            mixers = (*block.occurrences(equations.attention_mixer),
                      *block.occurrences(equations.recurrent_mixer))
            feed_forward = (*block.occurrences(equations.dense_feedforward),
                            *block.occurrences(equations.routed_feedforward))
            if len(mixers) != 1 or len(feed_forward) != 1:
                raise ValueError("traced Qwen block must contain one mixer and one feed-forward formula")
            blocks.append(BlockFormulas(block, mixers[0], feed_forward[0]))
        state_definitions = {equations._attention_state.ref, equations._recurrent_state.ref}
        state_only = tuple(child for child in root.children if child.definition in state_definitions)
        return cls(root, tuple(blocks), state_only)
