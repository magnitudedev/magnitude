"""Explicit Qwen input semantics over the upstream decoder, without shared request state."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm.models.qwen3_5.language import LanguageModel

from magnitude_engine.models.embeddings.replacement import replace
from magnitude_engine.models.inputs import ModelInputs

from .inputs import QwenInputs, batch_positions


@dataclass(frozen=True)
class QwenForward:
    model: LanguageModel

    def forward_inputs(
        self, inputs: tuple[ModelInputs, ...], cache: list, offsets: tuple[int, ...]
    ):
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([row.tokens for row in inputs])
        )
        positions = batch_positions(inputs, offsets)
        if positions is None:
            positions = mx.array(offsets, mx.int32)
        if positions.ndim == 1:
            positions = positions[:, None] + mx.arange(tokens.shape[1], dtype=mx.int32)[None]
        embeddings = self.model.model.embed_tokens(tokens)
        replacements = tuple(
            row.data.embeddings if isinstance(row.data, QwenInputs) else () for row in inputs
        )
        if any(replacements):
            embeddings = replace(embeddings, replacements)
        hidden = self.model.model(
            tokens, inputs_embeds=embeddings, cache=cache, position_ids=positions
        )
        return (
            self.model.model.embed_tokens.as_linear(hidden)
            if self.model.args.tie_word_embeddings
            else self.model.lm_head(hidden)
        )
