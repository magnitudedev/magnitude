"""Gemma decoder composition with explicit, rectangular local image visibility."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.gemma4.language import LanguageModel

from magnitude_engine.models.embeddings.replacement import replace

from .inputs import GemmaInputs, batch_key_ends


@dataclass(frozen=True)
class GemmaForward:
    model: LanguageModel

    def forward_inputs(self, inputs, cache, offsets):
        if not any(row.data is not None for row in inputs):
            tokens = (
                inputs[0].tokens if len(inputs) == 1 else mx.concatenate([r.tokens for r in inputs])
            )
            return self.model(tokens, cache=cache).logits
        if any(row.data is not None and not isinstance(row.data, GemmaInputs) for row in inputs):
            raise ValueError("Gemma native program requires Gemma input operands")
        tokens = mx.concatenate([row.tokens for row in inputs])
        language = mx.concatenate(
            [
                row.data.language
                if row.data is not None
                else mx.ones_like(row.tokens, dtype=mx.bool_)
                for row in inputs
            ]
        )
        ends = batch_key_ends(inputs, offsets)
        inner = self.model.model
        hidden = inner.embed_tokens(tokens) * inner.embed_scale
        hidden = replace(hidden, tuple(row.data.embeddings if row.data else () for row in inputs))
        per_layer = None
        if inner.hidden_size_per_layer_input:
            lookup = inner.get_per_layer_inputs(mx.where(language, tokens, 0))
            per_layer = inner.project_per_layer_inputs(hidden, lookup)
        caches = cache + [None] * (len(inner.layers) - len(cache))
        masks = {}
        for layer, state in zip(inner.layers, caches, strict=True):
            if layer.layer_type in masks:
                continue
            window = inner.window_size if layer.layer_type == "sliding_attention" else None
            mask = create_attention_mask(hidden, state, window_size=window, return_array=True)
            if window is not None and ends is not None:
                if not isinstance(mask, mx.array):
                    raise ValueError("Gemma image visibility requires an explicit cache mask")
                count = tokens.shape[1]
                # Cache masks describe their own packed key coordinates. Vision
                # only adds future keys in this same atomic forward; past keys
                # retain the cache's existing window and padding visibility.
                if hasattr(state, "positions"):
                    keys = mx.arange(mask.shape[-1])[None, None]
                    queries = (
                        mx.array(offsets, mx.int32)[:, None, None] + mx.arange(count)[None, :, None]
                    )
                else:
                    keys = mx.arange(mask.shape[-1])[None, None] - (mask.shape[-1] - count)
                    queries = mx.arange(count)[None, :, None]
                distances = keys - queries
                additional = (distances > 0) & (
                    distances
                    < (ends - (mx.array(offsets, mx.int32)[:, None] + mx.arange(count)[None]))[
                        :, :, None
                    ]
                )
                mask = mask | additional[:, None]
            masks[layer.layer_type] = mask
        intermediates: list[Any] = [(None, None)] * len(inner.layers)
        for index, (layer, state, previous) in enumerate(
            zip(inner.layers, caches, inner.previous_kvs, strict=True)
        ):
            keys, offset = intermediates[previous]
            hidden, keys, offset = layer(
                hidden,
                masks[layer.layer_type],
                state,
                per_layer_input=None if per_layer is None else per_layer[:, :, index],
                shared_kv=keys,
                offset=offset,
            )
            intermediates[index] = keys, offset
        hidden = inner.norm(hidden)
        if self.model.config.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(hidden)
        else:
            assert self.model.lm_head is not None
            logits = self.model.lm_head(hidden)
        cap = self.model.final_logit_softcapping
        return mx.tanh(logits / cap) * cap if cap is not None else logits
