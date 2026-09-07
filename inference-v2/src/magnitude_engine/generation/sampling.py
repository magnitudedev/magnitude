"""Position-addressed sampling shared by ordinary and speculative advancement."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.components import component

from .sampling_policy import SamplingPolicy


def position_key(seed: int, position: int) -> mx.array:
    """Preserve the PoC's SplitMix64 seed/position mapping exactly.

    A position receives the same draw regardless of request interleaving, batch
    shape, proposal width, or rejected speculative work at later positions.
    """
    if position < 0:
        raise ValueError("sampling position cannot be negative")
    mask = (1 << 64) - 1
    mixed = (seed * 0x9E3779B97F4A7C15 + (position + 1) * 0xBF58476D1CE4E5B9) & mask
    for shift, multiplier in ((30, 0xBF58476D1CE4E5B9), (27, 0x94D049BB133111EB)):
        mixed = ((mixed ^ (mixed >> shift)) * multiplier) & mask
    return mx.random.key(mixed ^ (mixed >> 31))


@component("GENERATION:SAMPLING:MAG:POSITION_KEYED")
class SequenceSampler:
    def __init__(self, policy: SamplingPolicy):
        self.policy = policy
        self._history: deque[int] = deque(maxlen=policy.history_window)

    def observe(self, tokens: tuple[int, ...]) -> None:
        if self.policy.uses_history:
            self._history.extend(tokens)

    def preview(self, tokens: tuple[int, ...]) -> SequenceSampler:
        preview = SequenceSampler(self.policy)
        preview._history.extend(self._history)
        preview.observe(tokens)
        return preview

    def logits(self, raw: mx.array, preview_tokens: mx.array | None = None) -> mx.array:
        if raw.ndim != 1 or raw.shape[0] == 0:
            raise ValueError("sampling requires one nonempty vocabulary vector")
        policy = self.policy
        # Greedy selection only compares values. Widening BF16/FP16 logits
        # preserves their ordering and adds a redundant full-vocabulary pass.
        # Arithmetic penalties and stochastic normalization still use FP32.
        result = (
            raw if policy.temperature == 0 and not policy.uses_history else raw.astype(mx.float32)
        )
        if preview_tokens is not None and (
            preview_tokens.ndim != 1 or preview_tokens.dtype != mx.int32
        ):
            raise ValueError("sampling preview requires one-dimensional int32 tokens")
        if self.policy.uses_history and (self._history or preview_tokens is not None):
            history = mx.array(list(self._history), dtype=mx.int32)
            if preview_tokens is not None:
                history = mx.concatenate([history, preview_tokens])[-policy.history_window :]
            counts = mx.zeros(result.shape, dtype=mx.float32).at[history].add(1)
            repeated = mx.where(
                result < 0, result * policy.repetition_penalty, result / policy.repetition_penalty
            )
            result = mx.where(counts > 0, repeated, result)
            result -= policy.presence_penalty * (counts > 0) + policy.frequency_penalty * counts
        if policy.temperature == 0:
            return result
        result = result / policy.temperature
        if policy.top_k:
            threshold = mx.sort(result)[-min(policy.top_k, result.shape[0])]
            result = mx.where(result >= threshold, result, -mx.inf)
        if policy.min_p:
            probabilities = mx.softmax(result)
            result = mx.where(probabilities >= policy.min_p * probabilities.max(), result, -mx.inf)
        if policy.top_p < 1:
            order = mx.argsort(-result)
            probabilities = mx.softmax(result[order])
            keep = mx.cumsum(probabilities) - probabilities < policy.top_p
            filtered = mx.full(result.shape, -mx.inf, dtype=result.dtype)
            filtered[order] = mx.where(keep, result[order], -mx.inf)
            result = filtered
        return result

    def sample(
        self, raw: mx.array, position: int, *, preview_tokens: mx.array | None = None
    ) -> mx.array:
        distribution = self.logits(raw, preview_tokens)
        if self.policy.temperature == 0:
            return mx.argmax(distribution).astype(mx.int32)
        key = None if self.policy.seed is None else position_key(self.policy.seed, position)
        return mx.random.categorical(distribution, key=key).astype(mx.int32)


@dataclass(frozen=True)
class TokenLogprobs:
    selected: mx.array
    top_ids: mx.array
    top_values: mx.array


def token_logprobs(raw: mx.array, token: mx.array, top: int = 0) -> TokenLogprobs:
    if top < 0:
        raise ValueError("top-logprob count cannot be negative")
    logits = raw.astype(mx.float32)
    probabilities = logits - mx.logsumexp(logits)
    ids = mx.argsort(-probabilities)[:top]
    return TokenLogprobs(probabilities[token], ids, probabilities[ids])
