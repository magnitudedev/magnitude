"""llguidance adapter: bounded compiled prototypes and isolated live matchers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import llguidance as llg
import mlx.core as mx
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

from .constraint_spec import ConstraintError, ConstraintSpec


class GuidanceConstraint:
    def __init__(self, matcher: llg.LLMatcher, vocabulary: int, eos: tuple[int, ...]):
        self.matcher: llg.LLMatcher | None = matcher
        self.vocabulary, self.eos = vocabulary, eos
        self.mask = allocate_token_bitmask(1, vocabulary)

    def _live(self) -> llg.LLMatcher:
        if self.matcher is None:
            raise RuntimeError("constraint session is closed")
        if self.matcher.is_error():
            raise ConstraintError(f"constraint matcher failed: {self.matcher.get_error()}")
        return self.matcher

    def fork(self) -> GuidanceConstraint:
        return GuidanceConstraint(self._live().deep_copy(), self.vocabulary, self.eos)

    def apply(self, logits: mx.array) -> mx.array:
        if logits.shape != (self.vocabulary,):
            raise ValueError("constraint vocabulary differs from model logits")
        matcher = self._live()
        # A completed language still permits only EOS, never unconstrained continuation.
        if matcher.is_stopped():
            ids = mx.arange(self.vocabulary, dtype=mx.int32)
            allowed = mx.zeros((self.vocabulary,), dtype=mx.bool_)
            for token in self.eos:
                allowed = allowed | (ids == token)
        else:
            fill_next_token_bitmask(matcher, self.mask)
            self._live()
            words = mx.array(self.mask[0]).astype(mx.uint32)
            ids = mx.arange(self.vocabulary, dtype=mx.uint32)
            allowed = ((words[ids // 32] >> (ids % 32)) & 1) != 0
        return mx.where(allowed, logits, -mx.inf)

    def consume(self, token: int) -> bool:
        return self._live().consume_token(token)

    def forced(self) -> tuple[int, ...]:
        matcher = self._live()
        if matcher.is_stopped():
            return ()
        tokens = matcher.compute_ff_tokens()
        self._live()
        # EOS belongs to generation's stopping policy; preserve a prefix, never skip a token.
        end = next((i for i, token in enumerate(tokens) if token in self.eos), len(tokens))
        return tuple(tokens[:end])

    def close(self) -> None:
        self.matcher = None


class GuidanceCompiler:
    def __init__(self, tokenizer: Callable[[], llg.LLTokenizer], capacity: int = 64):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("compiled grammar capacity must be a positive integer")
        self.load_tokenizer = tokenizer
        self.tokenizer: llg.LLTokenizer | None = None
        self.capacity = capacity
        self.prototypes: OrderedDict[str, llg.LLMatcher] = OrderedDict()

    def create(self, spec: ConstraintSpec) -> GuidanceConstraint:
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        prototype = self.prototypes.get(spec.lark)
        if prototype is None:
            grammar = llg.LLMatcher.grammar_from_lark(spec.lark)
            invalid, messages = llg.LLMatcher.validate_grammar_with_warnings(
                grammar, self.tokenizer
            )
            if invalid:
                raise ConstraintError("; ".join(messages))
            prototype = llg.LLMatcher(self.tokenizer, grammar, log_level=0)
            if prototype.is_error():
                raise ConstraintError(prototype.get_error())
            self.prototypes[spec.lark] = prototype
            if len(self.prototypes) > self.capacity:
                self.prototypes.popitem(last=False)
        self.prototypes.move_to_end(spec.lark)
        return GuidanceConstraint(
            prototype.deep_copy(), self.tokenizer.vocab_size, tuple(self.tokenizer.eos_tokens)
        )
