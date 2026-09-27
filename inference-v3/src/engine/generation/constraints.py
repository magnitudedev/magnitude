"""Worker-owned constraint vocabulary over the engine's exact byte BPE contract."""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter_ns
from typing import Literal

from llguidance import LLMatcher, LLParserLimits, LLTokenizer, TokenizerWrapper
from pydantic import Field

from engine.data import Record, TokenId
from engine.inputs.tokenizer import BPEConfig, ByteBPETokenizer, PieceKind, SpecialTokens
from engine.weights.identity import ArtifactIdentity
from templates.grammar import CONVERTER_SHA256, to_lark


class ConstraintCompilation(Record):
    conversion_ns: int = 0
    compilation_ns: int = 0
    prefix_ns: int = 0
    initial_mask_ns: int = 0
    cache_hit: bool = False


class ConstraintMetrics(Record):
    preparation: ConstraintCompilation
    mask_ns: int
    mask_calls: int
    mask_bytes: int
    transition_ns: int
    forced_query_ns: int
    forced_queries: int


class ConstraintPlan(Record):
    """Immutable symbolic input to worker admission; contains no live owners."""

    version: Literal[1] = 1
    dialect: Literal["gbnf"] = "gbnf"
    artifact_identity: ArtifactIdentity
    template_identity: str = Field(min_length=1)
    grammar: str = Field(min_length=1, max_length=8 * 1024 * 1024)
    initial_prefix: str = Field(default="", max_length=1024 * 1024)
    converter_sha256: str = CONVERTER_SHA256


class _Vocabulary:
    def __init__(self, tokenizer: ByteBPETokenizer, projection_vocabulary: int):
        self.tokenizer = tokenizer
        self.eos_token_id = min(tokenizer.stop_tokens)
        self.bos_token_id = None
        self.tokens = tuple(
            b""
            if kind == PieceKind.UNUSED
            or tokenizer.piece(TokenId(i), skip_control=False).startswith(b"\xff")
            else tokenizer.piece(TokenId(i), skip_control=False)
            for i, kind in enumerate(tokenizer.config.kinds)
        ) + (b"",) * (projection_vocabulary - tokenizer.vocabulary)
        # GBNF describes visible completion bytes, including control delimiters.
        # Keep these pieces literal in llguidance; only EOS means termination.
        # Engine token IDs/kinds still govern encoding and publication separately.
        self.special_token_ids = tuple(sorted(tokenizer.stop_tokens))

    def __call__(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, special=SpecialTokens.RECOGNIZE))


class ConstraintVocabulary:
    """One immutable vocabulary binding; each request gets a separate matcher."""

    def __init__(
        self,
        config: BPEConfig,
        *,
        projection_vocabulary: int,
        cache_entries: int = 8,
        cache_bytes: int = 8 * 1024 * 1024,
    ):
        if cache_entries < 0 or cache_bytes < 0:
            raise ValueError("constraint cache bounds cannot be negative")
        self.cache_entries, self.cache_limit_bytes = cache_entries, cache_bytes
        self.cache_bytes = self.cache_hits = self.cache_misses = 0
        self._compiled: OrderedDict[tuple[str, str], tuple[LLMatcher, int, bytes]] = OrderedDict()
        self.limits = LLParserLimits(
            max_lexer_states=8192, max_grammar_size=100_000, verbose_errors=False
        )
        if type(projection_vocabulary) is not int or projection_vocabulary < len(config.pieces):
            raise ValueError("model projection is smaller than the tokenizer vocabulary")
        if not config.stop_tokens:
            raise ValueError("constrained generation requires explicit EOS identities")
        if any(config.kinds[token] == PieceKind.UNUSED for token in config.stop_tokens):
            raise ValueError("EOS cannot be an unused token")
        self.tokenizer = ByteBPETokenizer(config)
        self.projection_vocabulary = projection_vocabulary
        # 0xFF cannot occur in valid UTF-8. llguidance reserves it as a
        # special-token marker; prohibit such pieces instead of reinterpreting
        # their bytes as a different token. Partial valid UTF-8 stays untouched.
        self.unused = frozenset(
            i
            for i, kind in enumerate(config.kinds)
            if kind == PieceKind.UNUSED
            or self.tokenizer.piece(TokenId(i), skip_control=False).startswith(b"\xff")
        )
        self._mask_bytes = ((projection_vocabulary + 31) // 32) * 4
        usable = bytearray(b"\xff" * ((self.tokenizer.vocabulary + 7) // 8))
        for token in self.unused:
            usable[token // 8] &= ~(1 << (token % 8))
        if self.tokenizer.vocabulary % 8:
            usable[-1] &= (1 << (self.tokenizer.vocabulary % 8)) - 1
        self._restricted_bytes = tuple(
            (index, value) for index, value in enumerate(usable) if value != 255
        )
        self._tail_offset = len(usable)
        self._padding = bytes(self._mask_bytes - len(usable))
        self._usable_bits = int.from_bytes(usable, "little") & (
            (1 << self.tokenizer.vocabulary) - 1
        )
        self.native = LLTokenizer(
            TokenizerWrapper(_Vocabulary(self.tokenizer, projection_vocabulary)),
            n_vocab=projection_vocabulary,
            eos_token=sorted(config.stop_tokens),
        )
        if self.native.vocab_size != projection_vocabulary:
            raise ValueError("constraint vocabulary changed the model projection size")
        if set(self.native.eos_tokens) != config.stop_tokens:
            raise ValueError("constraint vocabulary changed EOS identities")

    def bind(self, plan: ConstraintPlan) -> ConstraintState:
        if plan.artifact_identity != self.tokenizer.config.artifact_identity:
            raise ValueError("constraint plan belongs to another model artifact")
        if plan.converter_sha256 != CONVERTER_SHA256:
            raise ValueError("constraint plan requires an incompatible grammar converter")
        return ConstraintState(self, plan.grammar, initial_prefix=plan.initial_prefix)

    def matcher(self, gbnf: str, *, initial_prefix: str = "") -> LLMatcher:
        return self._prepare_matcher(gbnf, initial_prefix=initial_prefix)[0]

    def _prepare_matcher(
        self, gbnf: str, *, initial_prefix: str = ""
    ) -> tuple[LLMatcher, ConstraintCompilation, bytes]:
        key = (gbnf, initial_prefix)
        if key in self._compiled:
            self.cache_hits += 1
            self._compiled.move_to_end(key)
            prototype, _, mask = self._compiled[key]
            return prototype.deep_copy(), ConstraintCompilation(cache_hit=True), mask
        self.cache_misses += 1
        started = perf_counter_ns()
        grammar = LLMatcher.grammar_from_lark(to_lark(gbnf))
        converted = perf_counter_ns()
        result = LLMatcher(self.native, grammar, log_level=0, limits=self.limits)
        compiled = perf_counter_ns()
        if result.is_error():
            raise ValueError(f"Constraint compilation failed: {result.get_error()}")
        if result.get_grammar_warnings():
            raise ValueError(f"Constraint compilation warnings: {result.get_grammar_warnings()}")
        prefix = self.tokenizer.encode(initial_prefix)
        if (
            b"".join(self.tokenizer.piece(token, skip_control=False) for token in prefix)
            != initial_prefix.encode()
        ):
            raise ValueError("grammar prefix is not exactly representable by the tokenizer")
        if not result.consume_tokens(list(prefix)):
            raise ValueError("grammar does not accept its declared generation prefix")
        prefixed = perf_counter_ns()
        initial_mask = self.allowed_mask(result)
        masked = perf_counter_ns()
        size = len(gbnf.encode()) + len(initial_prefix.encode()) + len(grammar.encode()) + len(initial_mask)
        if self.cache_entries and size <= self.cache_limit_bytes:
            while self._compiled and (
                len(self._compiled) >= self.cache_entries
                or self.cache_bytes + size > self.cache_limit_bytes
            ):
                _, (_, removed, _) = self._compiled.popitem(last=False)
                self.cache_bytes -= removed
            self._compiled[key] = (result.deep_copy(), size, initial_mask)
            self.cache_bytes += size
        return result, ConstraintCompilation(
            conversion_ns=converted - started,
            compilation_ns=compiled - converted,
            prefix_ns=prefixed - compiled,
            initial_mask_ns=masked - prefixed,
        ), initial_mask

    def allowed_mask(self, matcher: LLMatcher) -> bytes:
        """Packed little-endian bits over projection IDs, with unusable IDs cleared."""
        if matcher.is_error():
            raise ValueError(f"Constraint matcher failed: {matcher.get_error()}")
        mask = matcher.compute_bitmask()
        if matcher.is_error():
            raise ValueError(f"Constraint mask failed: {matcher.get_error()}")
        if len(mask) < self._mask_bytes:
            raise ValueError("constraint mask does not cover the model projection")
        if len(self._restricted_bytes) <= 16:
            # Real vocabularies usually have very few unusable byte positions.
            # Verify those positions and the padding first; an already valid
            # native mask can pass through without a whole-vocabulary conversion.
            clean = all(mask[index] & ~usable == 0 for index, usable in self._restricted_bytes)
            if clean and mask[self._tail_offset:self._mask_bytes] == self._padding:
                return mask[:self._mask_bytes]
            filtered = bytearray(mask[:self._mask_bytes])
            for index, usable in self._restricted_bytes:
                filtered[index] &= usable
            filtered[self._tail_offset:] = self._padding
            return bytes(filtered)
        # One bulk host bit operation clears padding and unusable identities;
        # no Python per-token scan belongs on the decode path.
        return (int.from_bytes(mask, "little") & self._usable_bits).to_bytes(
            self._mask_bytes, "little"
        )


class ConstraintTransition:
    """Validated speculative state; discarded transitions have no observable effect."""

    def __init__(self, owner: ConstraintState, candidate: LLMatcher, count: int, terminal: bool):
        self._owner = owner
        self._before = owner._matcher
        self._candidate = candidate
        self._count = count
        self._terminal = terminal
        self._committed = False

    def check(self) -> None:
        if self._committed or self._owner._matcher is not self._before:
            raise RuntimeError("constraint transition is stale or already committed")

    def commit(self) -> None:
        self.check()
        self._owner._matcher = self._candidate
        self._owner._mask = None
        self._owner.position += self._count
        self._owner._terminal = self._terminal
        self._committed = True


class ConstraintState:
    """Accepted grammar progress, independent of numerical residency and replay."""

    _mask: bytes | None

    def __init__(self, vocabulary: ConstraintVocabulary, grammar: str, *, initial_prefix: str = ""):
        self.vocabulary = vocabulary
        self._matcher, self.compilation, self._mask = vocabulary._prepare_matcher(
            grammar, initial_prefix=initial_prefix
        )
        self.position = 0
        self._terminal = False
        self.mask_ns = self.mask_calls = self.mask_bytes = 0
        self.transition_ns = self.forced_query_ns = self.forced_queries = 0

    @property
    def metrics(self) -> ConstraintMetrics:
        return ConstraintMetrics(
            preparation=self.compilation,
            mask_ns=self.mask_ns, mask_calls=self.mask_calls, mask_bytes=self.mask_bytes,
            transition_ns=self.transition_ns,
            forced_query_ns=self.forced_query_ns, forced_queries=self.forced_queries,
        )

    @property
    def accepting(self) -> bool:
        return self._matcher.is_accepting()

    @property
    def stopped(self) -> bool:
        return self._terminal

    def mask(self) -> bytes:
        started = perf_counter_ns()
        try:
            if self._mask is None:
                self._mask = self.vocabulary.allowed_mask(self._matcher)
            result = self._mask
            self.mask_bytes += len(result)
            return result
        finally:
            self.mask_ns += perf_counter_ns() - started
            self.mask_calls += 1

    def stage(self, tokens: tuple[TokenId, ...]) -> ConstraintTransition:
        started = perf_counter_ns()
        try:
            return self._stage(tokens)
        finally:
            self.transition_ns += perf_counter_ns() - started

    def _stage(self, tokens: tuple[TokenId, ...]) -> ConstraintTransition:
        if not tokens:
            raise ValueError("constraint transition requires tokens")
        if self.stopped:
            raise ValueError("constraint matcher is already stopped")
        for token in tokens:
            if (
                type(token) is not int
                or not 0 <= token < self.vocabulary.tokenizer.vocabulary
                or token in self.vocabulary.unused
            ):
                raise ValueError("constraint transition contains an unusable token")
        if any(token in self.vocabulary.tokenizer.stop_tokens for token in tokens[:-1]):
            raise ValueError("constraint transition continues after EOS")
        candidate = self._matcher.deep_copy()
        terminal = tokens[-1] in self.vocabulary.tokenizer.stop_tokens
        content = [int(token) for token in (tokens[:-1] if terminal else tokens)]
        if content and (
            candidate.validate_tokens(content) != len(content)
            or not candidate.consume_tokens(content)
        ):
            raise ValueError("tokens violate the prepared constraint")
        if terminal:
            # validate_tokens treats an exhausted finite grammar differently
            # from consume_token. EOS admission is its accepting state plus the
            # exact EOS bit in the next-token mask, not exhaustion alone.
            mask = self.vocabulary.allowed_mask(candidate)
            eos = tokens[-1]
            if not candidate.is_accepting() or not mask[eos // 8] & (1 << (eos % 8)):
                raise ValueError("EOS violates the prepared constraint")
            if not candidate.consume_token(eos):
                raise ValueError("Constraint EOS transition failed")
        if candidate.is_error():
            raise ValueError(f"Constraint transition failed: {candidate.get_error()}")
        return ConstraintTransition(self, candidate, len(tokens), terminal)

    def forced(self, limit: int) -> tuple[TokenId, ...]:
        """A bounded grammar proposal. Neither history nor matcher is advanced."""
        started = perf_counter_ns()
        try:
            return self._forced(limit)
        finally:
            self.forced_query_ns += perf_counter_ns() - started
            self.forced_queries += 1

    def _forced(self, limit: int) -> tuple[TokenId, ...]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("forced-token allowance must be positive")
        if self.stopped:
            return ()
        tokens = self._matcher.compute_ff_tokens()[:limit]
        if self._matcher.is_error():
            raise ValueError(f"Constraint forced-token query failed: {self._matcher.get_error()}")
        for index, token in enumerate(tokens):
            if token in self.vocabulary.tokenizer.stop_tokens:
                tokens = tokens[:index]
                break
        result = tuple(TokenId(token) for token in tokens)
        if result:
            self.stage(result)  # Reject unusable IDs before proposing model work.
        return result

    def fork(self) -> ConstraintState:
        result = object.__new__(ConstraintState)
        # Only the matcher is mutable shared state; other fields are counters,
        # immutable records, or the intentionally shared vocabulary owner.
        result.__dict__ = self.__dict__.copy()
        result._matcher = self._matcher.deep_copy()
        return result
