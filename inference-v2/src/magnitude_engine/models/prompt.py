"""Semantic decoder history and legal advancement, independent of device storage.

Input adapters describe exceptional spans. Consumers use their identity and valid
boundaries without interpreting an architecture, a modality, or special token IDs.
Ordinary text needs no spans and retains its token-exact advancement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSpan:
    start: int
    end: int
    identity: bytes
    indivisible: bool = False
    language: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
            or not isinstance(self.identity, bytes)
            or not self.identity
            or type(self.indivisible) is not bool
            or type(self.language) is not bool
        ):
            raise ValueError("input span requires a nonempty range and semantic identity")


@dataclass(frozen=True)
class Prompt:
    tokens: tuple[int, ...]
    spans: tuple[InputSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or any(
            type(token) is not int or not 0 <= token < 2**31 for token in self.tokens
        ):
            raise ValueError("prompt tokens must be nonnegative int32 IDs")
        if not isinstance(self.spans, tuple):
            raise ValueError("prompt spans must be an immutable tuple")
        end = 0
        for span in self.spans:
            if not isinstance(span, InputSpan) or span.start < end or span.end > len(self.tokens):
                raise ValueError("input spans must be ordered, disjoint, and inside the prompt")
            end = span.end

    def boundary(self, position: int) -> bool:
        """Whether this prefix is independent of input beyond its boundary."""
        if type(position) is not int or not 0 <= position <= len(self.tokens):
            return False
        return not any(span.indivisible and span.start < position < span.end for span in self.spans)

    def advance(self, start: int, allowance: int, *, end: int | None = None) -> int:
        """Choose a legal end, extending a soft allowance only for an indivisible unit.

        Divisible spans do not require execution boundaries. If an indivisible
        span is the next unit, consume it whole; shrinking must not starve it.
        A caller can request a smaller allowance to retain a specific legal prefix.
        Hard resource/input limits are validated before this selection.
        """
        end = len(self.tokens) if end is None else end
        if not self.boundary(start) or not self.boundary(end) or start > end:
            raise ValueError("prompt advancement requires ordered legal boundaries")
        if type(allowance) is not int or allowance < 1:
            raise ValueError("prompt allowance must be positive")
        proposed = min(end, start + allowance)
        for span in self.spans:
            if span.indivisible and span.start < proposed < span.end:
                return span.start if span.start > start else span.end
        return proposed

    @property
    def retention_boundaries(self) -> tuple[int, ...]:
        """Useful independent prefixes, without prescribing execution chunking."""
        return tuple(sorted({p for span in self.spans for p in (span.start, span.end)}))

    @property
    def anchor_start(self) -> int:
        """Start of the final legal unit whose output can predict the first token."""
        if not self.tokens:
            raise ValueError("generation requires a nonempty prompt")
        last = len(self.tokens) - 1
        for span in self.spans:
            if span.indivisible and span.start <= last < span.end:
                return span.start
        return last

    def prefix(self, length: int) -> "Prompt":
        if not self.boundary(length):
            raise ValueError("cannot retain an input prefix with unresolved dependencies")
        spans = tuple(
            InputSpan(s.start, min(s.end, length), s.identity, s.indivisible, s.language)
            for s in self.spans
            if s.start < length
        )
        return Prompt(self.tokens[:length], spans)

    def extend(self, tokens: tuple[int, ...]) -> "Prompt":
        """Append actual continuation tokens without redefining earlier input semantics."""
        return Prompt((*self.tokens, *tokens), self.spans)

    def identities(self) -> tuple[tuple[int, bytes], ...]:
        identities = [(token, b"") for token in self.tokens]
        for span in self.spans:
            for position in range(span.start, span.end):
                # The relative position prevents treating repeated placeholder IDs
                # at different locations in one feature span as the same input.
                identity = (
                    span.identity
                    + bytes((span.indivisible, span.language))
                    + (position - span.start).to_bytes(8, "little")
                )
                identities[position] = (self.tokens[position], identity)
        return tuple(identities)

    def language_tokens(self, start: int = 0, end: int | None = None) -> tuple[int, ...]:
        end = len(self.tokens) if end is None else end
        if not 0 <= start <= end <= len(self.tokens):
            raise ValueError("language history range is outside the prompt")
        if not self.spans:
            return self.tokens[start:end]
        return tuple(
            token
            for position, token in enumerate(self.tokens[start:end], start)
            if not any(s.start <= position < s.end and not s.language for s in self.spans)
        )
