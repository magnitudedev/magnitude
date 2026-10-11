"""Chat requests over contiguous book passages, independent of session scheduling.

Two workloads share one sizing policy. ``prose-continue`` asks for new prose after the
passage; its output varies by engine and numerics. ``prose-repeat`` asks for a
paragraph-aligned suffix near the end, sized to cover the output budget.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from pydantic import JsonValue

from .contexts import Context, Counter, PreparedContext
from .interactions import Interaction
from .records import digest

PASSAGE_WORDS = 256
CONTINUATION_WORDS = 128
COUNT_PREFIX = "Supplied text:\n\n"
INSTRUCTION = "Continue the passage below in prose. Return only the continuation.\n\n"
REPEAT_INSTRUCTION = (
    'Starting with “{opening}”, repeat the supplied passage verbatim to the end. '
    "Output only the passage text."
)
REPEAT_START = "Call me Ishmael."
# Normalize typography and intra-paragraph whitespace, retaining paragraph boundaries.
STRAIGHT_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

ProseWorkload = Literal["prose-continue", "prose-repeat"]


@dataclass(frozen=True)
class Prose:
    text: str
    provenance: dict[str, str]
    workload: ProseWorkload = "prose-continue"

    @property
    def repeating(self) -> bool:
        return self.workload == "prose-repeat"

    @cached_property
    def body(self) -> str:
        if self.repeating:
            return "\n\n".join(
                " ".join(paragraph.split())
                for paragraph in re.split(r"\n\s*\n", self.text.translate(STRAIGHT_QUOTES))
                if paragraph.strip()
            )
        return self.text

    @cached_property
    def word_ends(self) -> tuple[int, ...]:
        return (0, *(m.end() for m in re.finditer(r"\S+\s*", self.body)))

    @cached_property
    def paragraph_starts(self) -> tuple[int, ...]:
        return (0, *(bisect_left(self.word_ends, m.end()) for m in re.finditer(r"\n\n", self.body)))

    @cached_property
    def start_word(self) -> int:
        if not self.repeating:
            return 0
        return self.word_ends.index(self.body.index(REPEAT_START))

    @property
    def fixture(self) -> str:
        return "prose.moby-dick.repeat" if self.repeating else "prose.moby-dick"

    @property
    def identity(self) -> str:
        if self.repeating:
            return digest({**self.provenance, "workload": self.workload, "recipe": "prose-repeat-history-v2"})
        return digest(self.provenance)


class ProseHistory:
    def __init__(self, source: Prose, identity: str, *, output_tokens: int = 256):
        if output_tokens < 1:
            raise ValueError("prose output budget must be positive")
        self.output_tokens = output_tokens
        self.copy_start = 0
        self.source, self.identity = source, identity
        self.ends = source.word_ends
        self.cursor = source.start_word
        self.messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": f"Reading session {identity}."}
        ]
        self.pending: Context | None = None
        self.start = self.cursor
        self.end = self.cursor
        self.current = Interaction(
            id=source.fixture,
            category="prose",
            messages=[],
            tools=[],
            expected=[],
            provenance=source.provenance,
        )

    def passage(self, start: int, end: int) -> str:
        text = self.source.body[self.ends[start] : self.ends[end]]
        return text.rstrip() if self.source.repeating else text

    async def prepare(self, target: int, counter: Counter, sizing_identity: str) -> PreparedContext:
        if target < 0:
            raise ValueError("context target cannot be negative")
        reserve = 0 if self.source.repeating else CONTINUATION_WORDS
        available = len(self.ends) - 1 - self.cursor - reserve
        if available < (1 if self.source.repeating else PASSAGE_WORDS):
            raise ValueError("Moby Dick has no remaining passage and canonical continuation")
        def copy_context(suffix: str) -> Context:
            # Counting APIs may require a user turn even for an assistant suffix.
            # The identical prelude is subtracted with the prefix-only framing.
            return Context(messages=[
                {"role": "user", "content": "Continue the supplied text."},
                {"role": "assistant", "content": COUNT_PREFIX + suffix},
            ])

        framing = 0
        if self.source.repeating:
            framing = await counter(copy_context(""))

        async def evaluate(words: int) -> tuple[Context, int, int, int, bool]:
            end = self.cursor + words
            copy_start, copy_tokens = self.cursor, 0
            if self.source.repeating:
                starts = [self.cursor, *(p for p in self.source.paragraph_starts if self.cursor < p < end)]
                for copy_start in reversed(starts):
                    suffix = self.passage(copy_start, end)
                    copy_tokens = await counter(copy_context(suffix)) - framing
                    if copy_tokens >= self.output_tokens:
                        break
                opening = " ".join(self.passage(copy_start, end).split()[:6])
                added = [
                    {"role": "assistant", "content": self.passage(self.cursor, end)},
                    {"role": "user", "content": REPEAT_INSTRUCTION.format(opening=opening)},
                ]
            else:
                added = [{"role": "user", "content": INSTRUCTION + self.passage(self.cursor, end)}]
            context = Context(messages=[*self.messages, *added])
            count = await counter(context)
            if type(count) is not int or count < 1:
                raise ValueError("context renderer returned an invalid token count")
            sufficient = not self.source.repeating or copy_tokens >= self.output_tokens
            return context, count, copy_start, copy_tokens, sufficient

        low, high = (0 if self.source.repeating else PASSAGE_WORDS - 1), min(PASSAGE_WORDS, available)
        prepared = await evaluate(high)
        while prepared[1] < target or not prepared[4]:
            if high == available:
                raise ValueError("context or output budget exceeds the remaining Moby Dick text")
            low, high = high, min(available, high * 2)
            prepared = await evaluate(high)
        while high - low > 1:
            middle = (low + high) // 2
            candidate = await evaluate(middle)
            if candidate[1] >= target and candidate[4]:
                high, prepared = middle, candidate
            else:
                low = middle
        context, count, self.copy_start, copy_tokens, _ = prepared
        self.start = self.cursor
        self.end = self.cursor + high
        self.pending = context
        self.current = self.current.model_copy(
            update={"messages": context.messages[-2:] if self.source.repeating else context.messages[-1:]}
        )
        return PreparedContext(
            content=context,
            tokens=count,
            provenance={
                **self.source.provenance,
                "fixture": self.source.fixture,
                "recipe": (
                    "prose-repeat-history-v2" if self.source.repeating else "prose-chat-history-v1"
                ),
                "history": self.identity,
                "passage_start_word": self.cursor,
                "passage_end_word": self.end,
                **({
                    "copy_start_word": self.copy_start,
                    "copy_available_tokens": copy_tokens,
                    "copy_output_tokens": self.output_tokens,
                    "copy_count_basis": "prefixed assistant suffix minus prefix-only framing",
                } if self.source.repeating else {"canonical_continuation_words": CONTINUATION_WORDS}),
                "requested_context_tokens": target,
                "actual_context_tokens": count,
                "sizing_identity": sizing_identity,
                "content_digest": digest(context.model_dump(mode="json")),
            },
        )

    def complete(self) -> None:
        if self.pending is None:
            raise ValueError("prose history has no prepared request")
        # Canonical history uses source text, independently of measured output.
        answer = self.copy_start if self.source.repeating else self.end
        answer_end = self.end if self.source.repeating else answer + CONTINUATION_WORDS
        self.messages = [
            *self.pending.messages,
            {
                "role": "assistant",
                "content": self.passage(answer, answer_end),
            },
        ]
        self.cursor = self.end if self.source.repeating else self.end + CONTINUATION_WORDS
        self.pending = None
