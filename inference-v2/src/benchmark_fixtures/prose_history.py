"""Chat continuations of contiguous book passages, independent of session scheduling."""

import re
from dataclasses import dataclass
from functools import cached_property

from pydantic import JsonValue

from .contexts import Context, Counter, PreparedContext
from .interactions import Interaction
from .records import digest

PASSAGE_WORDS = 256
CONTINUATION_WORDS = 128
INSTRUCTION = "Continue the passage below in prose. Return only the continuation.\n\n"


@dataclass(frozen=True)
class Prose:
    text: str
    provenance: dict[str, str]

    @cached_property
    def word_ends(self) -> tuple[int, ...]:
        return (0, *(m.end() for m in re.finditer(r"\S+\s*", self.text)))

    @property
    def identity(self) -> str:
        return digest(self.provenance)


class ProseHistory:
    def __init__(self, source: Prose, identity: str):
        self.source, self.identity = source, identity
        self.ends = source.word_ends
        self.cursor = 0
        self.messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": f"Reading session {identity}."}
        ]
        self.pending: Context | None = None
        self.end = 0
        self.current = Interaction(
            id="prose.moby-dick",
            category="prose",
            messages=[],
            tools=[],
            expected=[],
            provenance=source.provenance,
        )

    def passage(self, start: int, end: int) -> str:
        return self.source.text[self.ends[start] : self.ends[end]]

    async def prepare(self, target: int, counter: Counter, sizing_identity: str) -> PreparedContext:
        if target < 0:
            raise ValueError("context target cannot be negative")
        available = len(self.ends) - 1 - self.cursor - CONTINUATION_WORDS
        if available < PASSAGE_WORDS:
            raise ValueError("Moby Dick has no remaining passage and canonical continuation")

        async def evaluate(words: int) -> tuple[Context, int]:
            message: dict[str, JsonValue] = {
                "role": "user",
                "content": INSTRUCTION + self.passage(self.cursor, self.cursor + words),
            }
            context = Context(messages=[*self.messages, message])
            count = await counter(context)
            if type(count) is not int or count < 1:
                raise ValueError("context renderer returned an invalid token count")
            return context, count

        low, high = PASSAGE_WORDS - 1, PASSAGE_WORDS
        context, count = await evaluate(high)
        while count < target:
            if high == available:
                raise ValueError("context target exceeds the remaining Moby Dick text")
            low, high = high, min(available, high * 2)
            previous = count
            context, count = await evaluate(high)
            if count <= previous:
                raise ValueError("rendered context did not grow after adding prose")
        while high - low > 1:
            middle = (low + high) // 2
            candidate, size = await evaluate(middle)
            if size >= target:
                high, context, count = middle, candidate, size
            else:
                low = middle
        self.end = self.cursor + high
        self.pending = context
        self.current = self.current.model_copy(update={"messages": context.messages[-1:]})
        return PreparedContext(
            content=context,
            tokens=count,
            provenance={
                **self.source.provenance,
                "fixture": "prose.moby-dick",
                "recipe": "prose-chat-history-v1",
                "history": self.identity,
                "passage_start_word": self.cursor,
                "passage_end_word": self.end,
                "canonical_continuation_words": CONTINUATION_WORDS,
                "requested_context_tokens": target,
                "actual_context_tokens": count,
                "sizing_identity": sizing_identity,
                "content_digest": digest(context.model_dump(mode="json")),
            },
        )

    def complete(self) -> None:
        if self.pending is None:
            raise ValueError("prose history has no prepared request")
        self.messages = [
            *self.pending.messages,
            {
                "role": "assistant",
                "content": self.passage(self.end, self.end + CONTINUATION_WORDS),
            },
        ]
        self.cursor = self.end + CONTINUATION_WORDS
        self.pending = None
