"""Chat requests over contiguous book passages, independent of session scheduling.

Two workloads share one sizing policy. ``prose-continue`` asks for new prose after the
passage; its output varies by engine and numerics. ``prose-repeat`` asks for the passage
back, starting at the first sentence of "Loomings" at every checkpoint, so each request's
256-token output is the same text regardless of context size.
"""

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from pydantic import JsonValue

from .contexts import Context, Counter, PreparedContext
from .interactions import Interaction
from .records import digest

PASSAGE_WORDS = 256
CONTINUATION_WORDS = 128
INSTRUCTION = "Continue the passage below in prose. Return only the continuation.\n\n"
REPEAT_INSTRUCTION = "Copy the supplied passage exactly. Output only its text.\n\n"
REPEAT_START = "Call me Ishmael."
# The repeated text is single-spaced with ASCII quotes, as V3's Loomings repeat recipe rendered it.
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
            return " ".join(self.text.translate(STRAIGHT_QUOTES).split())
        return self.text

    @cached_property
    def word_ends(self) -> tuple[int, ...]:
        return (0, *(m.end() for m in re.finditer(r"\S+\s*", self.body)))

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
            return digest({**self.provenance, "workload": self.workload})
        return digest(self.provenance)


class ProseHistory:
    def __init__(self, source: Prose, identity: str):
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
        available = len(self.ends) - 1 - self.cursor - CONTINUATION_WORDS
        if available < PASSAGE_WORDS:
            raise ValueError("Moby Dick has no remaining passage and canonical continuation")
        instruction = REPEAT_INSTRUCTION if self.source.repeating else INSTRUCTION

        async def evaluate(words: int) -> tuple[Context, int]:
            message: dict[str, JsonValue] = {
                "role": "user",
                "content": instruction + self.passage(self.cursor, self.cursor + words),
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
        self.start = self.cursor
        self.end = self.cursor + high
        self.pending = context
        self.current = self.current.model_copy(update={"messages": context.messages[-1:]})
        return PreparedContext(
            content=context,
            tokens=count,
            provenance={
                **self.source.provenance,
                "fixture": self.source.fixture,
                "recipe": (
                    "prose-repeat-history-v1" if self.source.repeating else "prose-chat-history-v1"
                ),
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
        # The canonical answer repeats the passage's opening words, or continues after it.
        answer = self.start if self.source.repeating else self.end
        self.messages = [
            *self.pending.messages,
            {
                "role": "assistant",
                "content": self.passage(answer, answer + CONTINUATION_WORDS),
            },
        ]
        self.cursor = self.end if self.source.repeating else self.end + CONTINUATION_WORDS
        self.pending = None
