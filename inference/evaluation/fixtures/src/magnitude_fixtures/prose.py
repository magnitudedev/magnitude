"""Reading sessions over the pinned Moby Dick text: ``prose-continue`` and ``prose-repeat``.

The book body from "Call me Ishmael." is unwrapped into paragraphs with straight quotes and
cut into consecutive fixed word windows, one per turn, so turn sizes never depend on a
tokenizer:
- ``prose-continue``: the user supplies a passage and asks for a continuation; the
  canonical answer is the book's next words. Engine output varies, so serving validates
  only that prose was produced (``expected`` is empty).
- ``prose-repeat``: the user supplies a passage and asks for it to be repeated verbatim
  from a paragraph opening; the answer is the latest paragraph-aligned suffix of at least
  ``REPEAT_COPY_WORDS`` words, which covers the serving output budget for any tokenizer.

Each fixture's probe is a later window than every turn in its history; lanes probe
consecutive windows near the end of the book, so their sessions differ at the tail.
"""

import gzip
import hashlib
import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from .canonical import digest
from .fixture import Fixture, Generation, Message, Source, TurnSizing
from .sources import DATA

ProseFamily = Literal["prose-continue", "prose-repeat"]
START = "Call me Ishmael."
OUTPUT_TOKENS = 256
CONTINUE_PASSAGE_WORDS = 256
CONTINUATION_WORDS = 128
REPEAT_PASSAGE_WORDS = 512
REPEAT_COPY_WORDS = 256
CONTINUE_INSTRUCTION = "Continue the passage below in prose. Return only the continuation.\n\n"
REPEAT_INSTRUCTION = (
    "Starting with “{opening}”, repeat the supplied passage verbatim to the end. "
    "Output only the passage text."
)
STRAIGHT_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def gutenberg_body(content: bytes) -> str:
    text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(
        r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.MULTILINE
    )
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.MULTILINE)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("missing or unordered Gutenberg body markers")
    return text[start.end() : end.start()].strip()


def load() -> tuple[str, Source]:
    """The vendored book body text and its source identity.

    The public-domain Gutenberg file (license included) is vendored compressed, because the
    upstream URL is mutable; its decompressed bytes must match the locked digest.
    """
    lock = json.loads((DATA / "moby-dick.lock.json").read_text())
    content = gzip.decompress((DATA / lock["file"]).read_bytes())
    if hashlib.sha256(content).hexdigest() != lock["sha256"]:
        raise ValueError("vendored Moby Dick text does not match its lock")
    text = gutenberg_body(content)
    return text, Source(
        name="moby-dick",
        digest=digest({**lock, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}),
    )


@dataclass(frozen=True)
class Book:
    text: str

    @cached_property
    def body(self) -> str:
        paragraphs = re.split(r"\n\s*\n", self.text.translate(STRAIGHT_QUOTES))
        body = "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())
        if START not in body:
            raise ValueError(f"book body lacks its opening {START!r}")
        return body[body.index(START) :]

    @cached_property
    def word_ends(self) -> tuple[int, ...]:
        return (0, *(m.end() for m in re.finditer(r"\S+\s*", self.body)))

    @property
    def words(self) -> int:
        return len(self.word_ends) - 1

    @cached_property
    def paragraph_starts(self) -> tuple[int, ...]:
        return tuple(bisect_left(self.word_ends, m.end()) for m in re.finditer(r"\n\n", self.body))

    def passage(self, start: int, end: int) -> str:
        return self.body[self.word_ends[start] : self.word_ends[end]].rstrip()

    def copy_start(self, start: int, end: int) -> int:
        """The latest paragraph start whose suffix has at least ``REPEAT_COPY_WORDS`` words."""
        candidates = [p for p in self.paragraph_starts if start < p <= end - REPEAT_COPY_WORDS]
        return candidates[-1] if candidates else start


def continue_turn(book: Book, start: int, *, scored: bool) -> list[Message]:
    split = start + CONTINUE_PASSAGE_WORDS
    answer: Message = {
        "role": "assistant",
        "content": book.passage(split, split + CONTINUATION_WORDS),
    }
    if scored:
        answer["scored"] = True
    return [{"role": "user", "content": CONTINUE_INSTRUCTION + book.passage(start, split)}, answer]


def repeat_turn(book: Book, start: int, *, scored: bool) -> list[Message]:
    end = start + REPEAT_PASSAGE_WORDS
    copy = book.passage(book.copy_start(start, end), end)
    opening = " ".join(copy.split()[:6])
    answer: Message = {"role": "assistant", "content": copy}
    if scored:
        answer["scored"] = True
    return [
        {
            "role": "user",
            "content": "Supplied text:\n\n"
            + book.passage(start, end)
            + "\n\n"
            + REPEAT_INSTRUCTION.format(opening=opening),
        },
        answer,
    ]


def build(text: str, source: Source, family: ProseFamily, *, count: int) -> list[Fixture]:
    book = Book(text)
    window, turn = (
        (REPEAT_PASSAGE_WORDS, repeat_turn)
        if family == "prose-repeat"
        else (CONTINUE_PASSAGE_WORDS + CONTINUATION_WORDS, continue_turn)
    )
    turns = book.words // window
    if not 1 <= count < turns:
        raise ValueError(f"{family} fixtures need 1..{turns - 1} probes, got {count}")
    fixtures = []
    for lane in range(count):
        probe = turns - count + lane
        messages: list[Message] = [{"role": "system", "content": f"Reading session {lane}."}]
        for index in range(probe):
            messages.extend(turn(book, index * window, scored=False))
        final = turn(book, probe * window, scored=True)
        prefix = len(messages) + 1
        messages.extend(final)
        expected = {"text": final[-1]["content"]} if family == "prose-repeat" else {}
        fixtures.append(
            Fixture(
                id=f"{family}/moby-dick/{lane}",
                family=family,
                source=source,
                messages=messages,
                generation=Generation(prefix=prefix, max_tokens=OUTPUT_TOKENS, expected=expected),
                sizing=TurnSizing(keep_head=1),
            )
        )
    return fixtures
