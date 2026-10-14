"""The fixture contract: one deterministic chat conversation per JSON Lines record."""

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from .canonical import Record

Family = Literal[
    "tools", "prose-continue", "prose-repeat", "prose-shift", "retrieval", "code", "vision"
]
FAMILIES: tuple[Family, ...] = (
    "tools",
    "prose-continue",
    "prose-repeat",
    "prose-shift",
    "retrieval",
    "code",
    "vision",
)
Message = dict[str, JsonValue]


class Source(Record):
    name: str
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class Generation(Record):
    prefix: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    expected: dict[str, JsonValue]


class TurnSizing(Record):
    unit: Literal["turn"] = "turn"
    keep_head: int = Field(ge=0)


class FixedSizing(Record):
    unit: Literal["fixed"] = "fixed"


Sizing = Annotated[TurnSizing | FixedSizing, Field(discriminator="unit")]


RenderedLength = Callable[[list[Message], list[Message]], int]
FIXTURE_ONLY_FIELDS = ("scored", "tools")


def strip(message: Message) -> Message:
    """A message as a server receives it: without the fixture-only ``scored`` and ``tools``."""
    return {key: value for key, value in message.items() if key not in FIXTURE_ONLY_FIELDS}


def function_name(tool: Message) -> str:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(name := function.get("name"), str):
        raise ValueError("fixture tools must be named function tools")
    return name


class Conversation(Record):
    """A (possibly sized) fixture as a server receives it."""

    messages: list[Message]
    tools: list[Message]
    prefix: int | None
    """The serving request is ``messages[:prefix]``; ``None`` without ``generation``."""
    scored: list[int]
    """Indices of scored assistant messages within ``messages``."""


class Fixture(Record):
    id: str
    family: Family
    source: Source
    tools: list[Message] = Field(default_factory=list)
    messages: list[Message] = Field(min_length=1)
    generation: Generation | None = None
    sizing: Sizing

    @model_validator(mode="after")
    def conversation(self) -> Self:
        for message in self.messages:
            if "scored" in message and (
                message["scored"] is not True or message.get("role") != "assistant"
            ):
                raise ValueError("only assistant messages may carry scored: true")
        if not any(message.get("scored") for message in self.messages):
            raise ValueError("a fixture needs at least one scored assistant message")
        if self.generation is not None:
            if self.generation.prefix >= len(self.messages):
                raise ValueError("generation prefix must leave the answer outside the request")
            if self.messages[self.generation.prefix]["role"] != "assistant":
                raise ValueError("generation prefix must end where an assistant answer begins")
        history: set[int] = set()
        if isinstance(self.sizing, TurnSizing):
            *units, (probe, _) = self.units()
            history = {start for start, _ in units}
            if any(message.get("scored") for message in self.messages[:probe]):
                raise ValueError("scored messages must lie in the probe (the final sizing unit)")
            if self.generation is not None and self.generation.prefix <= probe:
                raise ValueError("the generation answer must lie in the probe")
        for index, message in enumerate(self.messages):
            if "tools" in message and (index not in history or not message["tools"]):
                raise ValueError("only a history unit's opening user message may carry tools")
        return self

    def units(self) -> list[tuple[int, int]]:
        """Half-open message ranges of the sizing units after the kept head; the last is the
        probe.

        A turn is one user message through its final assistant or tool reply.
        """
        if not isinstance(self.sizing, TurnSizing):
            raise ValueError(f"{self.id} is fixed-size and has no sizing units")
        head = self.sizing.keep_head
        starts = [i for i in range(head, len(self.messages)) if self.messages[i]["role"] == "user"]
        if head >= len(self.messages) or starts[:1] != [head]:
            raise ValueError("turn sizing requires a user message right after the kept head")
        return list(zip(starts, [*starts[1:], len(self.messages)], strict=True))

    def chat_messages(self, end: int | None = None) -> list[Message]:
        """``messages[:end]`` as a server receives them, without ``scored`` or unit ``tools``."""
        return [strip(message) for message in self.messages[:end]]

    def conversation_with(self, history: int | None = None) -> Conversation:
        """The conversation keeping the head, the probe and the ``history`` units right before
        the probe (all of them when ``None``), with its offered tools and shifted prefix."""
        if isinstance(self.sizing, FixedSizing):
            if history is not None:
                raise ValueError(f"{self.id} is fixed-size and cannot be resized")
            kept = list(range(len(self.messages)))
        else:
            *units, probe = self.units()
            if history is not None and not 0 <= history <= len(units):
                raise ValueError(f"{self.id} has {len(units)} history units, not {history}")
            chosen = units if history is None else units[len(units) - history :]
            kept = [
                *range(self.sizing.keep_head),
                *(i for start, end in [*chosen, probe] for i in range(start, end)),
            ]
        tools: dict[str, Message] = {}
        for tool in [
            *self.tools,
            *(tool for i in kept for tool in self.messages[i].get("tools", [])),
        ]:
            tools.setdefault(function_name(tool), tool)
        prefix = None
        if self.generation is not None:
            prefix = self.generation.prefix - (len(self.messages) - len(kept))
        return Conversation(
            messages=[strip(self.messages[i]) for i in kept],
            tools=list(tools.values()),
            prefix=prefix,
            scored=[i for i, original in enumerate(kept) if self.messages[original].get("scored")],
        )

    def sized(self, target: int, rendered_length: RenderedLength) -> Conversation:
        """Fit to ``target`` by the contract's rule: keep the head and the probe, then add
        history units newest first, keeping the most history whose rendered length is at or
        under ``target``. Fails when the head and probe alone exceed ``target``.

        ``rendered_length(messages, tools)`` must render with the target model's template.
        Rendered length grows with each added unit, so the largest fitting history is found
        by bracketing and bisection rather than one render per unit.
        """
        if isinstance(self.sizing, FixedSizing):
            return self.conversation_with()
        available = len(self.units()) - 1

        def fits(history: int) -> bool:
            conversation = self.conversation_with(history)
            length = rendered_length(conversation.messages, conversation.tools)
            if type(length) is not int or length < 1:
                raise ValueError("rendered_length must return a positive integer")
            return length <= target

        if not fits(0):
            raise ValueError(f"{self.id}: head and probe alone exceed {target}")
        # Invariant: ``low`` fits; ``high`` does not, once the bracket closes.
        low, high = 0, 1
        while True:
            if high >= available:
                high = available
                if fits(available):
                    return self.conversation_with(available)
                break
            if not fits(high):
                break
            low, high = high, high * 2
        while high - low > 1:
            middle = (low + high) // 2
            if fits(middle):
                low = middle
            else:
                high = middle
        return self.conversation_with(low)

    def line(self) -> str:
        """The fixture's JSON Lines record, in contract key order."""
        record: dict[str, object] = {
            "id": self.id,
            "family": self.family,
            "source": self.source.model_dump(),
        }
        if self.tools:
            record["tools"] = self.tools
        record["messages"] = self.messages
        if self.generation is not None:
            record["generation"] = self.generation.model_dump()
        record["sizing"] = self.sizing.model_dump()
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode_set(fixtures: Iterable[Fixture]) -> bytes:
    lines = [fixture.line() for fixture in fixtures]
    if not lines:
        raise ValueError("a fixture set needs at least one fixture")
    if len({json.loads(line)["id"] for line in lines}) != len(lines):
        raise ValueError("fixture ids must be unique within a set")
    return ("\n".join(lines) + "\n").encode()


def write_set(path: Path, fixtures: Iterable[Fixture]) -> str:
    """Write a fixture set and return its digest (SHA-256 of the file bytes)."""
    content = encode_set(fixtures)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def read_set(path: Path) -> tuple[list[Fixture], str]:
    """Read a fixture set and its digest."""
    content = path.read_bytes()
    fixtures = [Fixture.model_validate_json(line) for line in content.decode().splitlines()]
    return fixtures, hashlib.sha256(content).hexdigest()
