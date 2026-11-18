"""The session record: one deterministic chat conversation built from a fixture, its history
turns and a final probe."""

from collections.abc import Callable
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from .canonical import Record

Key = Literal["repeat", "twist", "retrieve", "code", "tools"]
#: Every template key: repeat and twist (Moby Dick), retrieve (RULER), code (BigCodeBench),
#: tools (BFCL v4). The only definition of the key set.
KEYS: tuple[Key, ...] = ("repeat", "twist", "retrieve", "code", "tools")
Message = dict[str, JsonValue]


class Source(Record):
    name: str
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class Generation(Record):
    prefix: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    expected: dict[str, JsonValue]


RenderedLength = Callable[[list[Message], list[Message]], int]
SESSION_ONLY_FIELDS = ("scored", "tools")


def strip(message: Message) -> Message:
    """A message as a server receives it: without the session-only ``scored`` and ``tools``."""
    return {key: value for key, value in message.items() if key not in SESSION_ONLY_FIELDS}


def function_name(tool: Message) -> str:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(name := function.get("name"), str):
        raise ValueError("session tools must be named function tools")
    return name


class Conversation(Record):
    """A (possibly sized) session as a server receives it."""

    messages: list[Message]
    tools: list[Message]
    prefix: int
    """The serving request is ``messages[:prefix]``."""
    scored: list[int]
    """Indices of scored assistant messages within ``messages``."""


class Session(Record):
    key: Key
    tools: list[Message] = Field(default_factory=list)
    messages: list[Message] = Field(min_length=1)
    generation: Generation
    keep_head: int = Field(ge=0)

    @model_validator(mode="after")
    def conversation(self) -> Self:
        for message in self.messages:
            if "scored" in message and (
                message["scored"] is not True or message.get("role") != "assistant"
            ):
                raise ValueError("only assistant messages may carry scored: true")
        if not any(message.get("scored") for message in self.messages):
            raise ValueError("a session needs at least one scored assistant message")
        if self.generation.prefix >= len(self.messages):
            raise ValueError("generation prefix must leave the answer outside the request")
        if self.messages[self.generation.prefix]["role"] != "assistant":
            raise ValueError("generation prefix must end where an assistant answer begins")
        *units, (probe, _) = self.units()
        history = {start for start, _ in units}
        if any(message.get("scored") for message in self.messages[:probe]):
            raise ValueError("scored messages must lie in the probe (the final sizing unit)")
        if self.generation.prefix <= probe:
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
        head = self.keep_head
        starts = [i for i in range(head, len(self.messages)) if self.messages[i]["role"] == "user"]
        if head >= len(self.messages) or starts[:1] != [head]:
            raise ValueError("a session needs a user message right after the kept head")
        return list(zip(starts, [*starts[1:], len(self.messages)], strict=True))

    def chat_messages(self, end: int | None = None) -> list[Message]:
        """``messages[:end]`` as a server receives them, without ``scored`` or unit ``tools``."""
        return [strip(message) for message in self.messages[:end]]

    def conversation_with(self, history: int | None = None) -> Conversation:
        """The conversation keeping the head, the probe and the ``history`` units right before
        the probe (all of them when ``None``), with its offered tools and shifted prefix."""
        *units, probe = self.units()
        if history is not None and not 0 <= history <= len(units):
            raise ValueError(f"the session has {len(units)} history units, not {history}")
        chosen = units if history is None else units[len(units) - history :]
        kept = [
            *range(self.keep_head),
            *(i for start, end in [*chosen, probe] for i in range(start, end)),
        ]
        tools: dict[str, Message] = {}
        for tool in [
            *self.tools,
            *(tool for i in kept for tool in self.messages[i].get("tools", [])),
        ]:
            tools.setdefault(function_name(tool), tool)
        return Conversation(
            messages=[strip(self.messages[i]) for i in kept],
            tools=list(tools.values()),
            prefix=self.generation.prefix - (len(self.messages) - len(kept)),
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
        available = len(self.units()) - 1

        def fits(history: int) -> bool:
            conversation = self.conversation_with(history)
            length = rendered_length(conversation.messages, conversation.tools)
            if type(length) is not int or length < 1:
                raise ValueError("rendered_length must return a positive integer")
            return length <= target

        if not fits(0):
            raise ValueError(f"the session's head and probe alone exceed {target}")
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
