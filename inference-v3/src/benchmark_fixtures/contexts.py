"""Deterministic tool histories sized by a consumer's actual tokenizer/template."""

from collections.abc import Awaitable, Callable
from copy import deepcopy

from pydantic import Field, JsonValue

from .interactions import Interaction
from .records import Record, digest


class Context(Record):
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]] = Field(default_factory=list)


Counter = Callable[[Context], Awaitable[int]]


class PreparedContext(Record):
    content: Context
    tokens: int
    provenance: dict[str, JsonValue]


def tool_name(tool: dict[str, JsonValue]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(name := function.get("name"), str):
        raise ValueError("tool fixture requires named function tools")
    return name


class History:
    def __init__(self, fixtures: list[Interaction], identity: str, current: Interaction):
        if not fixtures:
            raise ValueError("tool history needs a nonempty source")
        self.fixtures, self.identity, self.current = fixtures, identity, current
        self.messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": f"Session {identity}. Use the supplied tools."}
        ]
        self.tools = {tool_name(t): t for t in current.tools}
        self.cursor = self.index = 0
        self.corpus_digest = digest([f.model_dump(mode="json") for f in fixtures])

    def _extend(self, rounds: int):
        messages, tools = list(self.messages), dict(self.tools)
        cursor, index = self.cursor, self.index
        for _ in range(rounds):
            for _ in range(len(self.fixtures)):
                fixture = self.fixtures[cursor % len(self.fixtures)]
                cursor += 1
                if all(
                    tool_name(t) not in tools or tools[tool_name(t)] == t for t in fixture.tools
                ):
                    break
            else:
                raise ValueError("no compatible BFCL interaction can extend this context")
            tools.update({tool_name(t): t for t in fixture.tools})
            messages.extend(fixture.completed(f"{self.identity}_{index}"))
            index += 1
        return messages, tools, cursor, index

    async def prepare(self, target: int, counter: Counter, sizing_identity: str) -> PreparedContext:
        if target < 0:
            raise ValueError("context target cannot be negative")

        async def evaluate(rounds: int):
            state = self._extend(rounds)
            context = Context(
                messages=state[0] + self.current.messages, tools=list(state[1].values())
            )
            count = await counter(context)
            if type(count) is not int or count < 1:
                raise ValueError("context renderer returned an invalid token count")
            return state, context, count

        state, context, count = await evaluate(0)
        if count < target:
            # Bracket then bisect complete interactions. Expensive renderer calls
            # grow logarithmically with history length, never once per message.
            low, high = 0, 1
            while True:
                previous_count = count
                state, context, count = await evaluate(high)
                if count <= previous_count:
                    raise ValueError("rendered context did not grow after adding interactions")
                if count >= target:
                    break
                low, high = high, high * 2
                if high > 65536:
                    raise ValueError("context target exceeds the fixture preparation limit")
            while high - low > 1:
                middle = (low + high) // 2
                candidate = await evaluate(middle)
                if candidate[2] >= target:
                    high = middle
                    state, context, count = candidate
                else:
                    low = middle
        self.messages, self.tools, self.cursor, self.index = state
        return PreparedContext(
            content=context,
            tokens=count,
            provenance={
                "fixture": "tools.bfcl",
                "recipe": "bfcl-history-v1",
                "corpus_digest": self.corpus_digest,
                "history": self.identity,
                "decision": self.current.id,
                "rounds": self.index,
                "source_cursor": self.cursor,
                "requested_context_tokens": target,
                "actual_context_tokens": count,
                "sizing_identity": sizing_identity,
                "content_digest": digest(context.model_dump(mode="json")),
                "tool_response": "synthetic argument acknowledgement",
            },
        )

    def complete(self) -> None:
        self.messages.extend(deepcopy(self.current.completed(f"{self.identity}_turn{self.index}")))
        self.index += 1
