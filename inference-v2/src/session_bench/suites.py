"""Fixed session shapes; user selections choose context checkpoints, not engine settings."""

from itertools import cycle

from pydantic import JsonValue, TypeAdapter

from .policy import CHARACTERS_PER_TOKEN
from .sessions import Interaction, Plan, Request, Section, encoded

SECTIONS = {
    "single": "One natural tool decision; no added history, concurrency 1.",
    "context": "Independent full-prefill requests near each context checkpoint, concurrency 1.",
    "session": "One sequential session growing through the context checkpoints.",
    "parallel": "Four independent sessions released together at each checkpoint.",
    "fork": "Establish one history, then release four branches at each checkpoint.",
    "concurrency": "Independent sessions at concurrency 1, 2, 4, 8 at each checkpoint.",
    "memory": "Four sessions growing through checkpoints; process-tree footprint over time.",
}


def tool_name(tool: dict[str, JsonValue]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(name := function.get("name"), str):
        raise ValueError("session history requires named function tools")
    return name


class History:
    """One canonical prefix family, with consistent tool definitions throughout its history."""

    def __init__(self, fixtures: list[Interaction], identity: str, current: Interaction):
        self.identity, self.current = identity, current
        self.messages: list[dict[str, JsonValue]] = [
            {"role": "system", "content": f"Session {identity}. Use the supplied tools."}
        ]
        self.tools = {tool_name(t): t for t in current.tools}
        self.source = cycle(fixtures)
        self.index = 0

    def request(self, identity: str, section: Section, checkpoint: int, depends=()) -> Request:
        def size():
            return len(
                encoded(
                    {
                        "messages": self.messages + self.current.messages,
                        "tools": list(self.tools.values()),
                    }
                )
            )

        goal = checkpoint * CHARACTERS_PER_TOKEN
        while size() < goal:
            fixture = next(self.source)
            if any(
                tool_name(t) in self.tools and self.tools[tool_name(t)] != t
                for t in fixture.tools
            ):
                continue
            addition = fixture.completed(f"{self.identity}_{self.index}")
            before = size()
            old_tools = self.tools.copy()
            self.tools.update({tool_name(t): t for t in fixture.tools})
            self.messages.extend(addition)
            self.index += 1
            after = size()
            if before < goal <= after:
                if goal - before < after - goal:
                    del self.messages[-len(addition) :]
                    self.tools = old_tools
                break
        return Request(
            id=identity,
            section=section,
            session=self.identity,
            checkpoint=checkpoint,
            fixture_id=self.current.id,
            messages=self.messages + self.current.messages,
            tools=list(self.tools.values()),
            expected=self.current.expected,
            depends_on=tuple(depends),
        )

    def complete(self):
        self.messages.extend(self.current.completed(f"{self.identity}_turn{self.index}"))
        self.index += 1


def compile_plan(
    fixtures: list[Interaction],
    corpus_digest: str,
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    case: str | None = None,
) -> Plan:
    selected = [f for f in fixtures if case is None or f.id == case]
    if not selected:
        raise ValueError(f"unknown or excluded BFCL case: {case}")
    requests = []
    capacity = 1
    # The fixture selection for one section is independent of other selected sections.
    for section in TypeAdapter(tuple[Section, ...]).validate_python(sections):
        if section == "single":
            f = selected[0]
            requests.append(
                Request(
                    id="single",
                    section="single",
                    session="single",
                    checkpoint=0,
                    fixture_id=f.id,
                    messages=f.messages,
                    tools=f.tools,
                    expected=f.expected,
                )
            )
            continue
        if section in ("session", "memory"):
            lanes = 4 if section == "memory" else 1
            capacity = max(capacity, lanes)
            histories = [
                History(fixtures, f"{section}-{i}", selected[i % len(selected)])
                for i in range(lanes)
            ]
            previous: list[str | None] = [None] * lanes
            for checkpoint in contexts:
                for i, history in enumerate(histories):
                    identity = f"{section}-{i}-t{checkpoint}"
                    requests.append(
                        history.request(
                            identity, section, checkpoint, [previous[i]] if previous[i] else []
                        ).model_copy(update={"concurrency": lanes})
                    )
                    history.complete()
                    previous[i] = identity
            continue
        previous_group = []
        for checkpoint in contexts:
            for count in (
                (1, 2, 4, 8)
                if section == "concurrency"
                else (4,)
                if section in ("parallel", "fork")
                else (1,)
            ):
                capacity = max(capacity, count)
                group = f"{section}-t{checkpoint}-c{count}"
                shared = History(fixtures, group, selected[0]) if section == "fork" else None
                dependencies = previous_group
                if shared:
                    parent = shared.request(f"{group}-parent", section, checkpoint, dependencies)
                    requests.append(parent)
                    shared.complete()
                    dependencies = [parent.id]
                current_group = []
                for i in range(count):
                    identity = f"{group}-{i}"
                    history = shared or History(fixtures, identity, selected[i % len(selected)])
                    request = history.request(identity, section, checkpoint, dependencies)
                    requests.append(request.model_copy(update={"concurrency": count}))
                    current_group.append(identity)
                previous_group = current_group
    return Plan(requests=tuple(requests), parallel_sequences=capacity, corpus_digest=corpus_digest)
