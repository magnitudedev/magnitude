"""Traffic sections over a fixture set; selections choose context checkpoints, not engine settings.

Lane *k* is fixture *k* of the set (one independent session); the set's last fixture is the
qualification lane, sent unsized as each launch's warmup. A request at checkpoint *N* is its
lane's fixture sized to *N* rendered tokens by the contract rule (``Fixture.sized``), cut at
the generation prefix. Section shapes describe offered traffic only: requests that share a
lane share history, which is never evidence of retained-prefix reuse.
"""

from magnitude_fixtures import Conversation, Fixture
from pydantic import TypeAdapter

from .policy import LANES
from .requests import Plan, Request, Section
from .sizing import Lengths, size

SECTIONS = {
    "single": "One unsized request (kept head and probe only), concurrency 1.",
    "context": "One request sized to each context checkpoint in turn, concurrency 1.",
    "session": "One session's requests at each checkpoint in turn, each after the last.",
    "parallel": "Four independent sessions released together at each checkpoint.",
    "fork": "One request establishes a history, then four branches of it are released together.",
    "concurrency": "Independent sessions at concurrency 1, 2, 4, 8 at each checkpoint.",
    "memory": "Four sessions advancing through the checkpoints; process-tree footprint over time.",
}
CONCURRENCY = (1, 2, 4, 8)
WIDE = 4


def lanes_needed(sections: tuple[str, ...]) -> int:
    widths = {"single": 1, "context": 1, "session": 1, "fork": 1, "parallel": WIDE,
              "memory": WIDE, "concurrency": max(CONCURRENCY)}
    return max(widths[section] for section in sections)


def request(
    fixture: Fixture,
    conversation: Conversation,
    identity: str,
    section: Section,
    checkpoint: int,
    *,
    session: str | None = None,
    concurrency: int = 1,
    depends_on: tuple[str, ...] = (),
) -> Request:
    if fixture.generation is None or conversation.prefix is None:
        raise ValueError(f"fixture {fixture.id} has no generation request to serve")
    return Request(
        id=identity,
        section=section,
        session=session or identity,
        checkpoint=checkpoint,
        concurrency=concurrency,
        family=fixture.family,
        fixture_id=fixture.id,
        messages=conversation.messages[: conversation.prefix],
        tools=conversation.tools,
        generation=fixture.generation,
        depends_on=depends_on,
    )


async def compile_plan(
    fixtures: list[Fixture],
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    lengths: Lengths,
) -> Plan:
    """The requests of the selected sections; every sizing is rendered by ``lengths``."""
    selected = TypeAdapter(tuple[Section, ...]).validate_python(sections)
    if len(fixtures) != LANES + 1:
        raise ValueError(f"a bench fixture set holds {LANES} lanes and a qualification lane")
    *lanes, qualification = fixtures
    width = lanes_needed(selected)
    sized_contexts = contexts if set(selected) - {"single"} else ()
    pairs = [(lanes[lane], target) for lane in range(width) for target in sized_contexts]
    conversations = dict(
        zip(
            [(lane, target) for lane in range(width) for target in sized_contexts],
            await size(pairs, lengths),
            strict=True,
        )
    )

    def at(lane: int, checkpoint: int, identity: str, section: Section, **fields) -> Request:
        fixture = lanes[lane]
        conversation = (
            fixture.conversation_with(0) if checkpoint == 0 else conversations[(lane, checkpoint)]
        )
        return request(fixture, conversation, identity, section, checkpoint, **fields)

    requests: list[Request] = []
    capacity = 1
    for section in selected:
        if section == "single":
            requests.append(at(0, 0, "single", section))
            continue
        if section in ("session", "memory"):
            count = WIDE if section == "memory" else 1
            capacity = max(capacity, count)
            previous: list[tuple[str, ...]] = [()] * count
            for checkpoint in contexts:
                for lane in range(count):
                    identity = f"{section}-{lane}-t{checkpoint}"
                    requests.append(
                        at(lane, checkpoint, identity, section, session=f"{section}-{lane}",
                           concurrency=count, depends_on=previous[lane])
                    )
                    previous[lane] = (identity,)
            continue
        previous_group: tuple[str, ...] = ()
        for checkpoint in contexts:
            widths = CONCURRENCY if section == "concurrency" else (
                (WIDE,) if section in ("parallel", "fork") else (1,)
            )
            for count in widths:
                capacity = max(capacity, count)
                group = f"{section}-t{checkpoint}-c{count}"
                dependencies = previous_group
                if section == "fork":
                    parent = at(0, checkpoint, f"{group}-parent", section, session=group,
                                depends_on=dependencies)
                    requests.append(parent)
                    dependencies = (parent.id,)
                current = []
                for index in range(count):
                    identity = f"{group}-{index}"
                    fork = section == "fork"
                    requests.append(
                        at(0 if fork else index, checkpoint, identity, section,
                           session=group if fork else identity, concurrency=count,
                           depends_on=dependencies)
                    )
                    current.append(identity)
                previous_group = tuple(current)
    warmup = request(
        qualification, qualification.conversation_with(0), "warmup", "single", 0
    )
    return Plan(warmup=warmup, requests=tuple(requests), parallel_sequences=capacity)
