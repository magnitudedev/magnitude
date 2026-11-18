"""Traffic sections over a run's templates; selections choose context checkpoints, not engine
settings.

A run's template files are one fixture and response kind at several contexts: each a
checkpoint. Lane *k* is session *k* of the files (one independent session); their last lane is
the qualification lane, sent at the smallest context as each launch's warmup. A request at
checkpoint *N* is its lane's session sized to *N* by the contract rule, cut at the generation
prefix. Section shapes describe offered traffic only: requests that share a lane share history,
which is never evidence of retained-prefix reuse.
"""

from dataclasses import dataclass

from magnitude_fixtures import Conversation, Session, TemplateFile
from pydantic import TypeAdapter

from .policy import LANES
from .requests import Plan, Request, Section

SECTIONS = {
    "context": "One request sized to each context checkpoint in turn, concurrency 1.",
    "session": "One session's requests at each checkpoint in turn, each after the last.",
    "parallel": "Four independent sessions released together at each checkpoint.",
    "fork": "One request establishes a history, then four branches of it are released together.",
    "concurrency": "Independent sessions at concurrency 1, 2, 4, 8 at each checkpoint.",
    "memory": "Four sessions advancing through the checkpoints; process-tree footprint over time.",
}
CONCURRENCY = (1, 2, 4, 8)
WIDE = 4


@dataclass(frozen=True)
class Sized:
    """One lane of one template, sized: its conversation, and the digest of its rendered tokens
    where the catalog GGUF's rendering sized it."""

    conversation: Conversation
    prompt_sha256: str | None


def checked(files: list[TemplateFile]) -> list[TemplateFile]:
    """The run's template files by context: one key and source, the same sessions
    (``LANES`` traffic lanes and the qualification lane), at distinct contexts."""
    if not files:
        raise ValueError("a bench run measures at least one template")
    ordered = sorted(files, key=lambda file: file.context)
    first = ordered[0]
    for file in ordered:
        if len(file.sessions) != LANES + 1:
            raise ValueError(f"{file.template} holds {len(file.sessions)} lanes, not the bench's "
                             f"{LANES} and a qualification lane")
        if file.source != first.source or file.sessions != first.sessions:
            raise ValueError(f"{file.template} is not of {first.template}'s key and sessions")
    if len({file.context for file in ordered}) != len(ordered):
        raise ValueError("a bench run's templates are at distinct contexts")
    return ordered


def lanes_needed(sections: tuple[str, ...]) -> int:
    widths = {"context": 1, "session": 1, "fork": 1, "parallel": WIDE,
              "memory": WIDE, "concurrency": max(CONCURRENCY)}
    return max(widths[section] for section in sections)


def requested(files: list[TemplateFile], sections: tuple[str, ...]) -> list[tuple[TemplateFile, int]]:
    """The (template, lane) pairs a plan of ``sections`` serves, in order: the traffic lanes at
    every checkpoint, then the qualification lane at the smallest."""
    width = lanes_needed(sections)
    return [(file, lane) for file in files for lane in range(width)] + [(files[0], LANES)]


def request(
    file: TemplateFile,
    lane: int,
    sized: Sized,
    identity: str,
    section: Section,
    *,
    session: str | None = None,
    concurrency: int = 1,
    depends_on: tuple[str, ...] = (),
) -> Request:
    served: Session = file.sessions[lane]
    conversation = sized.conversation
    return Request(
        id=identity,
        section=section,
        session=session or identity,
        checkpoint=file.context,
        concurrency=concurrency,
        template=file.template,
        lane=lane,
        key=served.key,
        prompt_sha256=sized.prompt_sha256,
        messages=conversation.messages[: conversation.prefix],
        tools=conversation.tools,
        generation=served.generation,
        depends_on=depends_on,
    )


def compile_plan(
    files: list[TemplateFile],
    sections: tuple[str, ...],
    sized: dict[tuple[str, int], Sized],
) -> Plan:
    """The requests of the selected sections over ``files`` (``checked``), every lane sized in
    ``sized`` by (template, lane) (``requested``)."""
    if not sections or set(sections) - set(SECTIONS):
        raise ValueError(f"choose sections from {', '.join(SECTIONS)}")
    selected = TypeAdapter(tuple[Section, ...]).validate_python(sections)

    def at(lane: int, file: TemplateFile, identity: str, section: Section, **fields) -> Request:
        return request(file, lane, sized[(file.template, lane)], identity, section, **fields)

    requests: list[Request] = []
    capacity = 1
    for section in selected:
        if section in ("session", "memory"):
            count = WIDE if section == "memory" else 1
            capacity = max(capacity, count)
            previous: list[tuple[str, ...]] = [()] * count
            for file in files:
                for lane in range(count):
                    identity = f"{section}-{lane}-t{file.context}"
                    requests.append(
                        at(lane, file, identity, section, session=f"{section}-{lane}",
                           concurrency=count, depends_on=previous[lane])
                    )
                    previous[lane] = (identity,)
            continue
        previous_group: tuple[str, ...] = ()
        for file in files:
            widths = CONCURRENCY if section == "concurrency" else (
                (WIDE,) if section in ("parallel", "fork") else (1,)
            )
            for count in widths:
                capacity = max(capacity, count)
                group = f"{section}-t{file.context}-c{count}"
                dependencies = previous_group
                if section == "fork":
                    parent = at(0, file, f"{group}-parent", section, session=group,
                                depends_on=dependencies)
                    requests.append(parent)
                    dependencies = (parent.id,)
                current = []
                for index in range(count):
                    identity = f"{group}-{index}"
                    fork = section == "fork"
                    requests.append(
                        at(0 if fork else index, file, identity, section,
                           session=group if fork else identity, concurrency=count,
                           depends_on=dependencies)
                    )
                    current.append(identity)
                previous_group = tuple(current)
    warmup = at(LANES, files[0], "warmup", "context")
    return Plan(warmup=warmup, requests=tuple(requests), parallel_sequences=capacity)
