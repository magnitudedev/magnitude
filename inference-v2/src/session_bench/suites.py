"""Fixed session shapes; user selections choose context checkpoints, not engine settings."""

from typing import NamedTuple

from pydantic import TypeAdapter

from benchmark_fixtures.contexts import Context, Counter, History
from benchmark_fixtures.interactions import Interaction
from benchmark_fixtures.prose_history import Prose, ProseHistory
from benchmark_fixtures.records import digest
from benchmark_fixtures.ruler import RulerFixture

from .sessions import Plan, Request, Section

SECTIONS = {
    "single": "One short content request; no added history, concurrency 1.",
    "context": "Independent full-prefill requests near each context checkpoint, concurrency 1.",
    "session": "One sequential session growing through the context checkpoints.",
    "parallel": "Four independent sessions released together at each checkpoint.",
    "fork": "Establish one history, then release four branches at each checkpoint.",
    "concurrency": "Independent sessions at concurrency 1, 2, 4, 8 at each checkpoint.",
    "memory": "Four sessions growing through checkpoints; process-tree footprint over time.",
}


class SessionInput(NamedTuple):
    identity: str
    content: History | ProseHistory | RulerFixture
    needle_depth: float


async def history_request(
    history: SessionInput, identity, section, checkpoint, counter, sizing_identity, depends=()
):
    session, content, needle_depth = history
    if isinstance(content, RulerFixture):
        prepared = await content.prepare(
            checkpoint, counter, sizing_identity, needle_depth=needle_depth
        )
        return Request(
            id=identity,
            section=section,
            session=session,
            checkpoint=checkpoint,
            fixture_id=content.identity,
            workload="retrieval",
            messages=prepared.content.messages,
            tools=[],
            expected=prepared.expected,
            depends_on=tuple(depends),
            fixture_provenance=prepared.provenance,
        )
    prepared = await content.prepare(checkpoint, counter, sizing_identity)
    return Request(
        id=identity,
        section=section,
        session=session,
        checkpoint=checkpoint,
        fixture_id=content.current.id,
        workload="prose" if isinstance(content, ProseHistory) else "tools",
        messages=prepared.content.messages,
        tools=prepared.content.tools,
        expected=content.current.expected,
        depends_on=tuple(depends),
        fixture_provenance=prepared.provenance,
    )


def complete(history: SessionInput) -> None:
    # Retrieval snapshots never append answers: that would leak the probe into later inputs.
    if not isinstance(history.content, RulerFixture):
        history.content.complete()


async def compile_plan(
    fixtures: list[Interaction] | Prose | RulerFixture,
    corpus_digest: str,
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    case: str | None = None,
    *,
    counter: Counter,
    sizing_identity: str,
    needle_depth: float = 0.5,
) -> Plan:
    selected = (
        [f for f in fixtures if case is None or f.id == case] if isinstance(fixtures, list) else []
    )
    if not isinstance(fixtures, list):
        if case is not None:
            raise ValueError("--case is only supported for tool fixtures")
    elif not selected:
        raise ValueError(f"unknown or excluded BFCL case: {case}")

    def history_for(identity: str, index: int = 0) -> SessionInput:
        if isinstance(fixtures, Prose):
            content = ProseHistory(fixtures, identity)
        elif isinstance(fixtures, RulerFixture):
            # Lane selection is independent of section, checkpoint and sizing search.
            content = fixtures.model_copy(update={"seed": fixtures.seed + index})
        else:
            content = History(fixtures, identity, selected[index % len(selected)])
        return SessionInput(identity, content, needle_depth)

    requests = []
    capacity = 1
    # The fixture selection for one section is independent of other selected sections.
    for section in TypeAdapter(tuple[Section, ...]).validate_python(sections):
        if section == "single":
            if not isinstance(fixtures, list):
                requests.append(
                    await history_request(
                        history_for("single"), "single", section, 0, counter, sizing_identity
                    )
                )
                continue
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
                    fixture_provenance={
                        "fixture": "tools.bfcl",
                        "corpus_digest": corpus_digest,
                        "decision": f.id,
                    },
                )
            )
            continue
        if section in ("session", "memory"):
            lanes = 4 if section == "memory" else 1
            capacity = max(capacity, lanes)
            histories = [history_for(f"{section}-{i}", i) for i in range(lanes)]
            previous: list[str | None] = [None] * lanes
            for step, checkpoint in enumerate(contexts):
                for i, history in enumerate(histories):
                    identity = f"{section}-{i}-t{checkpoint}"
                    if isinstance(fixtures, RulerFixture):
                        identity += f"-s{step}"
                    requests.append(
                        (
                            await history_request(
                                history,
                                identity,
                                section,
                                checkpoint,
                                counter,
                                sizing_identity,
                                [previous[i]] if previous[i] else [],
                            )
                        ).model_copy(update={"concurrency": lanes})
                    )
                    complete(history)
                    previous[i] = identity
            continue
        previous_group = []
        for step, checkpoint in enumerate(contexts):
            for count in (
                (1, 2, 4, 8)
                if section == "concurrency"
                else (4,)
                if section in ("parallel", "fork")
                else (1,)
            ):
                capacity = max(capacity, count)
                group = f"{section}-t{checkpoint}-c{count}"
                if isinstance(fixtures, RulerFixture):
                    group += f"-s{step}"
                shared = history_for(group) if section == "fork" else None
                dependencies = previous_group
                if shared:
                    parent = await history_request(
                        shared,
                        f"{group}-parent",
                        section,
                        checkpoint,
                        counter,
                        sizing_identity,
                        dependencies,
                    )
                    requests.append(parent)
                    complete(shared)
                    dependencies = [parent.id]
                current_group = []
                for i in range(count):
                    identity = f"{group}-{i}"
                    history = shared or history_for(identity, i)
                    request = await history_request(
                        history,
                        identity,
                        section,
                        checkpoint,
                        counter,
                        sizing_identity,
                        dependencies,
                    )
                    requests.append(request.model_copy(update={"concurrency": count}))
                    current_group.append(identity)
                previous_group = current_group
    qualification = None
    if isinstance(fixtures, RulerFixture):
        qualification = await history_request(
            history_for("qualification", 1_000_000),
            "warmup",
            "single",
            0,
            counter,
            sizing_identity,
        )
        messages = [dict(message) for message in qualification.messages]
        messages[0]["content"] = "Independent retrieval qualification. " + str(
            messages[0]["content"]
        )
        qualification_context = Context(messages=messages)
        qualification = qualification.model_copy(
            update={
                "messages": messages,
                "fixture_provenance": {
                    **{
                        key: value
                        for key, value in qualification.fixture_provenance.items()
                        if key != "needle_prefix_render_tokens"
                    },
                    "recipe": "ruler-qualification-v1",
                    "actual_context_tokens": await counter(qualification_context),
                    "content_digest": digest(qualification_context.model_dump(mode="json")),
                },
            }
        )
    return Plan(
        requests=tuple(requests),
        parallel_sequences=capacity,
        corpus_digest=corpus_digest,
        qualification=qualification,
    )
