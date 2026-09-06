"""Fixed session shapes; user selections choose context checkpoints, not engine settings."""

from pydantic import TypeAdapter

from benchmark_fixtures.contexts import Counter, History
from benchmark_fixtures.interactions import Interaction
from benchmark_fixtures.prose_history import Prose, ProseHistory

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


async def history_request(
    history, identity, section, checkpoint, counter, sizing_identity, depends=()
):
    prepared = await history.prepare(checkpoint, counter, sizing_identity)
    return Request(
        id=identity,
        section=section,
        session=history.identity,
        checkpoint=checkpoint,
        fixture_id=history.current.id,
        workload="prose" if isinstance(history, ProseHistory) else "tools",
        messages=prepared.content.messages,
        tools=prepared.content.tools,
        expected=history.current.expected,
        depends_on=tuple(depends),
        fixture_provenance=prepared.provenance,
    )


async def compile_plan(
    fixtures: list[Interaction] | Prose,
    corpus_digest: str,
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    case: str | None = None,
    *,
    counter: Counter,
    sizing_identity: str,
) -> Plan:
    selected = (
        [] if isinstance(fixtures, Prose) else [f for f in fixtures if case is None or f.id == case]
    )
    if isinstance(fixtures, Prose):
        if case is not None:
            raise ValueError("--case is only supported for tool fixtures")
    elif not selected:
        raise ValueError(f"unknown or excluded BFCL case: {case}")

    def history_for(identity: str, index: int = 0) -> History | ProseHistory:
        if isinstance(fixtures, Prose):
            return ProseHistory(fixtures, identity)
        return History(fixtures, identity, selected[index % len(selected)])

    requests = []
    capacity = 1
    # The fixture selection for one section is independent of other selected sections.
    for section in TypeAdapter(tuple[Section, ...]).validate_python(sections):
        if section == "single":
            if isinstance(fixtures, Prose):
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
            for checkpoint in contexts:
                for i, history in enumerate(histories):
                    identity = f"{section}-{i}-t{checkpoint}"
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
                    shared.complete()
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
    return Plan(requests=tuple(requests), parallel_sequences=capacity, corpus_digest=corpus_digest)
