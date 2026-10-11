import json

import pytest
from bench_support import character_lengths, tools_fixture
from magnitude_bench.plan import compile_plan
from magnitude_bench.sizing import chat, size


class Recording:
    """A renderer that records each batch it is asked for."""

    def __init__(self):
        self.batches = []

    async def __call__(self, bodies):
        self.batches.append(len(bodies))
        return await character_lengths(bodies)


async def test_sizing_matches_fixture_rule_and_batches_renders(fixtures):
    renderer = Recording()
    targets = [(fixture, 1200) for fixture in fixtures[:4]] + [(fixtures[0], 2500)]
    sized = await size(targets, renderer)

    def length(messages, tools):
        return len(json.dumps(chat(messages, tools)))

    for (fixture, target), conversation in zip(targets, sized, strict=True):
        assert conversation == fixture.sized(target, length)
        assert length(conversation.messages, conversation.tools) <= target
    # Every sizing waits in lockstep: far fewer render calls than lengths asked for.
    assert len(renderer.batches) < sum(renderer.batches)
    assert max(renderer.batches) > 1


async def test_sizing_failure_reaches_every_waiter():
    async def broken(bodies):
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        await size([(tools_fixture(0), 1200), (tools_fixture(1), 1200)], broken)


async def test_contexts_a_fixture_cannot_fill_or_hold_are_refused(fixtures):
    with pytest.raises(ValueError, match="too little history"):
        await size([(fixtures[0], 10**7)], character_lengths)
    with pytest.raises(ValueError, match="head and probe alone exceed"):
        await size([(fixtures[0], 100)], character_lengths)


async def test_sections_shape_offered_traffic(fixtures):
    plan = await compile_plan(
        fixtures, ("single", "context", "session", "parallel", "fork", "concurrency", "memory"),
        (1200, 2400), character_lengths,
    )
    by_id = {request.id: request for request in plan.requests}
    assert plan.parallel_sequences == 8
    assert by_id["single"].checkpoint == 0 and len(by_id["single"].messages) == 2
    # The warmup is the qualification lane, never a measured lane.
    assert plan.warmup.fixture_id == fixtures[-1].id
    assert fixtures[-1].id not in {request.fixture_id for request in plan.requests}
    # Each request is its lane's fixture sized to the checkpoint, served up to the probe.
    context = by_id["context-t2400-c1-0"]
    assert context.messages[-1] == {"role": "user", "content": "Call echo with value 7."}
    assert context.depends_on == ("context-t1200-c1-0",)
    assert len(context.messages) > len(by_id["context-t1200-c1-0"].messages)
    # Sessions advance one lane through the checkpoints.
    assert by_id["memory-2-t2400"].depends_on == ("memory-2-t1200",)
    assert by_id["memory-2-t2400"].session == "memory-2"
    assert by_id["memory-2-t2400"].fixture_id == fixtures[2].id
    # Fork branches share the parent's history and start after it.
    branches = [by_id[f"fork-t1200-c4-{i}"] for i in range(4)]
    assert all(branch.depends_on == ("fork-t1200-c4-parent",) for branch in branches)
    assert len({json.dumps(branch.messages) for branch in branches}) == 1
    # Concurrency points use distinct lanes and follow the previous point.
    widest = [by_id[f"concurrency-t1200-c8-{i}"] for i in range(8)]
    assert len({request.fixture_id for request in widest}) == 8
    assert widest[0].depends_on == tuple(f"concurrency-t1200-c4-{i}" for i in range(4))
    assert all(request.tools for request in plan.requests)
    assert plan.identity == (
        await compile_plan(
            fixtures,
            ("single", "context", "session", "parallel", "fork", "concurrency", "memory"),
            (1200, 2400), character_lengths,
        )
    ).identity


async def test_a_set_without_the_qualification_lane_is_refused(fixtures):
    with pytest.raises(ValueError, match="qualification lane"):
        await compile_plan(fixtures[:-1], ("context",), (1200,), character_lengths)
