import json

import pytest
from bench_support import body_length, character_lengths, template_file, tools_session
from magnitude_bench.plan import Sized, checked, compile_plan, requested
from magnitude_bench.policy import LANES
from magnitude_bench.sizing import chat, size

SECTIONS = ("context", "session", "parallel", "fork", "concurrency", "memory")


class Recording:
    """A renderer that records each batch it is asked for."""

    def __init__(self):
        self.batches = []

    async def __call__(self, bodies):
        self.batches.append(len(bodies))
        return await character_lengths(bodies)


async def test_sizing_matches_the_session_rule_and_batches_renders(sessions):
    renderer = Recording()
    targets = [(session, 2048) for session in sessions[:4]] + [(sessions[0], 4096)]
    sized = await size(targets, renderer)

    def length(messages, tools):
        return len(json.dumps(chat(messages, tools)))

    for (session, target), conversation in zip(targets, sized, strict=True):
        assert conversation == session.sized(target, length)
        assert length(conversation.messages, conversation.tools) <= target
    # Every sizing waits in lockstep: far fewer render calls than lengths asked for.
    assert len(renderer.batches) < sum(renderer.batches)
    assert max(renderer.batches) > 1


async def test_sizing_failure_reaches_every_waiter():
    async def broken(bodies):
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        await size([(tools_session(0), 2048), (tools_session(1), 2048)], broken)


async def test_contexts_a_session_cannot_fill_or_hold_are_refused(sessions):
    with pytest.raises(ValueError, match="too little history"):
        await size([(sessions[0], 10**7)], character_lengths)
    with pytest.raises(ValueError, match="head and probe alone exceed"):
        await size([(sessions[0], 100)], character_lengths)


def plan_of(files, sections=SECTIONS):
    """The plan of ``sections`` over ``files``, every lane sized by the stand-in rendering."""
    ordered = checked(files)
    sized = {(file.template, lane): Sized(file.sessions[lane].sized(file.context, body_length), None)
             for file, lane in requested(ordered, sections)}
    return compile_plan(ordered, sections, sized)


def test_sections_shape_offered_traffic():
    files = [template_file(4), template_file(2)]
    plan = plan_of(files)
    by_id = {request.id: request for request in plan.requests}
    assert plan.parallel_sequences == 8
    # Every request is a template's lane: none is unsized.
    assert {request.checkpoint for request in plan.requests} == {2048, 4096}
    assert {request.template for request in plan.requests} == {"tools:2k", "tools:4k"}
    # The warmup is the qualification lane at the smallest context, never a measured lane.
    assert (plan.warmup.template, plan.warmup.lane) == ("tools:2k", LANES)
    assert LANES not in {request.lane for request in plan.requests}
    # Each request is its lane's session sized to the checkpoint, served up to the probe.
    context = by_id["context-t4096-c1-0"]
    assert context.messages[-1] == {"role": "user", "content": "Call echo with value 7."}
    assert context.depends_on == ("context-t2048-c1-0",)
    assert len(context.messages) > len(by_id["context-t2048-c1-0"].messages)
    # Sessions advance one lane through the checkpoints.
    assert by_id["memory-2-t4096"].depends_on == ("memory-2-t2048",)
    assert by_id["memory-2-t4096"].session == "memory-2"
    assert by_id["memory-2-t4096"].lane == 2
    # Fork branches share the parent's history and start after it.
    branches = [by_id[f"fork-t2048-c4-{i}"] for i in range(4)]
    assert all(branch.depends_on == ("fork-t2048-c4-parent",) for branch in branches)
    assert len({json.dumps(branch.messages) for branch in branches}) == 1
    # Concurrency points use distinct lanes and follow the previous point.
    widest = [by_id[f"concurrency-t2048-c8-{i}"] for i in range(8)]
    assert {request.lane for request in widest} == set(range(8))
    assert widest[0].depends_on == tuple(f"concurrency-t2048-c4-{i}" for i in range(4))
    assert all(request.tools and request.key == "tools" for request in plan.requests)
    assert plan.identity == plan_of(files).identity


def test_templates_of_another_shape_are_refused(sessions):
    with pytest.raises(ValueError, match="qualification lane"):
        checked([template_file(2, sessions[:-1])])
    with pytest.raises(ValueError, match="sessions"):
        checked([template_file(2), template_file(4, [tools_session(lane, 30) for lane in range(LANES + 1)])])
    with pytest.raises(ValueError, match="distinct contexts"):
        checked([template_file(2), template_file(2)])
    with pytest.raises(ValueError, match="choose sections"):
        compile_plan(checked([template_file(2)]), ("single",), {})
