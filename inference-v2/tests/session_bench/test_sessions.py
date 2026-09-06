from benchmark_fixtures.interactions import ExpectedCall
from benchmark_fixtures.records import encoded
from session_bench.policy import MAX_OUTPUT_TOKENS
from session_bench.suites import SECTIONS, compile_plan


async def count_context(context):
    return len(encoded(context.model_dump(mode="json")))


async def test_shared_deterministic_sessions_and_limits(interaction):
    args = ([interaction], "corpus", tuple(SECTIONS), (1024, 4096))
    plan = await compile_plan(*args, counter=count_context, sizing_identity="test-bytes")
    assert plan == (await compile_plan(*args, counter=count_context, sizing_identity="test-bytes"))
    assert plan.parallel_sequences == 8
    seen = set()
    for request in plan.requests:
        assert set(request.depends_on) <= seen
        seen.add(request.id)
        body = request.body("one")
        other = request.body("two")
        assert body.pop("model") == "one"
        other.pop("model")
        assert body == other
        assert body["max_tokens"] == MAX_OUTPUT_TOKENS == 32768
        assert "max_tokens" not in type(request).model_fields
    assert len(seen) == len(plan.requests)


async def test_sections_do_not_change_other_sections(interaction):
    both = await compile_plan(
        [interaction],
        "c",
        ("single", "context"),
        (1024,),
        counter=count_context,
        sizing_identity="test-bytes",
    )
    alone = await compile_plan(
        [interaction],
        "c",
        ("context",),
        (1024,),
        counter=count_context,
        sizing_identity="test-bytes",
    )
    assert tuple(r for r in both.requests if r.section == "context") == alone.requests


async def test_history_grows_and_calls_have_unique_ids(interaction):
    plan = await compile_plan(
        [interaction],
        "c",
        ("session",),
        (1024, 4096, 16384),
        counter=count_context,
        sizing_identity="test-bytes",
    )
    sizes = [len(encoded(r.messages)) for r in plan.requests]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    for request in plan.requests:
        calls = [c["id"] for m in request.messages for c in m.get("tool_calls", [])]
        assert len(calls) == len(set(calls))


async def test_tool_definition_conflicts_do_not_overwrite_current_decision(interaction):
    other = interaction.model_copy(
        update={
            "id": "other",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Different schema",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "expected": [ExpectedCall(name="echo", arguments={"other": ["wrong"]})],
        }
    )
    plan = await compile_plan(
        [interaction, other],
        "c",
        ("context",),
        (1024,),
        counter=count_context,
        sizing_identity="test-bytes",
    )
    assert plan.requests[0].tools == interaction.tools
    assert all("Different schema" not in encoded(r.messages) for r in plan.requests)
    assert len(encoded(plan.requests[0].body("test"))) < 6000


async def test_fork_establishes_parent_before_children(interaction):
    plan = await compile_plan(
        [interaction], "c", ("fork",), (1024,), counter=count_context, sizing_identity="test-bytes"
    )
    parent, *children = plan.requests
    assert len(children) == 4
    for child in children:
        assert child.depends_on == (parent.id,)
        assert child.messages[: len(parent.messages)] == parent.messages


async def test_capacity_includes_warmup_and_wire_order_matches_saved_input(interaction):
    import json

    from session_bench.sessions import Request, encoded

    plan = await compile_plan(
        [interaction],
        "corpus",
        ("single",),
        (1024,),
        counter=count_context,
        sizing_identity="test-bytes",
    )
    assert {r.id for r in plan.prepared_requests} == {"single", "warmup"}
    assert plan.warmup.messages != plan.requests[0].messages
    for request in plan.prepared_requests:
        saved = Request.model_validate_json(encoded(request.model_dump(mode="json")))
        # Template rendering can depend on tool-schema key order, so dict equality is insufficient.
        assert json.dumps(request.body("test")) == json.dumps(saved.body("test"))
        assert request.body("test")["max_tokens"] == 32768
