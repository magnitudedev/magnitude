import json

import pytest
from magnitude_fixtures import Source, tools
from magnitude_fixtures.tools import normalize_schema

SOURCE = Source(name="test", digest="0" * 64)


def test_optional_extension_is_represented_by_required_list():
    schema = {
        "type": "dict",
        "properties": {"name": {"type": "string"}, "rating": {"type": "float"}},
        "required": ["name"],
        "optional": ["rating"],
    }
    assert normalize_schema(schema) == {
        "type": "object",
        "properties": {"name": {"type": "string"}, "rating": {"type": "number"}},
        "required": ["name"],
    }


def test_undeclared_required_names_are_removed():
    schema = {
        "type": "object",
        "properties": {
            "population": {
                "type": "object",
                "description": "Adults, children, and single residents.",
                "required": ["adults", "children", "singles"],
            }
        },
        "required": ["population", "missing"],
    }
    assert normalize_schema(schema) == {
        "type": "object",
        "properties": {
            "population": {
                "type": "object",
                "description": "Adults, children, and single residents.",
            }
        },
        "required": ["population"],
    }


def test_string_format_stays_in_description_without_grammar_constraint():
    schema = {"type": "string", "format": "date", "description": "Date in YYYY-MM-DD form"}
    assert normalize_schema(schema) == {"type": "string", "description": "Date in YYYY-MM-DD form"}


def test_sessions_complete_every_other_decision_then_probe(interactions):
    fixtures = tools.build(interactions, SOURCE, count=3)
    assert [f.id for f in fixtures] == [f"tools/bfcl/simple_python_{i}" for i in range(3)]
    for lane, fixture in enumerate(fixtures):
        probe = interactions[lane]
        units = fixture.units()
        assert len(units) == len(interactions)
        questions = [fixture.messages[start]["content"] for start, _ in units]
        assert questions.count(probe.question) == 1
        assert questions[-1] == probe.question
        # History rotates from the decision after the probe, so lanes end differently.
        assert questions[0] == interactions[(lane + 1) % len(interactions)].question
        assert fixture.tools == probe.tools
        history = [interactions[(lane + 1 + i) % len(interactions)] for i in range(len(units) - 1)]
        for (start, _), interaction in zip(units, history, strict=False):
            assert fixture.messages[start]["tools"] == interaction.tools
        assert "tools" not in fixture.messages[units[-1][0]]
        assert [m for m in fixture.messages if m.get("scored")] == [
            fixture.messages[fixture.generation.prefix]
        ]
        assert fixture.messages[fixture.generation.prefix - 1] == {
            "role": "user",
            "content": probe.question,
        }
        assert fixture.generation.expected == {
            "calls": [{"name": probe.expected[0].name, "arguments": {"value": [lane]}}]
        }


def test_completed_calls_have_unique_ids_and_matching_replies(interactions):
    fixture = tools.build(interactions, SOURCE, count=1)[0]
    calls = [c for m in fixture.messages for c in m.get("tool_calls", [])]
    replies = [m["tool_call_id"] for m in fixture.messages if m["role"] == "tool"]
    assert len({c["id"] for c in calls}) == len(calls)
    assert replies == [c["id"] for c in calls]
    for call in calls:
        assert json.loads(call["function"]["arguments"]).keys() == {"value"}


def test_categories_select_probes_and_history(interactions):
    other = interactions[0].model_copy(update={"id": "parallel_0", "category": "parallel"})
    fixtures = tools.build([other, *interactions], SOURCE, count=1, categories=("parallel",))
    assert [f.id for f in fixtures] == ["tools/bfcl/parallel_0"]
    assert len(fixtures[0].units()) == 1
    with pytest.raises(ValueError, match="probes"):
        tools.build(interactions, SOURCE, count=len(interactions) + 1)
