import json

import pytest
from magnitude_fixtures import bfcl
from magnitude_fixtures.bfcl import normalize_schema


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
    sessions = bfcl.build(interactions, lanes=3)
    for lane, session in enumerate(sessions):
        probe = interactions[lane]
        units = session.units()
        assert len(units) == len(interactions)
        questions = [session.messages[start]["content"] for start, _ in units]
        assert questions.count(probe.question) == 1
        assert questions[-1] == probe.question
        # History rotates from the decision after the probe, so lanes end differently.
        assert questions[0] == interactions[(lane + 1) % len(interactions)].question
        assert session.tools == probe.tools
        history = [interactions[(lane + 1 + i) % len(interactions)] for i in range(len(units) - 1)]
        for (start, _), interaction in zip(units, history, strict=False):
            assert session.messages[start]["tools"] == interaction.tools
        assert "tools" not in session.messages[units[-1][0]]
        assert [m for m in session.messages if m.get("scored")] == [
            session.messages[session.generation.prefix]
        ]
        assert session.messages[session.generation.prefix - 1] == {
            "role": "user",
            "content": probe.question,
        }
        assert session.generation.expected == {
            "calls": [{"name": probe.expected[0].name, "arguments": {"value": [lane]}}]
        }


def test_completed_calls_have_unique_ids_and_matching_replies(interactions):
    session = bfcl.build(interactions, lanes=1)[0]
    calls = [c for m in session.messages for c in m.get("tool_calls", [])]
    replies = [m["tool_call_id"] for m in session.messages if m["role"] == "tool"]
    assert len({c["id"] for c in calls}) == len(calls)
    assert replies == [c["id"] for c in calls]
    for call in calls:
        assert json.loads(call["function"]["arguments"]).keys() == {"value"}


def test_categories_select_probes_and_history(interactions):
    other = interactions[0].model_copy(update={"id": "parallel_0", "category": "parallel"})
    (session,) = bfcl.build([other, *interactions], lanes=1, categories=("parallel",))
    assert session.messages[session.units()[-1][0]]["content"] == other.question
    assert len(session.units()) == 1
    with pytest.raises(ValueError, match="lanes"):
        bfcl.build(interactions, lanes=len(interactions) + 1)
