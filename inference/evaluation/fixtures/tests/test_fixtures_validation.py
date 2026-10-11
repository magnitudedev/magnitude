import json

import pytest
from magnitude_fixtures.validation import GeneratedCall, retrieval, tool_calls

EXPECTED = {
    "calls": [
        {"name": "area", "arguments": {"base": [10], "unit": ["units", ""]}},
        {"name": "area", "arguments": {"base": [3, 3.0]}},
    ]
}


def call(name: str, **arguments) -> GeneratedCall:
    return GeneratedCall(name=name, arguments=json.dumps(arguments))


def test_calls_match_distinct_expectations_in_any_order():
    assert tool_calls(EXPECTED, [call("area", base=3), call("area", base=10)]) is None
    assert (
        tool_calls(EXPECTED, [call("area", base=10, unit="units"), call("area", base=3.0)]) is None
    )


@pytest.mark.parametrize(
    "actual,reason",
    [
        ([call("area", base=10)], "expected 2 calls"),
        ([call("area", base=10), call("area", base=10)], "distinct matches"),
        ([call("area", base=True), call("area", base=3)], "distinct matches"),
        ([call("area", base=10), GeneratedCall(name="area", arguments="[3]")], "object"),
        ([call("area", base=10), GeneratedCall(name="area", arguments="{")], "valid JSON"),
        ([call("area", base=10), call("volume", base=3)], "distinct matches"),
    ],
)
def test_mismatched_calls_explain_why(actual, reason):
    assert reason in tool_calls(EXPECTED, actual)


@pytest.mark.parametrize(
    "text,exact,correct,valid",
    [
        ('{"a": "1", "b": "2"}', True, 2, True),
        ('{"a": "1", "b": "x"}', False, 1, True),
        ('{"a": "1", "a": "1", "b": "2"}', False, 0, False),
        ('{"a": 1, "b": "2"}', False, 0, False),
        ("not JSON", False, 0, False),
    ],
)
def test_retrieval_scores_strict_json(text, exact, correct, valid):
    score = retrieval({"values": {"a": "1", "b": "2"}}, text)
    assert (score.exact_match, score.correct, score.total, score.format_valid) == (
        exact,
        correct,
        2,
        valid,
    )
