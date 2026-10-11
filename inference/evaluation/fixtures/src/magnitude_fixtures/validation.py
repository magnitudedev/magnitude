"""Answer validation against a fixture's ``generation.expected``.

- ``tools``: ``{"calls": [{"name", "arguments": {key: [allowed values]}}]}``. Each generated
  call must match a distinct expected call; an argument may be omitted only when ``""`` or
  ``null`` is among its allowed values (BFCL possible-answer semantics).
- ``retrieval``: ``{"values": {key: value}}``. The answer must be a JSON object of strings
  with no duplicate keys; it is exact when it equals the expected values.
- ``prose-repeat``: ``{"text": …}`` is the canonical repeat; ``prose-continue``: ``{}``.
  Prose answers are valid when they contain text; the serving benchmark judges finish
  reasons against its output budget.
"""

import json

from pydantic import Field

from .canonical import Record, encoded
from .tools import ExpectedCall


class GeneratedCall(Record):
    name: str
    arguments: str


class RetrievalScore(Record):
    exact_match: bool
    correct: int = Field(ge=0)
    total: int = Field(gt=0)
    format_valid: bool


def json_equal(left, right) -> bool:
    # JSON has one number type; a boolean must not compare equal to numeric 0/1.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def tool_calls(expected: dict, actual: list[GeneratedCall]) -> str | None:
    """``None`` when ``actual`` matches the expected calls, else the reason it does not."""
    wanted = [ExpectedCall.model_validate(call) for call in expected["calls"]]
    if len(wanted) != len(actual):
        return f"expected {len(wanted)} calls, received {len(actual)}"
    decoded = []
    for call in actual:
        try:
            arguments = json.loads(call.arguments)
            encoded(arguments)  # Reject non-JSON NaN/Infinity as well.
        except (ValueError, TypeError):
            return "tool arguments are not valid JSON"
        if not isinstance(arguments, dict):
            return "tool arguments must be an object"
        decoded.append(arguments)

    def matches(call: ExpectedCall, index: int) -> bool:
        if call.name != actual[index].name:
            return False
        for key, alternatives in call.arguments.items():
            if key not in decoded[index]:
                if "" in alternatives or None in alternatives:
                    continue
                return False
            if not any(json_equal(decoded[index][key], value) for value in alternatives):
                return False
        return True

    candidates = [[i for i in range(len(actual)) if matches(call, i)] for call in wanted]
    assigned = [-1] * len(actual)

    def assign(index: int, visited: set[int]) -> bool:
        for partner in candidates[index]:
            if partner in visited:
                continue
            visited.add(partner)
            if assigned[partner] == -1 or assign(assigned[partner], visited):
                assigned[partner] = index
                return True
        return False

    if not all(assign(i, set()) for i in range(len(wanted))):
        return "tool calls do not have distinct matches with BFCL allowed arguments"
    return None


def retrieval(expected: dict, text: str) -> RetrievalScore:
    values: dict[str, str] = expected["values"]

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate answer key")
            result[key] = value
        return result

    try:
        answer = json.loads(text, object_pairs_hook=unique_object)
        valid = isinstance(answer, dict) and all(type(v) is str for v in answer.values())
    except (ValueError, RecursionError):
        answer, valid = {}, False
    return RetrievalScore(
        exact_match=valid and answer == values,
        correct=sum(answer.get(k) == v for k, v in values.items()) if valid else 0,
        total=len(values),
        format_valid=valid,
    )
