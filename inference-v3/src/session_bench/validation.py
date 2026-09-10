"""Protocol evidence and BFCL semantic correctness are independent gates."""

import json
import math

from .sessions import ExpectedCall, encoded


def json_equal(left, right) -> bool:
    # JSON has one number type; Python bool must not compare equal to numeric 0/1.
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


def tool_calls(expected: list[ExpectedCall], actual: list[dict]) -> str | None:
    if len(expected) != len(actual):
        return f"expected {len(expected)} calls, received {len(actual)}"
    decoded = []
    for call in actual:
        try:
            arguments = json.loads(call["arguments"])
            encoded(arguments)  # Reject non-JSON NaN/Infinity as well.
        except (ValueError, TypeError):
            return "tool arguments are not valid JSON"
        if not isinstance(arguments, dict):
            return "tool arguments must be an object"
        decoded.append(arguments)

    def matches(wanted, index):
        if wanted.name != actual[index]["name"]:
            return False
        for key, alternatives in wanted.arguments.items():
            if key not in decoded[index]:
                if "" in alternatives or None in alternatives:
                    continue
                return False
            if not any(json_equal(decoded[index][key], value) for value in alternatives):
                return False
        return True

    candidates = [[i for i in range(len(actual)) if matches(wanted, i)] for wanted in expected]
    assigned = [-1] * len(actual)

    def assign(index, visited):
        for partner in candidates[index]:
            if partner in visited:
                continue
            visited.add(partner)
            if assigned[partner] == -1 or assign(assigned[partner], visited):
                assigned[partner] = index
                return True
        return False

    if not all(assign(i, set()) for i in range(len(expected))):
        return "tool calls do not have distinct matches with BFCL allowed arguments"
    return None


def terminal(payload: dict) -> dict:
    if payload.get("choices") != []:
        raise ValueError("terminal usage event must have empty choices")
    usage, timing = payload.get("usage"), payload.get("timings")
    if not isinstance(usage, dict) or not isinstance(timing, dict):
        raise ValueError("missing terminal usage or native timings")
    try:
        counts = [usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        counts += [usage["prompt_tokens_details"]["cached_tokens"]]
        counts += [timing[k] for k in ("cache_n", "prompt_n", "predicted_n")]
        durations = [timing[k] for k in ("prompt_ms", "predicted_ms")]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing terminal field: {error}") from error
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("terminal counts must be nonnegative integers")
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in durations):
        raise ValueError("terminal times must be finite nonnegative milliseconds")
    prompt, output, total, cached, cache_n, prompt_n, predicted_n = counts
    if (
        total != prompt + output
        or prompt != cache_n + prompt_n
        or cached != cache_n
        or output != predicted_n
    ):
        raise ValueError("terminal token counts disagree")
    draft, accepted = timing.get("draft_n"), timing.get("draft_n_accepted")
    if (draft is None) != (accepted is None):
        raise ValueError("draft counters must be supplied together")
    if draft is not None and (
        type(draft) is not int or type(accepted) is not int or not 0 <= accepted <= draft
    ):
        raise ValueError("invalid draft counters")
    return {"usage": usage, "timings": timing}
