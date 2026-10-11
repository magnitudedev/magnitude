"""Terminal protocol evidence: native counters must be complete and agree with usage.

Answer semantics (tool calls, retrieval) are validated by ``magnitude_fixtures.validation``.
"""

import math


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
