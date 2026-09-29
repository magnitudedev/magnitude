"""Bounded reasoning-control discovery, ported from icn-reasoning's template probes."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from templates.events import Record
from templates.native import NativeError, Template

DETECTOR_VERSION = "v3-reasoning-1"
EFFORTS = {
    "minimal": ("minimal",),
    "low": ("low",),
    "medium": ("medium",),
    "high": ("high",),
    "xhigh": ("xhigh", "extra_high", "extra-high", "very_high"),
    "max": ("max",),
}
DISABLED = ("none", "off", "no_think", "disabled")
PROBE_TIME = 946684800
REASONING_CONTROLS = frozenset({"enable_thinking", "thinking", "thinking_mode", "reasoning_effort"})


class Control(Record):
    name: str
    value: str | bool


class EffortMapping(Record):
    effort: str
    controls: tuple[Control, ...]
    aliases: tuple[str, ...] = ()


class ReasoningProfile(Record):
    detector: str = DETECTOR_VERSION
    template_identity: str
    option_context: str = "{}"
    default_effort: str | None
    mappings: tuple[EffortMapping, ...]
    baseline_shapes: tuple[bool, ...]
    effort_domain: Literal["closed", "shared_fallback", "open_or_ignored"]
    supports_reasoning_output: bool | None
    supports_preserve_reasoning: bool

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def resolve(self, effort: str | None) -> dict[str, str | bool]:
        # Default classification is informational. Omission must stay omitted.
        if effort is None:
            return {}
        for mapping in self.mappings:
            if effort == mapping.effort or effort in mapping.aliases:
                return {control.name: control.value for control in mapping.controls}
        available = ", ".join(mapping.effort for mapping in self.mappings)
        raise ValueError(f"Unsupported reasoning effort {effort}; available: {available}")


@dataclass(frozen=True)
class Signature:
    prompt: str
    prefix: str
    parser: str
    grammar: str
    thinking: bool
    start: str
    ends: tuple[str, ...]


type ProbeShape = tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]


def _shapes() -> tuple[ProbeShape, ...]:
    tool: dict[str, JsonValue] = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    question: dict[str, JsonValue] = {"role": "user", "content": "What is the weather in Paris?"}
    return (
        ([{"role": "user", "content": "Explain why the sky appears blue."}], []),
        ([question], [tool]),
        (
            [
                question,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should check the weather tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "18 C and clear",
                    "name": "weather",
                    "tool_call_id": "call_1",
                },
            ],
            [tool],
        ),
    )


def _comparable(base, candidate):
    return any(item is not None for item in base) and all(
        a is None or b is not None for a, b in zip(base, candidate, strict=True)
    )


def _names(outcomes, effort):
    return any(
        re.search(
            r"(?<![a-z0-9_-])" + re.escape(name) + r"(?![a-z0-9_-])",
            item.prompt.lower() + "\n" + item.prefix.lower(),
        )
        for item in outcomes
        if item is not None
        for name in EFFORTS.get(effort, ())
    )


def inspect_reasoning(
    template: Template, *, template_arguments: dict[str, JsonValue] | None = None
) -> ReasoningProfile:
    """Compare complete prepared signatures across the Rust detector's three shapes.

    Rejected shapes compare as rejected regardless of diagnostic wording. Two
    random invalid effort strings distinguish a closed/fallback domain from a
    template that merely echoes arbitrary effort values.
    """
    shapes = _shapes()
    arguments = dict(template_arguments or {})
    if REASONING_CONTROLS.intersection(arguments):
        raise ValueError("reasoning probe context must omit reasoning controls")
    context_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cache = {}
    first_error = None

    def render(controls):
        nonlocal first_error
        key = tuple(sorted(controls.items()))
        if key not in cache:
            results = []
            for messages, tools in shapes:
                try:
                    with template.prepare(
                        messages,
                        tools=tools,
                        now=PROBE_TIME,
                        template_arguments={**arguments, **controls},
                    ) as plan:
                        d = plan.description
                        results.append(
                            Signature(
                                d.prompt,
                                d.generation_prefix,
                                d.parser,
                                d.grammar,
                                d.supports_thinking,
                                d.thinking_start,
                                d.thinking_ends,
                            )
                        )
                except NativeError as error:
                    if first_error is None:
                        first_error = str(error)
                    results.append(None)
            cache[key] = tuple(results)
        return cache[key]

    baseline = render({})
    if not any(item is not None for item in baseline):
        raise ValueError(f"Reasoning inspection rejected every conversation probe: {first_error}")
    toggle = None
    for disabled, enabled in (
        ({"enable_thinking": False}, {"enable_thinking": True}),
        ({"thinking": False}, {"thinking": True}),
        ({"thinking_mode": "chat"}, {"thinking_mode": "thinking"}),
        ({"thinking_mode": "disabled"}, {"thinking_mode": "enabled"}),
    ):
        off, on = render(disabled), render(enabled)
        if _comparable(baseline, off) and _comparable(baseline, on) and off != on:
            toggle = disabled, enabled, off, on
            break
    disabled, enabled, off, on = toggle or ({}, {}, baseline, baseline)
    adaptive = render({"thinking_mode": "adaptive"})
    has_adaptive = (
        toggle is not None and _comparable(baseline, adaptive) and adaptive not in (off, on)
    )
    effort_base = render(enabled)
    invalid_a = render(
        {**enabled, "reasoning_effort": "magnitude-invalid-" + secrets.token_hex(16)}
    )
    invalid_b = render(
        {**enabled, "reasoning_effort": "magnitude-invalid-" + secrets.token_hex(16)}
    )
    rejected = all(
        base is None or (a is None and b is None)
        for base, a, b in zip(effort_base, invalid_a, invalid_b, strict=True)
    )
    fallback = (
        _comparable(effort_base, invalid_a)
        and _comparable(effort_base, invalid_b)
        and invalid_a == invalid_b
    )

    # Each entry is (mapping, full render outcomes). Preserve aliases when
    # collapsing behavior so the public API does not invent intermediate levels.
    def observed(effort, controls, outcomes, aliases=()):
        return (
            EffortMapping(
                effort=effort,
                controls=tuple(
                    Control(name=key, value=value) for key, value in sorted(controls.items())
                ),
                aliases=aliases,
            ),
            outcomes,
        )

    def probe(effort, spellings):
        selected = None
        for spelling in spellings:
            controls = {**enabled, "reasoning_effort": spelling}
            outcomes = render(controls)
            if not _comparable(effort_base, outcomes):
                continue
            if not rejected and (
                not fallback or (outcomes == invalid_a and not _names(outcomes, effort))
            ):
                continue
            if selected is not None and selected[1] != outcomes:
                raise ValueError(f"Reasoning aliases for {effort} render differently")
            if selected is None:
                selected = observed(effort, controls, outcomes)
        return selected

    disabled_effort = probe("none", DISABLED) if rejected or fallback else None
    options = []
    if rejected or fallback:
        for effort, spellings in EFFORTS.items():
            option = probe(effort, spellings)
            if option is None:
                continue
            for index, existing in enumerate(options):
                if existing[1] != option[1]:
                    continue
                if _names(option[1], existing[0].effort) and not _names(option[1], effort):
                    options[index] = (
                        existing[0].model_copy(update={"aliases": (*existing[0].aliases, effort)}),
                        existing[1],
                    )
                    option = None
                else:
                    options.pop(index)
                    option = (
                        option[0].model_copy(
                            update={"aliases": (*existing[0].aliases, existing[0].effort)}
                        ),
                        option[1],
                    )
                break
            if option is not None:
                options.append(option)
    if not options and toggle:
        options = [observed("none", disabled, off)]
        if has_adaptive:
            options.append(observed("adaptive", {"thinking_mode": "adaptive"}, adaptive))
        options.append(observed("high", enabled, on))
    elif not options and disabled_effort is not None and disabled_effort[1] != effort_base:
        options = [disabled_effort, observed("high", enabled, effort_base)]
    elif options:
        if toggle:
            options.insert(0, observed("none", disabled, off))
        elif disabled_effort is not None:
            options.insert(0, disabled_effort)
    if not options:
        thinking = any(
            item is not None
            and (item.thinking or "<think>" in item.prompt or "<reasoning>" in item.prompt)
            for item in (*baseline, *on)
        )
        thinking |= template.capabilities().get("supports_preserve_reasoning", False)
        options = [observed("high" if thinking else "none", {}, baseline)]
    defaults = [mapping.effort for mapping, outcomes in options if outcomes == baseline]
    return ReasoningProfile(
        template_identity=template.identity,
        option_context=context_json,
        default_effort=defaults[0] if len(defaults) == 1 else None,
        mappings=tuple(mapping for mapping, _ in options),
        baseline_shapes=tuple(item is not None for item in baseline),
        effort_domain="closed"
        if rejected
        else "shared_fallback"
        if fallback
        else "open_or_ignored",
        supports_reasoning_output=True
        if any(
            item is not None and bool(item.thinking or item.start or item.ends)
            for item in (*baseline, *on)
        )
        else None,
        supports_preserve_reasoning=template.capabilities().get(
            "supports_preserve_reasoning", False
        ),
    )
