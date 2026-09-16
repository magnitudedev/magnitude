"""Rust detector fixtures, with v3's authored omission and strict resolution policy."""

import json
from pathlib import Path

import pytest

from templates import Template
from templates.reasoning import ReasoningProfile, inspect_reasoning

FIXTURES = json.loads(Path(__file__).with_name("reasoning-fixtures.json").read_text())


@pytest.mark.parametrize(
    "name,efforts",
    [
        ("BASIC", ["none"]),
        ("TOGGLE", ["none", "high"]),
        ("FIXED", ["high"]),
        ("THINKING_BOOL", ["none", "high"]),
        ("THINKING_MODE", ["none", "adaptive", "high"]),
        ("EFFORT_TOGGLE", ["none", "high"]),
        ("EFFORT_NONE_MATCHES_LOW", ["none", "low", "high"]),
        ("CLOSED_EFFORT", ["none", "low", "medium", "high"]),
        ("QWEN_3_8_EFFORT", ["none", "low", "medium", "xhigh"]),
        ("REVERSE_EFFORT_ALIAS", ["low", "medium", "high"]),
        ("ONE_ENABLED_EFFORT_BEHAVIOR", ["none", "max"]),
        ("SHARED_FALLBACK_EFFORT", ["none", "low", "high"]),
        ("NAMED_SHARED_FALLBACK_EFFORT", ["none", "high", "max"]),
        ("OPEN_EFFORT", ["none", "high"]),
    ],
)
def test_port_preserves_rust_behavior_classification(name, efforts):
    with Template(FIXTURES[name]) as template:
        profile = inspect_reasoning(template)
        assert [mapping.effort for mapping in profile.mappings] == efforts
        assert profile.resolve(None) == {}
        assert ReasoningProfile.model_validate_json(profile.model_dump_json()) == profile


def test_omitted_boolean_remains_authored_false_without_inventing_default_enablement():
    with Template(FIXTURES["TOGGLE"]) as template:
        profile = inspect_reasoning(template)
        assert profile.default_effort == "none"
        assert profile.resolve(None) == {}
        assert profile.resolve("high") == {"enable_thinking": True}
        with pytest.raises(ValueError, match="Unsupported reasoning effort"):
            profile.resolve("medium")


def test_recognized_alias_resolves_to_the_same_behavior_without_an_invented_level():
    with Template(FIXTURES["QWEN_3_8_EFFORT"]) as template:
        profile = inspect_reasoning(template)
        assert profile.default_effort == "xhigh"
        assert profile.resolve("high") == profile.resolve("xhigh")
        assert "high" not in [mapping.effort for mapping in profile.mappings]


def test_profiles_include_independent_template_options_in_behavior_and_identity():
    source = (
        "{% if extra_levels %}"
        + FIXTURES["CLOSED_EFFORT"]
        + "{% else %}"
        + FIXTURES["TOGGLE"]
        + "{% endif %}"
    )
    with Template(source) as template:
        plain = inspect_reasoning(template)
        extended = inspect_reasoning(template, template_arguments={"extra_levels": True})
        assert [mapping.effort for mapping in plain.mappings] == ["none", "high"]
        assert [mapping.effort for mapping in extended.mappings] == [
            "none",
            "low",
            "medium",
            "high",
        ]
        assert plain.template_identity == extended.template_identity
        assert plain.fingerprint != extended.fingerprint
        assert json.loads(extended.option_context) == {"extra_levels": True}
        assert (
            inspect_reasoning(template, template_arguments={"extra_levels": True}).fingerprint
            == extended.fingerprint
        )
