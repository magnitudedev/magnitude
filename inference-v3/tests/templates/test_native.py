"""Native boundary qualification, independent of engine/device imports."""

import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from templates import NativeError, Template
from templates.native import ABI_VERSION, UPSTREAM_REVISION, _Buffer, _get_library


def test_authored_default_and_explicit_boolean_controls_are_distinct():
    source = (
        "{% if enable_thinking is not defined %}default"
        "{% elif enable_thinking %}on{% else %}off{% endif %}"
    )
    with Template(source) as template:
        assert template.render({}, now=0) == "default"
        assert template.render({"enable_thinking": True}, now=0) == "on"
        assert template.render({"enable_thinking": False}, now=0) == "off"
        assert template.build_info.abi == ABI_VERSION
        assert template.build_info.upstream == UPSTREAM_REVISION


def test_render_preserves_utf8_and_embedded_zero_bytes():
    with Template("{{ messages[0].content }}") as template:
        text = "héllo 世界 🦙\0tail"
        assert template.render({"messages": [{"role": "user", "content": text}]}, now=0) == text


def test_special_tokens_are_artifact_owned_and_missing_values_stay_undefined():
    with Template(
        "{{ bos_token }}{{ eos_token is defined }}", special_tokens={"bos_token": "<B>"}
    ) as template:
        assert template.render({}, now=0) == "<B>False"
        with pytest.raises(NativeError, match="overrides"):
            template.render({"bos_token": "invented"}, now=0)


def test_explicit_time_is_repeatable_and_changes_rendering():
    with Template("{{ strftime_now('%Y') }}") as template:
        assert template.render({}, now=962409600) == "2000"
        assert template.render({}, now=994032000) == "2001"
        assert template.render({}, now=962409600) == "2000"


def test_prepared_request_owns_its_plan_after_template_release():
    fixtures = Path(__file__).resolve().parents[2] / "native/templates/upstream/models/templates"
    template = Template((fixtures / "Qwen-Qwen3-0.6B.jinja").read_text())
    messages = [{"role": "user", "content": "Hello"}]
    authored = template.render({"messages": messages, "add_generation_prompt": True}, now=946684800)
    prepared = template.prepare(messages, now=946684800)
    template.close()
    with prepared:
        assert "Hello" in prepared.description.prompt
        assert prepared.description.parser
        assert prepared.description.prompt == authored
        assert prepared.description.grammar_dialect == "gbnf"


def test_preparation_preserves_disabled_thinking_and_rejects_reserved_arguments():
    fixtures = Path(__file__).resolve().parents[2] / "native/templates/upstream/models/templates"
    with Template((fixtures / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}],
            now=946684800,
            template_arguments={"enable_thinking": False},
        ) as prepared:
            assert prepared.description.prompt.endswith("<think>\n\n</think>\n\n")
        with pytest.raises(NativeError, match="Reserved"):
            template.prepare(
                [{"role": "user", "content": "Hello"}],
                now=946684800,
                template_arguments={"messages": []},
            )


def test_lifecycle_and_concurrent_owners():
    def render(index):
        with Template("{{ value }}") as template:
            return template.render({"value": f"owner-{index}"}, now=0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(render, range(40))) == [f"owner-{i}" for i in range(40)]
    template = Template("hello")
    template.close()
    template.close()
    with pytest.raises(NativeError, match="closed"):
        template.render({}, now=0)


def test_template_errors_do_not_poison_subsequent_requests():
    with pytest.raises(NativeError):
        Template("{% if %}")
    with Template("{{ raise_exception('bad request') }}") as template:
        with pytest.raises(NativeError, match="bad request"):
            template.render({}, now=0)
    with Template("ok") as template:
        assert template.render({}, now=0) == "ok"


def test_null_oversize_version_mismatch_and_stale_handles():
    library = _get_library()
    for payload, length, expected in [
        (None, 0, 1),
        (b"x", 2**64 - 1, 1),
        (b'{"version":2}', 13, 5),
    ]:
        handle, error = ctypes.c_uint64(), _Buffer()
        status = library.lib.templates_template_create(
            payload, length, ctypes.byref(handle), ctypes.byref(error)
        )
        assert status == expected
        assert handle.value == 0
        assert library._take(error)
    error = _Buffer()
    assert library.lib.templates_template_release(2**64 - 1, ctypes.byref(error)) == 2
    assert library._take(error)
    assert library.lib.templates_buffer_release(2**64 - 1) == 2


def test_output_buffer_survives_other_calls_until_explicit_release():
    library = _get_library()
    output, error = _Buffer(), _Buffer()
    assert library.lib.templates_build_info(ctypes.byref(output), ctypes.byref(error)) == 0
    first = ctypes.string_at(output.data, output.size)
    with Template("second call") as template:
        template.capabilities()
    assert ctypes.string_at(output.data, output.size) == first
    assert json.loads(library._take(output))["abi"] == ABI_VERSION
    assert library.lib.templates_buffer_release(output.owner) == 2
