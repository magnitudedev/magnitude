"""Artifact template selection is independent of tokenizer-family support."""

import json
import struct

import pytest

from engine.inputs.formats.templates import directory_templates, gguf_templates
from engine.platform.storage import FileSource
from engine.weights.formats.gguf import read_directory
from templates.bundle import TemplateBundle, Variant


def test_directory_precedence_is_per_variant_and_selection_uses_effective_tools(tmp_path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "chat_template": {"default": "processor", "processor": "retained"},
                "bos_token": "<bos>",
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "default", "template": "tokenizer"},
                    {"name": "tool_use", "template": "tools"},
                ],
                "bos_token": {"content": "<s>", "special": True},
                "extra_special_tokens": {"image_token": "<image>"},
            }
        )
    )
    (tmp_path / "chat_templates").mkdir()
    (tmp_path / "chat_templates" / "default.jinja").write_text("named default")
    (tmp_path / "chat_templates" / "tool_use.jinja").write_text("named tools")
    (tmp_path / "chat_template.jinja").write_text("default file")
    bundle = directory_templates(tmp_path)
    assert bundle.select(tools_offered=False).source == "default file"
    assert bundle.select(tools_offered=True).source == "named tools"
    assert bundle.select(tools_offered=True, variant="processor").source == "retained"
    assert {token.name: token.text for token in bundle.special_tokens} == {
        "bos_token": "<s>",
        "image_token": "<image>",
    }
    assert bundle.select(tools_offered=True).provenance.endswith("chat_templates/tool_use.jinja")
    assert TemplateBundle.model_validate_json(bundle.model_dump_json()) == bundle
    assert directory_templates(tmp_path).fingerprint == bundle.fingerprint
    with pytest.raises(ValueError, match="Unknown template variant"):
        bundle.select(tools_offered=False, variant="missing")
    override = Variant(name="operator", source="override", provenance="configuration")
    assert bundle.select(tools_offered=True, override=override) == override
    with pytest.raises(ValueError, match="not both"):
        bundle.select(tools_offered=True, variant="default", override=override)
    (tmp_path / "chat_template.jinja").write_text("updated")
    assert directory_templates(tmp_path).fingerprint != bundle.fingerprint


@pytest.mark.parametrize(
    "value", [{"tool_use": "only"}, [{"name": "tool_use", "template": "only"}]]
)
def test_no_implicit_default(tmp_path, value):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": value}))
    with pytest.raises(ValueError, match="usable declared default"):
        directory_templates(tmp_path)


def test_null_config_template_and_unnamed_extra_tokens(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": None,
                "extra_special_tokens": ["<extra>"],
                "eos_token": None,
            }
        )
    )
    (tmp_path / "chat_template.jinja").write_text("authored")
    bundle = directory_templates(tmp_path)
    assert bundle.special_tokens == ()
    assert bundle.select(tools_offered=False).source == "authored"


def test_duplicate_configuration_names_are_rejected(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "default", "template": "first"},
                    {"name": "default", "template": "second"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate named"):
        directory_templates(tmp_path)


def test_gguf_real_container_metadata_without_tokenizer_family(tmp_path):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    # Actual GGUF v3 encoding exercises the container reader as well as selection.
    metadata = [
        ("tokenizer.chat_template", 8, string("default")),
        ("tokenizer.chat_template.default", 8, string("shadowed")),
        ("tokenizer.chat_template.tool_use", 8, string("tools")),
        (
            "tokenizer.chat_templates",
            9,
            struct.pack("<IQ", 8, 2) + string("default") + string("tool_use"),
        ),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + string("<s>") + string("</s>")),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 1)),
    ]
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, kind, value in metadata:
        data += string(key) + struct.pack("<I", kind) + value
    data += bytes((-len(data)) % 32)
    path = tmp_path / "metadata.gguf"
    path.write_bytes(data)
    with FileSource(path) as source:
        directory = read_directory(source)
    bundle = gguf_templates(directory, provenance=str(path))
    assert bundle.select(tools_offered=False).source == "default"
    assert bundle.select(tools_offered=True).source == "tools"
    assert {token.name: token.text for token in bundle.special_tokens} == {
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    bad = directory.model_copy(
        update={
            "metadata": tuple(
                item.model_copy(update={"value": 3}) if item.name.endswith("eos_token_id") else item
                for item in directory.metadata
            )
        }
    )
    with pytest.raises(ValueError, match="special-token ID"):
        gguf_templates(bad, provenance=str(path))
