"""Read template bundles independently of a model's tokenizer implementation."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from engine.platform.storage import FileSource
from engine.weights.formats.gguf import Directory
from templates.bundle import SpecialToken, TemplateBundle, Variant

MAX_METADATA_BYTES = 16 * 1024 * 1024


def _read(path: Path) -> str:
    with FileSource(path) as source:
        if source.size > MAX_METADATA_BYTES:
            raise ValueError(f"Template metadata exceeds the size limit: {path}")
        return source.read(0, source.size).decode("utf-8")


def template_file(path: Path, *, name: str) -> Variant:
    return Variant(name=name, source=_read(path), provenance=str(path.resolve()))


def _variants(value, provenance):
    if isinstance(value, str):
        return [Variant(name="default", source=value, provenance=provenance)]
    if isinstance(value, dict):
        return [
            Variant(name=name, source=source, provenance=provenance)
            for name, source in value.items()
        ]
    if isinstance(value, list):
        variants = [
            Variant(name=item["name"], source=item["template"], provenance=provenance)
            for item in value
        ]
        if len({variant.name for variant in variants}) != len(variants):
            raise ValueError("duplicate named templates in artifact configuration")
        return variants
    raise ValueError("unsupported artifact chat_template representation")


def directory_templates(path: Path) -> TemplateBundle:
    """Precedence: processor config, tokenizer config, named files, default file.

    Precedence applies per variant. The default is explicitly named `default`;
    another sole variant is never silently promoted to default.
    """
    variants = {}
    tokens = {}
    for filename in ("processor_config.json", "tokenizer_config.json"):
        source = path / filename
        if not source.is_file():
            continue
        config = json.loads(_read(source))
        if not isinstance(config, dict):
            raise ValueError(f"Template configuration must be an object: {source}")
        if config.get("chat_template") is not None:
            for variant in _variants(config["chat_template"], str(source)):
                variants[variant.name] = variant
        for name, value in config.items():
            if not name.endswith("_token") or isinstance(value, bool) or value is None:
                continue
            text = value.get("content") if isinstance(value, dict) else value
            if not isinstance(text, str):
                raise ValueError(f"invalid artifact special token {name}")
            tokens[name] = SpecialToken(name=name, text=text)
        extras = config.get("extra_special_tokens") or {}
        # Unnamed extra tokens have no Jinja variable to bind.
        if isinstance(extras, list):
            TypeAdapter(list[str]).validate_python(extras, strict=True)
            extras = {}
        if not isinstance(extras, dict):
            raise ValueError("extra_special_tokens must be a list or a named mapping")
        for name, value in extras.items():
            tokens[name] = SpecialToken(name=name, text=value)
    for source in sorted((path / "chat_templates").glob("*.jinja")):
        variant = template_file(source, name=source.stem)
        variants[variant.name] = variant
    source = path / "chat_template.jinja"
    if source.is_file():
        variants["default"] = template_file(source, name="default")
    return TemplateBundle(
        default="default",
        variants=tuple(variants[name] for name in sorted(variants)),
        special_tokens=tuple(tokens[name] for name in sorted(tokens)),
    )


def gguf_templates(directory: Directory, *, provenance: str) -> TemplateBundle:
    metadata = {item.name: item.value for item in directory.metadata}
    variants = {}
    if "tokenizer.chat_template" in metadata:
        variants["default"] = Variant(
            name="default",
            source=TypeAdapter(str).validate_python(
                metadata["tokenizer.chat_template"], strict=True
            ),
            provenance=provenance + "#tokenizer.chat_template",
        )
    prefix = "tokenizer.chat_template."
    for key, source in metadata.items():
        if key.startswith(prefix):
            name = key[len(prefix) :]
            # The explicit unsuffixed default wins over a named default.
            if name not in variants:
                variants[name] = Variant(
                    name=name,
                    source=TypeAdapter(str).validate_python(source, strict=True),
                    provenance=provenance + "#" + key,
                )
    tokens = []
    pieces = TypeAdapter(tuple[str, ...]).validate_python(
        metadata.get("tokenizer.ggml.tokens", ()), strict=True
    )
    for native, name in (
        ("bos", "bos_token"),
        ("eos", "eos_token"),
        ("unknown", "unk_token"),
        ("padding", "pad_token"),
        ("separator", "sep_token"),
        ("cls", "cls_token"),
        ("mask", "mask_token"),
    ):
        identity = metadata.get(f"tokenizer.ggml.{native}_token_id")
        if identity is None:
            continue
        if type(identity) is not int or not 0 <= identity < len(pieces):
            raise ValueError(f"invalid artifact special-token ID: {native}")
        tokens.append(SpecialToken(name=name, text=pieces[identity]))
    return TemplateBundle(
        default="default",
        variants=tuple(variants[name] for name in sorted(variants)),
        special_tokens=tuple(tokens),
    )
