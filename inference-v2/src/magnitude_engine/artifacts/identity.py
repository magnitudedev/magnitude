"""Content identity for text vocabulary, template and stopping configuration."""

import hashlib
from pathlib import Path


def tokenizer_identity(directory: Path) -> str:
    files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if not any((directory / name).is_file() for name in files):
        raise ValueError("model artifact has no tokenizer identity files")
    digest = hashlib.sha256()
    for name in (*files, "chat_template.jinja", "config.json", "generation_config.json"):
        path = directory / name
        if path.is_file():
            value = path.read_bytes()
            digest.update(name.encode())
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    return digest.hexdigest()


def processor_identity(directory: Path, implementation: str) -> str:
    """Bind media interpretation to its artifact settings and exact adapter dependency."""
    from importlib.metadata import version

    digest = hashlib.sha256(tokenizer_identity(directory).encode())
    for value in (implementation, version("transformers")):
        digest.update(value.encode())
    for name in ("preprocessor_config.json", "processor_config.json"):
        path = directory / name
        if path.is_file():
            value = path.read_bytes()
            digest.update(name.encode())
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    return digest.hexdigest()
