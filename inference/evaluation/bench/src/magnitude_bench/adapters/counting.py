"""Prompt token counts of a safetensors tag's text, from its own ``tokenizer.json``."""

from collections.abc import Callable
from pathlib import Path


def tokenizer_json_count(path: Path) -> Callable[[str], int]:
    """Tokens a ``tokenizer.json`` encodes a text to, added tokens recognized and no
    sequence-start token added."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
