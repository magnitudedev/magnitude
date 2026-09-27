"""Host-safe checkpoint tokenizer metadata, shared by prompting and constraint compilation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .identity import tokenizer_identity


@dataclass(frozen=True)
class TokenizerArtifact:
    tokenizer: Any
    family: str
    vocabulary: int
    eos_tokens: tuple[int, ...]
    identity: str = ""

    @classmethod
    def load(cls, directory: Path) -> "TokenizerArtifact":
        config = json.loads((directory / "config.json").read_text())
        text = config.get("text_config", config)
        generation_file = directory / "generation_config.json"
        generation = json.loads(generation_file.read_text()) if generation_file.exists() else {}
        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
        vocabulary = text["vocab_size"]
        if type(vocabulary) is not int or vocabulary < len(tokenizer):
            raise ValueError("tokenizer vocabulary exceeds the target projection")
        eos = []
        for value in (
            text.get("eos_token_id"),
            config.get("eos_token_id"),
            generation.get("eos_token_id"),
            tokenizer.eos_token_id,
        ):
            for token in value if isinstance(value, list) else (() if value is None else (value,)):
                if type(token) is not int or not 0 <= token < vocabulary:
                    raise ValueError("artifact declares an invalid EOS token")
                if token not in eos:
                    eos.append(token)
        if not eos:
            raise ValueError("artifact does not declare a generation EOS token")
        return cls(
            tokenizer,
            text.get("model_type", config.get("model_type", "")),
            vocabulary,
            tuple(eos),
            tokenizer_identity(directory),
        )
