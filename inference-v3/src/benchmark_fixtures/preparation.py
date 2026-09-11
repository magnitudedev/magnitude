"""Tokenizer-bound context and continuation preparation, outside measured execution."""

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from . import bfcl, prose
from .contexts import Context, History
from .records import Record, digest, encoded
from .storage import cache_root, write_atomic


class Fixture(Record):
    identity: Literal["prose.moby-dick", "tools.bfcl"]
    context_tokens: int = Field(gt=0)
    continuation_tokens: int = Field(default=256, gt=0)
    offset: int = Field(default=0, ge=0)


class PreparedTokens(Record):
    fixture: Fixture
    prompt: tuple[int, ...]
    continuation: tuple[int, ...]
    context: Context | None
    provenance: dict[str, JsonValue]


class Tokenization:
    """Checkpoint tokenization; serving adapters may supply their own counter."""

    def __init__(self, artifact: Path):
        from magnitude_engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
        from magnitude_engine.serving import template

        loaded = TokenizerArtifact.load(artifact)
        self.eos_tokens = loaded.eos_tokens
        self.tokenizer = loaded.tokenizer
        self.template = template.ChatTemplate(loaded)
        self.identity = digest(
            {
                "tokenizer": loaded.identity,
                "renderer": "magnitude-chat-v1",
                "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "renderer_source": hashlib.sha256(Path(template.__file__).read_bytes()).hexdigest(),
            }
        )

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def chat(self, context: Context) -> tuple[int, ...]:
        context = Context.model_validate_json(encoded(context.model_dump(mode="json")))
        return self.template.render(
            context.messages,
            tools=context.tools,
            tool_choice="required" if context.tools else "auto",
            chat_template_kwargs={"enable_thinking": False},
        ).tokens

    async def count(self, context: Context) -> int:
        return len(self.chat(context))


async def prepare(
    fixture: Fixture,
    tokenizer: Tokenization,
    *,
    cache: Path | None = None,
) -> PreparedTokens:
    root = cache or cache_root()
    if fixture.identity == "prose.moby-dick":
        text, source = await prose.prepare(cache=root)
        token_key = digest({"source": source, "tokenizer": tokenizer.identity})
        token_file = root / "tokens" / f"{token_key}.json"
        if token_file.exists():
            import json

            record = json.loads(token_file.read_text())
            tokens = tuple(record["tokens"])
            if record["digest"] != digest(tokens):
                raise ValueError("cached prose tokens failed their content digest")
        else:
            tokens = tokenizer.encode(text)
            write_atomic(token_file, encoded({"tokens": tokens, "digest": digest(tokens)}).encode())
        start, split = fixture.offset, fixture.offset + fixture.context_tokens
        end = split + fixture.continuation_tokens
        if end > len(tokens):
            raise ValueError(
                f"fixture window ends at {end}, but Moby Dick has {len(tokens)} tokens"
            )
        prompt, continuation = tokens[start:split], tokens[split:end]
        context = None
        provenance = {**source, "recipe": "contiguous-token-window-v1", "total_tokens": len(tokens)}
    else:
        interactions, corpus_digest = await bfcl.prepare(tuple(bfcl.CATEGORIES), cache=root)
        offset = fixture.offset % len(interactions)
        interactions = interactions[offset:] + interactions[:offset]
        history = History(interactions, f"bfcl-offset-{fixture.offset}", interactions[0])
        prepared = await history.prepare(
            fixture.context_tokens, tokenizer.count, tokenizer.identity
        )
        context, provenance = prepared.content, prepared.provenance
        provenance = {**provenance, "corpus_digest": corpus_digest}
        prompt = tokenizer.chat(context)
        # Complete the current tool round, preserving the original vocabulary/header.
        # Encoding a rendered completion must retain the exact prepared prompt prefix.
        messages = list(context.messages)
        completion = history.current.completed("replay-current")
        messages.extend(completion[len(history.current.messages) :])
        full = tokenizer.chat(Context(messages=messages, tools=context.tools))
        if full[: len(prompt)] != prompt:
            raise ValueError("tool completion changed the rendered prompt prefix")
        continuation = full[len(prompt) : len(prompt) + fixture.continuation_tokens]
    provenance = {
        **provenance,
        "fixture": fixture.identity,
        "offset": fixture.offset,
        "tokenizer": tokenizer.identity,
        "requested_context_tokens": fixture.context_tokens,
        "actual_context_tokens": len(prompt),
        "continuation_tokens": len(continuation),
        "prompt_digest": digest(prompt),
        "continuation_digest": digest(continuation),
    }
    result = PreparedTokens(
        fixture=fixture,
        prompt=prompt,
        continuation=continuation,
        context=context,
        provenance=provenance,
    )
    write_atomic(
        root / "prepared" / f"{digest(provenance)}.json", result.model_dump_json().encode()
    )
    return result
