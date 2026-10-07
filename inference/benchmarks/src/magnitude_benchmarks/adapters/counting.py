"""Prompt token counts from a model's own tokenizer, without loading the model.

A prompt's size is a property of its text and the model's tokenizer, so counting it never
needs a running engine. ``magnitude-count`` (built with ``magnitude-engine``) reads a GGUF's
tokenizer and chat templates; a safetensors tag's ``tokenizer.json`` is read with Hugging
Face ``tokenizers``.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from .base import stop

Count = Callable[[Any], Awaitable[int]]


@contextlib.asynccontextmanager
async def magnitude_count(binary: Path, mode: str, model: Path, log: Path) -> AsyncIterator[Count]:
    """A ``magnitude-count chat|text`` process: each value is sent as one JSON line and its
    count read back. Its diagnostics go to ``log``."""
    if not binary.is_file():
        raise ValueError(f"magnitude-count binary is not a file: {binary}")
    with log.open("ab") as diagnostics:
        process = await asyncio.create_subprocess_exec(
            str(binary),
            mode,
            "--model",
            str(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=diagnostics,
            start_new_session=True,
        )
        requests, replies = process.stdin, process.stdout
        if requests is None or replies is None:
            raise RuntimeError("magnitude-count pipes were not opened")
        lock = asyncio.Lock()

        async def count(value: Any) -> int:
            async with lock:
                requests.write(json.dumps(value, ensure_ascii=False).encode() + b"\n")
                await requests.drain()
                line = await replies.readline()
            if not line:
                code = await process.wait()
                raise RuntimeError(f"magnitude-count exited ({code}); see {log}")
            tokens = int(line)
            if tokens <= 0:
                raise ValueError(f"magnitude-count returned {tokens} tokens")
            return tokens

        try:
            yield count
        finally:
            await stop(process)


def tokenizer_json_count(path: Path) -> Callable[[str], int]:
    """Tokens a ``tokenizer.json`` encodes a text to, added tokens recognized and no
    sequence-start token added."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
