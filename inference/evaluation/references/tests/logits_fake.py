"""Synthetic logits outputs for the KL tests: rendered sessions, the true logits a reference
would produce for them, and contract logits files."""

import hashlib
import json
from pathlib import Path

import numpy as np

VOCABULARY = 64
MODEL = "fake:gguf:q4"
TARGET_SHA = "0" * 64
CHAT_TEMPLATE_SHA = "1" * 64
TEMPLATE = "repeat:4k"
SCORED = 12


def family() -> type:
    """The reference family the fake references are of: their target's architecture's (`write_target`)."""
    from magnitude_references.families.llama import LlamaReference
    return LlamaReference


def write_target(path: Path) -> Path:
    """A target GGUF of its header alone, of the architecture of `family`, which is all a cached reference's
    identity reads of it."""
    from gguf.constants import GGUFValueType

    from magnitude_references.gguf import Value, Writer
    writer = Writer(path, {"general.architecture": Value(GGUFValueType.STRING, "llama")})
    writer.begin()
    writer.close()
    return path


def rendered_lines(lanes: int, context: int, shift: bool) -> list[dict]:
    lines = []
    for lane in range(lanes):
        length = 40 + lane
        tokens = [
            (index * 7 + lane + (1 if shift and index == 5 else 0)) % VOCABULARY
            for index in range(length)
        ]
        lines.append(
            {
                "template": TEMPLATE,
                "lane": lane,
                "model": MODEL,
                "target_sha256": TARGET_SHA,
                "chat_template_sha256": CHAT_TEMPLATE_SHA,
                "request_options": {},
                "context": context,
                "history": 0,
                "tokens": tokens,
                "prompt_sha256": hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest(),
                "scored_spans": [[length - SCORED, length]],
                "generation_prefix_tokens": length - SCORED,
            }
        )
    return lines


def rows_of(lines: list[dict]) -> list[dict]:
    return [
        {"template": line["template"], "lane": line["lane"], "position": position}
        for line in lines
        for start, end in line["scored_spans"]
        for position in range(start - 1, end - 1)
    ]


def truth(rows: list[dict]) -> np.ndarray:
    """The true logits of rows, deterministic per (session, position)."""
    values = np.empty((len(rows), VOCABULARY), dtype="<f4")
    for index, row in enumerate(rows):
        seed = int(
            hashlib.sha256(f"{row['template']}/{row['lane']}/{row['position']}".encode()).hexdigest()[:8], 16
        )
        values[index] = np.random.default_rng(seed).normal(0, 3, VOCABULARY)
    return values


def write_logits(
    directory: Path, producer: str, rendered_digest: str, rows: list[dict], values: np.ndarray,
    sample: dict | None = None,
) -> None:
    np.ascontiguousarray(values, dtype="<f4").tofile(directory / "logits.f32")
    header = {
        "producer": producer,
        "model": MODEL,
        "rendered_digest": rendered_digest,
        "vocabulary": VOCABULARY,
        "rows": rows,
        "sample": sample,
    }
    (directory / "logits.json").write_text(json.dumps(header))
