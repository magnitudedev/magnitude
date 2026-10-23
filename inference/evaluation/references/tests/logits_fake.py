"""Synthetic logits outputs for the KL tests: rendered fixtures, the true logits a reference
would produce for them, and contract logits files."""

import hashlib
import json
from pathlib import Path

import numpy as np

VOCABULARY = 64
MODEL = "fake:gguf:q4"
TARGET_SHA = "0" * 64
TEMPLATE_SHA = "1" * 64
SCORED = 12


def rendered_lines(fixtures: Path, context: int, shift: bool) -> list[dict]:
    lines = []
    for number, line in enumerate(fixtures.read_text().splitlines()):
        fixture_id = json.loads(line)["id"]
        length = 40 + number
        tokens = [
            (index * 7 + number + (1 if shift and index == 5 else 0)) % VOCABULARY
            for index in range(length)
        ]
        lines.append(
            {
                "fixture_id": fixture_id,
                "fixture_set_digest": hashlib.sha256(fixtures.read_bytes()).hexdigest(),
                "model": MODEL,
                "target_sha256": TARGET_SHA,
                "template_sha256": TEMPLATE_SHA,
                "context_target": context,
                "tokens": tokens,
                "scored_spans": [[length - SCORED, length]],
                "generation_prefix_tokens": None,
            }
        )
    return lines


def rows_of(lines: list[dict]) -> list[dict]:
    return [
        {"fixture_id": line["fixture_id"], "position": position}
        for line in lines
        for start, end in line["scored_spans"]
        for position in range(start - 1, end - 1)
    ]


def truth(rows: list[dict]) -> np.ndarray:
    """The true logits of rows, deterministic per (fixture, position)."""
    values = np.empty((len(rows), VOCABULARY), dtype="<f4")
    for index, row in enumerate(rows):
        seed = int(
            hashlib.sha256(f"{row['fixture_id']}/{row['position']}".encode()).hexdigest()[:8], 16
        )
        values[index] = np.random.default_rng(seed).normal(0, 3, VOCABULARY)
    return values


def write_logits(
    directory: Path, producer: str, rendered_digest: str, rows: list[dict], values: np.ndarray
) -> None:
    np.ascontiguousarray(values, dtype="<f4").tofile(directory / "logits.f32")
    header = {
        "producer": producer,
        "model": MODEL,
        "rendered_digest": rendered_digest,
        "vocabulary": VOCABULARY,
        "rows": rows,
    }
    (directory / "logits.json").write_text(json.dumps(header))
