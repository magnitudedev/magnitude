"""RULER-derived key/value retrieval sessions with needles at the start of the context.

Adapted from NVIDIA/RULER's NIAH record-haystack and multi-query tasks:
https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
The recipe uses independent indexed records and strict JSON answers; its scores are not
official RULER scores. No upstream runtime is required.

The kept head holds the instruction and the first record block, which contains the needle
records; every later turn supplies a block of distractor records. The probe asks for the
needle values. Because sizing keeps the head and the probe, the needles are always the
farthest facts from the question, whatever the context length.
"""

import json
from typing import Literal

from .canonical import digest
from .fixture import Fixture, Generation, Message, Source, TurnSizing

SOURCE = {
    "repository": "https://github.com/NVIDIA/RULER",
    "commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
    "path": "scripts/data/synthetic/niah.py",
    "recipe": "ruler-record-retrieval-v2",
}
OUTPUT_TOKENS = 1_024
RECORDS_PER_TURN = 64
DISTRACTOR_TURNS = 192
INSTRUCTION = (
    "You will receive records in several messages. When asked, look up the requested keys "
    "in the records. Return only a JSON object mapping each requested key to its exact string "
    "value. Include every requested key, no other keys, no explanation and no markdown."
)
ACKNOWLEDGEMENT = "Noted."


def record(seed: int, index: int) -> tuple[str, str]:
    # Indexed facts do not depend on sizing, position or other records.
    fingerprint = digest({"seed": seed, "record": index})
    return f"{fingerprint[:8]}-{index:08x}", fingerprint[8:24]


def block(lines: list[str]) -> str:
    return "Records:\n" + "\n".join(lines)


def build(*, count: int, queries: int, seed: int = 42) -> list[Fixture]:
    """``count`` sessions; lane ``k`` draws its facts from seed ``seed + k``."""
    if count < 1:
        raise ValueError("retrieval fixtures need at least one probe")
    if not 1 <= queries <= 16:
        raise ValueError("retrieval queries must be within 1..16")
    variant: Literal["single", "multiquery"] = "single" if queries == 1 else "multiquery"
    source = Source(name="ruler", digest=digest(SOURCE))
    fixtures = []
    for lane in range(count):
        lane_seed = seed + lane
        needles = dict(record(lane_seed, i) for i in range(queries))
        total = RECORDS_PER_TURN * (1 + DISTRACTOR_TURNS)
        lines = [": ".join(record(lane_seed, i)) for i in range(total)]
        # Records 0..queries are the needles, placed in the middle of the first block.
        middle = queries + (RECORDS_PER_TURN - queries) // 2
        first = lines[queries:middle] + lines[:queries] + lines[middle:RECORDS_PER_TURN]
        messages: list[Message] = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": block(first)},
            {"role": "assistant", "content": ACKNOWLEDGEMENT},
        ]
        for start in range(RECORDS_PER_TURN, total, RECORDS_PER_TURN):
            messages.append(
                {"role": "user", "content": block(lines[start : start + RECORDS_PER_TURN])}
            )
            messages.append({"role": "assistant", "content": ACKNOWLEDGEMENT})
        prefix = len(messages) + 1
        messages.append(
            {"role": "user", "content": "Return the values for these keys: " + ", ".join(needles)}
        )
        messages.append({"role": "assistant", "content": json.dumps(needles), "scored": True})
        fixtures.append(
            Fixture(
                id=f"retrieval/ruler/{variant}-q{queries}-s{lane_seed}",
                family="retrieval",
                source=source,
                messages=messages,
                generation=Generation(
                    prefix=prefix, max_tokens=OUTPUT_TOKENS, expected={"values": needles}
                ),
                sizing=TurnSizing(keep_head=3),
            )
        )
    return fixtures
