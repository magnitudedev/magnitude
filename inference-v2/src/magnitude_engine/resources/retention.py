"""Opaque retained storage identities; the engine never interprets model addresses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetainedStorage:
    owner: object
    nbytes: int
