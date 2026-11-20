"""Fixtures: the only model inputs of evaluation. A template (``<key>:<context>``, e.g.
``repeat:8k``) names one benchmark request built from a pinned, checksummed source as
deterministic chat sessions: ``repeat`` and ``twist`` (Moby Dick), ``retrieve`` (RULER),
``code`` (BigCodeBench), and ``tools`` (BFCL v4)."""

from . import bfcl, bigcodebench, moby, ruler, validation
from .session import (
    KEYS,
    Conversation,
    Generation,
    Key,
    Session,
    Source,
)
from .template import Template, TemplateFile

__all__ = [
    "KEYS",
    "Conversation",
    "Generation",
    "Key",
    "Session",
    "Source",
    "Template",
    "TemplateFile",
    "bfcl",
    "bigcodebench",
    "moby",
    "ruler",
    "validation",
]
