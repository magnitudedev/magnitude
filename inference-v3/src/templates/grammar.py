"""Pinned GBNF conversion; grammar names have no semantic meaning at this boundary."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from importlib.metadata import version
from pathlib import Path

from llguidance import gbnf_to_lark as converter

from templates.regular import orient_regular_regions
from templates.lexical import whole_completion

CONVERTER_VERSION = "1.8.0"
CONVERTER_SHA256 = "df65c9e43ceafa24b24a7915feb2da359f3cf098c7048aff424498761b4c00c1"


@cache
def _verify_converter() -> None:
    if version("llguidance") != CONVERTER_VERSION:
        raise RuntimeError("Unqualified llguidance version")
    if hashlib.sha256(Path(converter.__file__).read_bytes()).hexdigest() != CONVERTER_SHA256:
        raise RuntimeError("Unqualified GBNF converter source")


class _GrammarParser(converter.GrammarParser):
    def _parse_char(self, pos):
        value, end = super()._parse_char(pos)
        # The pinned converter strips leading zeroes from Unicode escapes.
        # Lark literals and regexes require their original four/eight digits.
        if pos.peek(2) in ("\\u", "\\U"):
            return pos.peek(6 if pos.peek(2) == "\\u" else 10), end
        return value, end

    def _parse_literal(self, pos):
        if pos.current() != '"':
            raise converter.GbnfToLarkError(pos, "Expected literal")
        pos = pos.advance()
        characters = []
        while True:
            value, pos = self._parse_char(pos)
            if value == '"':
                break
            if value.startswith("\\"):
                if value[1] in "xuU":
                    scalar = int(value[2:], 16)
                    if scalar > 0x10FFFF or 0xD800 <= scalar <= 0xDFFF:
                        raise converter.GbnfToLarkError(pos, "Invalid Unicode scalar")
                    value = chr(scalar)
                else:
                    value = {"n": "\n", "r": "\r", "t": "\t"}.get(value[1], value[1])
            characters.append(value)
        # Lark strings accept JSON escapes, not GBNF's \x and \U spellings.
        encoded = json.dumps("".join(characters), ensure_ascii=False)[1:-1]
        return converter.LiteralNode(encoded), pos


def to_lark(grammar: str) -> str:
    """Alpha-rename before conversion to avoid reserved/case-normalized collisions.

    Upstream maps root to start and terminal names to uppercase. Native grammars
    can already have a start rule or distinct names that normalize identically.
    Rename parsed identifiers, never literal text or character classes.
    """
    _verify_converter()
    try:
        rules = _GrammarParser().parse(grammar)
    except converter.GbnfToLarkError as error:
        raise ValueError(f"Invalid GBNF grammar: {error}") from error
    if "root" not in rules:
        raise ValueError("GBNF grammar has no root rule")
    names = {name: "root" if name == "root" else f"g{i}" for i, name in enumerate(rules)}

    def rename(node):
        if isinstance(node, converter.RuleRefNode):
            if node.name not in names:
                raise ValueError(f"Undefined grammar rule: {node.name}")
            node.name = names[node.name]
        for child in node.children():
            rename(child)

    renamed = {}
    for name, rule in rules.items():
        rename(rule)
        rule.name = names[name]
        renamed[rule.name] = rule
    lexical = whole_completion(renamed)
    if lexical is not None:
        return lexical
    orient_regular_regions(renamed)
    converter.resolve(renamed)
    ordered = sorted(renamed.values(), key=lambda rule: rule.order)
    return "%llguidance {}\n\n" + "\n".join(str(rule) for rule in ordered) + "\n"
