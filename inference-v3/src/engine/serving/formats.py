"""Declared output wires; architecture aliases resolve at chat construction."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatFormat:
    arguments: Literal["json", "xml", "gemma"]
    call_open: str
    call_close: str
    reasoning_open: str
    reasoning_close: str
    reasoning_label: str = ""


def format_for(family: str) -> ChatFormat | None:
    if family in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen4_exp",
        "qwen3_next",
    }:
        return ChatFormat("xml", "<tool_call>", "</tool_call>", "<think>", "</think>")
    if family in {
        "qwen3",
        "qwen3_moe",
        "qwen2",
        "qwen2_moe",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
    }:
        return ChatFormat("json", "<tool_call>", "</tool_call>", "<think>", "</think>")
    if family in {"gemma4", "gemma4_text", "gemma4_unified"}:
        return ChatFormat(
            "gemma", "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "thought\n"
        )
    return None
