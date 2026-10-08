"""Ollama's chat renderers, reproduced offline for the families the benchmark serves.

Ollama renders a chat request for a tag whose config names a renderer with that Go renderer
(``model/renderers/*.go`` in Ollama v0.35.1) and sends the text to its runner. The prompt it
evaluates is therefore known without running the model: render here, then tokenize with the
tag's own tokenizer. Only what the benchmark sends is reproduced: text messages in the
system, user and assistant roles, thinking off, no tools and no images. Anything else is
refused rather than approximated.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

# Go's strings.TrimSpace removes unicode.IsSpace runes; Python's str.strip also removes the
# information separators U+001C..U+001F, which Go keeps.
GO_SPACE = "\t\n\v\f\r \x85\xa0                　"
ROLES = ("system", "user", "assistant")


def trim(text: str) -> str:
    return text.strip(GO_SPACE)


def text_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(role, content) pairs, refusing what the reproduced renderers do not cover."""
    if not messages:
        raise ValueError("Ollama rendering needs at least one message")
    pairs = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise ValueError(f"Ollama rendering is reproduced for text roles only, not {role}")
        if not isinstance(content, str):
            raise ValueError("Ollama rendering is reproduced for text content only")
        if set(message) - {"role", "content"}:
            raise ValueError(f"Ollama rendering does not reproduce {sorted(set(message) - {'role', 'content'})}")
        pairs.append((role, content))
    return pairs


# model/renderers/qwen35.go: "qwen3.5" (also served for Qwen 3.6) and "qwen3.8".
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def split_qwen_reasoning(content: str) -> tuple[str, str]:
    """splitQwen35ReasoningContent with tagged extraction and no message thinking."""
    reasoning = ""
    index = content.find("</think>")
    if index != -1:
        before = content[:index]
        opening = before.rfind("<think>")
        reasoning = before[opening + len("<think>") :] if opening != -1 else before
        content = content[index + len("</think>") :].lstrip("\n")
    return trim(reasoning), content


def qwen(messages: list[dict[str, Any]], variant38: bool) -> str:
    pairs = text_messages(messages)
    if variant38:
        # normalizeQwen38Messages: instructions fold into one leading system turn.
        instructions = [content for role, content in pairs if role == "system"]
        if len(instructions) > 1 or (instructions and pairs[0][0] != "system"):
            folded = "\n\n".join(t for t in (trim(c) for c in instructions) if t)
            pairs = [("system", folded), *((r, c) for r, c in pairs if r != "system")]
        if not any(role == "user" for role, _ in pairs):
            raise ValueError("qwen3.8 needs a user query")
    parts = []
    if pairs[0][0] == "system":
        system = trim(pairs[0][1])
        if not variant38 or system:
            parts.append(f"{IM_START}system\n{system}{IM_END}\n")
    last = len(pairs) - 1
    for index, (role, content) in enumerate(pairs):
        content = trim(content)
        prefill = index == last and role == "assistant"
        if role == "user" or (role == "system" and index != 0):
            parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
        elif role == "assistant":
            # Thinking is off, so only qwen3.8 renders an assistant think block, and it does
            # not extract tagged reasoning from the content.
            if variant38:
                parts.append(f"{IM_START}assistant\n<think>\n\n</think>\n\n{content}")
            else:
                parts.append(f"{IM_START}assistant\n{split_qwen_reasoning(content)[1]}")
            if not prefill:
                parts.append(f"{IM_END}\n")
        if index == last and not prefill:
            parts.append(f"{IM_START}assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


# model/renderers/gemma4.go: "gemma4-small" (E2B, E4B) and "gemma4-large" (12B and larger).
def strip_gemma_thinking(text: str) -> str:
    result = []
    while True:
        start = text.find("<|channel>")
        if start == -1:
            result.append(text)
            break
        result.append(text[:start])
        end = text.find("<channel|>", start)
        if end == -1:
            break
        text = text[end + len("<channel|>") :]
    return trim("".join(result))


def gemma4(messages: list[dict[str, Any]], large: bool) -> str:
    pairs = text_messages(messages)
    parts = ["<bos>"]
    if pairs[0][0] == "system":
        system, pairs = pairs[0][1], pairs[1:]
        parts.append("<|turn>system\n" + (trim(system) if system else "") + "<turn|>\n")
    previous = ""
    for index, (role, content) in enumerate(pairs):
        turn = "model" if role == "assistant" else role
        if not (turn == "model" and previous == "assistant"):
            parts.append(f"<|turn>{turn}\n")
        if turn == "model":
            if content:
                parts.append(strip_gemma_thinking(content))
        else:
            parts.append(trim(content))
        following = pairs[index + 1][0] if index + 1 < len(pairs) else ""
        if not (turn == "model" and following == "assistant"):
            parts.append("<turn|>\n")
        previous = role
    parts.append("<|turn>model\n")
    if large:
        parts.append("<|channel>thought\n<channel|>")
    return "".join(parts)


# model/renderers/nemotron3nano.go: "nemotron-3-nano" and "nemotron-3.5-nano" (v35).
def strip_think_toggles(text: str) -> str:
    text = text.replace("</think>", "<_end_think>")
    text = text.replace("/think", "").replace("/no_think", "")
    return text.replace("<_end_think>", "</think>")


def nemotron(messages: list[dict[str, Any]], v35: bool) -> str:
    pairs = text_messages(messages)
    if not v35 and any(
        role != "assistant" and ("/think" in content.replace("</think>", "") or "/no_think" in content)
        for role, content in pairs
    ):
        raise ValueError("nemotron-3-nano think toggles in the prompt are not reproduced")
    system = ""
    if pairs[0][0] == "system":
        system, pairs = pairs[0][1], pairs[1:]
        if not v35:
            system = strip_think_toggles(system)
    last_user = max((i for i, (role, _) in enumerate(pairs) if role == "user"), default=-1)
    parts = [f"{IM_START}system\n{system}{IM_END}\n"]
    for index, (role, content) in enumerate(pairs):
        if role == "assistant":
            if "<think>" not in content and "</think>" not in content:
                content = "<think></think>" + content
            if index < last_user and "<think>" in content and "</think>" in content:
                content = "<think></think>" + content.split("</think>")[-1]
            parts.append(f"{IM_START}assistant\n{trim(content)}{IM_END}\n")
        else:
            if not v35:
                content = trim(strip_think_toggles(content))
            parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n<think></think>")
    return "".join(parts)


# model/renderers/glimmer.go: "glimmer".
GLIMMER_BOS = "<|begin_of_text|>"
GLIMMER_REASONING = (
    ("Reasoning effort", "Reasoning strength"),
    ("Reasoning Effort", "Reasoning Strength"),
    ("reasoning effort", "reasoning strength"),
    ("REASONING EFFORT", "REASONING STRENGTH"),
)


def glimmer_system(content: str) -> str:
    # strings.NewReplacer replaces left to right without rescanning; the four patterns never
    # overlap one another, so sequential replacement is the same.
    for old, new in GLIMMER_REASONING:
        content = content.replace(old, new)
    strength = "" if "reasoning strength" in content.lower() else "\n\nReasoning strength: none."
    return (
        f"<|start|>system<|message|>{content}{strength}"
        '\n\n# Valid recipients: "self", "user".<|eot|>'
    )


def glimmer(messages: list[dict[str, Any]]) -> str:
    pairs = text_messages(messages)
    parts = [GLIMMER_BOS]
    if not any(role == "system" for role, _ in pairs):
        # Ollama uses time.Now().Format(time.DateOnly), in the host local timezone.
        parts.append(glimmer_system(
            "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04."
            f"\nCurrent date: {date.today().isoformat()}."
        ))
    for role, content in pairs:
        if role == "system":
            parts.append(glimmer_system(content))
        elif role == "user":
            parts.append(f"<|start|>user<|message|>{content}<|eot|>")
        else:
            parts.append(f"<|start|>assistant to=user<|message|>{content}<|eot|>")
    parts.append("<|start|>assistant")
    return "".join(parts)


# Muse Glimmer has no thinking-off switch: with ``think`` false its renderer only writes
# "Reasoning strength: none" and the model still reasons first. Thinking off is the reply
# opened at the answer channel's header, as Magnitude's engine opens it.
GLIMMER_ANSWER_HEADER = " to=user<|message|>"


@dataclass(frozen=True)
class Renderer:
    write: Callable[[list[dict[str, Any]]], str]
    #: The textual BOS the renderer emits, which the llama.cpp runner strips when its
    #: tokenizer adds BOS itself (``LeadingBOS``).
    leading_bos: str = ""
    #: The header a reply opens at with thinking off, for a format without a switch.
    answer_header: str = ""


RENDERERS = {
    "qwen3.5": Renderer(lambda messages: qwen(messages, False)),
    "qwen3.8": Renderer(lambda messages: qwen(messages, True)),
    "gemma4-small": Renderer(lambda messages: gemma4(messages, False), "<bos>"),
    "gemma4-large": Renderer(lambda messages: gemma4(messages, True), "<bos>"),
    "nemotron-3-nano": Renderer(lambda messages: nemotron(messages, False)),
    "nemotron-3.5-nano": Renderer(lambda messages: nemotron(messages, True)),
    "glimmer": Renderer(glimmer, GLIMMER_BOS, GLIMMER_ANSWER_HEADER),
}


def gemma4_renderer(name: str, model_type: str) -> str:
    """server/renderer_resolution.go: the legacy "gemma4" renderer resolved by size."""
    lower = name.lower()
    if "e2b" in lower or "e4b" in lower:
        return "gemma4-small"
    if "12b" in lower or "26b" in lower or "31b" in lower:
        return "gemma4-large"
    multipliers = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    if model_type and model_type[-1:].upper() in multipliers:
        try:
            count = float(model_type[:-1]) * multipliers[model_type[-1:].upper()]
        except ValueError:
            return "gemma4-small"
        return "gemma4-large" if int(count) >= 12_000_000_000 else "gemma4-small"
    return "gemma4-small"


def renderer(configured: str, name: str, model_type: str) -> str:
    """The renderer Ollama resolves for a tag (resolveRendererName)."""
    resolved = gemma4_renderer(name, model_type) if configured == "gemma4" else configured
    if resolved not in RENDERERS:
        raise ValueError(f"Ollama renderer {configured!r} is not reproduced")
    return resolved


def render(name: str, messages: list[dict[str, Any]], think: bool | None) -> str:
    """The prompt Ollama's renderer ``name`` writes for a chat request with thinking off."""
    if think is not False:
        raise ValueError("Ollama rendering is reproduced with thinking off only")
    return RENDERERS[name].write(messages)


def leading_bos(name: str) -> str:
    return RENDERERS[name].leading_bos


def answer_header(name: str) -> str:
    return RENDERERS[name].answer_header
