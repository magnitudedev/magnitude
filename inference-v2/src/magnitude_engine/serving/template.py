"""Checkpoint-owned prompt rendering, with explicit semantic output and constraint metadata."""

import json
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.composition import Blueprint, build
from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.models.preparation import ImagePreparation, PreparedMedia

from .formats import ChatFormat, format_for
from .grammar import chat_constraint
from .tool_choice import select_tools


@dataclass(frozen=True)
class PreparedChat:
    text: str
    tokens: tuple[int, ...]
    format: ChatFormat | None
    tools: list[dict]
    constraint: ConstraintSpec | None
    reasoning_prefilled: bool
    boundaries: tuple[int, ...]
    media: PreparedMedia | None = None


def normalize_messages(messages: list[dict], *, allow_images: bool = False) -> list[dict]:
    result = deepcopy(messages)
    if not result:
        raise ValueError("chat requires at least one message")
    for message in result:
        if message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("unsupported chat message role")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url" and allow_images:
                    if set(part) != {"type", "image_url"}:
                        raise ValueError(
                            "image content part fields differ from the request contract"
                        )
                    parts.append(part)
                    continue
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError(
                        "this text renderer requires a media-aware renderer for non-text parts"
                    )
                if not isinstance(part.get("text"), str):
                    raise ValueError("text content parts require a string")
                parts.append(part)
            message["content"] = (
                parts
                if any(part["type"] == "image_url" for part in parts)
                else "".join(part["text"] for part in parts)
            )
        elif content is None:
            message["content"] = ""
        elif not isinstance(content, str):
            raise ValueError("message content must be text or content parts")
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(arguments, dict):
                raise ValueError("historical tool call arguments must be a JSON object")
            function["arguments"] = arguments
    return result


class ChatTemplate:
    def __init__(
        self,
        artifact: TokenizerArtifact,
        image_processor: Blueprint[ImagePreparation] | None = None,
    ):
        self.artifact = artifact
        self._lifetime = ExitStack()
        self.image_processor = (
            None
            if image_processor is None
            else self._lifetime.enter_context(build(image_processor))
        )
        self.format = format_for(artifact.family)
        self.markers = (
            frozenset()
            if self.format is None
            else frozenset(
                marker
                for marker in (
                    self.format.call_open,
                    self.format.call_close,
                    self.format.reasoning_open,
                    self.format.reasoning_close,
                )
                if len(artifact.tokenizer.encode(marker, add_special_tokens=False)) == 1
            )
        )

    def close(self) -> None:
        self._lifetime.close()

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        parallel_tool_calls: bool = True,
        chat_template_kwargs: dict | None = None,
        response_format: dict | None = None,
    ) -> PreparedChat:
        tools = deepcopy(tools or [])
        names = set()
        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name")
            if (
                tool.get("type") != "function"
                or not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(function.get("parameters", {}), dict)
            ):
                raise ValueError("tools require uniquely named functions and parameter schemas")
            names.add(name)
        kwargs = dict(chat_template_kwargs or {})
        if {
            "tools",
            "conversation",
            "add_generation_prompt",
            "tokenize",
            "tool_choice",
            "parallel_tool_calls",
        } & kwargs.keys():
            raise ValueError("template kwargs cannot override chat rendering inputs")
        selection = select_tools(tools, tool_choice)
        tools = list(selection.tools)
        normalized = selection.instruct(
            normalize_messages(messages, allow_images=self.image_processor is not None),
            parallel=parallel_tool_calls,
        )
        images = []
        if self.image_processor is not None:
            from .images import replace_image_parts

            images = replace_image_parts(normalized)
        tokenizer = self.artifact.tokenizer
        text = tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
        media = None
        if images:
            assert self.image_processor is not None
            text, media = self.image_processor.process(text, images, tokenizer)
        tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
        if not tokens:
            raise ValueError("chat template produced an empty prompt")
        prefilled = False
        boundaries = ()
        if self.format is not None:
            before, found, after = text.rpartition(self.format.reasoning_open)
            prefilled = bool(found) and after.strip() == self.format.reasoning_label.strip()
            user_marker = "<|turn>user" if self.format.arguments == "gemma" else "<|im_start|>user"
            end = text.find(user_marker)
            if end > 0:
                prefix = tuple(tokenizer.encode(text[:end], add_special_tokens=False))
                if 0 < len(prefix) < len(tokens) - 1 and tokens[: len(prefix)] == prefix:
                    boundaries = (len(prefix),)
        constraint = chat_constraint(
            self.format,
            self.markers,
            tools,
            choice=tool_choice,
            parallel=parallel_tool_calls,
            reasoning_prefilled=prefilled,
            response_format=response_format,
        )
        return PreparedChat(
            text, tokens, self.format, tools, constraint, prefilled, boundaries, media
        )
