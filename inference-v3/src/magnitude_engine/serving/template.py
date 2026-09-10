"""Artifact-owned rendering, independent of numerical execution and scheduling."""

import json
from dataclasses import dataclass
from typing import Literal

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import TypeAdapter

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.data import TokenId
from magnitude_engine.inputs.tokenizer import ByteBPETokenizer
from magnitude_engine.serving.formats import ChatFormat
from magnitude_engine.serving.requests import Message, NamedChoice, TemplateOptions, Tool
from magnitude_engine.serving.tool_choice import select_tools


@dataclass(frozen=True)
class PreparedChat:
    text: str
    tokens: tuple[TokenId, ...]
    format: ChatFormat
    tools: list[dict]
    reasoning_prefilled: bool


class ChatTemplate:
    def __init__(self, artifact: TokenizerArtifact):
        self.artifact = artifact
        self.tokenizer = ByteBPETokenizer(artifact.config)
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def reject(message):
            raise TemplateError(message)

        def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
                allow_nan=False,
            )

        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = reject
        self._template = environment.from_string(artifact.chat_template)
        self.format = ChatFormat("xml", "<tool_call>", "</tool_call>", "<think>", "</think>")

    def render(
        self,
        messages,
        *,
        tools=None,
        tool_choice: Literal["auto", "required", "none"] | NamedChoice = "auto",
        parallel_tool_calls=True,
        chat_template_kwargs=None,
    ) -> PreparedChat:
        # Serialization here is an explicit wire normalization, not mutation of
        # caller-owned history. Typed public messages also share this route.
        messages = TypeAdapter(list[Message]).validate_python(messages)
        tools = TypeAdapter(list[Tool]).validate_python([] if tools is None else tools)
        options = TemplateOptions.model_validate(chat_template_kwargs or {})
        normalized = [message.model_dump(mode="json", exclude_none=True) for message in messages]
        for message in normalized:
            content = message["content"] if "content" in message else ""
            message["content"] = (
                "".join(part["text"] for part in content) if isinstance(content, list) else content
            )
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments) if arguments.strip() else {}
                if not isinstance(arguments, dict):
                    raise ValueError("historical tool arguments must be a JSON object")
                call["function"]["arguments"] = arguments
        if not normalized:
            raise ValueError("chat requires at least one message")
        offered = [
            tool.model_dump(mode="json", exclude_none=True, exclude_unset=True) for tool in tools
        ]
        names = [tool["function"]["name"] for tool in offered]
        if len(names) != len(set(names)):
            raise ValueError("tools must have unique names")
        choice = (
            tool_choice.model_dump(mode="json")
            if isinstance(tool_choice, NamedChoice)
            else tool_choice
        )
        selection = select_tools(offered, choice)
        normalized = selection.instruct(normalized, parallel=parallel_tool_calls)
        text = self._template.render(
            messages=normalized,
            tools=list(selection.tools),
            add_generation_prompt=True,
            **options.model_dump(),
        )
        tokens = self.tokenizer.encode(text)
        if not tokens:
            raise ValueError("chat template produced no input tokens")
        return PreparedChat(
            text, tokens, self.format, list(selection.tools), text.endswith("<think>\n")
        )

    def close(self) -> None:
        pass
