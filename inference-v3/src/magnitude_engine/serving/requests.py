"""Typed chat wire inputs; unsupported generation policies fail before admission."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from magnitude_engine.data import Record


class Function(Record):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strict: bool = False


class Tool(Record):
    type: Literal["function"] = "function"
    function: Function


class Invocation(Record):
    name: str = Field(min_length=1)
    arguments: str | dict[str, JsonValue]


class ToolCall(Record):
    id: str
    type: Literal["function"] = "function"
    function: Invocation


class TextPart(Record):
    type: Literal["text"] = "text"
    text: str


class Message(Record):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class NamedFunction(Record):
    name: str = Field(min_length=1)


class NamedChoice(Record):
    type: Literal["function"] = "function"
    function: NamedFunction


class TemplateOptions(Record):
    enable_thinking: bool = False
    add_vision_id: bool = False


class StreamOptions(Record):
    include_usage: bool = False


class ChatRequest(Record):
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=4096)
    tools: list[Tool] = Field(default_factory=list, max_length=512)
    tool_choice: Literal["auto", "required", "none"] | NamedChoice = "auto"
    parallel_tool_calls: bool = True
    chat_template_kwargs: TemplateOptions = TemplateOptions()
    max_tokens: int | None = Field(default=None, ge=0, le=0x7FFFFFFF)
    max_completion_tokens: int | None = Field(default=None, ge=0, le=0x7FFFFFFF)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    presence_penalty: float = Field(default=0.0, allow_inf_nan=False)
    frequency_penalty: float = Field(default=0.0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, lt=2**64)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions = StreamOptions()
    n: Literal[1] = 1

    @model_validator(mode="after")
    def compatible(self):
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens disagree")
        if len(self.stops) > 4 or any(not s or len(s) > 1024 for s in self.stops):
            raise ValueError("provide at most four nonempty stop strings up to 1024 characters")
        return self

    @property
    def output_limit(self) -> int:
        value = (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens
        )
        return 512 if value is None else value

    @property
    def stops(self) -> tuple[str, ...]:
        return (
            ()
            if self.stop is None
            else (self.stop,)
            if isinstance(self.stop, str)
            else tuple(self.stop)
        )

    def require_supported_generation(self) -> None:
        if (
            self.temperature not in (0, 1)
            or self.top_p != 1
            or self.top_k != 0
            or self.min_p != 0
            or self.repetition_penalty != 1
            or self.presence_penalty != 0
            or self.frequency_penalty != 0
        ):
            raise ValueError(
                "distribution transformations are not implemented; "
                "use temperature 0 or 1 with unmodified logits"
            )
        if self.tool_choice not in ("auto", "none"):
            raise ValueError("required/named tool constraints are not implemented")
