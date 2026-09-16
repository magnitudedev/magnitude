"""Append-only semantic events, shared by streaming and complete responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

TerminalCause = Literal["natural", "length", "user_stop", "cancelled", "failed"]
TERMINAL_CAUSES: tuple[TerminalCause, ...] = (
    "natural",
    "length",
    "user_stop",
    "cancelled",
    "failed",
)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ContentDelta(Record):
    kind: Literal["content"] = "content"
    text: str


class ReasoningDelta(Record):
    kind: Literal["reasoning"] = "reasoning"
    text: str


class ToolStart(Record):
    kind: Literal["tool_start"] = "tool_start"
    index: int
    name: str
    id: str


class ToolArguments(Record):
    kind: Literal["tool_arguments"] = "tool_arguments"
    index: int
    text: str


class ToolComplete(Record):
    kind: Literal["tool_complete"] = "tool_complete"
    index: int


class Finish(Record):
    kind: Literal["finish"] = "finish"
    cause: TerminalCause


Event = ContentDelta | ReasoningDelta | ToolStart | ToolArguments | ToolComplete | Finish
