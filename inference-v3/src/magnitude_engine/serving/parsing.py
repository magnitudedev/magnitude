"""Chunk-invariant semantic output events over model-specific marker wires."""

from dataclasses import dataclass
from typing import Literal

from .arguments import decode_call
from .formats import ChatFormat


@dataclass(frozen=True)
class TextDelta:
    channel: Literal["content", "reasoning"]
    text: str


@dataclass(frozen=True)
class ToolCall:
    index: int
    name: str
    arguments: dict


class OutputParser:
    def __init__(self, format: ChatFormat | None, tools: list[dict], *, reasoning_prefilled=False):
        self.format = format
        self.tools = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
        self.buffer = ""
        self.channel: Literal["content", "reasoning"] = (
            "reasoning" if reasoning_prefilled else "content"
        )
        self.in_call = False
        self.call_index = 0
        self._label = ""
        self._after_reasoning = False
        self._newlines = 0
        self.closed = False

    def feed(
        self, text: str, *, final: bool = False, truncated: bool = False
    ) -> list[TextDelta | ToolCall]:
        if self.closed:
            raise RuntimeError("output parser is closed")
        self.buffer += text
        events = []
        while self.buffer:
            if self._label:
                if not final and self._label.startswith(self.buffer):
                    break
                if self.buffer.startswith(self._label):
                    self.buffer = self.buffer[len(self._label) :]
                self._label = ""
            if self._after_reasoning:
                while self.buffer.startswith("\n") and self._newlines < 2:
                    self.buffer = self.buffer[1:]
                    self._newlines += 1
                if not self.buffer and not final and self._newlines < 2:
                    break
                self._after_reasoning = False
            if self.in_call:
                assert self.format is not None
                end = self.buffer.find(self.format.call_close)
                if end < 0:
                    if final:
                        if truncated:
                            events.append(TextDelta("content", self.format.call_open + self.buffer))
                            self.buffer, self.in_call = "", False
                            break
                        raise ValueError("generation ended inside a tool call")
                    break
                name, arguments = decode_call(self.buffer[:end], self.format.arguments, self.tools)
                events.append(ToolCall(self.call_index, name, arguments))
                self.call_index += 1
                self.buffer = self.buffer[end + len(self.format.call_close) :]
                self.in_call = False
                continue
            markers = (
                ()
                if self.format is None
                else (
                    self.format.call_open,
                    self.format.call_close,
                    self.format.reasoning_open,
                    self.format.reasoning_close,
                )
            )
            matches = [
                (self.buffer.find(marker), marker) for marker in markers if marker in self.buffer
            ]
            if matches:
                position, marker = min(matches)
                if position:
                    events.append(TextDelta(self.channel, self.buffer[:position]))
                self.buffer = self.buffer[position + len(marker) :]
                assert self.format is not None
                if marker == self.format.call_open:
                    self.in_call = True
                elif marker == self.format.reasoning_open:
                    self.channel = "reasoning"
                    self._label = self.format.reasoning_label
                elif marker == self.format.reasoning_close:
                    self.channel = "content"
                    self._after_reasoning, self._newlines = True, 0
                continue
            keep = (
                0
                if final
                else max(
                    (
                        n
                        for marker in markers
                        for n in range(1, min(len(marker), len(self.buffer) + 1))
                        if self.buffer.endswith(marker[:n])
                    ),
                    default=0,
                )
            )
            count = len(self.buffer) - keep
            if count:
                events.append(TextDelta(self.channel, self.buffer[:count]))
                self.buffer = self.buffer[count:]
            break
        if final:
            if self.in_call:
                if not truncated:
                    raise ValueError("generation ended inside a tool call")
                assert self.format is not None
                events.append(TextDelta("content", self.format.call_open))
                self.in_call = False
            self.closed = True
        return events
