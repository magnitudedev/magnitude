"""Incremental tokenizer text and bounded stop-string lookbehind."""

from typing import Any


class TokenText:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.context: tuple[int, ...] = ()
        self.pending: list[int] = []
        self.closed = False

    def feed(self, tokens: tuple[int, ...], *, final: bool = False) -> str:
        if self.closed:
            raise RuntimeError("incremental text decoder is closed")
        self.pending.extend(tokens)
        decode = self.tokenizer.decode
        prefix = decode(self.context, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        whole = decode(
            (*self.context, *self.pending),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not whole.startswith(prefix):
            raise ValueError("tokenizer changed already published text")
        if not final and whole.endswith("\ufffd"):
            return ""
        output = whole[len(prefix) :]
        if output:
            self.context = tuple(self.pending)
            self.pending.clear()
        if final:
            self.closed = True
        return output


class StopText:
    def __init__(self, stops: tuple[str, ...]):
        if any(not isinstance(stop, str) or not stop for stop in stops):
            raise ValueError("stop strings must be nonempty")
        self.stops = stops
        self.pending = ""
        self.matched: str | None = None
        self.closed = False

    def feed(self, text: str, *, final: bool = False) -> str:
        if self.closed:
            raise RuntimeError("stop filter is closed")
        if self.matched is not None:
            return ""
        self.pending += text
        matches = [
            (position + len(stop), index, position, stop)
            for index, stop in enumerate(self.stops)
            if (position := self.pending.find(stop)) >= 0
        ]
        if matches:
            # Select the first completed match, just as if characters arrived individually.
            # Overlapping patterns therefore do not change behavior with transport chunk size.
            _, _, position, self.matched = min(matches)
            output, self.pending = self.pending[:position], ""
            return output
        keep = (
            0
            if final
            else max(
                (
                    size
                    for stop in self.stops
                    for size in range(1, min(len(stop), len(self.pending) + 1))
                    if self.pending.endswith(stop[:size])
                ),
                default=0,
            )
        )
        end = len(self.pending) - keep
        output, self.pending = self.pending[:end], self.pending[end:]
        self.closed = final
        return output
