"""Bounded, chunk-invariant stop-string lookbehind from v2."""


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
