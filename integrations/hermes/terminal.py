"""Non-blocking status in the ordinary Hermes terminal's scrollback.

Use prompt_toolkit's public output API: it schedules output above the application,
then restores its input. No Hermes layout objects, private methods, or ANSI cursor
sequences are touched. Capture at the host callback, before crossing worker threads.
"""

from contextvars import copy_context
from dataclasses import dataclass
from threading import Lock
from typing import Callable


def plain_line(value: str, limit: int = 300) -> str:
    """Keep untrusted model/error labels from injecting terminal control characters."""
    return "".join(char if char.isprintable() else " " for char in value)[:limit]


@dataclass(frozen=True)
class TerminalOutput:
    write: Callable[[str], None]

    @staticmethod
    def capture():
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.application import get_app_or_none
        from prompt_toolkit.formatted_text import FormattedText

        app = get_app_or_none()
        if app is None or not app.is_running:
            # Gateway, Desktop, headless and JSON output must remain uncontaminated.
            return None
        context = copy_context()

        def write(text: str) -> None:
            if app.is_running:
                context.copy().run(
                    print_formatted_text,
                    FormattedText([("ansicyan", "Magnitude · "), ("", plain_line(text))]),
                )

        return TerminalOutput(write)


class TerminalStatus:
    """One request's output lifetime. Only changes of phase enter scrollback.

    Numerical updates belong to a persistent UI, not an ever-growing transcript.
    Closing a request prevents late observations from printing stale phase updates.
    """

    def __init__(self, output: TerminalOutput):
        self._output = output
        self._phase = None
        self._closed = False
        self._lock = Lock()

    def phase(self, phase: str, text: str) -> None:
        with self._lock:
            if self._closed or self._phase == phase:
                return
            self._phase = phase
            self._output.write(text)

    def finish(self, text: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if text is not None:
                self._output.write(text)
