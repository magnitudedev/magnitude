"""Plain terminal rendering and explicitly defined per-turn diagnostics."""

from typing import TextIO

from magnitude_engine.engine.delivery import Finished, PrefillProgress
from magnitude_engine.serving.parsing import TextDelta


def duration(ns: int | None) -> str:
    return "n/a" if ns is None else f"{ns / 1e9:.3f}s"


def rate(tokens: int, ns: int) -> str:
    return "n/a" if ns <= 0 or tokens <= 0 else f"{tokens * 1e9 / ns:,.1f} tok/s"


class Terminal:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.status_visible = False
        self.channel: str | None = None

    def write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def status(self, text: str) -> None:
        if self.stream.isatty():
            self.write("\r\x1b[2K" + text)
            self.status_visible = True
        else:
            self.write(text + "\n")

    def clear_status(self) -> None:
        if self.status_visible:
            self.write("\r\x1b[2K")
            self.status_visible = False

    def begin(self) -> None:
        self.channel = None
        self.status("Preparing prompt / waiting for admission…")

    def progress(self, event: PrefillProgress) -> None:
        percent = (
            100 if not event.total_tokens else 100 * event.completed_tokens / event.total_tokens
        )
        self.status(
            f"Prefill {event.completed_tokens:,}/{event.total_tokens:,} tokens ({percent:.0f}%)"
            f" · cached {event.cached_tokens:,}"
            f" · {rate(event.completed_tokens - event.cached_tokens, event.elapsed_ns)}"
            + (" · generating first token…" if event.completed_tokens == event.total_tokens else "")
        )

    def text(self, event: TextDelta) -> None:
        if not event.text:
            return
        self.clear_status()
        text = event.text
        if event.channel != self.channel:
            if self.channel is not None:
                self.write("\n")
            text = ("Thinking: " if event.channel == "reasoning" else "Assistant: ") + text
            self.channel = event.channel
        if event.channel == "reasoning" and self.stream.isatty():
            text = f"\x1b[2m{text}\x1b[0m"
        self.write(text)

    def finish(self, event: Finished, wall_ns: int) -> None:
        self.clear_status()
        new_prompt = event.prompt_tokens - event.cached_tokens
        prompt_ns = event.prefill_ns + event.first_decode_ns
        decode_ns = event.decode_ns - event.first_decode_ns
        self.write(
            f"\n\n[{event.reason}] Prompt {event.prompt_tokens:,}"
            f" · cached {event.cached_tokens:,} · new {new_prompt:,}"
            f" · generated {event.generated_tokens:,}\n"
            f"TTFT {duration(event.first_token_ns)} · queue {duration(event.queued_ns)}"
            f" · worker total {duration(event.finished_ns)} · wall {duration(wall_ns)}\n"
            f"Prefill + first token: {duration(prompt_ns)} · {rate(new_prompt, prompt_ns)}\n"
            f"Decode after first token: {duration(decode_ns)}"
            f" · {rate(max(0, event.generated_tokens - 1), decode_ns)}\n"
        )
        if event.proposed_tokens:
            self.write(
                f"Draft acceptance: {event.accepted_tokens}/{event.proposed_tokens} "
                f"({100 * event.accepted_tokens / event.proposed_tokens:.1f}%)\n"
            )
        if event.forced_tokens:
            self.write(f"Forced tokens: {event.forced_tokens}\n")
        self.write("\n")
