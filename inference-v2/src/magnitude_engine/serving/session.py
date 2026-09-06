"""Host request lifetime: demand-driven worker output to semantic chat events."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from functools import partial

import anyio
from anyio import to_thread

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.engine.delivery import Finished, PrefillProgress
from magnitude_engine.worker.host import Worker

from .parsing import OutputParser, TextDelta, ToolCall
from .requests import ChatRequest
from .template import ChatTemplate, PreparedChat
from .text import StopText, TokenText


@dataclass(frozen=True)
class ChatFinished:
    native: Finished
    reason: str
    string_stop: str | None


class ChatService:
    def __init__(self, host: Worker, artifact: TokenizerArtifact, model: str):
        self.host, self.artifact, self.model = host, artifact, model
        self.template = ChatTemplate(artifact)
        if artifact.vocabulary != host.properties["vocab_size"]:
            raise ValueError("host tokenizer vocabulary differs from the ready worker")
        if artifact.identity != host.properties["tokenizer_identity"]:
            raise ValueError("host tokenizer identity differs from the ready worker")

    def prepare(self, request: ChatRequest) -> PreparedChat:
        prompt = self.template.render(
            request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            chat_template_kwargs=request.chat_template_kwargs,
            response_format=request.response_format,
        )
        if len(prompt.tokens) + request.output_limit > self.host.properties["context_tokens"]:
            raise ValueError("rendered prompt and output allowance exceed the configured context")
        return prompt

    async def events(
        self,
        request: ChatRequest,
        prompt: PreparedChat,
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncGenerator[TextDelta | ToolCall | ChatFinished]:
        remote = await to_thread.run_sync(
            partial(
                self.host.submit,
                prompt.tokens,
                request.sampling(),
                request.output_limit,
                self.artifact.eos_tokens,
                constraint=prompt.constraint,
            )
        )
        decoder = TokenText(self.artifact.tokenizer)
        stops = StopText(request.stops)
        parser = OutputParser(
            prompt.format, prompt.tools, reasoning_prefilled=prompt.reasoning_prefilled
        )
        finish = None
        try:
            while not await disconnected():
                try:
                    event = await to_thread.run_sync(partial(remote.next, timeout=0.25))
                except TimeoutError:
                    continue
                if isinstance(event, PrefillProgress):
                    continue
                if isinstance(event, Finished):
                    finish = event
                    break
                tokens = tuple(t for t in event.values if t not in self.artifact.eos_tokens)
                for parsed in parser.feed(stops.feed(decoder.feed(tokens))):
                    yield parsed
                if stops.matched is not None:
                    finish = await to_thread.run_sync(remote.cancel)
                    if finish is None:
                        raise RuntimeError("cancelled generation supplied no completion evidence")
                    break
            if finish is None:
                return
            if finish.reason == "error":
                raise ValueError(finish.message or "worker generation failed")
            if finish.reason == "cancelled" and stops.matched is None:
                raise RuntimeError("generation was cancelled before completing")
            reason = "stop" if stops.matched is not None else finish.reason
            tail = stops.feed(decoder.feed((), final=True), final=True)
            for parsed in parser.feed(
                tail, final=True, truncated=reason == "length" or stops.matched is not None
            ):
                yield parsed
            if reason == "stop" and parser.call_index:
                reason = "tool_calls"
            yield ChatFinished(finish, reason, stops.matched)
        finally:
            # ASGI cancels streaming generators on disconnect. Cleanup must outlive that
            # cancellation and use the independent worker control lane, not an output read.
            with anyio.CancelScope(shield=True):
                await to_thread.run_sync(remote.cancel)
