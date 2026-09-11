"""Async transport lifetime over typed execution-owner calls and token publication."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial

from magnitude_engine.generation.plain import FinishReason, Options
from magnitude_engine.operations.sampling import SamplingSeed, SelectionKind
from magnitude_engine.platform.host.worker import Worker
from magnitude_engine.service.engine import Snapshot
from magnitude_engine.serving.parsing import OutputParser, TextDelta, ToolCall
from magnitude_engine.serving.requests import ChatRequest
from magnitude_engine.serving.runtime import Config, Runtime, ServerProperties, open_runtime
from magnitude_engine.serving.template import ChatTemplate, PreparedChat
from magnitude_engine.serving.text import StopText


@dataclass(frozen=True)
class ChatFinished:
    reason: str
    prompt_tokens: int
    native: Snapshot
    string_stop: str | None


class ChatService:
    def __init__(
        self, worker: Worker[Runtime], properties: ServerProperties, template: ChatTemplate
    ):
        self.worker, self.properties, self.template = worker, properties, template
        self.model = properties.model

    @classmethod
    async def open(cls, config: Config):
        worker = Worker(partial(open_runtime, config))
        try:
            await asyncio.shield(asyncio.wrap_future(worker.ready))
            ready = await asyncio.wrap_future(worker.call(lambda owner: owner.ready))
            template = ChatTemplate(ready.tokenizer)
            return cls(worker, ready.properties, template)
        except BaseException:
            await asyncio.to_thread(worker.close)
            raise

    def prepare(self, body: ChatRequest) -> PreparedChat:
        body.require_supported_generation()
        prompt = self.template.render(
            body.messages,
            tools=body.tools,
            tool_choice=body.tool_choice,
            parallel_tool_calls=body.parallel_tool_calls,
            chat_template_kwargs=body.chat_template_kwargs.model_dump(),
        )
        if len(prompt.tokens) > self.properties.context_tokens:
            raise ValueError("rendered prompt exceeds the configured context limit")
        return prompt

    async def events(
        self, body: ChatRequest, prompt: PreparedChat
    ) -> AsyncGenerator[TextDelta | ToolCall | ChatFinished, None]:
        options = Options(
            max_tokens=min(
                body.output_limit, self.properties.context_tokens - len(prompt.tokens) + 1
            ),
            stop_tokens=self.template.tokenizer.stop_tokens,
            selection=SelectionKind.GREEDY if body.temperature == 0 else SelectionKind.CATEGORICAL,
            seed=SamplingSeed(body.seed),
            output_capacity=self.properties.output_capacity,
        )
        admission = asyncio.wrap_future(
            self.worker.call(lambda owner: owner.admit(prompt.tokens, options))
        )
        identity = None
        decoder = self.template.tokenizer.decoder(skip_control=False)
        parser = OutputParser(
            prompt.format, prompt.tools, reasoning_prefilled=prompt.reasoning_prefilled
        )
        stops = StopText(body.stops)
        try:
            # A cancelled await must not orphan an admission that already began.
            identity = await asyncio.shield(admission)
            request_id = identity
            while True:
                receiver = await asyncio.wrap_future(
                    self.worker.call(lambda owner: owner.receive(request_id))
                )
                publication = await asyncio.wrap_future(receiver)
                state = publication.state
                for item in publication.tokens:
                    for event in parser.feed(stops.feed(decoder.push(item.token))):
                        yield event
                if stops.matched is not None:
                    state = await asyncio.wrap_future(
                        self.worker.call(lambda owner: owner.stop(request_id))
                    )
                if stops.matched is not None or (
                    state.finish is not None and state.queued_output == 0
                ):
                    if state.finish == FinishReason.FAILED:
                        raise RuntimeError(
                            state.failure.message
                            if state.failure is not None
                            else "model execution failed"
                        )
                    reason = (
                        "stop"
                        if stops.matched is not None or state.finish == FinishReason.STOP
                        else "length"
                    )
                    tail = stops.feed(decoder.finish(), final=True)
                    for event in parser.feed(tail, final=True, truncated=reason == "length"):
                        yield event
                    if parser.call_index:
                        reason = "tool_calls" if reason == "stop" else reason
                    yield ChatFinished(reason, len(prompt.tokens), state, stops.matched)
                    return
        finally:
            if identity is None:
                try:
                    identity = await asyncio.shield(admission)
                except Exception:
                    pass
            if identity is not None:
                await asyncio.shield(
                    asyncio.wrap_future(self.worker.call(lambda owner: owner.release(identity)))
                )

    async def close(self) -> None:
        await asyncio.to_thread(self.worker.close)
        self.template.close()
