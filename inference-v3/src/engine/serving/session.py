"""Async transport lifetime over typed execution-owner calls and token publication."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from threading import Lock

from engine.generation.plain import FinishReason, Options
from engine.operations.sampling import SamplingSeed, SelectionKind
from engine.platform.host.worker import Worker
from engine.service.engine import Snapshot
from engine.serving.metrics import ParsingMetrics, PreparationMetrics
from engine.serving.requests import ChatRequest, ImagePart, SchemaFormat
from engine.serving.runtime import Config, Runtime, ServerProperties, open_runtime
from engine.serving.template import ChatTemplate, PreparedChat
from engine.serving.text import StopText
from templates.events import ContentDelta, ReasoningDelta, ToolArguments, ToolComplete, ToolStart


@dataclass(frozen=True)
class ChatFinished:
    reason: str
    prompt_tokens: int
    native: Snapshot
    string_stop: str | None
    preparation: PreparationMetrics = PreparationMetrics()
    parsing: ParsingMetrics = ParsingMetrics()


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
            template = ChatTemplate(
                ready.tokenizer,
                variant=config.template_variant,
                override=config.template_override,
                image_directory=ready.image_directory,
            )
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
            chat_template_kwargs=body.chat_template_kwargs,
            reasoning_effort=body.reasoning_effort,
            json_schema=(
                body.response_format.json_schema.schema_
                if isinstance(body.response_format, SchemaFormat)
                else {"type": "object"}
                if body.response_format.type == "json_object"
                else None
            ),
        )
        if len(prompt.tokens) > self.properties.context_tokens:
            prompt.close()
            raise ValueError("rendered prompt exceeds the configured context limit")
        return prompt

    async def prepare_async(self, body: ChatRequest) -> PreparedChat:
        if not any(
            isinstance(message.content, list)
            and any(isinstance(part, ImagePart) for part in message.content)
            for message in body.messages
        ):
            return self.prepare(body)

        # Cancelling the await does not stop host image preparation. Keep the
        # result owned until the event loop takes it, or close it on abandonment.
        lock = Lock()
        abandoned = False
        prompt: PreparedChat | None = None

        def build():
            nonlocal prompt
            prepared = self.prepare(body)
            with lock:
                if not abandoned:
                    prompt = prepared
                    return
            prepared.close()

        try:
            await asyncio.to_thread(build)
        except BaseException:
            with lock:
                abandoned = True
                discarded, prompt = prompt, None
            if discarded is not None:
                discarded.close()
            raise
        with lock:
            result, prompt = prompt, None
        assert result is not None
        return result

    async def events(
        self, body: ChatRequest, prompt: PreparedChat
    ) -> AsyncGenerator[
        ContentDelta | ReasoningDelta | ToolStart | ToolArguments | ChatFinished, None
    ]:
        options = Options(
            max_tokens=min(
                body.output_limit, self.properties.context_tokens - len(prompt.tokens) + 1
            ),
            stop_tokens=self.template.tokenizer.stop_tokens,
            selection=SelectionKind.GREEDY if body.temperature == 0 else SelectionKind.CATEGORICAL,
            seed=SamplingSeed(body.seed),
            output_capacity=self.properties.output_capacity,
            forced_quantum=self.properties.forced_quantum,
        )
        admission = None
        identity = None
        parser = None
        calls_completed = False
        try:
            decoder = self.template.tokenizer.decoder(skip_control=False)
            parser = prompt.native.stream()
            stops = StopText(body.stops + prompt.native.description.additional_stops)
            admission = asyncio.wrap_future(
                self.worker.call(
                    lambda owner: owner.admit(
                        prompt.tokens, options, prompt.constraint, prompt.media
                    )
                )
            )
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
                    try:
                        decoded = decoder.push(item.token)
                    except IndexError as error:
                        raise RuntimeError(
                            f"tokenizer cannot decode model token {int(item.token)}"
                        ) from error
                    for event in parser.feed(stops.feed(decoded).encode("utf-8")):
                        if isinstance(event, ToolComplete):
                            calls_completed = True
                        if isinstance(
                            event, (ContentDelta, ReasoningDelta, ToolStart, ToolArguments)
                        ):
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
                    cause = (
                        "user_stop"
                        if stops.matched in body.stops
                        else "length"
                        if reason == "length"
                        else "natural"
                    )
                    for event in parser.feed(tail.encode("utf-8")) + parser.finish(cause):
                        if isinstance(event, ToolComplete):
                            calls_completed = True
                        if isinstance(
                            event, (ContentDelta, ReasoningDelta, ToolStart, ToolArguments)
                        ):
                            yield event
                    if calls_completed and cause == "natural":
                        reason = "tool_calls"
                    yield ChatFinished(
                        reason,
                        len(prompt.tokens),
                        state,
                        stops.matched,
                        prompt.metrics,
                        ParsingMetrics(
                            elapsed_ns=parser.elapsed_ns,
                            input_bytes=parser.input_bytes,
                            calls=parser.calls,
                        ),
                    )
                    return
        finally:
            if parser is not None:
                parser.close()
            prompt.close()
            if identity is None and admission is not None:
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
