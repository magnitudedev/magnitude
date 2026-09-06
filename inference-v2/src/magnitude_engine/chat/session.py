"""Text-only conversation state and streaming request lifetime."""

from collections.abc import Generator

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.engine.delivery import Finished, PrefillProgress
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.serving.parsing import OutputParser, TextDelta, ToolCall
from magnitude_engine.serving.template import ChatTemplate
from magnitude_engine.serving.text import TokenText
from magnitude_engine.worker.host import Worker


class ChatSession:
    def __init__(
        self, worker: Worker, artifact: TokenizerArtifact, system: str = "",
        *, thinking: bool | None = None,
    ):
        self.worker, self.artifact = worker, artifact
        if (
            artifact.identity != worker.properties["tokenizer_identity"]
            or artifact.vocabulary != worker.properties["vocab_size"]
        ):
            raise ValueError("chat tokenizer differs from the loaded worker")
        self.template = ChatTemplate(artifact)
        self.system = system
        self.thinking = thinking
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}] if self.system else []

    def respond(
        self,
        text: str,
        sampling: SamplingPolicy,
        max_tokens: int,
    ) -> Generator[PrefillProgress | TextDelta | Finished, None, None]:
        messages = [*self.messages, {"role": "user", "content": text}]
        prompt = self.template.render(
            messages, tool_choice="none",
            chat_template_kwargs=(
                {} if self.thinking is None else {"enable_thinking": self.thinking}
            ),
        )
        if len(prompt.tokens) + max_tokens > self.worker.properties["context_tokens"]:
            raise ValueError(
                f"{len(prompt.tokens):,} prompt + {max_tokens:,} output tokens exceed the "
                f"{self.worker.properties['context_tokens']:,}-token context; use /reset "
                "or a smaller --max-tokens"
            )
        decoder = TokenText(self.artifact.tokenizer)
        parser = OutputParser(prompt.format, [], reasoning_prefilled=prompt.reasoning_prefilled)
        content: list[str] = []
        reasoning: list[str] = []
        remote = self.worker.submit(
            prompt.tokens,
            sampling,
            max_tokens,
            self.artifact.eos_tokens,
            progress=True,
        )

        def parse(text: str, *, final=False, truncated=False):
            for event in parser.feed(text, final=final, truncated=truncated):
                if isinstance(event, ToolCall):
                    raise ValueError("the text-only chat received an unsolicited tool call")
                (content if event.channel == "content" else reasoning).append(event.text)
                yield event

        try:
            while True:
                event = remote.next()
                if isinstance(event, PrefillProgress):
                    yield event
                elif isinstance(event, Finished):
                    if event.reason not in ("stop", "length"):
                        raise RuntimeError(event.message or f"generation ended: {event.reason}")
                    yield from parse(
                        decoder.feed((), final=True), final=True, truncated=event.reason == "length"
                    )
                    assistant = {"role": "assistant", "content": "".join(content)}
                    if reasoning:
                        assistant["reasoning_content"] = "".join(reasoning)
                    self.messages = [*messages, assistant]
                    yield event
                    return
                else:
                    tokens = tuple(t for t in event.values if t not in self.artifact.eos_tokens)
                    yield from parse(decoder.feed(tokens))
        finally:
            remote.cancel()
