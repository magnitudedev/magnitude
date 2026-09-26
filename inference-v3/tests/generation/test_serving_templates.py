"""Native preparation, exact tokenizer binding and wire event integration."""

from pathlib import Path

import pytest
from test_constraints import vocabulary

from engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from engine.serving.requests import ChatRequest, NamedChoice, NamedFunction
from engine.serving.responses import ChatResponse
from engine.serving.template import ChatTemplate
from templates.bundle import TemplateBundle, Variant
from templates.events import ContentDelta, ReasoningDelta, ToolArguments, ToolStart

FIXTURES = Path(__file__).resolve().parents[2] / "native/templates/upstream/models/templates"
MESSAGES = [{"role": "user", "content": "Hello"}]
TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def renderer():
    binding = vocabulary()
    source = (FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()
    bundle = TemplateBundle(
        default="default",
        variants=(
            Variant(name="default", source=source, provenance="fixture"),
            Variant(name="tool_use", source=source, provenance="tool fixture"),
        ),
    )
    return ChatTemplate(
        TokenizerArtifact(config=binding.tokenizer.config, templates=bundle)
    ), binding


def test_authored_thinking_default_and_strict_effort_resolution():
    template, _ = renderer()
    try:
        authored = template.render(MESSAGES, now=0)
        disabled = template.render(MESSAGES, reasoning_effort="none", now=0)
        try:
            assert authored.text.endswith("<|im_start|>assistant\n")
            assert disabled.text.endswith("<think>\n\n</think>\n\n")
            assert authored.profile.default_effort == "high"
            assert template.tokenizer.encode(authored.text) == authored.tokens
        finally:
            authored.close()
            disabled.close()
        with pytest.raises(ValueError, match="Unsupported reasoning effort"):
            template.render(MESSAGES, reasoning_effort="low", now=0)
        with pytest.raises(ValueError, match="conflicts"):
            template.render(
                MESSAGES,
                reasoning_effort="none",
                chat_template_kwargs={"enable_thinking": True},
                now=0,
            )
        with pytest.raises(ValueError, match="Reserved"):
            template.render(MESSAGES, chat_template_kwargs={"messages": []}, now=0)
    finally:
        template.close()


def test_effective_tools_select_variant_and_named_choice_constrains_output():
    template, binding = renderer()
    try:
        none = template.render(MESSAGES, tools=[TOOL], tool_choice="none", now=0)
        named = template.render(
            MESSAGES,
            tools=[TOOL],
            tool_choice=NamedChoice(function=NamedFunction(name="weather")),
            reasoning_effort="none",
            now=0,
        )
        try:
            assert none.variant.name == "default"
            assert named.variant.name == "tool_use"
            assert "weather" not in none.text
            assert named.constraint is not None
            state = binding.bind(named.constraint)
            output = '<tool_call>\n{"name":"weather","arguments":{"city":"Paris"}}\n</tool_call>'
            ids = binding.tokenizer.encode(output)
            state.stage(ids).commit()
            assert state.accepting
            events = named.native.parse(output.encode())
            starts = [event for event in events if isinstance(event, ToolStart)]
            assert len(starts) == 1 and starts[0].name == "weather"
        finally:
            none.close()
            named.close()
    finally:
        template.close()


def test_wire_arguments_preserve_exact_incremental_text_and_authored_call_id(monkeypatch):
    from types import SimpleNamespace

    response = ChatResponse("fixture")
    monkeypatch.setattr(response, "evidence", lambda event: {})
    events = [
        ReasoningDelta(text="think"),
        ContentDelta(text="answer"),
        ToolStart(index=0, name="weather", id="authored-id"),
        ToolArguments(index=0, text='{"city":'),
        ToolArguments(index=0, text=' "Paris"}'),
    ]
    chunks = [response.semantic(event, retain=True) for event in events]
    complete = response.complete(SimpleNamespace(reason="tool_calls"))
    assert complete["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "authored-id",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": '{"city": "Paris"}',
            },
        }
    ]
    assert (
        chunks[-1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == ' "Paris"}'
    )
    assert response.content == ["answer"] and response.reasoning == ["think"]


def test_wire_defaults_do_not_inject_controls_and_json_tool_combination_rejects():
    body = ChatRequest.model_validate({"model": "fixture", "messages": MESSAGES})
    assert body.chat_template_kwargs == {} and body.reasoning_effort is None
    body.require_supported_generation()
    body = ChatRequest.model_validate(
        {
            "model": "fixture",
            "messages": MESSAGES,
            "response_format": {"type": "json_object"},
            "tools": [TOOL],
        }
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        body.require_supported_generation()
    body.model_copy(update={"tool_choice": "none"}).require_supported_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["natural", "length", "user_stop"])
async def test_service_uses_prepared_parser_and_passes_constraint_before_publication(cause):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from engine.generation.plain import FinishReason, OutputToken
    from engine.serving.runtime import Publication
    from engine.serving.session import ChatFinished, ChatService

    template, _ = renderer()
    body = ChatRequest.model_validate(
        {
            "model": "fixture",
            "messages": MESSAGES,
            "tools": [TOOL],
            "tool_choice": "required",
            "reasoning_effort": "none",
            "stop": "ris" if cause == "user_stop" else None,
        }
    )
    output = '<tool_call>\n{"name":"weather","arguments":{"city":"Paris"}}\n</tool_call>'
    if cause == "length":
        output = output[: output.index("Paris") + 2]
    tokens = template.tokenizer.encode(output)

    class Owner:
        def __init__(self):
            self.received = 0
            self.released = False
            self.plan = None

        def admit(self, prompt, options, plan, media):
            assert media is None
            self.plan = plan
            return 7

        def receive(self, identity):
            assert identity == 7
            index = self.received
            self.received += 1
            final = self.received == len(tokens)
            result = Future()
            result.set_result(
                Publication(
                    tokens=(OutputToken(index=index, token=tokens[index]),),
                    state=SimpleNamespace(
                        finish=(FinishReason.LENGTH if cause == "length" else FinishReason.STOP)
                        if final
                        else None,
                        queued_output=0,
                    ),
                )
            )
            return result

        def release(self, identity):
            assert identity == 7
            self.released = True

        def stop(self, identity):
            assert identity == 7
            return SimpleNamespace(finish=FinishReason.CANCELLED, queued_output=0)

    owner = Owner()

    class Worker:
        def call(self, operation):
            result = Future()
            try:
                result.set_result(operation(owner))
            except BaseException as error:
                result.set_exception(error)
            return result

    service = ChatService(
        Worker(),
        SimpleNamespace(
            model="fixture", context_tokens=32768, output_capacity=16, forced_quantum=32
        ),
        template,
    )
    prompt = service.prepare(body)
    try:
        events = [event async for event in service.events(body, prompt)]
        assert owner.plan == prompt.constraint and owner.plan is not None
        assert owner.released
        assert isinstance(events[-1], ChatFinished)
        assert (
            events[-1].reason
            == {"length": "length", "natural": "tool_calls", "user_stop": "stop"}[cause]
        )
        assert any(isinstance(event, ToolStart) for event in events)
        assert not any(
            isinstance(event, ContentDelta) and "<tool_call>" in event.text for event in events
        )
        arguments = "".join(event.text for event in events if isinstance(event, ToolArguments))
        assert arguments == ('{"city":"Paris"}' if cause == "natural" else '{"city":"Pa')
    finally:
        prompt.close()
        template.close()


@pytest.mark.parametrize(
    "filename,output",
    [
        (
            "Qwen3.5-4B.jinja",
            "consider</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        ),
        (
            "openai-gpt-oss-120b.jinja",
            ' to=functions.weather<|channel|>commentary json<|message|>{"city":"Paris"}',
        ),
        (
            "deepseek-ai-DeepSeek-V3.1.jinja",
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{"city":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ),
        (
            "mistralai-Mistral-Nemo-Instruct-2407.jinja",
            '[TOOL_CALLS][{"name":"weather","arguments":{"city":"Paris"},"id":"abc123xyz"}]',
        ),
    ],
)
def test_serving_preparation_generalizes_across_template_families(filename, output):
    from templates.bundle import SpecialToken

    binding = vocabulary()
    bundle = TemplateBundle(
        default="default",
        variants=(
            Variant(name="default", source=(FIXTURES / filename).read_text(), provenance=filename),
        ),
        special_tokens=(
            SpecialToken(name="bos_token", text="<s>"),
            SpecialToken(name="eos_token", text="</s>"),
        ),
    )
    template = ChatTemplate(TokenizerArtifact(config=binding.tokenizer.config, templates=bundle))
    try:
        prompt = template.render(MESSAGES, tools=[TOOL], now=946684800)
        try:
            assert prompt.constraint is not None
            state = binding.bind(prompt.constraint)
            state.stage(binding.tokenizer.encode(output)).commit()
            assert state.accepting
            events = prompt.native.parse(output.encode())
            starts = [event for event in events if isinstance(event, ToolStart)]
            assert len(starts) == 1 and starts[0].name == "weather"
            profile = template.describe()["profiles"]["default"]
            assert profile["reasoning_fingerprint"] == prompt.profile.fingerprint
            assert profile["identity"] == prompt.constraint.template_identity
        finally:
            prompt.close()
    finally:
        template.close()
