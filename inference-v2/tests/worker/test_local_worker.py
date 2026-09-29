"""Opt-in production model execution across the disposable worker boundary."""

import os

import pytest
from transformers import AutoTokenizer

from magnitude_engine.engine.delivery import Finished
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.worker.host import Worker
from tests.worker.test_worker import compose_engine


@pytest.mark.model
def test_local_qwen_mtp_worker_concurrency_restore_and_disposal():
    target, head = (
        os.environ.get("MAGNITUDE_TEST_MTP_TARGET"),
        os.environ.get("MAGNITUDE_TEST_MTP_HEAD"),
    )
    if target is None or head is None:
        pytest.skip("set MAGNITUDE_TEST_MTP_TARGET and MAGNITUDE_TEST_MTP_HEAD to local artifacts")
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    prompt = tuple(
        tokenizer.encode("Write a Python function that adds two numbers.\n\ndef add(a, b):")
    )
    config = compose_engine(target, head, context_tokens=128, max_active=2, output_capacity=4)

    def collect(request):
        tokens = []
        for _ in range(40):
            event = request.next(10)
            if isinstance(event, Finished):
                return tokens, event
            tokens.extend(event.values)
        raise AssertionError("worker stream did not terminate")

    with Worker(config) as host:
        blocked = host.submit(prompt, SamplingPolicy(temperature=0), 32)
        request = host.submit(prompt, SamplingPolicy(temperature=0), 8)
        tokens, finish = collect(request)
        assert tokens[:6] == [198, 262, 460, 264, 478, 292]
        assert len(tokens) == 8 and finish.reason == "length"
        blocked.cancel()
        warm_tokens, warm = collect(host.submit(prompt, SamplingPolicy(temperature=0), 8))
        assert warm_tokens == tokens and warm.cached_tokens == len(prompt) - 1
        print(
            {
                "target": target,
                "head": head,
                "worker_pid": host.process.pid,
                "tokens": tokens,
                "text": tokenizer.decode(tokens),
                "cached_tokens": warm.cached_tokens,
            }
        )
    assert host.process.poll() == 0, host.stderr


@pytest.mark.model
def test_local_qwen_mtp_required_tool_call_through_private_worker():
    from dataclasses import asdict
    from pathlib import Path

    from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
    from magnitude_engine.serving.parsing import OutputParser, ToolCall
    from magnitude_engine.serving.template import ChatTemplate

    target = os.environ.get("MAGNITUDE_TEST_MTP_TARGET")
    head = os.environ.get("MAGNITUDE_TEST_MTP_HEAD")
    if target is None or head is None:
        pytest.skip("set MAGNITUDE_TEST_MTP_TARGET and MAGNITUDE_TEST_MTP_HEAD to local artifacts")
    artifact = TokenizerArtifact.load(Path(target))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "enum": ["San Francisco"]},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    prompt = ChatTemplate(artifact).render(
        [{"role": "user", "content": "Get the current weather in San Francisco in celsius."}],
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    config = compose_engine(target, head, context_tokens=2048, output_capacity=8)
    with Worker(config) as host:
        request = host.submit(
            prompt.tokens,
            SamplingPolicy(temperature=0),
            160,
            artifact.eos_tokens,
            constraint=prompt.constraint,
        )
        tokens = []
        for _ in range(170):
            event = request.next(30)
            if isinstance(event, Finished):
                break
            tokens.extend(event.values)
        else:
            raise AssertionError("constrained worker did not terminate")
        assert event.reason == "stop", event
        text = artifact.tokenizer.decode(
            [t for t in tokens if t not in artifact.eos_tokens], skip_special_tokens=False
        )
        parser = OutputParser(prompt.format, tools, reasoning_prefilled=prompt.reasoning_prefilled)
        events = []
        for char in text:
            events.extend(parser.feed(char))
        events.extend(parser.feed("", final=True))
        calls = [e for e in events if isinstance(e, ToolCall)]
        assert calls == [ToolCall(0, "get_weather", {"city": "San Francisco", "unit": "celsius"})]
        assert event.forced_tokens > 0
        print(
            {
                "target": target,
                "head": head,
                "prompt_tokens": len(prompt.tokens),
                "output": text,
                "calls": [asdict(call) for call in calls],
                "finish": asdict(event),
            }
        )
    assert host.process.poll() == 0, host.stderr
