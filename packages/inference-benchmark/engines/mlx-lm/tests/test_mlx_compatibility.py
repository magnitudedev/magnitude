import threading

import magnitude_mlx_benchmark_server.generation as generation
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.server import (
    SequenceStateMachine,
    ToolCallFormatter,
    _process_control_tokens,
    process_message_content,
)


def test_pinned_mlx_lm_internal_adapter_surface_exists():
    assert callable(make_prompt_cache)
    assert callable(_process_control_tokens)
    assert callable(process_message_content)
    assert callable(SequenceStateMachine)
    assert callable(ToolCallFormatter)
    assert callable(LRUPromptCache)


def test_native_message_normalization_handles_openai_tool_history():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"value"}'},
                }
            ],
        }
    ]
    process_message_content(messages)
    assert messages[0]["content"] == ""
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}


def test_generation_worker_loads_and_generates_on_one_thread(monkeypatch):
    owner = None

    class FakeGenerator:
        def __init__(self, *_args, **_kwargs):
            nonlocal owner
            owner = threading.get_ident()

        def generate(self, _request, _cancelled):
            assert threading.get_ident() == owner
            yield generation.GeneratedEvent("content", "ok")

    monkeypatch.setattr(generation, "MlxGenerator", FakeGenerator)
    received = []
    completed = threading.Event()
    worker = generation.MlxGenerationWorker(
        "model", prefill_step_size=1, prompt_cache_entries=1
    )

    def emit(event):
        received.append(event)
        if event is None:
            completed.set()

    worker.submit(object(), threading.Event(), emit)
    assert completed.wait(timeout=1)
    worker.close()
    assert received[0] == generation.GeneratedEvent("content", "ok")
    assert received[1] is None
