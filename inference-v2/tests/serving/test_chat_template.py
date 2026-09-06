from copy import deepcopy

import pytest

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.serving.template import ChatTemplate, normalize_messages


def test_message_normalization_preserves_tool_results_and_does_not_mutate_history():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "f",
                        "arguments": '{"x": 2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    before = deepcopy(messages)
    normalized = normalize_messages(messages)
    assert messages == before
    assert normalized[0]["content"] == "onetwo"
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"x": 2}
    assert normalized[2] == messages[2]
    with pytest.raises(ValueError, match="media-aware"):
        normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,...",
                            },
                        }
                    ],
                }
            ]
        )


def test_actual_rendered_tail_determines_reasoning_and_none_omits_template_tools():
    class Tokenizer:
        def __init__(self):
            self.received = None

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            if text in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
                return [900]
            return [ord(c) for c in text]

        def apply_chat_template(self, messages, **kwargs):
            self.received = (messages, kwargs)
            tail = "<think>\n" if kwargs.get("enable_thinking") else "<think>\n</think>\n"
            return "system\n<|im_start|>user\nquestion\n<|im_start|>assistant\n" + tail

    tokenizer = Tokenizer()
    template = ChatTemplate(TokenizerArtifact(tokenizer, "qwen3_5", 1000, (999,)))
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    prompt = template.render(
        [{"role": "user", "content": "question"}],
        tools=tools,
        tool_choice="required",
        chat_template_kwargs={"enable_thinking": True},
    )
    assert prompt.reasoning_prefilled and prompt.constraint is not None
    assert tokenizer.received[0][0]["role"] == "system"
    assert "Call" in tokenizer.received[0][0]["content"]
    assert tokenizer.received[1]["tool_choice"] == "required"
    assert prompt.boundaries == (7,) and prompt.tokens == tuple(ord(c) for c in prompt.text)
    off = template.render(
        [{"role": "user", "content": "question"}],
        tools=tools,
        tool_choice="none",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert not off.reasoning_prefilled and off.constraint is None
    assert tokenizer.received[1]["tools"] is None
    with pytest.raises(ValueError, match="override"):
        template.render(
            [{"role": "user", "content": "question"}], chat_template_kwargs={"tokenize": True}
        )


def test_named_tool_selection_is_shared_by_prompt_intent_and_grammar_without_mutating_history():
    from magnitude_engine.serving.tool_choice import select_tools

    tools = [{"type": "function", "function": {"name": name}} for name in ("first", "second")]
    messages = [{"role": "system", "content": "Original system context."},
                {"role": "user", "content": "Question."}]
    before = deepcopy((messages, tools))
    selected = select_tools(tools, {"type": "function", "function": {"name": "second"}})
    assert [tool["function"]["name"] for tool in selected.tools] == ["second"]
    assert selected.required
    rendered = selected.instruct(messages, parallel=False)
    assert "second" in rendered[0]["content"] and "first" not in rendered[0]["content"]
    assert rendered[0]["content"].startswith(messages[0]["content"])
    assert rendered[1:] == messages[1:]
    assert (messages, tools) == before
    for choice in ("auto", "none"):
        automatic = select_tools(tools, choice)
        assert not automatic.required
        assert automatic.instruct(messages, parallel=True) == messages
    assert select_tools(tools, "none").tools == ()
    for choice in ({"function": "second"}, {"function": {"name": "missing"}}):
        with pytest.raises(ValueError, match="unavailable"):
            select_tools(tools, choice)
