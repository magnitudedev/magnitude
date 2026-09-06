from types import SimpleNamespace

import pytest

from magnitude_engine.chat.session import ChatSession
from magnitude_engine.engine.delivery import Finished, PrefillProgress, Tokens
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.serving.parsing import TextDelta


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['tools'] is None and kwargs['tool_choice'] == 'none'
        return str(messages)

    def encode(self, text, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, **kwargs):
        return ''.join({4: 'Hello', 5: '!'}[t] for t in tokens)


class Remote:
    def __init__(self, events):
        self.events = iter(events)
        self.cancelled = False

    def next(self):
        return next(self.events)

    def cancel(self):
        self.cancelled = True


class Host:
    properties = {'vocab_size': 16, 'tokenizer_identity': 'test', 'context_tokens': 32}

    def __init__(self, events):
        self.remote = Remote(events)
        self.submitted = False

    def submit(self, tokens, sampling, max_tokens, eos, *, progress):
        assert progress and tokens == (1, 2, 3) and eos == (0,)
        self.submitted = True
        return self.remote


def session(events):
    host = Host(events)
    artifact = SimpleNamespace(identity='test', vocabulary=16, family='unknown',
                               eos_tokens=(0,), tokenizer=Tokenizer())
    return ChatSession(host, artifact, 'Be concise.'), host


def test_chat_streams_without_waiting_for_finish_and_commits_only_complete_turns():
    finish = Finished('stop', 3, 3, 0, 0, 0, 0, 1, 2)
    chat, host = session([PrefillProgress(2, 2, 0, 1), Tokens((4,)), Tokens((5, 0)), finish])
    stream = chat.respond('Hi', SamplingPolicy(temperature=0), 4)
    assert isinstance(next(stream), PrefillProgress)
    assert next(stream) == TextDelta('content', 'Hello')
    assert len(chat.messages) == 1
    assert list(stream) == [TextDelta('content', '!'), finish]
    assert chat.messages[-2:] == [
        {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'},
    ]
    assert host.remote.cancelled
    chat.reset()
    assert chat.messages == [{'role': 'system', 'content': 'Be concise.'}]


def test_interrupted_turn_cancels_worker_and_leaves_history_unchanged():
    chat, host = session([Tokens((4,))])
    original = list(chat.messages)
    stream = chat.respond('Hi', SamplingPolicy(), 4)
    assert next(stream) == TextDelta('content', 'Hello')
    with pytest.raises(KeyboardInterrupt):
        stream.throw(KeyboardInterrupt)
    assert host.remote.cancelled and chat.messages == original


def test_overlong_request_is_rejected_before_submission():
    chat, host = session([])
    with pytest.raises(ValueError, match='exceed'):
        list(chat.respond('Hi', SamplingPolicy(), 32))
    assert not host.submitted


def test_worker_failure_does_not_add_a_partial_turn_to_history():
    chat, host = session([Tokens((4,)), Finished('error', 3, 1, 0, 0, 0, 0, 1, 2,
                                              message='device failed')])
    with pytest.raises(RuntimeError, match='device failed'):
        list(chat.respond('Hi', SamplingPolicy(), 4))
    assert host.remote.cancelled and len(chat.messages) == 1
