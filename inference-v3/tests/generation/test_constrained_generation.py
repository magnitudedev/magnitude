"""Logical acceptance tests use no numerical implementation or device."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
from test_constraints import vocabulary

from engine.data import TokenId
from engine.generation.constraints import ConstraintState
from engine.generation.plain import FinishReason, Generation, GenerationBatch, Options, WorkKind
from engine.inputs.layout import InputLayout
from engine.models.sequence import LogitsSelection


class Sequence:
    def __init__(self, model, position=0):
        self.model, self.position = model, position
        self.layout = InputLayout(count=1)
        self.context_limit = 100
        self.context = SimpleNamespace(check=lambda: None, check_thread=lambda: None)

    def close(self):
        pass

    def checkpoint(self):
        model, position = self.model, self.position
        return SimpleNamespace(fork=lambda: Sequence(model, position), close=lambda: None)


class Model:
    def __init__(self):
        self.next_token = ord("y")
        self.refuse = False
        self.commit_fails = False
        self.requests = ()

    def prepare(self, requests):
        if self.refuse:
            raise RuntimeError("capacity retry")
        self.requests = tuple(
            replace(request, allowed_tokens=request.allowed_tokens.mask())
            if request.allowed_tokens is not None and not isinstance(request.allowed_tokens, bytes)
            else request
            for request in requests
        )
        advances = []
        for request in requests:

            def commit(request=request):
                if self.commit_fails:
                    raise RuntimeError("commit failure")
                request.sequence.position += len(request.tokens)

            sampled = None if request.selection == LogitsSelection.NONE else (self.next_token, 0)
            advances.append(
                SimpleNamespace(
                    read_sample=lambda sampled=sampled: sampled,
                    commit=commit,
                    close=lambda: None,
                )
            )
        return SimpleNamespace(
            advances=tuple(advances),
            completion=SimpleNamespace(done=True, wait=lambda: None),
            close=lambda: None,
        )


def generation():
    model = Model()
    state = ConstraintState(vocabulary(), 'root ::= "yes"')
    request = Generation(
        Sequence(model),
        (TokenId(10),),
        Options(
            max_tokens=10, stop_tokens=frozenset({TokenId(256), TokenId(257)}), forced_quantum=0
        ),
        constraint=state,
    )
    return model, request


def advance(model, request, token):
    model.next_token = token
    ready = request.ready(1)
    batch = GenerationBatch.prepare((ready,))
    batch.finish()
    return ready


def test_constraint_is_checked_before_model_commit_or_publication():
    model, request = generation()
    with pytest.raises(ValueError, match="violate"):
        advance(model, request, ord("n"))
    assert request.processed == 0
    assert request.generated == [] and request.constraint.position == 0
    assert request.take(10) == ()
    assert request.finish_reason == FinishReason.FAILED


def test_capacity_retry_and_commit_failure_leave_constraint_unadvanced():
    model, request = generation()
    ready = request.ready(1)
    model.refuse = True
    with pytest.raises(RuntimeError, match="capacity"):
        GenerationBatch.prepare((ready,))
    assert request.ready(1) == ready
    assert request.constraint.position == 0
    model.refuse = False
    model.commit_fails = True
    with pytest.raises(RuntimeError, match="commit failure"):
        advance(model, request, ord("y"))
    assert request.constraint.position == request.processed == 0
    assert request.generated == []


def test_packed_mask_acceptance_replay_checkpoint_and_eos_stay_consistent():
    model, request = generation()
    advance(model, request, ord("y"))
    mask = model.requests[0].allowed_tokens
    assert mask[ord("y") // 8] & (1 << (ord("y") % 8))
    advance(model, request, ord("e"))
    assert request.constraint.position == len(request.generated) == 2
    assert tuple(token.token for token in request.take(10)) == tuple(b"ye")
    checkpoint = request.checkpoint()
    fork = checkpoint.fork()
    checkpoint.close()
    request.evict()
    request.restore(Sequence(model))
    while request.rebuilding:
        ready = advance(model, request, 999)
        assert ready.kind == WorkKind.REPLAY
        assert model.requests[0].allowed_tokens is None
        assert request.constraint.position == 2
    advance(model, request, ord("s"))
    assert request.constraint.accepting and not fork.constraint.accepting
    advance(model, request, 257)
    assert request.constraint.stopped
    assert request.finish_reason == FinishReason.STOP
    assert request.constraint.position == len(request.generated) == 4
    assert tuple(token.token for token in request.take(10)) == (ord("s"),)
    assert fork.constraint.position == 2
    request.close()
    fork.close()
