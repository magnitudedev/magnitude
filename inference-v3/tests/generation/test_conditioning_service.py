"""Original image work shares service accounting and cancellation ownership."""

from types import SimpleNamespace

from test_constrained_generation import Model, Sequence

from engine.data import TokenId
from engine.generation.plain import FinishReason, Options
from engine.service.engine import Engine, Idle, PendingInput, Status
from engine.service.policy import Limits, Phase


class Source:
    def __init__(self, model, conditioned=False):
        self.model = model
        self.prompt = (TokenId(10),)
        self.conditioned = conditioned
        self.finished = not conditioned
        self.closed = False
        self.ticket = SimpleNamespace(done=False)
        self.finish_count = self.close_count = 0

    def prepare(self):
        if self.finished:
            return None
        return SimpleNamespace(completion=self.ticket, finish=self.finish, close=self.retire)

    def finish(self):
        assert self.ticket.done and not self.closed
        self.finished = True
        self.finish_count += 1

    def retire(self):
        self.close_count += 1

    def open(self):
        assert self.finished and not self.closed
        return Sequence(self.model)

    def close(self):
        self.closed = True


def engine():
    model = Model()
    model.context = SimpleNamespace(check=lambda: None, check_thread=lambda: None)
    return Engine(model, Limits(max_batch=2, prefill_tokens=32, decode_tokens=1))


def test_conditioning_cancellation_does_not_publish_or_open_decoder_state():
    owner = engine()
    source = Source(owner.model, True)
    identity = owner.admit(source, Options(max_tokens=5))
    submission = owner.step()
    assert isinstance(owner.pending, PendingInput)
    assert submission.phase == Phase.PREFILL and submission.tokens == 0
    assert owner.snapshot(identity).status == Status.COMPLETION
    owner.cancel(identity)
    source.ticket.done = True
    assert isinstance(owner.step(), Idle)
    assert source.finish_count == 0 and source.close_count == 1
    assert owner.snapshot(identity).finish == FinishReason.CANCELLED
    assert owner.requests[identity].generation is None
    owner.remove(identity)
    owner.close()


def test_conditioning_completion_returns_to_shared_scheduler_before_text_prefill():
    owner = engine()
    image = Source(owner.model, True)
    image_id = owner.admit(image, Options(max_tokens=5))
    first = owner.step()
    assert first.requests == (image_id,)
    text_id = owner.admit(Source(owner.model), Options(max_tokens=5))
    assert owner.step() is first
    image.ticket.done = True
    next_work = owner.step()
    assert image.finish_count == image.close_count == 1
    assert set(next_work.requests) == {image_id, text_id}
    assert next_work.phase == Phase.PREFILL and next_work.tokens == 2
    assert len(owner.model.requests) == 2
    assert owner.scheduler.completed_service_ns >= 0
    owner.close()
