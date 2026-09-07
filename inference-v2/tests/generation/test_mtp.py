import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from magnitude_engine.generation.execution import run
from magnitude_engine.generation.methods.contracts import Verification
from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.qwen35.mtp.program import MTPProgram
from magnitude_engine.models.embeddings.resident import ResidentEmbedding
from magnitude_engine.models.execution import ExecutionOwner, MLXCompletion
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput, ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


class CompletionLog(MLXCompletion):
    def __init__(self):
        self.waits = []

    def complete(self, arrays):
        self.waits.append(tuple(a.shape for a in arrays))
        super().complete(arrays)


class TargetProgram:
    features = frozenset({"residual:1"})
    conditioning = frozenset()

    def forward(self, inputs, state, request, scope):
        caches = state.caches if state.batch is None else state.store.batch_caches((state,))
        return self._forward(inputs.tokens, caches, request)

    def forward_batch(self, inputs, states, request, scope):
        return self._forward(mx.concatenate([row.tokens for row in inputs]),
                             states[0].store.batch_caches(states), request)

    def _forward(self, tokens, caches, request):
        value = tokens.astype(mx.float32)[..., None]
        cache_values = value[:, None]
        caches[0].update_and_fetch(cache_values, cache_values)
        logits = -mx.square(mx.arange(128) - (value + 1) % 128) * 3
        return ModelOutput(
            logits if request.logits else None, {"residual:1": value} if request.features else {}
        )


def setup():
    budget = MemoryBudget(4 << 20)
    backend = CompletionLog()
    owner = ExecutionOwner(backend)
    target = ModelRuntime(
        TargetProgram(), LibraryStateStore(lambda: [KVCache()], budget, lambda n, q: 8192), owner
    )
    calls = []

    def layer(hidden, cache):
        calls.append(hidden)
        value = hidden[:, None]
        cache.update_and_fetch(value, value)
        return (hidden + 1) % 128

    def project(hidden):
        return -mx.square(mx.arange(128) - hidden) * 3

    pairs = []

    def combine(values):
        pairs.append(values)
        return values[..., :1]

    program = MTPProgram(
        ResidentEmbedding(mx.arange(128).astype(mx.float32)[:, None]),
        lambda x: x,
        lambda x: x,
        combine,
        (layer,),
        lambda x: x,
        project,
    )
    head = ModelRuntime(
        program, LibraryStateStore(lambda: [KVCache()], budget, lambda n, q: 8192), owner
    )
    method = MTPMethod(
        target=target,
        head=head,
        target_feature="residual:1",
        project=project,
        capacity=5,
        budget=budget,
        identity="test-head",
    )
    return target, method, budget, backend, calls, pairs


def observation(inputs, accepted_inputs, bonus):
    return Verification(
        inputs,
        accepted_inputs,
        bonus,
        {"residual:1": mx.array(inputs, dtype=mx.float32).reshape(1, -1, 1)},
    )


def test_mtp_seeding_chaining_rejection_and_buffered_alignment():
    target, method, budget, _, calls, pairs = setup()
    session = method.create(target=target)
    assert run(session.propose((1, 2, 3), 3)).host() == ()
    assert not calls and session.position == 0
    session.observe(observation((7,), 1, 8))
    assert session.position == 0  # observation never runs the private head
    assert run(session.propose((7, 8), 3)).host() == (9, 10, 11)
    assert len(calls) == 3  # one flush, then only two chained forwards
    assert pairs[0].tolist() == [[[8, 7]]]
    assert pairs[1].tolist() == [[[9, 9]]]
    assert session.position == 3
    session.observe(observation((8, 9, 10, 11), 2, 99))
    assert session.position == session.row.state.position == 2
    assert session.row.state.caches[0].state[0].reshape(-1).tolist() == [8, 9]
    assert run(session.propose((8, 9, 99), 1)).host() == (100,)
    assert pairs[-1].tolist() == [[[99, 9]]]
    with pytest.raises(RuntimeError, match="idle"):
        run(session.propose((8, 9, 99), 1))
    session.observe(observation((99, 100), 2, 101))
    session.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_lazy_head_steps_do_not_complete_or_release_growth_before_publication():
    target, method, budget, backend, _, _ = setup()
    row = method.head.create()
    inputs = ModelInputs(mx.array([[5]], dtype=mx.int32), {"previous_hidden": mx.array([[[4.0]]])})
    first = method.head.forward(row, inputs, ForwardRequest(False, frozenset({"draft_hidden"})))
    first.accept_all_lazily()
    assert not backend.waits
    assert row.state.position == 1 and not row.state.active
    second = method.head.forward(row, inputs, ForwardRequest(False, frozenset({"draft_hidden"})))
    current = row.state.active
    first.complete()
    assert row.state.active is current  # old retirement cannot clear the new transaction
    second.accept_all_lazily()
    row.complete_committed()
    assert row.state.position == 2 and not row.state.active
    method.head.rewind(row, 1)
    assert row.state.position == 1
    row.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_mtp_checkpoint_keeps_unflushed_observation_and_detaches_ownership():
    target, method, budget, _, _, _ = setup()
    session = method.create(target=target)
    session.observe(observation((3,), 1, 4))
    checkpoint = session.checkpoint()
    restored = method.create(checkpoint, target=target)
    checkpoint.close()
    session.close()
    assert run(restored.propose((3, 4), 3)).host() == (5, 6, 7)
    restored.observe(observation((4, 5, 6, 7), 1, 22))
    restored.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("temperature", [0, 0.7])
def test_bound_mtp_generation_matches_plain_and_observes_when_proposals_disabled(temperature):
    target, method, budget, _, calls, _ = setup()
    policy = SamplingPolicy(temperature=temperature, seed=13)
    speculative = GenerationRuntime(target, method).create((1, 2, 3), policy, 19)
    plain = GenerationRuntime(target, PlainMethod()).create((1, 2, 3), policy, 19)
    assert len(calls) == 1 and calls[0].shape[1] == 1  # shifted prompt head prefill
    a, b, accepted = [], [], 0
    rounds = 0
    while not speculative.finished:
        result = speculative.step(1 if rounds < 2 else 5)
        rounds += 1
        a.extend(result.tokens)
        accepted += result.accepted
    while not plain.finished:
        b.extend(plain.step().tokens)
    assert a == b and accepted > 5
    assert calls[1].shape[1] == 2  # committed catch-up precedes the actual anchor
    speculative.close()
    plain.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_conditioning_validation_precedes_state_mutation_and_slices_with_replay():
    target, method, budget, _, _, _ = setup()
    row = method.head.create()
    with pytest.raises(ValueError, match="conditioning"):
        method.head.forward(row, (1,))
    assert budget.snapshot().reserved == 0 and not row.state.active
    inputs = ModelInputs(
        mx.array([[1, 2, 3]], dtype=mx.int32),
        {"previous_hidden": mx.array([[[4.0], [5.0], [6.0]]])},
    )
    assert inputs.prefix(2).conditioning["previous_hidden"].tolist() == [[[4.0], [5.0]]]
    with pytest.raises(ValueError, match="align"):
        ModelInputs(mx.array([[1]], dtype=mx.int32), {"previous_hidden": mx.ones((1, 2, 1))})
    with pytest.raises(ValueError, match="int32"):
        ModelInputs.from_tokens((2**32,))
    row.close()
    target.owner.close()


def test_mtp_rejects_a_different_target_binding():
    target, method, budget, _, _, _ = setup()
    other = ModelRuntime(TargetProgram(), target.states, target.owner)
    with pytest.raises(ValueError, match="another target"):
        method.create(target=other)
    assert budget.snapshot().reserved == 0
    target.owner.close()


def test_mtp_state_transitions_match_reviewed_reference(poc_module):
    source = poc_module("spec.mtp")
    target, method, budget, _, _, _ = setup()

    class ReferenceHead:
        layers = [None]

        def __call__(self, embedded, hidden, cache):
            values = embedded[:, None]
            cache[0].update_and_fetch(values, values)
            return (embedded + 1) % 128

    reference = object.__new__(source.MTPDrafter)
    reference.head = ReferenceHead()
    reference.embed = lambda tokens: tokens.astype(mx.float32)[..., None]
    reference.lm_head = method.project
    reference.last_layer = 0
    reference.max_block = 5
    original = reference.new_state([1, 2, 3])
    ours = method.create(target=target)
    context = [1, 2, 3]

    def observe(inputs, accepted, bonus):
        features = mx.array(inputs, dtype=mx.float32).reshape(1, -1, 1)
        reference.observe(original, list(inputs), accepted, bonus, {0: features})
        ours.observe(Verification(inputs, accepted + 1, bonus, {"residual:1": features}))
        assert ours.position == original.position
        # The reference binds bonus immediately. Ours retains its conditioning
        # without binding a token outside the checkpoint's consumed prefix.
        assert [token for token, _ in ours.buffer] == [token for token, _ in original.buffer[:-1]]
        for (_, actual), (_, expected) in zip(ours.buffer, original.buffer[:-1], strict=True):
            assert mx.array_equal(actual.value, expected).item()
        assert original.buffer[-1][0] == bonus
        assert mx.array_equal(ours.pending.value, original.buffer[-1][1]).item()

    observe((3,), 0, 4)
    context.append(4)
    for width, accepted, bonus in ((4, 0, 20), (3, 2, 30), (5, 5, 40), (1, 0, 50)):
        expected = tuple(reference.propose(original, context, width))
        actual = run(ours.propose(context, width)).host()
        assert actual == expected
        assert ours.position == original.position
        inputs = (context[-1], *actual)
        observe(inputs, accepted, bonus)
        context.extend((*actual[:accepted], bonus))
    ours.close()
    reference.release(original)
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_mtp_proposal_is_not_read_back_before_target_verification(monkeypatch):
    from magnitude_engine.generation.proposals import Proposal

    target, method, budget, _, _, _ = setup()
    ready = False
    forward = target.program.forward
    materialize = Proposal.host

    def target_forward(*args, **kwargs):
        nonlocal ready
        ready = True
        return forward(*args, **kwargs)

    def host(proposal):
        assert ready, "proposal readback preceded target verification"
        return materialize(proposal)

    monkeypatch.setattr(target.program, "forward", target_forward)
    monkeypatch.setattr(Proposal, "host", host)
    sequence = GenerationRuntime(target, method).create(
        (1, 2, 3), SamplingPolicy(temperature=0), 13
    )
    while not sequence.finished:
        ready = False
        sequence.step(5)
    sequence.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("chunks", [(7,), (1, 1, 1, 1, 1, 1, 1), (2, 3, 2)])
def test_prompt_conditioning_is_shifted_across_chunks(chunks):
    target, method, budget, _, _, pairs = setup()
    session = method.create(target=target)
    tokens = tuple(range(10, 17))
    offset = 0
    for count in chunks:
        values = tokens[offset : offset + count]
        run(session.prefill(values, observation(values, count, 99).features))
        offset += count
    assert session.position == 6
    assert session.pending.value.item() == 16
    actual = mx.concatenate(pairs, axis=1)
    assert actual.tolist() == [[[token, token - 1] for token in range(11, 17)]]
    assert run(session.propose((*tokens, 77), 2)).host() == (78, 79)
    assert pairs[-2].tolist() == [[[77, 16]]]
    session.observe(observation((77, 78, 79), 1, 80))
    assert session.position == 7
    session.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("after_decode", [False, True])
@pytest.mark.parametrize("extension", [(), (43, 44, 45)])
def test_checkpoint_reuses_only_committed_history_with_a_different_next_token(after_decode, extension):
    target, method, budget, _, _, pairs = setup()
    runtime = GenerationRuntime(target, method)
    policy = SamplingPolicy(temperature=0)
    original = runtime.create((1, 2, 3), policy, 8)
    if after_decode:
        original.step(4)
    checkpoint = original.checkpoint()
    prompt = (*checkpoint.tokens, *extension, 77)
    original.close()
    restored = runtime.create(prompt, policy, 8, checkpoint=checkpoint, chunk_size=1)
    checkpoint.close()
    pair_start = len(pairs)
    deferred = len(restored.method.buffer)
    result = restored.step(4)
    assert result.tokens == (78, 79, 80, 81)
    consumed = mx.concatenate(pairs[pair_start:], axis=1).tolist()[0]
    assert consumed[deferred:] == [[77, prompt[-2]], [78, 78], [79, 79]]
    assert restored.target_position == len(prompt) + 3
    restored.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0
