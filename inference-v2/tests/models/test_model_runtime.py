import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.qwen3 import Model, ModelArgs

from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ForwardRequest, ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


def hybrid():
    budget = MemoryBudget(1 << 20)

    def call(tokens, caches):
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        caches[0].update_and_fetch(values, values)
        initial = caches[1][0]
        recurrence = mx.cumsum(tokens.astype(mx.float32), axis=1)
        if initial is not None:
            recurrence += initial
        caches[1][0] = recurrence[:, -1:]
        return -((recurrence[..., None] - mx.arange(16)) ** 2)

    store = LibraryStateStore(lambda: [KVCache(), ArraysCache(1)], budget, lambda n: 2048 + n * 8)
    return ModelRuntime(LibraryProgram(call), store, ExecutionOwner()), budget


def test_hybrid_partial_commit_replays_only_accepted_inputs():
    runtime, budget = hybrid()
    row = runtime.create()
    runtime.prefill(row, (1, 2))
    checkpoint = row.checkpoint()
    advance = runtime.forward(row, (3, 4, 5))
    advance.submit()
    advance.accept(1)
    assert row.state.position == 3
    assert row.state.caches[0].offset == 3
    assert row.state.caches[1][0].item() == 6
    next_step = runtime.forward(row, (2,))
    assert mx.argmax(next_step.output.logits, axis=-1).item() == 8
    next_step.accept(1)
    restored = runtime.create(checkpoint)
    assert restored.state.caches[1][0].item() == 3
    runtime.prefill(restored, (5,))
    assert row.state.caches[1][0].item() == 8
    checkpoint.close()
    restored.close()
    row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_zero_commit_on_initial_recurrent_state_and_pending_guards():
    runtime, budget = hybrid()
    row = runtime.create()
    advance = runtime.forward(row, (2,))
    with pytest.raises(RuntimeError, match="reconciled"):
        row.checkpoint()
    with pytest.raises(ValueError, match="idle"):
        runtime.forward(row, (3,))
    with pytest.raises(ValueError, match="outside"):
        advance.accept(2)
    advance.accept(0)
    assert row.state.position == 0
    assert row.state.caches[0].offset == 0
    assert row.state.caches[1][0] is None
    with pytest.raises(RuntimeError, match="reconciled"):
        advance.accept(0)
    row.close()
    assert budget.snapshot().reserved == 0


def test_rotating_cache_rollback_restores_overwritten_history():
    budget = MemoryBudget(1 << 20)
    store = LibraryStateStore(lambda: [RotatingKVCache(max_size=4)], budget, lambda n: 4096)

    def call(tokens, caches):
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        keys, _ = caches[0].update_and_fetch(values, values)
        return keys.reshape(1, 1, -1)

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    for token in range(6):
        runtime.prefill(row, (token,))
    original = mx.array(row.state.caches[0].keys)
    step = runtime.forward(row, (9, 10, 11))
    step.accept(0)
    assert row.state.caches[0].offset == 6
    assert mx.array_equal(original, row.state.caches[0].keys).item()
    row.close()
    assert budget.snapshot().reserved == 0


def test_state_reservation_failure_precedes_forward():
    runtime, budget = hybrid()
    row = runtime.create()
    budget.limit = 1
    with pytest.raises(MemoryError):
        runtime.forward(row, (1,))
    assert row.state.position == 0
    assert not row.state.active and not row.failed
    assert budget.snapshot().reserved == 0
    budget.limit = 1 << 20
    runtime.prefill(row, (1,))
    row.close()
    assert budget.snapshot().reserved == 0


def test_prepaid_native_capacity_still_accounts_for_actual_replacement_peaks():
    runtime, budget = hybrid()
    row = runtime.create()
    runtime.reserve(row, 10)
    reserved = runtime.states.capacity(10)
    assert budget.snapshot().reserved == reserved
    assert row.state.allocated_bytes == 0
    assert row.state.position == 0 and row.state.caches[0].keys is None
    runtime.prefill(row, (1, 2))
    allocated = runtime.states.capacity(2)
    assert row.state.allocated_bytes == allocated
    assert budget.snapshot().reserved == reserved
    # The new cache fits the prepaid allowance, but its old physical allocation
    # remains live during replacement and must be reserved separately.
    budget.limit = reserved + allocated - 1
    with pytest.raises(MemoryError, match="library-cache-growth"):
        runtime.prefill(row, (3,))
    assert row.state.position == 2 and row.state.active is None
    assert budget.snapshot().reserved == reserved
    budget.limit += 1
    runtime.prefill(row, (3,))
    assert row.state.position == 3
    row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_forward_failure_disposes_sequence_and_releases_completed_resources():
    budget = MemoryBudget(1 << 20)
    store = LibraryStateStore(lambda: [KVCache()], budget, lambda n: 4096)

    def call(tokens, caches):
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        caches[0].update_and_fetch(values, values)
        raise ValueError("injected forward failure")

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    with pytest.raises(ValueError, match="injected"):
        runtime.forward(row, (1,))
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.forward(row, (2,))
    row.close()
    assert budget.snapshot().reserved == 0


def test_real_qwen_attention_program_chunk_checkpoint_and_verify_agree():
    mx.random.seed(41)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            rope_theta=10000,
            head_dim=16,
            tie_word_embeddings=True,
        )
    )
    budget = MemoryBudget(1 << 20)
    store = LibraryStateStore(lambda: [KVCache(), KVCache()], budget, lambda n: 65536)
    runtime = ModelRuntime(
        LibraryProgram(lambda ids, cache: model(ids, cache=cache)), store, ExecutionOwner()
    )
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    checkpoint = row.checkpoint()
    block = runtime.forward(row, (4, 5, 6))
    block.accept(1)
    resumed = runtime.forward(row, (7,))
    resumed.complete()
    expected = model(mx.array([[1, 2, 3, 4, 7]]))[:, -1:]
    assert mx.allclose(resumed.output.logits, expected, atol=1e-5).item()
    resumed.accept(1)
    branch = runtime.create(checkpoint)
    branch_step = runtime.forward(branch, (4, 7))
    branch_step.complete()
    assert mx.allclose(branch_step.output.logits[:, -1:], expected, atol=1e-5).item()
    branch.close()
    checkpoint.close()
    row.close()
    assert budget.snapshot().reserved == 0


def test_missing_feature_rejected_before_state_mutation():
    runtime, budget = hybrid()
    row = runtime.create()
    with pytest.raises(ValueError, match="features"):
        runtime.forward(row, (1,), ForwardRequest(features=frozenset({"hidden:3"})))
    assert not row.state.active and budget.snapshot().reserved == 0
    row.close()


def test_native_rollback_image_is_lazy_until_rejection(monkeypatch):
    runtime, budget = hybrid()
    row = runtime.create()
    runtime.prefill(row, (1, 2))
    original_eval = mx.eval
    evaluations = []

    def record_eval(*arrays):
        evaluations.append(arrays)
        return original_eval(*arrays)

    monkeypatch.setattr(mx, "eval", record_eval)
    advance = runtime.forward(row, (3, 4))
    assert not evaluations, "staging rollback must not submit an extra device evaluation"
    advance.accept(2)
    assert row.state.caches[1][0].item() == 10
    evaluations.clear()
    rejected = runtime.forward(row, (5, 6))
    assert not evaluations
    rejected.accept(0)
    assert row.state.position == 4
    assert row.state.caches[1][0].item() == 10
    row.close()
    assert budget.snapshot().reserved == 0


def test_lazy_snapshot_preserves_in_place_recurrent_array_mutation():
    budget = MemoryBudget(1 << 20)
    store = LibraryStateStore(lambda: [ArraysCache(1)], budget, lambda n: 64)

    def call(tokens, caches):
        if caches[0][0] is None:
            caches[0][0] = mx.zeros((1,), dtype=mx.float32)
        for index in range(tokens.shape[1]):
            caches[0][0][:] += tokens[0, index]
        return caches[0][0].reshape(1, 1, 1)

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    runtime.prefill(row, (1, 2))
    checkpoint = row.checkpoint()
    for tokens, accepted, expected in (((3, 4), 1, 6), ((9, 10), 0, 6), ((2, 3), 2, 11)):
        advance = runtime.forward(row, tokens)
        advance.accept(accepted)
        assert row.state.caches[0][0].item() == expected
    restored = runtime.create(checkpoint)
    assert restored.state.caches[0][0].item() == 3
    restored.close()
    checkpoint.close()
    row.close()
    assert budget.snapshot().reserved == 0


def test_causal_native_advance_omits_rollback_and_enforces_its_commitment():
    runtime, budget = hybrid()
    row = runtime.create()
    runtime.prefill(row, (1, 2))
    before = budget.snapshot().reserved
    advance = runtime.forward(row, (3, 4), ForwardRequest(committed_inputs=2))
    assert advance.transaction.snapshot is None
    with pytest.raises(ValueError, match="commitment"):
        advance.accept(1)
    assert row.pending is advance and not row.failed
    advance.accept(2)
    assert row.state.caches[1][0].item() == 10
    partial = runtime.forward(row, (5, 6), ForwardRequest(committed_inputs=1))
    assert partial.transaction.snapshot is not None
    partial.accept(1)
    assert row.state.caches[1][0].item() == 15
    with pytest.raises(ValueError, match="committed prefix"):
        runtime.forward(row, (7,), ForwardRequest(committed_inputs=2))
    assert row.pending is None
    row.close()
    assert budget.snapshot().reserved == 0 and before > 0
