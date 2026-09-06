from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import KVCache

from benchmarks.subjects.upstream import runtime
from magnitude_engine.artifacts.source import LocalArtifact


@pytest.fixture
def upstream(monkeypatch):
    calls = []

    class Language:
        def __call__(self, *, inputs, cache):
            calls.append(tuple(inputs[0].tolist()))
            offset = cache[0].offset
            values = inputs[:, None, :, None].astype(mx.float32)
            cache[0].update_and_fetch(values, values)
            expected = (inputs + mx.arange(inputs.shape[1])[None, :] + offset + 1) % 32
            logits = -(mx.arange(32)[None, None, :] - expected[..., None]).astype(mx.float32) ** 2
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        runtime, "load_model", lambda *args, **kwargs: SimpleNamespace(language_model=Language())
    )
    monkeypatch.setattr(runtime, "make_prompt_cache", lambda model: [KVCache()])
    return calls


@pytest.mark.parametrize("service_tokens", [1, 4, 64])
def test_upstream_decode_has_exact_input_budget_and_same_causal_tokens(upstream, service_tokens):
    trace = runtime.DecodeTrace(
        artifact=LocalArtifact("unused"), prompt_tokens=7, output_tokens=7,
        service_tokens=service_tokens, prefill_tokens=4, cache_limit_bytes=256 << 20,
    )
    try:
        expected = []
        anchor = 7
        for position in range(6, 13):
            anchor = (anchor + position + 1) % 32
            expected.append(anchor)
        for _ in range(2):
            trace.reset()
            start = len(upstream)
            trace.invoke()
            trace.complete()
            observation = trace.observe()
            assert observation.evidence["output_tokens"] == expected
            assert trace.caches[0].offset == 13
            assert len(upstream) - start == 7
            assert all(len(inputs) == 1 for inputs in upstream[start:])
    finally:
        trace.close()
    assert trace.model is None
    assert not trace.caches


def test_upstream_prefill_excludes_prefix_and_probe_from_query(upstream):
    trace = runtime.PrefillTrace(
        artifact=LocalArtifact("unused"), prefix_tokens=7, input_tokens=5,
        prefill_tokens=4, cache_limit_bytes=256 << 20,
    )
    try:
        digests = []
        for _ in range(2):
            trace.reset()
            assert trace.caches[0].offset == 7
            start = len(upstream)
            trace.invoke()
            trace.complete()
            assert trace.caches[0].offset == 12
            assert upstream[start:] == [(1, 2, 3, 4, 5)]
            observation = trace.observe()
            assert trace.caches[0].offset == 13
            assert observation.evidence["continuation_token"] == 14
            digests.append(observation.output_digest)
        assert digests[0] == digests[1]
    finally:
        trace.close()


def test_failed_upstream_load_restores_process_allocator_policy(monkeypatch):
    limits = []
    monkeypatch.setattr(runtime.mx, "set_cache_limit", lambda value: limits.append(value) or 1234)

    def fail(*args, **kwargs):
        raise ValueError("unsupported model")

    monkeypatch.setattr(runtime, "load_model", fail)
    with pytest.raises(ValueError, match="unsupported model"):
        runtime.DecodeTrace(
            artifact=LocalArtifact("unused"), prompt_tokens=7, output_tokens=7,
            service_tokens=4, prefill_tokens=4, cache_limit_bytes=5678,
        )
    assert limits == [5678, 1234]


@pytest.mark.parametrize('termination', ['length', 'stop'])
def test_upstream_waves_use_declared_cold_work_and_reject_early_termination(
    monkeypatch, termination,
):
    from benchmarks.subjects.upstream import batch

    instances = []

    class Generator:
        def __init__(self, model, processor, **kwargs):
            assert kwargs['compute_logprobs'] is False
            assert kwargs['greedy_sampling'] is True
            assert 'apc_manager' not in kwargs and 'draft_model' not in kwargs
            self.limit = kwargs['max_tokens']
            self.step = 0
            self.closed = False
            instances.append(self)

        def insert(self, prompts, *, prompt_kwargs):
            assert prompts == [[1, 2, 3, 1, 2, 3, 1]] * 2
            assert all('inputs_embeds' in kwargs for kwargs in prompt_kwargs)
            return [0, 1]

        def next(self):
            self.step += 1
            return [], [SimpleNamespace(
                uid=uid, token=self.step + uid,
                finish_reason=termination if self.step == self.limit else None,
            ) for uid in (0, 1)]

        def close(self):
            self.closed = True

    monkeypatch.setattr(batch, 'BatchGenerator', Generator)
    monkeypatch.setattr(
        batch, 'load_model', lambda *a, **k: SimpleNamespace(
            language_model=object(), get_input_embeddings=lambda tokens, _: SimpleNamespace(
                to_dict=lambda: {'inputs_embeds': tokens[..., None]},
            ),
        ),
    )
    monkeypatch.setattr(batch, 'load_processor', lambda *a, **k: SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda _: [1, 2, 3]),
    ))
    trace = batch.BatchTrace(
        artifact=LocalArtifact('unused'), prompt_text='test', prompt_tokens=7,
        output_tokens=3, rows=2, prefill_tokens=4, cache_limit_bytes=256 << 20,
    )
    try:
        trace.reset()
        if termination == 'stop':
            with pytest.raises(ValueError, match='before its fixed output allowance'):
                trace.invoke()
        else:
            trace.invoke()
            observation = trace.observe()
            assert observation.counters['output_tokens'] == 12
            assert observation.counters['cached_tokens'] == 0
            assert observation.evidence['outputs'] == [[1, 2, 3], [2, 3, 4]] * 2
            assert len(instances) == 2
    finally:
        trace.close()
    assert all(instance.closed for instance in instances)
