"""Opt-in integrated artifact construction and generation on a released target/head pair."""

import os
from pathlib import Path

import mlx.core as mx
import pytest
from transformers import AutoTokenizer

from magnitude_engine.engine.delivery import Tokens
from magnitude_engine.engine.prefixes.radix import Radix
from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed
from magnitude_engine.engine.requests import GenerationRequest
from magnitude_engine.engine.runtime import Engine
from magnitude_engine.engine.scheduler.time_shared import TimeShared
from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.qwen35.attention.binding import Attention
from magnitude_engine.models.architectures.qwen35.feedforward.binding import MoE
from magnitude_engine.models.architectures.qwen35.loading import load_qwen35
from magnitude_engine.models.architectures.qwen35.mtp.loading import load_mtp
from magnitude_engine.models.architectures.qwen35.recurrence.binding import Mixer
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.embeddings.binding import Resident as ResidentEmbeddingFactory
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.binding import Resident as ResidentExpertFactory
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.arena import KVArena
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


@pytest.mark.model
def test_local_mtp_pair_loads_generates_and_releases_all_reservations():
    target_path, head_path = (
        os.environ.get("MAGNITUDE_TEST_MTP_TARGET"),
        os.environ.get("MAGNITUDE_TEST_MTP_HEAD"),
    )
    if not target_path or not head_path:
        pytest.skip("set MAGNITUDE_TEST_MTP_TARGET and MAGNITUDE_TEST_MTP_HEAD to local artifacts")
    budget, reader, owner = MemoryBudget(28 << 30), PositionalReader(workers=4), ExecutionOwner()
    mx.set_cache_limit(256 << 20)
    loaded = load_qwen35(
        Path(target_path),
        budget=budget,
        reader=reader,
        attention=Attention(GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        embedding_factory=ResidentEmbeddingFactory(),
    )
    head = load_mtp(Path(head_path), loaded, budget=budget, reader=reader)
    arena = KVArena(
        loaded.attention,
        page_size=16,
        slab_pages=16,
        max_pages=128,
        budget=budget,
        dtype=loaded.state_dtype,
    )
    target = ModelRuntime(
        loaded.program,
        HybridStateStore(PageStore(arena), loaded.recurrence, budget),
        owner,
    )
    drafter = ModelRuntime(head.program, head.state_store(budget), owner)
    method = MTPMethod(
        target=target,
        head=drafter,
        target_feature=head.target_feature,
        project=head.vocabulary.project,
        capacity=head.capacity,
        budget=budget,
        identity=str(Path(head_path).resolve()),
    )
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    prompt = tuple(
        tokenizer.encode("Write a Python function that adds two numbers.\n\ndef add(a, b):")
    )
    results, accepted, proposed = [], 0, 0
    for strategy in (PlainMethod(), method):
        sequence = GenerationRuntime(target, strategy).create(
            prompt,
            SamplingPolicy(temperature=0),
            24,
        )
        tokens = []
        try:
            while not sequence.finished:
                result = sequence.step(head.capacity + 1)
                tokens.extend(result.tokens)
                accepted += result.accepted
                proposed += result.proposed
            results.append(tokens)
        finally:
            sequence.close()
    assert results[0] == results[1]
    assert accepted > 0 and proposed > 0
    engine = Engine(
        GenerationRuntime(target, method),
        prefixes=Radix(retention=LeastRecentlyUsed(32, None)),
        namespace=(loaded.tokenizer_identity + method.identity).encode(),
        scheduler=TimeShared(decode_tokens=head.capacity + 1),
    )
    policy = SamplingPolicy(temperature=0)
    concurrent = (
        engine.submit(GenerationRequest(prompt, policy, 24), output_capacity=4),
        engine.submit(GenerationRequest(prompt, policy, 8), output_capacity=3),
    )
    scheduled = [[], []]
    for _ in range(100):
        engine.tick()
        for index, handle in enumerate(concurrent):
            while True:
                try:
                    event = handle.delivery.take(0)
                    if isinstance(event, Tokens):
                        scheduled[index].extend(event.values)
                except (TimeoutError, StopIteration):
                    break
        if all(handle.delivery.finish is not None for handle in concurrent):
            break
    assert scheduled == [results[0], results[0][:8]]
    warm = engine.submit(GenerationRequest(prompt, policy, 8))
    warm_tokens = []
    for _ in range(30):
        engine.tick()
        while True:
            try:
                event = warm.delivery.take(0)
                if isinstance(event, Tokens):
                    warm_tokens.extend(event.values)
            except (TimeoutError, StopIteration):
                break
        if warm.delivery.finish is not None:
            break
    assert warm_tokens == results[0][:8]
    assert warm.delivery.finish is not None
    assert warm.delivery.finish.cached_tokens == len(prompt) - 1
    print(
        {
            "concurrent_output_lengths": [len(values) for values in scheduled],
            "warm_cached_tokens": warm.delivery.finish.cached_tokens,
        }
    )
    engine.close()
    usage = budget.snapshot()
    print(
        {
            "target": target_path,
            "head": head_path,
            "tokens": results[0],
            "text": tokenizer.decode(results[0]),
            "accepted": accepted,
            "proposed": proposed,
            "reserved_bytes": usage.reserved,
            "peak_reserved_bytes": usage.peak,
            "mlx_peak_bytes": mx.get_peak_memory(),
        }
    )
    owner.close()
    arena.close()
    head.close()
    loaded.close()
    reader.close()
    assert budget.snapshot().reserved == 0
