from types import SimpleNamespace

import pytest

from performance.assembly import inspect_component, inspect_engine
from performance.records import Assembly, Node


def test_graph_identity_preserves_aliases_but_not_occurrence_paths():
    leaf = Node("MODEL:ATTENTION:MAG:PAGED", "leaf")
    shared = Assembly(
        "root",
        {
            "root": Node("MODEL:QWEN35:MAG:LAYERWISE", "parent", children={"a": "one", "b": "one"}),
            "one": leaf,
        },
        "shared",
    )
    separate = Assembly(
        "root",
        {
            "root": Node("MODEL:QWEN35:MAG:LAYERWISE", "parent", children={"a": "one", "b": "two"}),
            "one": leaf,
            "two": leaf,
        },
        "separate",
    )
    renamed = Assembly(
        "top",
        {
            "top": Node(
                "MODEL:QWEN35:MAG:LAYERWISE", "parent", children={"a": "bottom", "b": "bottom"}
            ),
            "bottom": leaf,
        },
        "renamed",
    )
    assert shared.identity != separate.identity
    assert shared.revision != separate.revision
    assert shared.identity == renamed.identity
    assert shared.component_keys()["one"] == separate.component_keys()["two"]


@pytest.fixture
def engine(monkeypatch):
    from magnitude_engine.engine.prefixes.radix import Radix
    from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed
    from magnitude_engine.engine.runtime import Engine
    from magnitude_engine.engine.scheduler.time_shared import TimeShared
    from magnitude_engine.generation.methods.plain.runtime import PlainMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from tests.models.architectures.qwen35.test_hybrid_model import setup

    model, runtime, arena, budget = setup()
    generation = GenerationRuntime(runtime, PlainMethod())
    engine = Engine(
        generation,
        namespace=b"tiny",
        scheduler=TimeShared(prefill_tokens=8),
        prefixes=Radix(retention=LeastRecentlyUsed(2, None)),
    )
    monkeypatch.setattr(
        "performance.assembly.artifact_identity", lambda path: {"test": "tiny-qwen"}
    )
    residency = SimpleNamespace(
        output_capacity=8,
        engine=engine,
        budget=budget,
        properties={"target_path": "/test/tiny-qwen", "speculative_backend": None},
    )
    yield residency
    engine.close()
    arena.close()


def test_actual_qwen_engine_relationships(engine):
    bound = inspect_engine(engine)
    graph = bound.graph
    assert graph.nodes["target"].dependencies["state"] == "target.state"
    assert graph.nodes["generation"].children["target"] == "target"
    assert len([p for p in graph.nodes if p.endswith("mixer")]) == 4
    assert graph.nodes["target.layers.1.mixer.attention"].parameters["key_width"] == 16
    assert graph.nodes["target.state"].children["kv"] == "target.state.kv"
    assert graph.nodes["target.state.kv"].parameters["layers"][0]["heads"] == 2
    assert graph.identity == inspect_engine(engine).graph.identity
    assert graph.revision == inspect_engine(engine).graph.revision


def test_actual_gemma_sharing():
    from magnitude_engine.models.attention.gathered import GatheredAttention
    from tests.models.architectures.gemma4.test_gemma_model import compose_gemma

    _, runtime, arena, _ = compose_gemma(bits=4, shared=True, attention=GatheredAttention())
    try:
        graph = inspect_component(runtime.program, artifacts={"test": "gemma-tiny"}).graph
        assert (
            graph.nodes["component.layers.2.mixer"].dependencies["kv_producer"]
            == "component.layers.0.mixer.producer"
        )
        assert (
            graph.nodes["component.layers.3.inputs"].dependencies["prepared"] == "component.inputs"
        )
        assert graph.nodes["component.layers.1.mixer.attention"].parameters["key_width"] == 64
    finally:
        arena.close()


def test_model_and_restore_boundaries(engine, tmp_path):
    from performance.benchmarks.model import benchmark
    from performance.benchmarks.state import restore
    from performance.records import Profile

    options = dict(
        context_tokens=8,
        prompt=tuple(range(8)),
        continuation=(9, 10, 11, 12),
        profile=Profile({"hostname": "test"}, {}),
        output=tmp_path,
        warmup=0,
        repetitions=1,
    )
    for mode in ("replay", "prefill", "generate"):
        result = benchmark(engine, measured_tokens=2, mode=mode, **options)
        assert result.record["status"] == "complete"
    for advance, accepted in ((0, 0), (2, 1)):
        result = restore(engine, advance_tokens=advance, accepted_tokens=accepted, **options)
        assert result.record["status"] == "complete"
        assert "RESTORE" in result.record["samples"][0]["observation"]["metrics"]


def test_real_layer_region_controls(engine, tmp_path):
    from performance.benchmarks.regions import benchmark, capture
    from performance.records import Profile

    options = dict(
        context_tokens=8,
        query_tokens=1,
        profile=Profile({"hostname": "test"}, {}),
        output=tmp_path,
        warmup=0,
        repetitions=1,
    )
    with capture(
        engine, context_tokens=8, query_tokens=1, prompt=tuple(range(8)), continuation=(9,)
    ) as inputs:
        for path in (
            "target.embedding",
            "target.layers.0.mixer",
            "target.layers.1.mixer",
            "target.layers.0.feedforward",
            "target.readout",
        ):
            result = benchmark(engine, path, prepared=inputs, **options)
            assert result.record["status"] == "complete"


def test_whole_engine_waves(engine, tmp_path):
    from performance.benchmarks.engine import benchmark
    from performance.records import Profile

    run = benchmark(
        engine,
        context_tokens=8,
        output_tokens=2,
        prompts=[tuple(range(8))],
        warmup=0,
        repetitions=1,
        output=tmp_path,
        profile=Profile({"hostname": "test"}, {}),
    )
    assert run.record["status"] == "complete"
    assert set(run.record["samples"][0]["observation"]["metrics"]) == {"RATE", "TTFT", "GAP"}


def test_attached_head_boundaries(engine, tmp_path):
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import TextModelArgs

    from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from magnitude_engine.models.architectures.qwen35.mtp.loading import (
        AttentionStep,
        MTPParameters,
    )
    from magnitude_engine.models.architectures.qwen35.mtp.program import MTPProgram
    from magnitude_engine.models.runtime import ModelRuntime
    from magnitude_engine.models.state.native import LibraryStateStore
    from performance.benchmarks.mtp import benchmark
    from performance.records import Profile

    target = engine.engine.generation.model
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        full_attention_interval=1,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
    )
    p = MTPParameters(args)
    program = MTPProgram(
        target.program.embedding,
        p.pre_fc_norm_embedding,
        p.pre_fc_norm_hidden,
        p.fc,
        tuple(AttentionStep(layer) for layer in p.layers),
        p.norm,
        target.program.output,
    )
    head = ModelRuntime(
        program,
        LibraryStateStore(lambda: [KVCache()], engine.budget, lambda n, q: 65536),
        target.owner,
    )
    method = MTPMethod(
        target=target,
        head=head,
        target_feature="residual:4",
        project=target.program.output,
        capacity=2,
        budget=engine.budget,
        identity="test-head",
    )
    engine.engine.generation = GenerationRuntime(target, method)
    engine.properties["speculative_backend"] = "mtp"
    for mode, reference in (("execute", False), ("execute", True), ("restore", False)):
        result = benchmark(
            engine,
            context_tokens=8,
            history_tokens=2,
            query_tokens=1,
            mode=mode,
            reference=reference,
            prompt=tuple(range(8)),
            continuation=(9,),
            output=tmp_path,
            profile=Profile({"hostname": "test"}, {}),
            warmup=0,
            repetitions=1,
        )
        assert result.record["status"] == "complete"


def test_batched_and_generation_benchmarks(engine, tmp_path):
    from performance.benchmarks.generation import benchmark
    from performance.benchmarks.model import prefill_batch
    from performance.records import Profile

    record = dict(output=tmp_path, profile=Profile({}, {}), warmup=0, repetitions=1)
    for execution in ("shared", "independent"):
        result = prefill_batch(
            engine,
            context_tokens=4,
            input_tokens=2,
            rows=2,
            prompt=(1, 2, 3, 4),
            continuation=(5, 6),
            execution=execution,
            **record,
        )
        assert result.record["status"] == "complete"
        result = benchmark(
            engine,
            context_tokens=4,
            rows=2,
            output_tokens=2,
            prompts=[(1, 2, 3, 4), (1, 2, 3, 4, 5)],
            execution=execution,
            **record,
        )
        assert result.record["status"] == "complete"


def test_control_operators_and_native_checkpoint(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from magnitude_engine.models.state.native import LibraryStateStore
    from magnitude_engine.resources.budget import MemoryBudget
    from performance.benchmarks import control, state
    from performance.records import Profile

    record = dict(output=tmp_path, profile=Profile({}, {}), warmup=0, repetitions=1)
    assert control.ready_assembly(rows=8, capacity=2, **record).record["status"] == "complete"
    assert control.scheduling(rounds=4, **record).record["status"] == "complete"
    store = LibraryStateStore(lambda: [KVCache()], MemoryBudget(1 << 20), lambda p, c: 4096)
    row = store.create()
    store.reserve(row, 4)
    values = mx.ones((1, 1, 4, 8), mx.bfloat16)
    row.caches[0].update_and_fetch(values, values)
    row.position = 4
    saved = store.checkpoint(row)
    store.release(row)
    try:
        result = state.checkpoint(
            store,
            saved,
            retained_shapes=[
                {"identity": k, "shape": [1, 1, 4, 8], "element_bytes": 2} for k in ("k", "v")
            ],
            **record,
        )
        assert result.record["samples"][0]["observation"]["metrics"]["MEM"] >= 128
    finally:
        saved.close()


def test_upstream_inspection_ignores_execution_caches(monkeypatch):
    import mlx.core as mx

    from performance.assembly import inspect_upstream
    from tests.models.architectures.qwen35.test_hybrid_model import setup

    model, _, arena, _ = setup()
    monkeypatch.setattr("performance.assembly.artifact_identity", lambda _: {"test": "weights"})
    try:
        before = inspect_upstream(model, artifact="/test").graph
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2]]), cache=cache))
        after = inspect_upstream(model, artifact="/test").graph
        assert before.identity == after.identity
        assert before.revision == after.revision
    finally:
        arena.close()


def test_model_generation_honors_explicit_eos(engine, tmp_path):
    from performance.benchmarks.model import benchmark
    from performance.records import Profile

    result = benchmark(
        engine,
        context_tokens=4,
        measured_tokens=3,
        mode="generate",
        prompt=(1, 2, 3, 4),
        eos_tokens=range(64),
        warmup=0,
        repetitions=1,
        output=tmp_path,
        profile=Profile({}, {}),
    )
    sample = result.record["samples"][0]["observation"]
    assert sample["counters"]["input_tokens"] == sample["counters"]["output_tokens"] == 1
    assert result.record["workload"]["stopping"] == "eos-or-limit"


def test_windowed_attention_binds_actual_work_and_input_identity(tmp_path):
    from magnitude_engine.models.attention.gathered import GatheredAttention
    from performance.benchmarks.attention import benchmark
    from performance.records import Profile

    result = benchmark(
        GatheredAttention(),
        context_tokens=5,
        query_tokens=3,
        geometry=dict(
            query_heads=2,
            kv_heads=1,
            key_width=32,
            value_width=32,
            element_bytes=4,
            dtype="float32",
            window=3,
        ),
        warmup=0,
        repetitions=1,
        output=tmp_path,
        profile=Profile({}, {}),
    )
    assert result.record["status"] == "complete"
    assert result.record["workload"]["input_digest"]
