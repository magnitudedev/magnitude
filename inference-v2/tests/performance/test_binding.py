import pytest

from magnitude_engine.components import ComponentId, component, component_id
from magnitude_engine.engine.runtime import Engine
from performance.assembly import inspect_component, inspect_engine
from performance.facts import AttentionGeometry, Configuration
from performance.records import Assembly, Node


def test_graph_identity_preserves_aliases_but_not_occurrence_paths():
    leaf = Node(ComponentId("MODEL:ATTENTION:MAG:PAGED"), "leaf")
    shared = Assembly(
        "root",
        {
            "root": Node(
                ComponentId("MODEL:QWEN35:MAG:LAYERWISE"),
                "parent",
                children={"a": "one", "b": "one"},
            ),
            "one": leaf,
        },
        "shared",
    )
    separate = Assembly(
        "root",
        {
            "root": Node(
                ComponentId("MODEL:QWEN35:MAG:LAYERWISE"),
                "parent",
                children={"a": "one", "b": "two"},
            ),
            "one": leaf,
            "two": leaf,
        },
        "separate",
    )
    renamed = Assembly(
        "top",
        {
            "top": Node(
                ComponentId("MODEL:QWEN35:MAG:LAYERWISE"),
                "parent",
                children={"a": "bottom", "b": "bottom"},
            ),
            "bottom": leaf,
        },
        "renamed",
    )
    assert shared.identity == separate.identity
    assert shared.revision != separate.revision
    assert shared.identity == renamed.identity
    assert shared.component_keys()["one"] == separate.component_keys()["two"]


@pytest.fixture
def engine(monkeypatch):
    from magnitude_engine.engine.prefixes.radix import Radix
    from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed
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
    from magnitude_engine.engine.binding import EngineResidency

    residency = EngineResidency.__new__(EngineResidency)
    values = dict(
        output_capacity=8,
        engine=engine,
        budget=budget,
        properties={"target_path": "/test/tiny-qwen", "speculative_backend": None},
    )
    residency.__dict__.update(values)
    yield residency
    engine.close()
    arena.close()


def test_actual_qwen_engine_relationships(engine):
    bound = inspect_engine(engine)
    graph = bound.graph
    assert graph.nodes["target"].dependencies["state"] == "target.state"
    assert graph.nodes["generation"].children["target"] == "target"
    assert len([p for p in graph.nodes if p.endswith("mixer")]) == 4
    assert graph.nodes["target.layers.1.mixer.attention"].parameters.key_width == 16
    assert graph.nodes["target.state"].children["kv"] == "target.state.kv"
    assert graph.nodes["target.state.kv"].parameters.layers[0].heads == 2
    assert graph.identity == inspect_engine(engine).graph.identity
    assert graph.revision == inspect_engine(engine).graph.revision
    standalone = inspect_component(
        engine.engine.generation.model.program, artifacts=graph.artifacts
    ).graph
    assert (
        graph.component_keys()["target.embedding"]
        == standalone.component_keys()["component.embedding"]
    )


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
        assert graph.nodes["component.layers.1.mixer.attention"].parameters.key_width == 64
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
    from magnitude_engine.models.ownership import OwnedProgram
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
    owner = OwnedProgram(target.program, (), vocabulary=(
        "test-vocabulary", 64, target.program.embedding, target.program.output,
    ))
    vocabulary = owner.borrow_vocabulary()
    p = MTPParameters(args)
    program = MTPProgram(
        vocabulary,
        p.pre_fc_norm_embedding,
        p.pre_fc_norm_hidden,
        p.fc,
        tuple(AttentionStep(layer) for layer in p.layers),
        p.norm,
        vocabulary.project,
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
        project=vocabulary.project,
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

    vocabulary.close()
    owner.close()


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
        dtype="float32",
        geometry=AttentionGeometry(
            query_heads=2,
            kv_heads=1,
            key_width=32,
            value_width=32,
            element_bytes=4,
            window=3,
        ),
        warmup=0,
        repetitions=1,
        output=tmp_path,
        profile=Profile({}, {}),
    )
    assert result.record["status"] == "complete"
    assert result.record["workload"]["input_digest"]


def test_blueprint_selection_is_the_captured_execution_component():
    from magnitude_engine.composition import build, dumps, loads
    from magnitude_engine.models.attention.blueprint import Gathered, Paged

    geometry = AttentionGeometry(
        query_heads=8, kv_heads=2, key_width=128, value_width=128, element_bytes=2
    )
    captures = []
    for selected in (Gathered(), Paged(heads_per_group=2)):
        with build(loads(dumps(selected))) as live:
            bound = inspect_component(live, context=geometry)
            assert bound.at("component").instance is live
            assert bound.graph.nodes["component"].binding == component_id(live)
            captures.append(bound.graph)
    assert captures[0].nodes["component"].implementation.endswith(":GATHERED")
    assert captures[1].nodes["component"].implementation.endswith(":PAGED")
    assert captures[1].nodes["component"].children == {"fallback": "component.fallback"}


def test_engine_captures_budget_policy_and_suffix_method(engine):
    from magnitude_engine.engine.memory.policy import Budgeted, EvictPrefixesBeforeRejecting
    from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod

    engine.budget = Budgeted(limit_bytes=1 << 30, pressure=EvictPrefixesBeforeRejecting())
    engine.engine.generation.method = SuffixMethod(minimum=2, maximum=8)
    graph = inspect_engine(engine).graph
    assert graph.nodes["memory"].implementation == "MEMORY:ACCOUNTING:MAG:BUDGETED"
    assert graph.nodes["generation"].implementation == "GENERATION:SPECULATION:MAG:SUFFIX"
    assert graph.nodes["generation"].parameters.settings == {"minimum": 2, "maximum": 8}
    assert "acceptance" in graph.nodes["generation"].children
    assert "draft" not in graph.nodes["generation"].children


def test_shared_kernel_uses_preserve_distinct_geometries(monkeypatch):
    from magnitude_engine.models.attention.gathered import GatheredAttention
    from performance.bindings import SCHEMAS, Fields, Use, schema

    @component("ENGINE:INFERENCE:MAG:TEST")
    class AssemblyFixture:
        def __init__(self):
            self.attention = GatheredAttention()

    monkeypatch.setitem(SCHEMAS, AssemblyFixture, None)
    del SCHEMAS[AssemblyFixture]

    @schema(AssemblyFixture)
    def fields(a: AssemblyFixture, _: None) -> Fields[Configuration]:
        return Fields(
            Configuration(),
            children={
                str(i): Use(
                    a.attention,
                    AttentionGeometry(
                        query_heads=8,
                        kv_heads=2,
                        key_width=16 * (i + 1),
                        value_width=16,
                        element_bytes=2,
                    ),
                )
                for i in range(24)
            },
            sources=(GatheredAttention,),
        )

    bound = inspect_component(AssemblyFixture())
    assert len(bound.graph.nodes) == 25
    assert [bound.graph.nodes[f"component.{i}"].parameters.key_width for i in range(24)] == [
        16 * (i + 1) for i in range(24)
    ]
    assert len({id(bound.objects[f"component.{i}"]) for i in range(24)}) == 1


def test_library_capture_reads_the_executed_model_binding():
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn

    from magnitude_engine.models.architectures.mlx_vlm.program import LibraryForward, LibraryProgram

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def __call__(self, tokens, *, cache, position_ids):
            return SimpleNamespace(logits=self.embedding(tokens) + position_ids[..., None])

    model = LanguageModel()
    program = LibraryProgram(LibraryForward(model))
    tokens = mx.array([[1, 2], [3, 4]])
    positions = mx.array([[5, 6], [9, 10]])
    output = program.call(tokens, [SimpleNamespace(offset=mx.array([5, 9]))])
    assert mx.allclose(output, model.embedding(tokens) + positions[..., None]).item()
    bound = inspect_component(program, artifacts={"target": {"fixture": "library"}})
    assert bound.at("component").instance is program
    assert (
        sum(t.bytes for t in bound.graph.nodes["component"].parameters.arrays.values())
        == model.embedding.weight.nbytes
    )


def test_compiled_qwen_has_a_real_identified_child_with_shared_blocks():
    from magnitude_engine.models.architectures.qwen35.decode import ResidentDecode
    from magnitude_engine.models.architectures.qwen35.program import Qwen35Program
    from magnitude_engine.models.attention.metal import MetalPagedAttention
    from tests.models.architectures.qwen35.test_hybrid_model import setup

    _, runtime, arena, _ = setup(attention=MetalPagedAttention(), head_width=32)
    try:
        bound = inspect_component(runtime.program, artifacts={"target": {"fixture": "compiled"}})
        root = bound.graph.nodes["component"]
        decoded = bound.graph.nodes[root.children["decode"]]
        assert root.binding == component_id(Qwen35Program)
        assert decoded.binding == component_id(ResidentDecode)
        assert bound.at(root.children["decode"]).instance is runtime.program.decode
        assert decoded.children["layers.0.mixer"] == root.children["layers.0.mixer"]
        assert decoded.children["embedding"] == root.children["embedding"]
    finally:
        arena.close()


def test_generation_binds_actual_rendered_fixture_lengths(engine, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from performance.benchmarks import generation
    from performance.records import Profile

    monkeypatch.setattr(
        generation,
        "tokens",
        lambda *args, **kwargs: SimpleNamespace(
            prompt=(1, 2, 3, 4, 5, 6), provenance={"fixture": "tools.bfcl"}
        ),
    )
    result = generation.benchmark(
        engine,
        context_tokens=4,
        output_tokens=2,
        validation="workload",
        profile=Profile({}, {}),
        output=tmp_path,
        warmup=0,
        repetitions=1,
    )
    assert result.record["status"] == "complete"
    workload = result.record["workload"]
    assert workload["requested_context_tokens"] == 4
    assert workload["context_tokens"] == 6
    assert workload["histories"] == [6]
    assert result.record["inputs"]["provenance"] == [{"fixture": "tools.bfcl"}]


def test_explicit_external_control_captures_its_own_source_and_constants(tmp_path):
    import importlib.util
    import sys

    from performance.assembly import source_key

    path = tmp_path / 'external_control.py'
    path.write_text('FACTOR = 2\ndef execute(x):\n    return x * FACTOR\n')
    spec = importlib.util.spec_from_file_location('external_control', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        original, files = source_key((module.execute,))
        assert files == {'external_control': path.read_text()}
        module.FACTOR = 3
        changed, _ = source_key((module.execute,))
        assert original != changed
    finally:
        del sys.modules[spec.name]
