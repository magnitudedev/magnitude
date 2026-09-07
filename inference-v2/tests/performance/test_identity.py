from dataclasses import replace

import pytest

from magnitude_engine.artifacts.blueprint import Local
from magnitude_engine.blueprints import dumps, loads
from magnitude_engine.components import component, component_id, component_of
from magnitude_engine.composition import digest as blueprint_digest
from magnitude_engine.models.architectures.qwen35.definition import DEFINITION
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.resources.budget import MemoryBudget
from performance.assembly import source_key
from performance.assessment import rebuild
from performance.facts import AttentionGeometry, Configuration
from performance.records import CompositionOrigin
from performance.theory.resources import Demands, DependentPhases, Extent, dependent_time_bound
from performance.theory.sensitivity import predict
from tests.performance.test_system import PROFILE, binding, measured


def test_default_identity_and_factory_are_one_definition():
    selected = DEFINITION.default(Local(path="/fixture"))
    assert selected.definition == DEFINITION.identity
    assert blueprint_digest(loads(dumps(selected))) == blueprint_digest(selected)
    candidate = replace(selected.executor, state=replace(selected.executor.state, page_size=32))
    with pytest.raises(ValueError, match="production default"):
        replace(selected, executor=candidate)


def test_default_history_preserves_siblings_and_rejects_candidate_promotion(tmp_path):
    original = binding()
    original.graph = replace(
        original.graph, origin=CompositionOrigin("QWEN35", "engine", selection="default")
    )
    old = measured(original, "attention", tmp_path)
    changed = binding(parent="new-parent")
    changed.graph = replace(changed.graph, origin=original.graph.origin)
    current = measured(changed, "root", tmp_path)
    candidate = binding(child="experimental")
    candidate.graph = replace(
        candidate.graph, origin=replace(original.graph.origin, selection="candidate")
    )
    experiment = measured(candidate, "root", tmp_path)
    state = rebuild([old.record, current.record, experiment.record])
    composition = state["compositions"][original.graph.identity]
    assert composition["current_revision"] == changed.graph.revision
    key = composition["current_assessments"]["attention"]["EXEC"]
    assert state["components"][key]["dimensions"]["EXEC"]["evidence"] == [old.record["id"]]
    promoted = replace(candidate.graph, origin=original.graph.origin)
    candidate.graph = promoted
    promotion = measured(candidate, "root", tmp_path)
    state = rebuild([old.record, current.record, experiment.record, promotion.record])
    composition = state["compositions"][promoted.identity]
    key = composition["current_assessments"]["attention"]["EXEC"]
    assert state["components"][key]["dimensions"]["EXEC"]["observed"] is None
    historic = composition["historical_assessments"]["attention"]["EXEC"]
    assert state["components"][historic]["dimensions"]["EXEC"]["evidence"] == [old.record["id"]]


def test_bound_direction_and_phase_certificate():
    phases = DependentPhases(
        (Demands((Extent("w", 0, 100),)),) * 3, "test contract: mandatory post-barrier reads"
    )
    profile = replace(PROFILE, capacities={"dram_bytes_per_second": 1000, "fast_storage_bytes": 20})
    bound = dependent_time_bound(phases, profile)
    assert bound.value == pytest.approx(0.24)
    assert bound.direction == "lower"  # Time floor implies throughput ceiling.
    optimistic = dependent_time_bound(
        phases, replace(profile, capacities={**profile.capacities, "fast_storage_bytes": 200})
    )
    assert optimistic.value == 0
    with pytest.raises(ValueError):
        DependentPhases(phases.demands, "")


def test_finite_sensitivity_handles_multiplicity_and_parallel_crossover():
    args = dict(children={"a": 4.0, "b": 3.0}, child="a", seconds_saved=2.0)
    assert (
        predict(
            execution="serial", parent_seconds=11, invocations={"a": 2}, **args
        ).predicted_seconds_saved
        == 4
    )
    assert predict(execution="parallel", parent_seconds=4, **args).predicted_seconds_saved == 1
    assert predict(execution="serial", parent_seconds=4, **args).issue
    assert predict(execution="joint", parent_seconds=4, **args).predicted_seconds_saved is None


def test_source_hash_tracks_cached_kernel_and_ignores_unreferenced_global(monkeypatch):
    from magnitude_engine.models.attention import metal

    first, _ = source_key((metal.MetalPagedAttention,))
    monkeypatch.setattr(metal, "irrelevant_global", 123, raising=False)
    assert source_key((metal.MetalPagedAttention,))[0] == first
    monkeypatch.setattr(metal, "_partials", replacement_kernel)
    assert source_key((metal.MetalPagedAttention,))[0] != first


def replacement_kernel():
    return "different kernel"


def test_typed_contract_rejects_wrong_geometry():
    from performance.assembly import inspect_component

    with pytest.raises(TypeError, match="requires AttentionGeometry"):
        inspect_component(GatheredAttention(), context=Configuration())


def test_migration_preserves_raw_history_and_is_idempotent(tmp_path):
    import json

    from performance.migration import migrate
    from performance.records import digest
    from performance.store import Store

    run = measured(binding(), "attention", tmp_path / "source").record
    old = {**run, "schema_version": 1}
    old.pop("checksum")
    old["checksum"] = digest(old)
    path = tmp_path / "destination" / "runs" / old["id"] / "run.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(old)
    path.write_text(original)
    store = Store(tmp_path / "destination")
    first = migrate(store)
    assert not path.exists()
    archived = store.root / "archive" / "schema-1" / old["id"] / "run.json"
    assert archived.read_text() == original
    assert migrate(store) == first
    records = store.runs()
    assert len(records) == 1
    assert records[0]["samples"] == old["samples"]
    assert records[0]["assembly"]["origin"]["selection"] == "historical"


def test_launch_configuration_changes_evidence_without_renaming_component():
    from magnitude_engine.models.attention.metal import MetalPagedAttention
    from performance.assembly import inspect_component

    geometry = AttentionGeometry(
        query_heads=8, kv_heads=2, key_width=128, value_width=128, element_bytes=2
    )
    a = inspect_component(MetalPagedAttention(heads_per_group=1), context=geometry).graph
    b = inspect_component(MetalPagedAttention(heads_per_group=2), context=geometry).graph
    assert a.identity == b.identity
    assert a.revision != b.revision


def test_component_declaration_leaves_execution_unchanged_and_is_explicit():
    class Operation:
        def __call__(self, x):
            return x + 1

    original_call = Operation.__call__
    declared = component("GENERATION:PLAIN:MAG:TEST")(Operation)
    assert declared is Operation
    assert declared.__call__ is original_call
    assert declared()(3) == 4
    assert str(component_id(declared())) == "GENERATION:PLAIN:MAG:TEST"

    class Undeclared(Operation):
        pass

    with pytest.raises(TypeError, match="undeclared"):
        component_of(Undeclared())


def test_binding_schema_must_match_declared_contract():
    from performance.bindings import Fields, schema

    @component("MODEL:ATTENTION:MAG:TEST")
    class WrongSchema:
        pass

    with pytest.raises(TypeError, match="wrong parameter schema"):

        @schema(WrongSchema)
        def fields(a: WrongSchema, _: None) -> Fields[Configuration]:
            return Fields(Configuration())


def test_reporting_tag_does_not_change_executable_fingerprint(monkeypatch):
    import inspect

    from magnitude_engine.models.attention.metal import MetalPagedAttention

    before = source_key((MetalPagedAttention,))[0]
    getsource = inspect.getsource

    def metadata_change(owner):
        source = getsource(owner)
        return (
            source.replace("MODEL:ATTENTION:MAG:PAGED", "MODEL:ATTENTION:MAG:RENAMED")
            if owner is MetalPagedAttention
            else source
        )

    monkeypatch.setattr(inspect, "getsource", metadata_change)
    assert source_key((MetalPagedAttention,))[0] == before


def test_benchmark_uses_actual_component_variant():
    from magnitude_engine.engine.memory.policy import Budgeted, EvictPrefixesBeforeRejecting
    from performance.assembly import bind_operation, inspect_component

    policy = Budgeted(limit_bytes=1024, pressure=EvictPrefixesBeforeRejecting())
    standalone = bind_operation(policy, MemoryBudget)
    captured = inspect_component(policy).at("component")
    assert standalone.node.implementation == "MEMORY:ACCOUNTING:MAG:BUDGETED"
    assert standalone.node == captured.node


@pytest.mark.parametrize(
    "identity",
    ["qwen", "MODEL:QWEN35:MAG", "MODEL:QWEN35:OTHER:STANDARD", "MODEL:QWEN35:MAG:bad-variant"],
)
def test_invalid_component_ids_fail_at_declaration(identity):
    with pytest.raises(ValueError, match="invalid component ID"):
        component(identity)


def test_identity_is_declared_once_and_explicit_sharing_uses_the_class():
    @component("MODEL:ATTENTION:MAG:DECLARATION_TEST")
    class Original:
        pass

    @component(Original)
    class Shared:
        pass

    assert component_id(Shared) is component_id(Original)
    with pytest.raises(ValueError, match="already declared"):

        @component(str(component_id(Original)))
        class Conflicting:
            pass


def test_blueprint_references_the_identity_of_its_actual_constructor():
    from magnitude_engine.composition import build
    from magnitude_engine.models.attention.blueprint import Paged
    from magnitude_engine.models.attention.metal import MetalPagedAttention

    selected = Paged()
    assert selected.implementation() is MetalPagedAttention
    with build(selected) as live:
        assert component_id(live) is component_id(selected.implementation())
