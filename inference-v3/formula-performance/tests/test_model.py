import sys

import pytest
from formula_performance.derive import contributions, roofline
from formula_performance.evidence import affected, evaluate
from formula_performance.records import (
    Capacity,
    Capture,
    Expression,
    Formula,
    Hardware,
    Manifest,
    Obligation,
    Observation,
    Publication,
    Quantity,
    Region,
    Transfer,
    Unit,
    identity,
)

TOKEN = Unit(name="token", dimension="count")
FLOP = Unit(name="FLOP", dimension="floating-work")
BYTE = Unit(name="byte", dimension="storage")


def fixture():
    quantity = Quantity(
        name="tokens",
        unit=TOKEN,
        expression=Expression.parameter("rows"),
        meaning="tokens processed",
    )
    manifest = Manifest(
        graph="graph",
        parameters={"rows": 1, "work": 4e9, "traffic": 100e6},
        formulas=(
            Formula(
                component="",
                parent=None,
                definition="decoder",
                version=1,
                semantics="root",
                primary=quantity,
                nodes=("a", "b"),
                label="Model",
            ),
            Formula(
                component="ffn",
                parent="",
                definition="feedforward",
                version=1,
                semantics="ffn",
                primary=quantity,
                nodes=("a",),
                label="Feed-forward",
            ),
            Formula(
                component="other",
                parent="",
                definition="other",
                version=1,
                semantics="other",
                primary=quantity,
                nodes=("b",),
                label="Other",
            ),
        ),
        obligations=(
            Obligation(
                identity="arithmetic",
                origins=("a",),
                amount=Expression.parameter("work"),
                unit=FLOP,
                resource="compute",
                mappings=("compute",),
                rule="fixture necessary work",
            ),
            Obligation(
                identity="traffic",
                origins=("a",),
                amount=Expression.parameter("traffic"),
                unit=BYTE,
                resource="memory",
                mappings=("memory",),
                rule="fixture necessary movement",
            ),
        ),
    )

    def hardware(name, compute, memory):
        return Hardware(
            identity=name,
            label=name,
            capacities=(
                Capacity(
                    parameter="compute",
                    pool="arithmetic",
                    unit=Unit(name="FLOP/s", dimension="floating-work/time"),
                    value=compute,
                    kind="upper-bound",
                    provenance="fixture",
                ),
                Capacity(
                    parameter="memory",
                    pool="memory",
                    unit=Unit(name="byte/s", dimension="storage/time"),
                    value=memory,
                    kind="upper-bound",
                    provenance="fixture",
                ),
            ),
        )

    return manifest, hardware("A", 20e12, 200e9), hardware("B", 40e12, 800e9)


def observation(
    manifest,
    hardware,
    name,
    seconds,
    component="ffn",
    created="2026-01-01",
    implementation="first",
    **kw,
):
    return Observation(
        identity=name,
        manifest=identity(manifest),
        component=component,
        hardware=identity(hardware),
        implementation=implementation,
        created=created,
        coordinates={"rows": 1},
        samples=(seconds,),
        boundary="complete-operation",
        correctness="passed",
        status="complete",
        **kw,
    )


def test_hardware_normalized_relation():
    m, a, b = fixture()
    p = Publication(
        manifests=(m,),
        hardware=(a, b),
        observations=(observation(m, a, "a", 0.001), observation(m, b, "b", 0.00025)),
    )
    report = evaluate([p])
    assert [v["attainment"] for v in report["components"]["ffn"]["points"]] == [0.5, 0.5]
    assert report["components"]["ffn"]["attainment"]["points"] == 2
    assert report["components"][""]["children"] == ["ffn", "other"]
    assert affected(report, ["a"]) == ["", "ffn"]
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from formula_performance.evidence import evaluate; evaluate([]); "
            "assert not ({'ops','torch','tilelang','engine'} & set(sys.modules))",
        ],
        check=True,
    )


def test_empirical_observation_cannot_become_theoretical_capacity():
    m, a, _ = fixture()
    a = a.model_copy(
        update={
            "capacities": tuple(c.model_copy(update={"kind": "achieved"}) for c in a.capacities)
        }
    )
    bound = roofline(m, m.formulas[1], a)
    assert bound["ceiling"] is None
    assert bound["reference_rate"] == 2000


def test_missing_alternative_mapping_cannot_tighten_bound():
    m, a, _ = fixture()
    obligation = m.obligations[0].model_copy(
        update={"mappings": ("compute", "another-legal-realization")}
    )
    m = m.model_copy(update={"obligations": (obligation,)})
    assert roofline(m, m.formulas[1], a)["ceiling"] is None


def test_shared_occurrences_count_once_and_capacity_monotonicity():
    m, a, _ = fixture()
    root = roofline(m, m.formulas[0], a)
    child = roofline(m, m.formulas[1], a)
    assert root["floor_seconds"] == child["floor_seconds"]
    faster = a.model_copy(
        update={
            "capacities": tuple(c.model_copy(update={"value": c.value * 2}) for c in a.capacities)
        }
    )
    assert roofline(m, m.formulas[0], faster)["ceiling"] == root["ceiling"] * 2


def test_fused_and_overlapping_regions_are_not_double_counted():
    m, _, _ = fixture()
    capture = Capture(
        identity="capture",
        manifest=identity(m),
        clock="native",
        coverage="complete",
        regions=(
            Region(identity="joint", owners=("ffn", "other"), start=0, end=2),
            Region(identity="ffn", owners=("ffn",), start=1, end=3),
        ),
    )
    values = contributions(m, capture)
    assert values[""]["inclusive_seconds"] == 3
    assert values[""]["joint_regions"] == ["joint"]
    assert values["ffn"]["inclusive_seconds"] == 2
    assert values["other"]["inclusive_seconds"] == 0


def test_prediction_preserves_history_and_is_validated_by_later_evidence():
    m, a, _ = fixture()
    child = observation(m, a, "child", 0.0005)
    parent = observation(m, a, "parent", 0.01, component="")
    candidate = observation(
        m, a, "new-child", 0.0003, created="2026-01-02", implementation="second"
    )
    actual = observation(
        m, a, "new-parent", 0.0099, component="", created="2026-01-03", implementation="second"
    )
    transfer = Transfer(
        identity="serial",
        child="ffn",
        parent="",
        baseline_child="child",
        baseline_parent="parent",
        contribution=0.0008,
        relationship="conditional-serial",
        assumptions=("unchanged serial boundary costs",),
        evidence=("child", "parent"),
    )
    report = evaluate(
        [
            Publication(
                manifests=(m,),
                hardware=(a,),
                observations=(child, parent, candidate, actual),
                transfers=(transfer,),
            )
        ]
    )
    prediction = report["components"][""]["predictions"][0]
    assert prediction["seconds"] == pytest.approx(0.0098)
    assert prediction["validation"][0]["error_seconds"] == pytest.approx(0.0001)
    assert report["components"][""]["points"][0]["seconds"] == 0.01
    assert next(
        c for c in report["components"][""]["constraints"] if c["kind"] == "serial-accounting"
    )["remaining_seconds"] == pytest.approx(0.0092)


def test_expression_and_record_reject_bad_evidence():
    with pytest.raises(ValueError):
        Expression(op="parameter")
    with pytest.raises(ValueError):
        Region(identity="bad", owners=("x",), start=2, end=1)
    with pytest.raises(ValueError):
        m, _, _ = fixture()
        Manifest(**{**m.model_dump(), "obligations": [m.obligations[0].model_dump()] * 2})


def test_updates_propagate_through_multiple_composition_levels():
    m, a, _ = fixture()
    block = m.formulas[1].model_copy(
        update={"component": "block", "parent": "", "definition": "block", "semantics": "block"}
    )
    ffn = m.formulas[1].model_copy(update={"parent": "block"})
    m = Manifest(**{**m.model_dump(), "formulas": (m.formulas[0], block, ffn, m.formulas[2])})
    base = observation(m, a, "leaf", 0.0005)
    middle = observation(m, a, "middle", 0.004, component="block")
    parent = observation(m, a, "root", 0.01, component="")
    candidate = observation(m, a, "new-leaf", 0.0003, created="2026-01-02", implementation="second")
    transfers = (
        Transfer(
            identity="one",
            child="ffn",
            parent="block",
            baseline_child="leaf",
            baseline_parent="middle",
            contribution=0.0005,
            relationship="conditional-serial",
            assumptions=("serial",),
            evidence=("leaf",),
        ),
        Transfer(
            identity="two",
            child="block",
            parent="",
            baseline_child="middle",
            baseline_parent="root",
            contribution=0.004,
            relationship="conditional-serial",
            assumptions=("serial",),
            evidence=("middle",),
        ),
    )
    p = Publication(
        manifests=(m,),
        hardware=(a,),
        observations=(base, middle, parent, candidate),
        transfers=transfers,
    )
    report = evaluate([p])
    assert report["components"]["block"]["predictions"][0]["seconds"] == pytest.approx(0.0038)
    assert report["components"][""]["predictions"][0]["seconds"] == pytest.approx(0.0098)
    assert report["components"][""]["predictions"][0]["inputs"] == ["new-leaf"]


def test_shared_capture_does_not_manufacture_complete_child_throughput():
    m, a, _ = fixture()
    c = Capture(
        identity="capture",
        manifest=identity(m),
        clock="gpu",
        coverage="complete",
        regions=(
            Region(identity="joint", owners=("ffn", "other"), start=0, end=0.002),
            Region(identity="leaf", owners=("ffn",), start=0.002, end=0.003),
        ),
    )
    root = observation(m, a, "root", 0.004, component="", captures=("capture",))
    p = Publication(manifests=(m,), hardware=(a,), observations=(root,), captures=(c,))
    report = evaluate([p])
    assert not report["components"]["ffn"]["points"]
    assert len(report["components"][""]["points"]) == 2
    assert report["components"]["ffn"]["contributions"][0]["crossing_regions"] == ["joint"]


def test_replay_is_deterministic_and_duplicate_publications_are_idempotent():
    m, a, b = fixture()
    first = Publication(
        manifests=(m,), hardware=(a,), observations=(observation(m, a, "a", 0.001),)
    )
    second = Publication(
        manifests=(m,), hardware=(b,), observations=(observation(m, b, "b", 0.00025),)
    )
    assert evaluate([first, second]) == evaluate([second, first, first])


def test_total_only_measurement_constrains_children_without_inventing_attribution():
    m, a, _ = fixture()
    p = Publication(
        manifests=(m,), hardware=(a,), observations=(observation(m, a, "root", 0.01, component=""),)
    )
    report = evaluate([p])
    child = report["components"]["ffn"]
    assert not child["points"] and child["attainment"] is None
    assert child["constraints"][0]["kind"] == "joint-total"
    assert child["constraints"][0]["children"] == ["ffn", "other"]
    assert child["constraints"][0]["remaining_seconds"] == 0.01


def test_serial_barrier_tightens_only_its_certified_region():
    from formula_performance.records import SerialStages

    m, a, _ = fixture()
    ordinary = roofline(m, m.formulas[1], a)
    serial = SerialStages(
        identity="barrier",
        component="ffn",
        stages=(("arithmetic",), ("traffic",)),
        rule="fixture requires complete arithmetic before transfer",
    )
    m = Manifest(**{**m.model_dump(), "serial_stages": (serial,)})
    bound = roofline(m, m.formulas[1], a)
    assert ordinary["floor_seconds"] == pytest.approx(0.0005)
    assert bound["floor_seconds"] == pytest.approx(0.0007)
    assert roofline(m, m.formulas[0], a)["floor_seconds"] == ordinary["floor_seconds"]


def test_hardware_behavior_keeps_prior_hypothesis_when_later_hardware_disagrees():
    m, a, b = fixture()
    baseline = observation(m, a, "a", 0.001, created="2026-01-01")
    later = observation(m, b, "b", 0.0002, created="2026-01-02")
    report = evaluate(
        [Publication(manifests=(m,), hardware=(a, b), observations=(baseline, later))]
    )
    relation = report["components"]["ffn"]
    first = relation["behavior"][0]
    assert first["efficiency"] == 0.5
    assert first["validation"][0]["predicted_rate"] == 4000
    assert first["validation"][0]["observed_rate"] == 5000
    assert first["validation"][0]["error_rate"] == 1000
    assert relation["behavior"][1]["efficiency"] == 0.625


def test_isolated_capture_never_becomes_an_enclosing_observation():
    m, a, _ = fixture()
    c = Capture(
        identity="isolated",
        manifest=identity(m),
        clock="gpu",
        component="ffn",
        regions=(Region(identity="kernel", owners=("ffn",), start=1, end=1.0001),),
        coverage="complete",
    )
    o = observation(m, a, "isolated", 0.001, captures=(c.identity,))
    report = evaluate(
        [Publication(manifests=(m,), hardware=(a,), captures=(c,), observations=(o,))]
    )
    assert report["components"][""]["points"] == []
    assert report["components"][""]["contributions"] == []
    assert len(report["components"]["ffn"]["points"]) == 2
    assert report["components"]["ffn"]["contributions"][0]["parent_seconds"] is None


def test_transfer_rejects_unresolvable_evidence():
    m, a, _ = fixture()
    t = Transfer(
        identity="bad",
        child="ffn",
        parent="",
        baseline_child="child",
        baseline_parent="parent",
        contribution=0.0001,
        relationship="conditional-serial",
        assumptions=("serial",),
        evidence=("missing",),
    )
    with pytest.raises(ValueError, match="transfer evidence closure"):
        evaluate(
            [
                Publication(
                    manifests=(m,),
                    hardware=(a,),
                    transfers=(t,),
                    observations=(
                        observation(m, a, "child", 0.001),
                        observation(m, a, "parent", 0.01, component=""),
                    ),
                )
            ]
        )


def test_resource_bound_is_sound_over_exhaustive_small_legal_mappings():
    from itertools import product

    m, _, _ = fixture()
    for rates in product((1, 2), repeat=2):
        h = Hardware(
            identity="small",
            label="small",
            capacities=tuple(
                Capacity(
                    parameter=name,
                    pool=name,
                    value=rate,
                    unit=Unit(name="FLOP/s", dimension="floating-work/time"),
                    kind="upper-bound",
                    provenance="exact toy machine",
                )
                for name, rate in zip(("A", "B"), rates, strict=True)
            ),
        )
        for mappings in product((("A",), ("B",), ("A", "B")), repeat=4):
            obligations = tuple(
                Obligation(
                    identity=str(i),
                    origins=("a",),
                    amount=Expression.constant(i + 1),
                    unit=FLOP,
                    resource="compute",
                    mappings=choices,
                    rule="one nonpreemptive operation",
                )
                for i, choices in enumerate(mappings)
            )
            small = m.model_copy(update={"obligations": obligations})
            # Each pool is serial; independent pools may overlap. Enumerate every
            # legal mapping. Ordering within a pool cannot change its total load.
            legal_times = []
            for assignment in product(*mappings):
                loads = {"A": 0, "B": 0}
                for i, name in enumerate(assignment):
                    loads[name] += (i + 1) / rates[("A", "B").index(name)]
                legal_times.append(max(loads.values()))
            assert roofline(small, small.formulas[0], h)["floor_seconds"] <= min(legal_times)


def test_sequential_workloads_compose_into_the_same_model_with_explicit_units():
    from formula_performance.composition import sequential

    m, a, _ = fixture()
    composite = sequential((m, m))
    assert [f.component for f in composite.formulas] == [f.component for f in m.formulas]
    original = roofline(m, m.formulas[0], a)
    combined = roofline(composite, composite.formulas[0], a)
    assert combined["quantity"] == 2 * original["quantity"]
    assert combined["floor_seconds"] == 2 * original["floor_seconds"]
    assert combined["ceiling"] == original["ceiling"]
    assert combined["serial_certificates"]
    assert (
        sequential((m,)).formulas[0].primary.expression.evaluate(sequential((m,)).parameters) == 1
    )
    assert Manifest.model_validate_json(composite.model_dump_json()) == composite
    report = evaluate(
        [
            Publication(
                manifests=(m, composite),
                hardware=(a,),
                observations=(observation(composite, a, "ordinary", 0.002, component=""),),
            )
        ]
    )
    assert report["components"][""]["points"][0]["rate"] == 1000


def test_unmeasured_formulas_still_expose_hardware_parameterized_mathematics():
    m, _, _ = fixture()
    report = evaluate([Publication(manifests=(m,))])
    relation = report["components"]["ffn"]
    assert relation["points"] == []
    math = relation["realizations"][0]["model"]
    assert math["quantity"] == 1
    assert {d["resource"]: d["known_amount"] for d in math["demands"]} == {
        "compute": 4e9,
        "memory": 100e6,
    }
