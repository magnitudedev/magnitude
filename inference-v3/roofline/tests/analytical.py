"""Small numerical composition used by transport, query and browser tests."""

from formula_performance.records import (
    Capacity,
    Expression,
    Formula,
    Hardware,
    Manifest,
    Obligation,
    Observation,
    Publication,
    Quantity,
    Unit,
    identity,
)


def graph(blocks=1):
    quantity = Quantity(
        name="tokens",
        unit=Unit(name="token", dimension="count"),
        expression=Expression.parameter("rows"),
        meaning="processed tokens",
    )
    nodes = tuple(str(i) for i in range(blocks))
    return Manifest(
        graph="production",
        parameters={"rows": 1, "bytes": 100e6},
        formulas=(
            Formula(
                component="",
                parent=None,
                definition="model",
                version=1,
                semantics="model",
                primary=quantity,
                nodes=nodes,
                label="Model",
            ),
            *(
                Formula(
                    component=f"block[{i}]",
                    parent="",
                    definition="block",
                    version=1,
                    semantics=f"block{i}",
                    primary=quantity,
                    nodes=(str(i),),
                    label=f"Block {i}",
                )
                for i in range(blocks)
            ),
        ),
        obligations=tuple(
            Obligation(
                identity=f"work{i}",
                origins=(str(i),),
                amount=Expression.parameter("bytes"),
                unit=Unit(name="byte", dimension="storage"),
                resource="memory",
                mappings=("bandwidth",),
                rule="test demand",
            )
            for i in range(blocks)
        ),
    )


def publication(name="A", bandwidth=200e9, seconds=0.001, phase="decode"):
    m = graph()
    h = Hardware(
        identity=name,
        label=name,
        capacities=(
            Capacity(
                parameter="bandwidth",
                pool="memory",
                unit=Unit(name="byte/s", dimension="storage/time"),
                value=bandwidth,
                kind="upper-bound",
                provenance="synthetic test hardware",
            ),
        ),
    )
    o = Observation(
        identity=name + phase,
        manifest=identity(m),
        hardware=identity(h),
        component="block[0]",
        implementation="source",
        created="2026-01-01",
        coordinates={"phase": phase},
        samples=(seconds,),
        boundary="complete-operation",
        correctness="passed",
        status="complete",
        evidence=(name + phase,),
    )
    return Publication(manifests=(m,), hardware=(h,), observations=(o,))
