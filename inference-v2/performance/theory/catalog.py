"""Typed formulation definitions; the only ID decoding is at the record boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from magnitude_engine import components as c
from performance.records import Node, Profile, digest
from performance.theory import composition, engine, neural, state
from performance.theory.resources import (
    Bound,
    Demands,
    Extent,
    join,
    tightened_time_bound,
    time_bound,
)
from performance.theory.workloads import (
    AttentionWorkload,
    ControlWorkload,
    NeuralWorkload,
    RecurrentWorkload,
    ServiceWorkload,
    StateWorkload,
    Workload,
)

METRICS = {
    "EXEC": {"unit": "seconds", "meaning": "operation through required completion"},
    "LAT": {"unit": "seconds", "meaning": "boundary latency"},
    "MEM": {"unit": "bytes", "meaning": "unique retained physical backing"},
    "RESTORE": {"unit": "seconds", "meaning": "requested saved state ready for use"},
    "REUSE": {"unit": "tokens", "meaning": "eligible prefix work recovered"},
    "RATE": {"unit": "tokens/second", "meaning": "completed outputs per workload duration"},
    "TTFT": {"unit": "seconds", "meaning": "maximum request time to first publication"},
    "GAP": {"unit": "seconds", "meaning": "maximum adjacent publication gap"},
}


@dataclass(frozen=True)
class Model[P: c.Facts, W: BaseModel]:
    contract: c.Contract[P]
    workload: type[W]
    dimensions: tuple[str, ...]
    demands: Callable[[P, W, dict[str, Demands]], Demands]
    bounds: Callable[[P, W, Profile, Demands, dict[str, dict[str, Bound]]], dict[str, Bound]]

    def inputs(self, node: Node, raw: dict) -> tuple[P, W]:
        import json

        w = TypeAdapter(self.workload).validate_json(json.dumps(raw))
        p = node.parameters
        # Standalone geometry is an explicit binding, not guessed from an operator.
        if p is None and self.contract in (c.ATTENTION, c.RECURRENCE) and "geometry" in raw:
            p = self.contract.read(raw["geometry"])
        if not isinstance(p, self.contract.parameters):
            raise ValueError(f"missing {self.contract.identity} parameters")
        return p, w


MODELS: dict[c.Contract[Any], Model[Any, Any]] = {}


def register[P: c.Facts, W: BaseModel](model: Model[P, W]) -> None:
    if model.contract in MODELS:
        raise ValueError(f"duplicate theory for {model.contract.identity}")
    MODELS[model.contract] = model


def execution(p, w, profile, demand, children):
    return {"EXEC": tightened_time_bound(demand, profile, w.dependent_phases)}


def bookkeeping(p, w, children):
    return Demands(assumptions=("local bookkeeping may fuse into its owner",))


def explicit(p, w: Workload, children):
    if w.required_inputs is None and not children:
        return Demands(missing=("mathematical required_inputs or child regions",))
    return join(Demands(w.required_inputs or (), w.required_outputs), *children.values())


register(
    Model(
        c.ATTENTION,
        AttentionWorkload,
        ("EXEC",),
        lambda p, w, ch: neural.attention(p, w),
        execution,
    )
)
register(
    Model(
        c.RECURRENCE,
        RecurrentWorkload,
        ("EXEC",),
        lambda p, w, ch: neural.recurrence(p, w),
        execution,
    )
)
for contract in c.CONTRACTS:
    if contract.parameters is c.NeuralParameters:
        register(Model(contract, NeuralWorkload, ("EXEC",), composition.model, execution))
register(Model(c.FORWARD, Workload, ("EXEC",), explicit, execution))


def storage_bounds(p, w, profile, demand, children):
    return {
        "MEM": state.retained(p, w, {k: v for k, v in children.items() if "MEM" in v}),
        "RESTORE": state.restore(p, w, profile),
    }


for contract in (c.RECURRENT_STATE, c.NATIVE_STATE, c.HYBRID_STATE):
    register(Model(contract, StateWorkload, ("MEM", "RESTORE"), bookkeeping, storage_bounds))
register(
    Model(
        c.KV_STORE,
        StateWorkload,
        ("MEM",),
        bookkeeping,
        lambda p, w, profile, d, ch: {"MEM": state.retained(p, w)},
    )
)
register(
    Model(c.KV_APPEND, StateWorkload, ("EXEC",), lambda p, w, ch: state.append(p, w), execution)
)


def service_bounds(p, w, profile, demand, children):
    return {
        dimension: engine.service(dimension, w, demand, profile)
        for dimension in ("RATE", "TTFT", "GAP")
    }


register(
    Model(
        c.ENGINE,
        ServiceWorkload,
        ("RATE", "TTFT", "GAP"),
        lambda p, w, ch: engine.service_information(explicit(p, w, ch)),
        service_bounds,
    )
)
register(Model(c.SCHEDULING, ServiceWorkload, ("RATE", "TTFT", "GAP"), bookkeeping, service_bounds))
register(
    Model(
        c.PREFIX,
        ControlWorkload,
        ("REUSE",),
        bookkeeping,
        lambda p, w, profile, d, ch: {"REUSE": engine.reuse(w)},
    )
)
for contract, dimension in (
    (c.MEMORY, "EXEC"),
    (c.BATCHING, "EXEC"),
    (c.KV_BRANCH, "EXEC"),
    (c.ADMISSION, "LAT"),
):
    register(
        Model(
            contract,
            ControlWorkload,
            (dimension,),
            bookkeeping,
            lambda p, w, profile, d, ch, dimension=dimension: {
                dimension: Bound(
                    0, "seconds", assumptions=("bookkeeping may disappear into its owner",)
                )
            },
        )
    )
register(Model(c.PREFILL, Workload, ("EXEC",), bookkeeping, execution))
for contract in (c.GENERATION, c.SPECULATION):
    register(
        Model(
            contract,
            Workload,
            ("EXEC",),
            lambda p, w, ch: join(*(ch[k] for k in ("target", "draft") if k in ch)),
            execution,
        )
    )


def sampling(p, w: ControlWorkload, children):
    if w.vocabulary is None or w.positions is None or w.element_bytes is None:
        return Demands(missing=("vocabulary, positions and element_bytes",))
    return Demands(
        (Extent("sampling:logits", 0, w.vocabulary * w.positions * w.element_bytes),),
        assumptions=("arbitrary-logit sampling; all candidates may matter",),
    )


def acceptance(p, w: ControlWorkload, children):
    if w.width is None or w.rounds is None:
        return Demands(missing=("width and rounds",))
    return Demands(
        (Extent("acceptance:prefix", 0, w.rounds * (12 if w.width else 4)),),
        assumptions=("earliest possible mismatch; int32 tokens",),
    )


def device(p, w: ControlWorkload, children):
    return (
        bookkeeping(p, w, children)
        if w.elements is None
        else Demands(
            (Extent("device:input", 0, w.elements * 4),),
            assumptions=("closed-form affine graph; intermediate passes eliminated",),
        )
    )


register(Model(c.SAMPLING, ControlWorkload, ("EXEC",), sampling, execution))
register(Model(c.ACCEPTANCE, ControlWorkload, ("EXEC",), acceptance, execution))
register(Model(c.DEVICE, ControlWorkload, ("EXEC",), device, execution))
register(
    Model(
        c.LOADING,
        StateWorkload,
        ("LAT", "MEM"),
        explicit,
        lambda p, w, profile, d, ch: {"LAT": time_bound(d, profile), "MEM": state.retained(p, w)},
    )
)

if set(MODELS) != set(c.CONTRACTS):
    raise RuntimeError("every production component contract needs a theoretical definition")


def revision() -> str:
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def requirements(node: Node, workload: dict, children: dict[str, Demands]) -> Demands:
    model = MODELS[node.component]
    return model.demands(*model.inputs(node, workload), children)


def evaluate(
    node: Node, workload: dict, profile: Profile, demand: Demands, children
) -> dict[str, Bound]:
    model = MODELS[node.component]
    return model.bounds(*model.inputs(node, workload), profile, demand, children)
