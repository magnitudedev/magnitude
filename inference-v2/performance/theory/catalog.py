"""Typed formulation definitions; the only ID decoding is at the record boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from magnitude_engine.components import component_id
from magnitude_engine.engine.prefixes.radix import Radix
from magnitude_engine.engine.runtime import Engine
from magnitude_engine.engine.scheduler.time_shared import TimeShared
from magnitude_engine.generation.acceptance import accept_prefix
from magnitude_engine.generation.execution import serve
from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling import SequenceSampler
from magnitude_engine.models.architectures.gemma4.program import (
    ExpertBranch,
    GeGLU,
    Gemma4Program,
    GemmaAttention,
    GemmaFeedForward,
    KVProducer,
    PerLayerInputs,
)
from magnitude_engine.models.architectures.gemma4.program import readout as gemma_readout
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.architectures.qwen35.attention.operation import GatedAttention
from magnitude_engine.models.architectures.qwen35.feedforward.operation import RoutedFeedForward
from magnitude_engine.models.architectures.qwen35.mtp.program import MTPProgram
from magnitude_engine.models.architectures.qwen35.program import Qwen35Program
from magnitude_engine.models.architectures.qwen35.program import readout as qwen_readout
from magnitude_engine.models.architectures.qwen35.recurrence.operation import RecurrentMixer
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.embeddings.resident import ResidentEmbedding
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.computation import ResidentExperts
from magnitude_engine.models.loading.parameters import load_resident_parameters
from magnitude_engine.models.recurrence.reference import DeltaReference
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.models.state.pages import PageStore, SequencePages
from magnitude_engine.models.state.recurrent import RecurrentImage
from magnitude_engine.resources.budget import MemoryBudget
from performance.facts import (
    AttentionGeometry,
    Configuration,
    Facts,
    KVStorage,
    NativeStorage,
    NeuralParameters,
    OpaqueParameters,
    RecurrentGeometry,
    RecurrentStorage,
)
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
class Model[P: Facts, W: BaseModel]:
    parameters: type[P]
    workload: type[W]
    dimensions: tuple[str, ...]
    demands: Callable[[P, W, dict[str, Demands]], Demands]
    bounds: Callable[[P, W, Profile, Demands, dict[str, dict[str, Bound]]], dict[str, Bound]]

    def inputs(self, node: Node, raw: dict) -> tuple[P, W]:
        import json

        w = TypeAdapter(self.workload).validate_json(json.dumps(raw))
        p = node.parameters
        # Standalone geometry is an explicit binding, not guessed from an operator.
        if (
            p is None
            and self.parameters in (AttentionGeometry, RecurrentGeometry)
            and "geometry" in raw
        ):
            p = TypeAdapter(self.parameters).validate_python(raw["geometry"])
        if not isinstance(p, self.parameters):
            raise ValueError(f"missing {self.parameters.__name__} parameters")
        return p, w


MODELS: dict[str, Model[Any, Any]] = {}


def register[P: Facts, W: BaseModel](owner: object, model: Model[P, W]) -> None:
    kind = component_id(owner).kind
    if kind in MODELS:
        raise ValueError(f"duplicate theory for {kind}")
    MODELS[kind] = model


def parameter_type(kind: str) -> type[Facts]:
    return MODELS[kind].parameters


def read_parameters(kind: str, raw: dict) -> Facts:
    import json

    return TypeAdapter(parameter_type(kind)).validate_json(json.dumps(raw), strict=True)


def execution(p, w, profile, demand, children):
    return {"EXEC": tightened_time_bound(demand, profile, w.dependent_phases)}


def bookkeeping(p, w, children):
    return Demands(assumptions=("local bookkeeping may fuse into its owner",))


def explicit(p, w: Workload, children):
    if w.required_inputs is None and not children:
        return Demands(missing=("mathematical required_inputs or child regions",))
    return join(Demands(w.required_inputs or (), w.required_outputs), *children.values())


register(
    GatheredAttention,
    Model(
        AttentionGeometry,
        AttentionWorkload,
        ("EXEC",),
        lambda p, w, ch: neural.attention(p, w),
        execution,
    ),
)
register(
    DeltaReference,
    Model(
        RecurrentGeometry,
        RecurrentWorkload,
        ("EXEC",),
        lambda p, w, ch: neural.recurrence(p, w),
        execution,
    ),
)
for owner in (
    ResidentEmbedding,
    ResidentExperts,
    GatedAttention,
    RecurrentMixer,
    RoutedFeedForward,
    qwen_readout,
    MTPProgram,
    PerLayerInputs,
    GemmaAttention,
    KVProducer,
    GemmaFeedForward,
    GeGLU,
    ExpertBranch,
    gemma_readout,
    ModelRuntime,
):
    register(
        owner, Model(NeuralParameters, NeuralWorkload, ("EXEC",), composition.model, execution)
    )
for owner in (Qwen35Program, Gemma4Program):
    register(
        owner, Model(NeuralParameters, NeuralWorkload, ("EXEC",), composition.program, execution)
    )
register(LibraryProgram, Model(OpaqueParameters, Workload, ("EXEC",), explicit, execution))


def storage_bounds(p, w, profile, demand, children):
    return {
        "MEM": state.retained(p, w, {k: v for k, v in children.items() if "MEM" in v}),
        "RESTORE": state.restore(p, w, profile),
    }


for owner, parameters in (
    (RecurrentImage, RecurrentStorage),
    (LibraryStateStore, NativeStorage),
    (HybridStateStore, Configuration),
):
    register(
        owner, Model(parameters, StateWorkload, ("MEM", "RESTORE"), bookkeeping, storage_bounds)
    )
register(
    PageStore,
    Model(
        KVStorage,
        StateWorkload,
        ("MEM",),
        bookkeeping,
        lambda p, w, profile, d, ch: {"MEM": state.retained(p, w)},
    ),
)
register(
    SequencePages.write,
    Model(KVStorage, StateWorkload, ("EXEC",), lambda p, w, ch: state.append(p, w), execution),
)


def service_bounds(p, w, profile, demand, children):
    return {
        dimension: engine.service(dimension, w, demand, profile)
        for dimension in ("RATE", "TTFT", "GAP")
    }


register(
    Engine,
    Model(
        Configuration,
        ServiceWorkload,
        ("RATE", "TTFT", "GAP"),
        lambda p, w, ch: engine.service_information(explicit(p, w, ch)),
        service_bounds,
    ),
)
register(
    TimeShared,
    Model(Configuration, ServiceWorkload, ("RATE", "TTFT", "GAP"), bookkeeping, service_bounds),
)
register(
    Radix,
    Model(
        Configuration,
        ControlWorkload,
        ("REUSE",),
        bookkeeping,
        lambda p, w, profile, d, ch: {"REUSE": engine.reuse(w)},
    ),
)
for owner, dimension in (
    (MemoryBudget, "EXEC"),
    (serve, "EXEC"),
    (PageStore.create, "EXEC"),
    (Engine.submit, "LAT"),
):
    register(
        owner,
        Model(
            Configuration,
            ControlWorkload,
            (dimension,),
            bookkeeping,
            lambda p, w, profile, d, ch, dimension=dimension: {
                dimension: Bound(
                    0, "seconds", assumptions=("bookkeeping may disappear into its owner",)
                )
            },
        ),
    )
register(
    GenerationRuntime.prefill_many,
    Model(Configuration, Workload, ("EXEC",), bookkeeping, execution),
)
for owner in (PlainMethod, MTPMethod):
    register(
        owner,
        Model(
            Configuration,
            Workload,
            ("EXEC",),
            lambda p, w, ch: join(*(ch[k] for k in ("target", "draft") if k in ch)),
            execution,
        ),
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


register(SequenceSampler, Model(Configuration, ControlWorkload, ("EXEC",), sampling, execution))
register(accept_prefix, Model(Configuration, ControlWorkload, ("EXEC",), acceptance, execution))
register(ExecutionOwner, Model(Configuration, ControlWorkload, ("EXEC",), device, execution))
register(
    load_resident_parameters,
    Model(
        Configuration,
        StateWorkload,
        ("LAT", "MEM"),
        explicit,
        lambda p, w, profile, d, ch: {"LAT": time_bound(d, profile), "MEM": state.retained(p, w)},
    ),
)


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
