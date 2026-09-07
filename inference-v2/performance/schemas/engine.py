"""Read engine relationships from their actual construction bindings."""

from magnitude_engine import components as c
from magnitude_engine.engine.binding import EngineResidency
from magnitude_engine.engine.memory.policy import Budgeted
from magnitude_engine.engine.prefixes.radix import Radix
from magnitude_engine.engine.runtime import Engine
from magnitude_engine.engine.scheduler.time_shared import TimeShared
from magnitude_engine.generation.acceptance import accept_prefix
from magnitude_engine.generation.execution import serve
from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling import SequenceSampler
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.resources.budget import MemoryBudget
from performance.bindings import Fields, Use, operation, port, schema


def model(runtime) -> Use:
    return Use(runtime.program, dependencies={"state": Use(runtime.states)})


@schema(Engine, context=EngineResidency)
def engine(a: Engine, binding: EngineResidency) -> Fields[c.Configuration]:
    return Fields(
        c.Configuration(
            settings={
                k: v
                for k, v in binding.properties.items()
                if k
                not in (
                    "target_path",
                    "tokenizer_identity",
                    "memory_bytes",
                    "program_implementation",
                )
            }
        ),
        children={
            "generation": Use(a.generation.method, a.generation),
            "admission": operation(a.submit),
            "batching": operation(serve),
            "execution": Use(a.generation.model.owner),
            "scheduling": Use(a.scheduler, a.generation),
            "prefixes": Use(a.prefixes),
            "memory": Use(binding.budget),
        },
    )


def generation(
    method: PlainMethod | MTPMethod | SuffixMethod, a: GenerationRuntime
) -> Fields[c.Configuration]:
    children = {
        "target": model(a.model),
        "sampling": port(SequenceSampler.__init__, SequenceSampler),
    }
    if isinstance(a.method, MTPMethod):
        children["draft"] = model(a.method.head)
    if isinstance(a.method, (MTPMethod, SuffixMethod)):
        children["acceptance"] = operation(accept_prefix)
    settings: dict[str, int | float | bool | str | None] = (
        {"minimum": a.method.minimum, "maximum": a.method.maximum}
        if isinstance(a.method, SuffixMethod)
        else {}
    )
    return Fields(
        c.Configuration(settings=settings),
        children=children,
        dependencies={"execution": Use(a.model.owner)},
        sources=(a,),
    )


for method_type in (PlainMethod, MTPMethod, SuffixMethod):
    schema(method_type, context=GenerationRuntime)(generation)


@schema(TimeShared, context=GenerationRuntime | None)
def scheduler(a: TimeShared, generation: GenerationRuntime | None) -> Fields[c.Configuration]:
    return Fields(
        c.Configuration(
            settings={
                "max_active": a.max_active,
                "max_queued": a.max_queued,
                "prefill_tokens": a.prefill_tokens,
                "decode_share": a.decode_share,
            }
        ),
        children={} if generation is None else {"prefill": operation(generation.prefill_many)},
    )


@schema(Radix)
def prefix(a: Radix, _: None) -> Fields[c.Configuration]:
    return Fields(
        c.Configuration(
            settings={"max_entries": a.retention.max_entries, "max_bytes": a.retention.max_bytes}
        ),
        sources=(a.retention,),
    )


@schema(Budgeted)
def budgeted(a: Budgeted, _: None) -> Fields[c.Configuration]:
    return Fields(c.Configuration(settings={"limit": a.limit}), sources=(a.pressure,))


@schema(MemoryBudget)
def budget(a: MemoryBudget, _: None) -> Fields[c.Configuration]:
    return Fields(c.Configuration(settings={"limit": a.limit}))


@schema(ExecutionOwner)
def device(a: ExecutionOwner, _: None) -> Fields[c.Configuration]:
    return Fields(c.Configuration())
