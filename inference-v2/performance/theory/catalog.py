"""Component contracts choose functions, not implementations or construction graphs."""

from dataclasses import asdict
from pathlib import Path

from performance.records import Node, Profile, digest
from performance.theory import composition, engine, neural, state
from performance.theory.resources import Bound, Demands, Extent, join, time_bound

# A metric has one unit and population meaning throughout collection and display.
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

DIMENSIONS = {
    "ENGINE:INFERENCE": ("RATE", "TTFT", "GAP"),
    "SCHEDULING:ADMISSION": ("LAT",),
    "SCHEDULING:SERVICE": ("RATE", "TTFT", "GAP"),
    "SCHEDULING:PREFILL": ("EXEC",),
    "BATCHING:ASSEMBLY": ("EXEC",),
    "EXECUTION:DEVICE": ("EXEC",),
    "MEMORY:ACCOUNTING": ("EXEC",),
    "CACHE:PREFIX": ("REUSE",),
    "KV:STORE": ("MEM",),
    "KV:APPEND": ("EXEC",),
    "KV:BRANCH": ("EXEC",),
    "STATE:RECURRENT": ("MEM", "RESTORE"),
    "STATE:CHECKPOINTS": ("MEM", "RESTORE"),
    "STATE:QWEN35": ("MEM", "RESTORE"),
    "GENERATION:PLAIN": ("EXEC",),
    "GENERATION:SPECULATION": ("EXEC",),
    "GENERATION:SAMPLING": ("EXEC",),
    "GENERATION:ACCEPTANCE": ("EXEC",),
    "MODEL:LOADING": ("LAT", "MEM"),
}
for component in (
    "EXECUTOR",
    "FORWARD",
    "EMBEDDING",
    "ATTENTION",
    "GATED_DELTA",
    "EXPERTS",
    "QWEN35",
    "QWEN35.ATTENTION",
    "QWEN35.RECURRENCE",
    "QWEN35.FEEDFORWARD",
    "QWEN35.READOUT",
    "QWEN35.MTP",
    "GEMMA4",
    "GEMMA4.INPUTS",
    "GEMMA4.ATTENTION",
    "GEMMA4.KV",
    "GEMMA4.FEEDFORWARD",
    "GEMMA4.MLP",
    "GEMMA4.EXPERT_BRANCH",
    "GEMMA4.READOUT",
):
    DIMENSIONS[f"MODEL:{component}"] = ("EXEC",)


def revision() -> str:
    return digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})


def requirements(node: Node, workload: dict, children: dict[str, Demands]) -> Demands:
    p = node.parameters | workload.get("geometry", {})
    if node.component == "MODEL:ATTENTION":
        return neural.attention(p, workload)
    if node.component == "MODEL:GATED_DELTA":
        return neural.recurrence(p, workload)
    if node.component in (
        "MEMORY:ACCOUNTING",
        "BATCHING:ASSEMBLY",
        "KV:BRANCH",
        "SCHEDULING:ADMISSION",
        "CACHE:PREFIX",
        "SCHEDULING:PREFILL",
        "SCHEDULING:SERVICE",
        "EXECUTION:DEVICE",
    ):
        if node.component != "EXECUTION:DEVICE" or "elements" not in workload:
            return Demands(assumptions=("local bookkeeping may fuse into its owner",))
    if node.component == "KV:APPEND":
        return state.append(p, workload)
    if node.component == "GENERATION:SAMPLING":
        missing = tuple(
            k for k in ("vocabulary", "positions", "element_bytes") if k not in workload
        )
        if missing:
            return Demands(missing=missing)
        size = workload["vocabulary"] * workload["positions"] * workload["element_bytes"]
        return Demands(
            (Extent("sampling:logits", 0, size),),
            assumptions=("arbitrary-logit sampling; all candidates may matter",),
        )
    if node.component == "GENERATION:ACCEPTANCE":
        if not all(k in workload for k in ("width", "rounds")):
            return Demands(missing=("width and rounds",))
        size = workload["rounds"] * (12 if workload["width"] else 4)
        return Demands(
            (Extent("acceptance:prefix", 0, size),),
            assumptions=("earliest possible mismatch; int32 tokens",),
        )
    if node.component == "EXECUTION:DEVICE" and "elements" in workload:
        return Demands(
            (Extent("device:input", 0, workload["elements"] * 4),),
            assumptions=("closed-form affine graph; intermediate passes eliminated",),
        )
    if node.component in ("GENERATION:PLAIN", "GENERATION:SPECULATION") and "target" in children:
        # Sampling and acceptance consume internally produced tensors. Relax their
        # cost rather than pretending a missing standalone input crosses this boundary.
        return join(*(children[k] for k in ("target", "draft") if k in children))
    if node.component == "MODEL:FORWARD" and p.get("opaque"):
        if "required_inputs" not in workload:
            return Demands(missing=("opaque forward: mathematical required_inputs binding",))
        return Demands(
            tuple(Extent(**e) for e in workload["required_inputs"]),
            assumptions=("explicit opaque-forward mathematical binding",),
        )
    if node.component.startswith("MODEL:") and "arrays" in p:
        return composition.model(node.component, p, workload, children)
    # Public region bindings allow a new component's mathematical projections to
    # reuse the same equations without embedding equations in a benchmark.
    local = [
        neural.projection(
            workload.get("batch_size", 1) * workload.get("query_tokens", 1),
            r["input_width"],
            r["output_width"],
            identity=r["identity"],
            encoded_bytes=r["encoded_bytes"],
            element_bytes=r["element_bytes"],
        )
        for r in p.get("projections", ())
    ]
    if "required_inputs" in workload:
        local.append(
            Demands(
                tuple(Extent(**e) for e in workload["required_inputs"]),
                assumptions=("declared required input domain",),
            )
        )
    if local or children:
        combined = join(
            *local,
            *children.values(),
            retained=tuple(Extent(**e) for e in workload.get("required_outputs", ())),
        )
        return (
            engine.service_information(combined)
            if node.component == "ENGINE:INFERENCE"
            else combined
        )
    return Demands(missing=(f"{node.component} required input/region binding",))


def evaluate(
    node: Node,
    workload: dict,
    profile: Profile,
    demand: Demands,
    children: dict[str, dict[str, Bound]],
) -> dict[str, Bound]:
    if node.component not in DIMENSIONS:
        raise ValueError(f"undefined component contract: {node.component}")
    result = {}
    for dimension in DIMENSIONS[node.component]:
        if dimension == "MEM":
            bound = state.retained(
                node.parameters, workload, {k: v for k, v in children.items() if "MEM" in v}
            )
        elif dimension == "RESTORE":
            bound = state.restore(node.parameters, workload, profile)
        elif dimension == "REUSE":
            bound = engine.reuse(workload)
        elif node.component in (
            "MEMORY:ACCOUNTING",
            "BATCHING:ASSEMBLY",
            "KV:BRANCH",
            "SCHEDULING:ADMISSION",
        ):
            bound = Bound(0, "seconds", assumptions=("bookkeeping may disappear into its owner",))
        elif dimension in ("RATE", "TTFT", "GAP"):
            bound = engine.service(dimension, workload, demand, profile)
        else:
            bound = time_bound(demand, profile)
        result[dimension] = bound
    return result


def serialized_bounds(bounds):
    return {key: asdict(value) for key, value in bounds.items()}
