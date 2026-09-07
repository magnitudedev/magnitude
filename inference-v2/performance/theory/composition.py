"""Optimistic model composition from captured parameters and semantic children.

The information relaxation permits complete fusion and ideal reuse. It charges no
intermediate activation transfer, no mandatory launch, and no assumed CPU fence.
Fixed encoded parameter bindings are preserved; changing that representation is a
new formulation contract. Nonlinear arithmetic is omitted, never calibrated from
reference throughput. Conditional expert selection uses a minimum legal active set
unless the workload supplies distinct selected experts.
"""

from dataclasses import replace

from performance.theory.resources import Demands, Extent, join


def weights(component: str, p: dict, w: dict) -> Demands:
    arrays = p.get("arrays")
    if arrays is None:
        return Demands(missing=(f"{component}: captured parameter tensors",))
    extents = []
    for item in arrays.values():
        size = item["bytes"]
        shape = item["shape"]
        if component == "MODEL:EMBEDDING":
            if not shape or shape[0] < 1:
                raise ValueError("embedding requires a positive vocabulary")
            rows = w.get("distinct_input_tokens", 1)
            if not 1 <= rows <= min(shape[0], w.get("batch_size", 1) * w.get("query_tokens", 1)):
                raise ValueError("distinct embedding rows exceed input domain")
            size = size // shape[0] * rows
        elif component == "MODEL:EXPERTS":
            if not shape or shape[0] < 1:
                raise ValueError("expert tensors require a positive expert axis")
            count = w.get("distinct_experts", p.get("top_k"))
            if count is None:
                return Demands(missing=("top_k or distinct_experts",))
            if not 1 <= count <= shape[0]:
                raise ValueError("selected experts exceed expert domain")
            size = size // shape[0] * count
        extents.append(Extent("weight:" + item["identity"], 0, size))
    operations = {}
    if w.get("conventional_arithmetic"):
        uses = w.get("batch_size", 1) * w.get("query_tokens", 1)
        if w.get("mode") in ("generate", "replay"):
            uses *= w.get("measured_tokens", 1)
        operations["scalar"] = sum(
            uses
            * matrix["output_width"]
            * (2 * matrix["input_width"] - 1)
            * (p.get("top_k", 1) if matrix["experts"] else 1)
            for matrix in p.get("matrices", ())
        )
    return Demands(
        tuple(extents),
        operations=operations,
        assumptions=(
            "fixed encoded parameter representation; ideal reuse and fusion",
            "minimum legal distinct embedding/expert rows unless explicitly conditioned",
            "nonlinear arithmetic relaxed away",
        ),
    )


def model(component: str, p: dict, w: dict, children: dict[str, Demands]) -> Demands:
    local = weights(component, p, w)
    # Child activation operands are produced by this composite. Parameter and
    # historical-state identities survive; their overlap is unioned by JOIN.
    parts = [
        replace(
            c,
            inputs=tuple(e for e in c.inputs if not e.identity.startswith("activation:")),
            outputs=(),
        )
        for c in children.values()
    ]
    return join(local, *parts)
