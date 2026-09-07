"""Optimistic model composition from captured parameters and semantic children.

The information relaxation permits complete fusion and ideal reuse. It charges no
intermediate activation transfer, no mandatory launch, and no assumed CPU fence.
Fixed encoded parameter bindings are preserved; changing that representation is a
new formulation contract. Nonlinear arithmetic is omitted, never calibrated from
reference throughput. Conditional expert selection uses a minimum legal active set
unless the workload supplies distinct selected experts.
"""

from dataclasses import replace

from performance.facts import NeuralParameters, WeightUse
from performance.theory.resources import Demands, Extent, join
from performance.theory.workloads import NeuralWorkload


def weights(p: NeuralParameters, w: NeuralWorkload) -> Demands:
    arrays = p.arrays
    extents = []
    for item in arrays.values():
        size = item.bytes
        shape = item.shape
        if p.weight_use == WeightUse.EMBEDDING:
            if not shape or shape[0] < 1:
                raise ValueError("embedding requires a positive vocabulary")
            rows = w.distinct_input_tokens
            if not 1 <= rows <= min(shape[0], w.batch_size * w.query_tokens):
                raise ValueError("distinct embedding rows exceed input domain")
            size = size // shape[0] * rows
        elif p.weight_use == WeightUse.EXPERTS:
            if not shape or shape[0] < 1:
                raise ValueError("expert tensors require a positive expert axis")
            count = w.distinct_experts if w.distinct_experts is not None else p.top_k
            if count is None:
                return Demands(missing=("top_k or distinct_experts",))
            if not 1 <= count <= shape[0]:
                raise ValueError("selected experts exceed expert domain")
            size = size // shape[0] * count
        extents.append(Extent("weight:" + item.identity, 0, size))
    operations = {}
    if w.conventional_arithmetic:
        uses = w.batch_size * w.query_tokens
        if w.mode in ("generate", "replay"):
            uses *= w.measured_tokens
        operations["scalar"] = sum(
            uses
            * matrix.output_width
            * (2 * matrix.input_width - 1)
            * ((p.top_k or 1) if matrix.experts else 1)
            for matrix in p.matrices
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


def model(p: NeuralParameters, w: NeuralWorkload, children: dict[str, Demands]) -> Demands:
    local = weights(p, w)
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
