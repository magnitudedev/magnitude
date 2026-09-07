"""Materialized state bounds under explicit representation contracts."""

import math

from performance.facts import Facts, KVStorage, RecurrentStorage
from performance.theory.resources import Bound, Demands, Extent, time_bound
from performance.theory.workloads import StateWorkload


def retained(p: Facts, w: StateWorkload, children=None) -> Bound:
    if children:
        terms = {}
        for name, dimensions in children.items():
            child = dimensions.get("MEM")
            if child is None or child.value is None:
                return Bound(None, "bytes", missing=(f"{name}/MEM binding",))
            terms.update(child.terms)
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("union of required materialized child allocations",),
        )
    if w.retained_shapes is not None:
        terms = {}
        for item in w.retained_shapes:
            if any(d < 0 for d in item.shape):
                raise ValueError("invalid retained tensor shape")
            size = math.prod(item.shape) * item.element_bytes
            if item.identity in terms and terms[item.identity] != size:
                raise ValueError("inconsistent shared retained shape")
            terms[item.identity] = size
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("required materialized representation",),
        )
    if isinstance(p, RecurrentStorage) and w.retained_rows is not None:
        terms: dict[str, float] = {
            t.identity: t.bytes * w.retained_rows for layout in p.layouts for t in layout
        }
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("distinct logically required recurrent rows",),
        )
    if isinstance(p, KVStorage) and w.retained_positions is not None:
        terms: dict[str, float] = {
            f"kv.{i}": w.retained_positions
            * layer.heads
            * (layer.key_width + layer.value_width)
            * p.element_bytes
            for i, layer in enumerate(p.layers)
        }
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("uncompressed declared KV representation",),
        )
    return Bound(
        None, "bytes", missing=("retained shapes or bound storage and retained rows/positions",)
    )


def append(p: KVStorage, w: StateWorkload) -> Demands:
    if w.append_tokens is None:
        return Demands(missing=("append_tokens",))
    logical = retained(p, StateWorkload(retained_positions=w.append_tokens))
    return Demands(
        (Extent("append:inputs", 0, int(logical.value or 0)),),
        assumptions=("new logical KV payload only; COW copies may be eliminated",),
    )


def restore(p: Facts, w: StateWorkload, profile) -> Bound:
    if w.restore_mode == "saved_boundary":
        return Bound(0, "seconds", assumptions=("immutable state may be selected by reference",))
    if w.restore_mode == "accepted_prefix":
        return Bound(
            0,
            "seconds",
            assumptions=("advance may retain the accepted prefix; selection can be by reference",),
        )
    if w.reconstruction_inputs is not None:
        return time_bound(
            Demands(
                w.reconstruction_inputs,
                operations=w.reconstruction_operations,
                assumptions=("declared reconstruction-only input boundary",),
            ),
            profile,
        )
    return Bound(None, "seconds", missing=("restore_mode or reconstruction_inputs",))
