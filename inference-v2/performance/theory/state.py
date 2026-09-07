"""Materialized retained information; bookkeeping and restoration are distinct."""

import math

from performance.theory.resources import Bound, Demands, Extent, time_bound


def retained(p: dict, w: dict, children: dict[str, dict[str, Bound]] | None = None) -> Bound:
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
    if "retained_shapes" in w:
        terms = {}
        for item in w["retained_shapes"]:
            if any(d < 0 for d in item["shape"]) or item["element_bytes"] <= 0:
                raise ValueError("invalid retained tensor shape/encoding")
            size = math.prod(item["shape"]) * item["element_bytes"]
            if item["identity"] in terms and terms[item["identity"]] != size:
                raise ValueError("inconsistent shared retained shape")
            terms[item["identity"]] = size
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("required materialized representation",),
        )
    if "layouts" in p and "retained_rows" in w:
        rows = w["retained_rows"]
        if rows < 0:
            raise ValueError("negative retained row count")
        terms = {
            f"recurrent.{i}.{j}": t["bytes"] * rows
            for i, layout in enumerate(p["layouts"])
            for j, t in enumerate(layout)
        }
        return Bound(
            sum(terms.values()),
            "bytes",
            terms=terms,
            assumptions=("distinct logically required recurrent rows",),
        )
    if "layers" not in p or "retained_positions" not in w:
        return Bound(
            None,
            "bytes",
            missing=("retained_shapes, or bound layouts and retained rows/positions",),
        )
    positions = w["retained_positions"]
    if positions < 0:
        raise ValueError("negative retained position count")
    terms = {
        f"kv.{i}": positions
        * layer.get("kv_heads", layer.get("heads"))
        * (layer["key_width"] + layer["value_width"])
        * layer.get("element_bytes", p.get("element_bytes", 2))
        for i, layer in enumerate(p["layers"])
    }
    return Bound(
        sum(terms.values()),
        "bytes",
        terms=terms,
        assumptions=("uncompressed declared KV representation",),
    )


def append(p: dict, w: dict) -> Demands:
    if "layers" not in p or "append_tokens" not in w:
        return Demands(missing=("layer geometry and append_tokens",))
    logical = retained(p, {"retained_positions": w["append_tokens"]})
    return Demands(
        (Extent("append:inputs", 0, int(logical.value or 0)),),
        assumptions=("new logical KV payload only; COW copies may be eliminated",),
    )


def restore(p: dict, w: dict, profile=None) -> Bound:
    if w.get("restore_mode") == "saved_boundary":
        return Bound(0, "seconds", assumptions=("immutable state may be selected by reference",))
    if w.get("restore_mode") == "accepted_prefix":
        # A legal implementation may retain each committed prefix during advance.
        # This contract does not require reconstruction after the timed boundary.
        return Bound(
            0,
            "seconds",
            assumptions=("advance may retain the accepted prefix; selection can be by reference",),
        )
    if "reconstruction_inputs" in w and profile is not None:
        return time_bound(
            Demands(
                tuple(Extent(**item) for item in w["reconstruction_inputs"]),
                operations=w.get("reconstruction_operations", {}),
                assumptions=("declared reconstruction-only input boundary",),
            ),
            profile,
        )
    return Bound(None, "seconds", missing=("restore_mode or reconstruction_inputs",))
