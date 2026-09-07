"""Service relaxations use necessary information, never observed overhead.

A complete workload may fuse, overlap and share all legal work. Its resource
relaxation reads each necessary parameter only once. This intentionally loose bound
is stronger only when a workload proves additional causal/residency constraints.
Publication GAP has a zero floor if the contract permits buffering outputs; a
positive bound requires an immediate-publication dependency contract.
"""

from performance.theory.resources import Bound, Demands, time_bound


def reuse(w: dict) -> Bound:
    eligible = w.get("eligible_prefix_tokens")
    if eligible is None:
        return Bound(None, "tokens", "upper", missing=("eligible_prefix_tokens",))
    if eligible < 0:
        raise ValueError("negative eligible prefix work")
    return Bound(eligible, "tokens", "upper", assumptions=("fixed eligible prefix workload",))


def service(dimension, w, demand, profile):
    if dimension == "GAP":
        return Bound(
            0,
            "seconds",
            assumptions=(
                "publication buffering is permitted by the unconstrained service contract",
                "a positive GAP bound needs an explicit immediate-publication dependency contract",
            ),
        )
    lower = time_bound(demand, profile)
    if dimension == "TTFT":
        if w.get("statistics", {}).get("TTFT") != "max":
            return Bound(None, "seconds", missing=("supported TTFT population statistic: max",))
        # The parameter-read relaxation also applies to the first publication:
        # it does not charge later outputs or internal state built by prefill.
        return lower
    units = w.get("output_tokens")
    if units is None:
        return Bound(None, "tokens/second", "upper", missing=("output_tokens",))
    units *= w.get("rows", 1) * w.get("waves", 1)
    if lower.value is None:
        return Bound(
            None, "tokens/second", "upper", missing=lower.missing, assumptions=lower.assumptions
        )
    if lower.value == 0:
        return Bound(
            None,
            "tokens/second",
            "upper",
            kind="unbounded",
            assumptions=("unbounded rate in the zero-time relaxation; no finite efficiency",),
        )
    return Bound(
        units / lower.value,
        "tokens/second",
        "upper",
        terms=lower.terms,
        missing=lower.missing,
        assumptions=lower.assumptions,
    )


def neural_point(w):
    """Minimal neural obligation of a nonempty service, independent of scheduling."""
    if "output_tokens" not in w:
        return w
    return {
        **w,
        "query_tokens": w.get("query_tokens", 1),
        "batch_size": w.get("batch_size", 1),
        "histories": w.get("histories", [w.get("context_tokens", 0)]),
    }


def service_information(demand: Demands) -> Demands:
    # Parameters are external. Context KV and recurrent state are produced inside
    # a cold service and are not compulsory incoming information at this boundary.
    return Demands(
        tuple(e for e in demand.inputs if e.identity.startswith("weight:")),
        assumptions=demand.assumptions + ("one ideal parameter read for the service",),
        missing=demand.missing,
    )
