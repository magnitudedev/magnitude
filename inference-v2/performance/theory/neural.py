"""Neural requirements under explicit representation and arithmetic assumptions."""

from performance.facts import AttentionGeometry, RecurrentGeometry
from performance.theory.resources import Demands, Extent
from performance.theory.workloads import AttentionWorkload, RecurrentWorkload


def attention(p: AttentionGeometry, w: AttentionWorkload) -> Demands:
    q, histories = w.query_tokens, w.histories
    if q < 1 or not histories or any(h < 0 for h in histories):
        raise ValueError("invalid attention workload")
    window = p.window
    pairs = sum(
        sum(min(h + j, window) if window else h + j for j in range(1, q + 1)) for h in histories
    )
    visible = sum(min(h + q, window + q - 1) if window else h + q for h in histories)
    b, hq, hk, dk, dv, size = (
        len(histories),
        p.query_heads,
        p.kv_heads,
        p.key_width,
        p.value_width,
        p.element_bytes,
    )
    if min(hq, hk, dk, dv, size) < 1 or hq % hk or (window is not None and window < 1):
        raise ValueError("invalid attention geometry")
    namespace = w.information_domain
    kv_namespace = w.kv_information_domain or namespace
    kv = visible * hk * (dk + dv) * size
    ops = hq * pairs * (2 * dk - 1) + hq * dv * (2 * pairs - b * q)
    return Demands(
        (
            Extent(kv_namespace + ".kv", 0, kv - b * q * hk * (dk + dv) * size),
            Extent("activation:" + namespace + ".new_kv", 0, b * q * hk * (dk + dv) * size),
            Extent("activation:" + namespace + ".q", 0, b * q * hq * dk * size),
        ),
        (Extent(namespace + ".output", 0, b * q * hq * dv * size),),
        {"scalar": ops} if w.conventional_arithmetic else {},
        ("fixed representation; arbitrary content-dependent attention",)
        + (("conventional dot-product arithmetic",) if w.conventional_arithmetic else ()),
    )


def projection(
    m: int, k: int, n: int, *, identity: str, encoded_bytes: int, element_bytes: int = 2
) -> Demands:
    if min(m, k, n, encoded_bytes, element_bytes) < 1:
        raise ValueError("invalid projection geometry")
    return Demands(
        (
            Extent(identity + ".weight", 0, encoded_bytes),
            Extent(identity + ".input", 0, m * k * element_bytes),
        ),
        (Extent(identity + ".output", 0, m * n * element_bytes),),
        {"scalar": m * n * (2 * k - 1)},
        ("conventional dense dot products",),
    )


def affine_bytes(n: int, k: int, bits: int, group_size: int, metadata_bytes: int = 4) -> int:
    if min(n, k, group_size, metadata_bytes) < 1 or bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError("invalid affine encoding")
    return n * ((k * bits + 7) // 8 + (k + group_size - 1) // group_size * metadata_bytes)


def recurrence(p: RecurrentGeometry, w: RecurrentWorkload) -> Demands:
    b, q = w.batch_size, w.query_tokens
    hk, hv, dk, dv, size = p.key_heads, p.value_heads, p.key_width, p.value_width, p.element_bytes
    if min(b, q, hk, hv, dk, dv, size) < 1 or hv % hk:
        raise ValueError("invalid recurrence geometry")
    state = b * hv * dk * dv * 4
    incoming = b * q * (2 * hk * dk * size + hv * dv * size + hv * (size + 4))
    ns = w.information_domain
    return Demands(
        (Extent(ns + ".state", 0, state), Extent("activation:" + ns + ".inputs", 0, incoming)),
        (Extent(ns + ".next_state", 0, state), Extent(ns + ".outputs", 0, b * q * hv * dv * size)),
        assumptions=("fixed materialized recurrent-state representation",),
    )
