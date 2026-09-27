"""Physical feasibility of retention decisions, without choosing retention policy."""

from __future__ import annotations

from dataclasses import dataclass

from .pages import KVCheckpoint, PageStore
from .placement import RepackedLayout


@dataclass(frozen=True)
class ReclamationPlan:
    layout: RepackedLayout | None
    victims: tuple[KVCheckpoint, ...]
    copy_budget: int
    ownership: tuple
    reason: str


@dataclass(frozen=True)
class ReclamationResult:
    retired: tuple[KVCheckpoint, ...]
    pages_discarded: int
    pages_moved: int
    bytes_released: int


def plan_reclamation(
    store: PageStore, eligible: frozenset[KVCheckpoint], *, max_copy_bytes: int
) -> ReclamationPlan:
    """Retain active pages; move cold survivors only into existing lower holes.

    The prefix owner supplies eligible checkpoints. The state store resolves which
    physical allocations those handles pin. Eligibility is rechecked at execution;
    merely removing a retained entry is never reported as released slab memory.
    """
    if max_copy_bytes < 0:
        raise ValueError("reclamation copy budget cannot be negative")
    allocator = store.arena.allocator
    ownership = tuple(
        (
            p.address,
            p.writer,
            tuple(sorted(p.readers.items())),
            tuple(sorted(p.checkpoints.items())),
        )
        for p in sorted(store._pages.values(), key=lambda p: p.address)
    )

    def declined(reason: str) -> ReclamationPlan:
        return ReclamationPlan(None, (), max_copy_bytes, ownership, reason)

    if not allocator.capacity:
        return declined("empty")
    boundary = allocator.capacity - allocator.slab_pages
    trailing = [p for p in store._pages.values() if p.address >= boundary]
    if any(p.writer is not None or p.readers for p in trailing):
        return declined("active pages pin trailing slab")
    if {p for p in allocator.owned if p >= boundary} != {p.address for p in trailing}:
        return declined("unregistered allocation pins trailing slab")
    candidates = {
        checkpoint._identity
        for checkpoint in eligible
        if checkpoint._store is store and not checkpoint.closed
    }
    victims = tuple(
        store._checkpoints[identity]
        for identity in sorted(
            candidates & {identity for p in trailing for identity in p.checkpoints}
        )
    )
    identities = {checkpoint._identity for checkpoint in victims}
    released = tuple(
        sorted(
            p.address
            for p in store._pages.values()
            if p.writer is None and not p.readers and set(p.checkpoints) <= identities
        )
    )
    survivors = sorted(p.address for p in trailing if p.address not in released)
    if len(survivors) * store.arena.page_bytes > max_copy_bytes:
        return declined("copy budget")
    try:
        destinations = allocator.plan(len(survivors), grow=False, before=boundary).pages
    except MemoryError:
        return declined("insufficient existing lower holes")
    layout = allocator.plan_repack(released, tuple(zip(survivors, destinations, strict=True)))
    return ReclamationPlan(layout, victims, max_copy_bytes, ownership, "ready")


def reclaim(
    store: PageStore, plan: ReclamationPlan, eligible: frozenset[KVCheckpoint]
) -> ReclamationResult:
    """Commit on the execution owner between steps, after retention eligibility is revalidated.

    Allocation and relocation are completed before any checkpoint is retired. A
    failed allocation leaves both logical retention and physical addresses intact.
    The caller removes returned retired handles from its semantic prefix index in
    the same owner turn, before admitting another sequence.
    """
    store.arena._idle()
    if plan.layout is None:
        raise ValueError(f"reclamation is not feasible: {plan.reason}")
    current = plan_reclamation(store, eligible, max_copy_bytes=plan.copy_budget)
    if current != plan:
        raise RuntimeError("reclamation ownership or retention eligibility changed")
    released_bytes = store.arena.repack(plan.layout)
    for checkpoint in plan.victims:
        for page in checkpoint._pages:
            del page.checkpoints[checkpoint._identity]
            page.retained_tails.pop(checkpoint._identity, None)
        del store._checkpoints[checkpoint._identity]
        checkpoint.closed = True
    addresses = dict(plan.layout.moves)
    survivors = {}
    for page in store._pages.values():
        if page.address in plan.layout.released:
            continue
        page.address = addresses.get(page.address, page.address)
        if page.writer is None:
            page.written = page.protected
        survivors[page.address] = page
    store._pages = survivors
    return ReclamationResult(
        plan.victims, len(plan.layout.released), len(plan.layout.moves), released_bytes
    )
