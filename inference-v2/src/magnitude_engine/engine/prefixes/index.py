"""Semantic prefix matching and retention, independent of physical model storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Protocol

from magnitude_engine.resources.retention import RetainedStorage


class Checkpoint(Protocol):
    """A complete generation checkpoint at a legal resume boundary."""

    @property
    def reclaimable(self) -> bool: ...

    @property
    def length(self) -> int: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...
    def retained_storage(self) -> tuple[RetainedStorage, ...]: ...


type TokenIdentity = tuple[int, bytes]


@dataclass(frozen=True)
class PrefixIdentity:
    """namespace binds model, tokenizer, execution state format, and generation method.

    Token semantics carry media/position identity where token IDs alone are
    insufficient. Request randomness and output constraints are not cached state.
    """

    namespace: bytes
    tokens: tuple[TokenIdentity, ...]

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("prefix identity requires an explicit compatibility namespace")


@dataclass(eq=False)
class _Entry[C: Checkpoint]:
    identity: int
    checkpoint: C
    node: _Node[C]
    touched: int
    borrowers: int = 0


@dataclass(eq=False)
class _Node[C: Checkpoint]:
    label: tuple[TokenIdentity, ...]
    parent: _Node[C] | None
    children: dict[TokenIdentity, _Node[C]] = field(default_factory=dict)
    entry: _Entry[C] | None = None


class PrefixLease[C: Checkpoint]:
    def __init__(self, entry: _Entry[C]):
        self._entry = entry
        self.closed = False
        entry.borrowers += 1

    @property
    def checkpoint(self) -> C:
        if self.closed:
            raise RuntimeError("prefix lease is closed")
        return self._entry.checkpoint

    def close(self) -> None:
        if not self.closed:
            self._entry.borrowers -= 1
            self.closed = True


class PrefixStore[C: Checkpoint]:
    """A compressed token trie with explicit retention and restore leases.

    The generation runtime supplies complete checkpoints; it decides legal
    boundaries and how their state is represented. Retention never reads page
    IDs, recurrent arrays, drafter internals, or physical memory estimates.
    """

    def __init__(self):
        self._roots: dict[bytes, _Node[C]] = {}
        self._entries: dict[int, _Entry[C]] = {}
        self._clock = count()
        self._identities = count()

    def __len__(self) -> int:
        return len(self._entries)

    def retain(self, identity: PrefixIdentity, checkpoint: C) -> C:
        if checkpoint.closed or checkpoint.length != len(identity.tokens) or checkpoint.length == 0:
            raise ValueError("checkpoint must cover the entire nonempty retained prefix")
        node = self._roots.setdefault(identity.namespace, _Node((), None))
        position = 0
        while position < len(identity.tokens):
            atom = identity.tokens[position]
            child = node.children.get(atom)
            remaining = identity.tokens[position:]
            if child is None:
                child = _Node(remaining, node)
                node.children[atom] = child
                node = child
                break
            common = 0
            while (
                common < min(len(remaining), len(child.label))
                and remaining[common] == child.label[common]
            ):
                common += 1
            if common < len(child.label):
                middle = _Node(child.label[:common], node)
                node.children[atom] = middle
                child.label = child.label[common:]
                child.parent = middle
                middle.children[child.label[0]] = child
                child = middle
            position += common
            node = child
        if node.entry is not None:
            existing = node.entry.checkpoint
            if existing.closed:
                del self._entries[node.entry.identity]
                node.entry = None
            else:
                if existing is not checkpoint:
                    checkpoint.close()
                node.entry.touched = next(self._clock)
                return existing
        if any(entry.checkpoint is checkpoint for entry in self._entries.values()):
            raise ValueError("a checkpoint cannot be owned by two retained identities")
        entry = _Entry(next(self._identities), checkpoint, node, next(self._clock))
        node.entry = entry
        self._entries[entry.identity] = entry
        return checkpoint

    def match(self, identity: PrefixIdentity, *, exclude_last: int = 1) -> PrefixLease[C] | None:
        if exclude_last < 0:
            raise ValueError("excluded suffix cannot be negative")
        node = self._roots.get(identity.namespace)
        limit = max(0, len(identity.tokens) - exclude_last)
        position, best = 0, None
        while node is not None and position < limit:
            child = node.children.get(identity.tokens[position])
            if child is None or position + len(child.label) > limit:
                break
            if identity.tokens[position : position + len(child.label)] != child.label:
                break
            position += len(child.label)
            node = child
            if node.entry is not None and not node.entry.checkpoint.closed:
                best = node.entry
        if best is None:
            return None
        best.touched = next(self._clock)
        return PrefixLease(best)

    def eligible(self) -> tuple[C, ...]:
        """Least recently used first; physical storage determines actual reclaimability."""
        return tuple(
            entry.checkpoint
            for entry in sorted(self._entries.values(), key=lambda e: e.touched)
            if not entry.borrowers and not entry.checkpoint.closed
        )

    def discard(self, checkpoints: tuple[C, ...]) -> None:
        wanted = {id(checkpoint) for checkpoint in checkpoints}
        entries = [entry for entry in self._entries.values() if id(entry.checkpoint) in wanted]
        if any(entry.borrowers for entry in entries):
            raise RuntimeError("prefix restoration still holds a retention lease")
        for entry in entries:
            entry.checkpoint.close()
            del self._entries[entry.identity]
            node = entry.node
            node.entry = None
            while node.parent is not None and node.entry is None:
                parent = node.parent
                if not node.children:
                    del parent.children[node.label[0]]
                    node = parent
                elif len(node.children) == 1:
                    child = next(iter(node.children.values()))
                    child.label = node.label + child.label
                    child.parent = parent
                    parent.children[node.label[0]] = child
                    break
                else:
                    break

    def close(self) -> None:
        if any(entry.borrowers for entry in self._entries.values()):
            raise RuntimeError("prefix restoration leases must retire before shutdown")
        self.discard(tuple(entry.checkpoint for entry in self._entries.values()))
        self._roots.clear()
