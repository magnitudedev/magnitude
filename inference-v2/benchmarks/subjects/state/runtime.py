"""Deterministic KV branch trace, including tensor writes and state completion."""

import hashlib
import json
from typing import cast

import mlx.core as mx

from benchmarks.contracts import Observation
from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import KVCheckpoint, PageStore, SequencePages
from magnitude_engine.resources.budget import MemoryBudget


class AppendTrace:
    """Compare logical run appends with explicit per-page reference calls."""

    def __init__(self, *, granularity: str, prefix_tokens: int, append_tokens: int):
        if granularity not in ("runs", "pages") or min(prefix_tokens, append_tokens) <= 0:
            raise ValueError("invalid KV append case")
        self.granularity, self.prefix, self.count = granularity, prefix_tokens, append_tokens
        total = prefix_tokens + append_tokens
        self.arena = KVArena(
            (LayerGeometry(2, 256, 256),) * 10,
            page_size=16,
            slab_pages=32,
            max_pages=(total + 15) // 16 + 32,
            budget=MemoryBudget(2 << 30),
        )
        self.state = PageStore(self.arena).create()
        self.state.reserve(total)
        initial = mx.full((2, prefix_tokens, 256), 0.25, mx.bfloat16)
        self.keys = mx.full((2, append_tokens, 256), 2, mx.bfloat16)
        self.values = -self.keys
        for layer in range(10):
            self.state.write(layer, 0, initial, -initial)
        self.state.commit(prefix_tokens)
        mx.eval(self.keys, self.values)
        self.arena.complete()
        self.before = 0

    def reset(self) -> None:
        self.arena.complete()
        self.state.trim(self.prefix)
        self.state.reserve(self.prefix + self.count)
        self.before = self.arena.counters["kv_write_runs"]

    def invoke(self) -> None:
        for layer in range(10):
            offset = 0
            while offset < self.count:
                count = (
                    self.count - offset
                    if self.granularity == "runs"
                    else min(self.count - offset, 16 - (self.prefix + offset) % 16)
                )
                self.state.write(
                    layer,
                    self.prefix + offset,
                    self.keys[:, offset : offset + count],
                    self.values[:, offset : offset + count],
                )
                offset += count
        self.state.commit(self.prefix + self.count)

    def complete(self) -> None:
        self.arena.complete()

    def observe(self) -> Observation:
        expected = mx.concatenate(
            [mx.full((self.prefix,), 0.25, mx.bfloat16), mx.full((self.count,), 2, mx.bfloat16)]
        )[None, :, None]
        sums = []
        for layer in range(10):
            keys, values = self.state.read(layer)
            if not mx.all(keys == expected).item() or not mx.all(values == -expected).item():
                raise ValueError("KV append corrupted the prefix or appended values")
            sums.append(cast(float, mx.sum(keys.astype(mx.float32)).item()))
        return Observation(
            hashlib.sha256(json.dumps(sums).encode()).hexdigest(),
            {
                "kv_write_runs": self.arena.counters["kv_write_runs"] - self.before,
                "logical_append_bytes": 10 * (self.keys.nbytes + self.values.nbytes),
                "reserved_bytes": self.arena.budget.snapshot().reserved,
                "prefix_tokens": self.prefix,
                "append_tokens": self.count,
            },
        )

    def close(self) -> None:
        self.arena.complete()
        self.state.close()
        self.arena.close()


class BranchTrace:
    def __init__(self, *, page_size: int, prefix_tokens: int, branch_tokens: int):
        if min(page_size, prefix_tokens, branch_tokens) <= 0:
            raise ValueError("trace sizes must be positive")
        self.page_size, self.prefix_tokens, self.branch_tokens = (
            page_size,
            prefix_tokens,
            branch_tokens,
        )
        self.store: PageStore | None = None
        self.sequences: list[SequencePages] = []
        self.checkpoint: KVCheckpoint | None = None

    def reset(self) -> None:
        self.close()
        self.store = PageStore(
            KVArena(
                (LayerGeometry(4, 64, 64),),
                page_size=self.page_size,
                slab_pages=32,
                max_pages=1024,
                budget=MemoryBudget(512 << 20),
                dtype=mx.float32,
            )
        )

    def _append(self, state: SequencePages, tokens: int, value: float) -> None:
        end = state.length + tokens
        state.reserve(end)
        state.write(
            0, state.length, mx.full((4, tokens, 64), value), mx.full((4, tokens, 64), -value)
        )
        state.commit(end)

    def invoke(self) -> None:
        assert self.store is not None
        original = self.store.create()
        self.sequences.append(original)
        self._append(original, self.prefix_tokens, 1.0)
        self.checkpoint = original.checkpoint()
        original.close()
        for value in (2.0, 3.0):
            branch = self.store.create(self.checkpoint)
            self.sequences.append(branch)
            self._append(branch, self.branch_tokens, value)

    def complete(self) -> None:
        assert self.store is not None
        self.store.arena.complete()

    def observe(self) -> Observation:
        assert self.store is not None
        self.store.validate()
        outputs = []
        for value, branch in zip((2.0, 3.0), self.sequences[1:], strict=True):
            actual = branch.read(0)[0][0, :, 0].tolist()
            expected = [1.0] * self.prefix_tokens + [value] * self.branch_tokens
            if actual != expected:
                raise ValueError("branch changed shared prefix or returned incorrect KV")
            outputs.append(actual)
        digest = hashlib.sha256(json.dumps(outputs).encode()).hexdigest()
        return Observation(
            digest,
            {
                **self.store.arena.counters,
                "allocated_pages": len(self.store.arena.allocator.owned),
                "reserved_bytes": self.store.arena.budget.snapshot().reserved,
                "peak_reserved_bytes": self.store.arena.budget.snapshot().peak,
            },
        )

    def close(self) -> None:
        if self.checkpoint:
            self.checkpoint.close()
            self.checkpoint = None
        for sequence in self.sequences:
            sequence.close()
        self.sequences.clear()
        if self.store:
            self.store.arena.close()
            self.store = None
