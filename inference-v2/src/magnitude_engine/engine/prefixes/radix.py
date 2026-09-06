"""Retain semantic prefixes under the injected retention policy."""

from .contracts import PrefixIndex, RetentionPolicy


class Radix(PrefixIndex):
    def __init__(self, *, retention: RetentionPolicy):
        super().__init__()
        self.retention = retention

    @property
    def enabled(self) -> bool:
        return self.retention.max_entries > 0 and self.retention.max_bytes != 0

    @property
    def retained_bytes(self) -> int:
        blocks = {}
        for entry in self._entries.values():
            for block in entry.checkpoint.retained_storage():
                blocks[id(block.owner)] = block.nbytes
        return sum(blocks.values())

    def retain(self, identity, checkpoint):
        retained = super().retain(identity, checkpoint)
        self.maintain()
        return retained

    def maintain(self) -> None:
        """Revisit deferred eviction on the execution owner after leases/pins retire."""
        while len(self) > self.retention.max_entries or (
            self.retention.max_bytes is not None and self.retained_bytes > self.retention.max_bytes
        ):
            candidates = tuple(c for c in self.eligible() if c.reclaimable)
            victims = self.retention.select(candidates)
            if not victims:
                break
            self.discard(victims)
