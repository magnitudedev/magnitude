"""Select prefix-retention victims from eligible checkpoints."""

from dataclasses import dataclass

from .contracts import RetentionPolicy
from .index import Checkpoint


@dataclass(frozen=True)
class LeastRecentlyUsed(RetentionPolicy):
    max_entries: int
    max_bytes: int | None

    def select(self, eligible: tuple[Checkpoint, ...]) -> tuple[Checkpoint, ...]:
        return eligible[:1]
