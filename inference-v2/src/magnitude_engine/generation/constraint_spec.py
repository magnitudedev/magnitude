"""Serializable constraint input; no tokenizer or device runtime in the host contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintSpec:
    lark: str

    def __post_init__(self) -> None:
        if not isinstance(self.lark, str) or not self.lark or len(self.lark.encode()) > 1 << 20:
            raise ValueError("constraint grammar must contain between 1 byte and 1 MiB")


class ConstraintError(ValueError):
    """A request grammar cannot be constructed for this residency."""
