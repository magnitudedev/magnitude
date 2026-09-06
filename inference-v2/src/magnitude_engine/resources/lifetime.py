from typing import Protocol


class Closable(Protocol):
    def close(self) -> None: ...
