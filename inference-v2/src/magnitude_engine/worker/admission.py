"""Admission and lifetime of numerical payloads received by a model worker."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

from magnitude_engine.resources.budget import Reservation

from .framing import Frame, read_header


@dataclass
class AdmittedFrame:
    frame: Frame
    lease: Reservation | None = None
    error: MemoryError | None = None

    def close(self) -> None:
        self.frame.close()
        if self.lease is not None:
            self.lease.close()
            self.lease = None


def read_admitted_frame(
    stream: BinaryIO,
    generation: str,
    *,
    reserve: Callable[[dict, int], Reservation],
) -> AdmittedFrame:
    header = read_header(stream, generation)
    lease = None
    if header.lengths:
        try:
            lease = reserve(header.message, sum(header.lengths))
        except MemoryError as error:
            header.discard(stream)
            return AdmittedFrame(Frame(header.generation, header.message), error=error)
    try:
        return AdmittedFrame(header.read(stream), lease)
    except BaseException:
        if lease is not None:
            lease.close()
        raise
