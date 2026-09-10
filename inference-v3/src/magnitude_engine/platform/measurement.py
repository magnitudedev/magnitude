"""Host observation and exclusive measurement ownership."""

import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_owner: tuple[int, int] | None = None


def clock_ns() -> int:
    return time.perf_counter_ns()


@contextmanager
def exclusive_measurement():
    import fcntl

    global _owner
    identity = os.getpid(), threading.get_ident()
    if _owner == identity:
        # The outer run includes binding/loading; nested component collection
        # borrows its ownership rather than opening a competing file lock.
        yield
        return
    if _owner is not None:
        raise RuntimeError("another thread owns this process's inference measurement")

    # Shared with the verbatim v2/v3 session-bench machine lock.
    path = Path(tempfile.gettempdir()) / f"magnitude-inference-measurement-{os.getuid()}.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another inference benchmark owns this machine") from error
        _owner = identity
        try:
            yield
        finally:
            _owner = None
            fcntl.flock(stream, fcntl.LOCK_UN)
