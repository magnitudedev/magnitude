"""Keep preparation and source scans outside cooperative inference measurements."""

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def preparation_reservation(timeout=1800):
    # This is the existing Ops/session-bench host reservation, without importing
    # their numerical runtime into coordinator and installation paths.
    path = Path(tempfile.gettempdir()) / f"magnitude-inference-measurement-{os.getuid()}.lock"
    with path.open("a+") as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "host preparation waited too long for active measurements"
                    ) from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
