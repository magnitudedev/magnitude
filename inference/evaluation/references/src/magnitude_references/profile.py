"""Where a reference build's time goes, per batch: its wall seconds and the thread-seconds its workers spent reading
stored bytes in (the pages of the GGUF, touched as `Weights.stored` returns them) and dequantizing them, one JSON
line a batch on stderr. What is left of the wall is the forward (the matmuls, norms, routing), which overlaps the
workers' time, so a share is a worker's thread-seconds over the wall times `model.WORKERS`.

A measurement outside the numerical implementation (`digest.OUTSIDE`): it wraps the functions from here and
touches one byte of each page it reads, so it changes no value and no cached reference's identity. Installed by
`MAGNITUDE_REFERENCES_PROFILE=1` (`obtain --profile` sets it for the builds it starts)."""
from __future__ import annotations

import json
import sys
import threading
import time

import numpy as np

from magnitude_references import model, run

ENV = "MAGNITUDE_REFERENCES_PROFILE"
PAGE = 4096


def install():
    """Wrap the reads, the dequantization and the batches; returns the function that undoes it."""
    lock = threading.Lock()
    spent = {"read": 0.0, "dequantize": 0.0}
    stored, dequantize, run_batch = model.Weights.stored, model.dequantize, run.run_batch

    def add(kind: str, started: float) -> None:
        with lock:
            spent[kind] += time.perf_counter() - started

    def timed_stored(self, *arguments, **named):
        started = time.perf_counter()
        data = stored(self, *arguments, **named)
        if data.size:  # the page-in a decode would take, touched here so it is not counted as the decode's
            np.ascontiguousarray(data).reshape(-1)[::PAGE].max()
        add("read", started)
        return data

    def timed_dequantize(*arguments, **named):
        started = time.perf_counter()
        try:
            return dequantize(*arguments, **named)
        finally:
            add("dequantize", started)

    def timed_batch(reference, group, writer, sanity):
        with lock:
            spent.update(read=0.0, dequantize=0.0)
        started = time.perf_counter()
        run_batch(reference, group, writer, sanity)
        wall = time.perf_counter() - started
        with lock:
            line = {"profile": "batch", "fixtures": len(group), "wall_s": round(wall, 3), "workers": model.WORKERS,
                    "read_thread_s": round(spent["read"], 3), "dequantize_thread_s": round(spent["dequantize"], 3)}
        print(json.dumps(line), file=sys.stderr, flush=True)

    model.Weights.stored, model.dequantize, run.run_batch = timed_stored, timed_dequantize, timed_batch

    def uninstall() -> None:
        model.Weights.stored, model.dequantize, run.run_batch = stored, dequantize, run_batch

    return uninstall
