"""The existing sampling formula at the model's actual vocabulary size."""

from functools import partial
from pathlib import Path

import numpy as np

import ops
from ops.lab import Configuration, Fixture, MeasurementProtocol
from engine import DevicePlan
from engine.models.qwen35.description import Geometry
from engine.operations.sampling import Draw


def sampling(geometry: Geometry, *, draw: Draw, plan: DevicePlan, store: Path,
             seed: int = 181) -> Configuration:
    signature = ops.Signature((
        ops.Argument(ops.TensorSpec((1, geometry.vocabulary), ops.DType.F32), "logits"),
        ops.Argument(ops.TensorSpec((1, 6), ops.DType.U32), "draws"),
    ))
    graph = ops.trace(ops.sample, signature)
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-8, 8, size=(1, geometry.vocabulary)).astype(np.float32)
    logits[:, ::127] = -np.inf
    draws = np.asarray([draw.words()], dtype=np.uint32)
    return Configuration(
        label=f"Synthetic Qwen sampling · {geometry.vocabulary} tokens · {draw.kind.name.lower()}",
        fixture=Fixture(graph, dict(zip(graph.inputs, (logits, draws), strict=True))),
        device=partial(ops.DeviceRuntime.open, plan),
        options=ops.CompileOptions(mode="decode"), store=store,
        protocol=MeasurementProtocol(absolute_tolerance=0, relative_tolerance=0),
        prepared_limit=1, reference_bytes=16 << 20,
    )
