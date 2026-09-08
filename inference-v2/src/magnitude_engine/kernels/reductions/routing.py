"""Rounded softmax, stable route selection and native ordered normalization."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Source

ROUTING = Source("reductions/routing.metal")


@kernel(source=ROUTING)
def routes(logits, *, top_k, normalize):
    arguments = {"logits": logits}
    inputs = (logits.value.tensor,)
    (logits,) = inputs
    experts = logits.shape[-1] - 1
    if not 1 <= top_k <= min(16, experts) or not 1 <= experts <= 1024:
        raise ValueError("routing requires up to 1024 experts and 16 selected routes")
    shape = (*logits.shape[:-1], top_k)
    inferred = (
        Tensor(shape, mx.uint32),
        Tensor(shape, logits.dtype),
        Tensor((*logits.shape[:-1], 1), logits.dtype),
    )
    (logits,) = inputs
    experts = logits.shape[-1] - 1
    threads = max(32, (experts + 127) // 128 * 32)
    rows = logits.size // logits.shape[-1]
    return Dispatch(
        arguments,
        dict(zip(("indices", "scores", "shared"), inferred, strict=True)),
        Launch((threads, rows, 1), (threads, 1, 1)),
        (
            ("T", logits.dtype),
            ("EXPERTS", experts),
            ("TOPK", top_k),
            ("GROUPS", threads // 32),
            ("NORMALIZE", normalize),
        ),
    )


def select(logits: mx.array, top_k: int, normalize: bool) -> tuple[mx.array, mx.array, mx.array]:
    indices, scores, shared = routes(logits, top_k=top_k, normalize=normalize)
    return (indices, scores, shared)
