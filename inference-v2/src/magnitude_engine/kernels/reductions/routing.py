"""Rounded softmax, stable route selection and native ordered normalization."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import Tensor, signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Source
from ..core.primitive import Primitive

ROUTING = Source("reductions/routing.metal")


@dataclass(frozen=True)
class Routes(Primitive):
    """Select largest rounded (probability, index) pairs, emitted in ascending order."""

    top_k: int
    normalize: bool

    def infer(self, inputs):
        (logits,) = inputs
        experts = logits.shape[-1] - 1
        if not 1 <= self.top_k <= min(16, experts) or not 1 <= experts <= 1024:
            raise ValueError("routing requires up to 1024 experts and 16 selected routes")
        shape = (*logits.shape[:-1], self.top_k)
        return (
            Tensor(shape, mx.uint32),
            Tensor(shape, logits.dtype),
            Tensor((*logits.shape[:-1], 1), logits.dtype),
        )

    def lower(self, inputs):
        (logits,) = inputs
        experts = logits.shape[-1] - 1
        threads = max(32, ((experts + 127) // 128) * 32)
        rows = logits.size // logits.shape[-1]
        return Kernel(
            signature(("logits",), inputs),
            signature(("indices", "scores", "shared"), self.infer(inputs)),
            ROUTING,
            Launch((threads, rows, 1), (threads, 1, 1)),
            (
                ("T", logits.dtype),
                ("EXPERTS", experts),
                ("TOPK", self.top_k),
                ("GROUPS", threads // 32),
                ("NORMALIZE", self.normalize),
            ),
        ).bind()


def select(logits: mx.array, top_k: int, normalize: bool) -> tuple[mx.array, mx.array, mx.array]:
    indices, scores, shared = Routes(top_k, normalize)(logits)
    return indices, scores, shared
