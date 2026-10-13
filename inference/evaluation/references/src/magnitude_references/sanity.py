"""The reference sanity gate: is the dense model in a sane regime on this exact input?

A reference is only worth comparing against when the dense model itself predicts the fixture well.
On out-of-distribution input (for example an instruct model fed untemplated raw text) the dense
distributions are garbage or near-flat, and any comparison against them measures noise
(`handoffs/26-10-08/gemma-kv-investigation-postmortem.md`: Gemma 4 12B on raw text had dense
perplexity 1,232; on chat-templated text 1.90).

Measured on every emitted row (row at position p, actual next token tokens[p+1], i.e. every scored
token):
- perplexity: exp(mean negative log-likelihood of the actual next token);
- top-1 accuracy: fraction of rows whose argmax is the actual next token.

Default bounds, from Gemma 4 QAT references (float32, MPS), per fixture:

| input                                                          | perplexity | top-1 |
|----------------------------------------------------------------|-----------:|------:|
| 12B, chat template, the model's own assistant turns            |        2.1 |  0.84 |
| 12B, chat template, Moby-Dick continuation (human prose)       |     61-129 |  0.31-0.35 |
| E2B, chat template, Moby-Dick continuation (prose-continue set)|     51-143 |  0.21-0.33 |
| 12B, no template, the model's own answers                      |        5.2 |  0.67 |
| 12B, no template, Moby-Dick (the postmortem's broken regime)   |  1437-2712 |  0.16-0.20 |

Gemma 4 instruct models are sharply peaked: on human literary prose their predictions are coherent
English but miss the author's word choice with near-zero probability, which is a sane regime with a
high perplexity and a low top-1. Perplexity separates the regimes by an order of magnitude; top-1
does not (the sane and broken ranges nearly touch), so it only guards against degenerate output:
- `perplexity_max` 300.0: about 2x above the hardest sane prose measured and 4.8x below the best
  broken-regime fixture;
- `top1_min` 0.10: below every sane fixture measured, above a near-flat or constant prediction.

Both must hold for every fixture on its own rows, so one broken fixture fails the reference however
sane the rest are; the aggregate over the set is recorded too. Bounds can be overridden per run and
are recorded with the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # torch loads only where rows are measured, so the bounds stay cheap to import
    import torch

PERPLEXITY_MAX = 300.0
TOP1_MIN = 0.10


@dataclass(frozen=True)
class Bounds:
    perplexity_max: float = PERPLEXITY_MAX
    top1_min: float = TOP1_MIN

    def to_json(self) -> dict:
        return {"perplexity_max": self.perplexity_max, "top1_min": self.top1_min}


class Accumulator:
    """Running negative log-likelihood and top-1 hits over scored rows."""

    def __init__(self):
        self.nll = 0.0
        self.hits = 0
        self.count = 0

    def add(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """logits [n, V] (any device), targets [n] int64: the actual next token of each row."""
        import torch
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(f"{logits.shape[0]} rows for {targets.shape[0]} targets")
        if logits.shape[0] == 0:
            return
        # float64 on the CPU (MPS has no float64), a block of rows at a time to bound memory.
        targets = targets.to("cpu")
        for start in range(0, targets.shape[0], 256):
            values = logits[start:start + 256].to("cpu", torch.float64)
            chosen = targets[start:start + 256]
            self.nll -= float(torch.log_softmax(values, dim=-1).gather(1, chosen[:, None]).sum())
            self.hits += int((values.argmax(-1) == chosen).sum())
        self.count += int(targets.shape[0])

    def merge(self, other: "Accumulator") -> None:
        self.nll += other.nll
        self.hits += other.hits
        self.count += other.count

    def measure(self, bounds: Bounds) -> dict:
        """{perplexity, top1_accuracy, passed} of the rows so far."""
        if self.count == 0:
            raise ValueError("no scored rows: the sanity gate cannot be evaluated")
        perplexity = math.exp(self.nll / self.count)
        top1 = self.hits / self.count
        passed = bool(np.isfinite(perplexity)) and perplexity <= bounds.perplexity_max and top1 >= bounds.top1_min
        return {"perplexity": perplexity, "top1_accuracy": top1, "passed": passed}


class Gate:
    """The sanity gate over a fixture set: each fixture's rows measured on their own, and the set."""

    def __init__(self):
        self.fixtures: dict[str, Accumulator] = {}

    def add(self, fixture_id: str, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.fixtures.setdefault(fixture_id, Accumulator()).add(logits, targets)

    @property
    def count(self) -> int:
        return sum(accumulator.count for accumulator in self.fixtures.values())

    def result(self, bounds: Bounds) -> dict:
        """The `sanity` record: aggregate perplexity and top-1, bounds, per-fixture values, and
        `passed` only when every fixture passes on its own."""
        total = Accumulator()
        fixtures = []
        for fixture_id, accumulator in self.fixtures.items():
            total.merge(accumulator)
            fixtures.append({"fixture_id": fixture_id, **accumulator.measure(bounds)})
        aggregate = total.measure(bounds)
        return {"perplexity": aggregate["perplexity"], "top1_accuracy": aggregate["top1_accuracy"],
                "bounds": bounds.to_json(), "passed": all(fixture["passed"] for fixture in fixtures),
                "fixtures": fixtures}

    @staticmethod
    def failing(sanity: dict) -> list[str]:
        return [fixture["fixture_id"] for fixture in sanity["fixtures"] if not fixture["passed"]]
