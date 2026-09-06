"""Target-sampled prefix acceptance, independent of the proposal implementation."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class AcceptedPrefix:
    count: mx.array
    bonus: mx.array

    @property
    def target_inputs_committed(self) -> mx.array:
        # The verification input is [anchor, proposals...]. The sampled bonus
        # is emitted now but becomes a target input in the following advancement.
        return self.count + 1


def accept_prefix(
    proposed: mx.array, target_samples: mx.array, stop_tokens: tuple[int, ...] = ()
) -> AcceptedPrefix:
    """Keep target work lazy; host publication is a separate generation boundary.

    Target samples use the ordinary sampler's position keys. Proposal probabilities
    are not used. Accept until the first unequal or terminal proposal, then publish
    that position's target sample as the bonus. This preserves seeded plain-decode
    output when target logits and their constraint/penalty contexts are identical.
    """
    if (
        proposed.ndim != 1
        or target_samples.ndim != 1
        or target_samples.shape[0] != proposed.shape[0] + 1
    ):
        raise ValueError("verification needs one target sample per proposal plus the bonus")
    if proposed.shape[0] == 0:
        return AcceptedPrefix(mx.array(0, dtype=mx.int32), target_samples[:1])
    matches = proposed == target_samples[: proposed.shape[0]]
    for token in stop_tokens:
        matches = matches & (proposed != token)
    count = mx.sum(mx.cumprod(matches.astype(mx.int32))).astype(mx.int32)
    return AcceptedPrefix(count, mx.take(target_samples, count).reshape(1))
