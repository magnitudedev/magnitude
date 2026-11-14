"""`unit_selection`, the references' one definition of a pooled indexer's selected rows, against a direct reading of
its rule: a row reads the rows of its best min(top, count) complete units, ties to the lower unit, and its tail."""

from __future__ import annotations

import pytest
import torch

from magnitude_references.model import unit_selection


def read_rows(slots: torch.Tensor, rows: int) -> torch.Tensor:
    """The rows [B, T, rows] slots [B, T, K] name, each named once."""
    named = slots.masked_fill(slots < 0, rows)
    taken = named.sort(-1).values
    assert not ((taken[..., 1:] == taken[..., :-1]) & (taken[..., 1:] < rows)).any(), "a row named twice"
    return torch.zeros(*slots.shape[:2], rows + 1, dtype=torch.bool).scatter_(2, named, True)[..., :rows]


def brute_force(scores: torch.Tensor, ends: list[int], top: int, rate: int, rows: int) -> torch.Tensor:
    batch, steps, _ = scores.shape
    out = torch.zeros(batch, steps, rows, dtype=torch.bool)
    for b in range(batch):
        for t in range(steps):
            count = ends[t] // rate
            # Rank each complete unit by (score descending, unit ascending).
            order = sorted(range(count), key=lambda u: (-scores[b, t, u].item(), u))
            for u in order[:min(top, count)]:
                out[b, t, u * rate:(u + 1) * rate] = True
            out[b, t, count * rate:ends[t]] = True
    return out


@pytest.mark.parametrize("rate", [1, 4])
def test_unit_selection_reads_the_best_units_and_the_tail(rate: int) -> None:
    generator = torch.Generator().manual_seed(7)
    steps, top = 29, 3
    # Token rows at rate 4; at rate 1 a pooled source's rows, one per two tokens.
    rows = steps if rate > 1 else steps // 2
    units = rows // rate
    # Integer scores in a small range: many ties, which go to the lower unit.
    scores = torch.randint(0, 4, (2, steps, units), generator=generator).float()
    positions = torch.arange(steps)
    ends = positions + 1 if rate > 1 else (positions + 1) // 2
    slots = unit_selection(scores, ends, top, rate)
    assert slots.shape[-1] == min(top, units) * rate + rate - 1
    got = read_rows(slots, rows)
    want = brute_force(scores, ends.tolist(), top, rate, rows)
    assert torch.equal(got, want)
    # A row with at most `top` complete units reads every visible row.
    dense = ends // rate <= top
    visible = torch.arange(rows)[None, :] < ends[:, None]
    assert torch.equal(got[:, dense], visible[dense].expand(2, -1, -1))


def test_unit_selection_before_the_first_unit_reads_only_the_tail() -> None:
    scores = torch.zeros(1, 3, 0)
    positions = torch.arange(3)
    got = read_rows(unit_selection(scores, positions + 1, 2, 4), 3)
    assert torch.equal(got[0], torch.arange(3)[None, :] <= positions[:, None])
