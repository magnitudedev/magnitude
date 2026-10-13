import pytest
import torch

from magnitude_references.families import qwen35


def recurrence(q, k, v, g, beta):
    """The gated delta rule token by token (its definition)."""
    batch, steps, heads, width = k.shape
    state = torch.zeros(batch, heads, width, v.shape[-1], dtype=torch.float64)
    out = []
    for t in range(steps):
        state = state * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", state, k[:, t])) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        out.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("steps", [1, 63, 64, 150])
def test_chunked_gated_delta_is_the_recurrence(steps: int) -> None:
    generator = torch.Generator().manual_seed(steps)
    shape = (2, steps, 3)
    q = torch.nn.functional.normalize(torch.randn(*shape, 16, generator=generator), dim=-1) / 4
    k = torch.nn.functional.normalize(torch.randn(*shape, 16, generator=generator), dim=-1)
    v = torch.randn(*shape, 8, generator=generator)
    g = -torch.rand(*shape, generator=generator) * 2
    beta = torch.rand(*shape, generator=generator)
    expected = recurrence(*(x.double() for x in (q, k, v, g, beta)))
    torch.testing.assert_close(qwen35.gated_delta(q, k, v, g, beta).double(), expected, rtol=1e-4, atol=1e-5)


def test_grouped_inverts_the_converter_reorder() -> None:
    """The converter's `_reorder_v_heads` (grouped -> tiled), then `grouped`, is the identity."""
    k_heads, per_k, width = 2, 3, 4
    original = torch.arange(k_heads * per_k * width * 5).reshape(k_heads * per_k * width, 5)
    tiled = original.reshape(k_heads, per_k, width, 5).permute(1, 0, 2, 3).reshape(original.shape)
    assert torch.equal(qwen35.grouped(tiled, 0, k_heads, per_k, width), original)
