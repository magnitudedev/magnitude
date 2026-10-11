"""Numerical and provenance controls, using small tensors rather than model inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from magnitude_references.precision import (checked_reference, compare_logits, make_error, reflect_error,
                                           stable_seed, subspace_correction)


def test_weight_draw_reused_across_fixtures_but_output_noise_is_not():
    x = torch.tensor([[1.0, 2.0], [3.0, -4.0]])
    weight = torch.zeros(3, 2)
    y, steps = x @ weight.T, torch.ones_like(weight)
    def draw(kind, fixture):
        return make_error(x, y, weight, steps, kind, 9, "model", "unit", fixture)[0]
    assert torch.equal(draw("uniform_weight", "a"), draw("uniform_weight", "b"))
    assert torch.equal(draw("gaussian_weight", "a"), draw("gaussian_weight", "b"))
    assert not torch.equal(draw("gaussian_output", "a"), draw("gaussian_output", "b"))
    assert torch.equal(draw("row_sign_weight", "a").square(), draw("uniform_weight", "a").square())
    assert stable_seed("model", "unit", 9, "weight") != stable_seed("model", "other", 9, "weight")


def test_weight_and_output_gaussians_share_marginal_variance_not_token_covariance():
    x = torch.tensor([[1.0], [2.0]])
    weight = torch.zeros(1, 1)
    y, steps = x @ weight.T, torch.full_like(weight, 12**0.5)
    samples = {}
    for kind in ("gaussian_weight", "gaussian_output"):
        samples[kind] = torch.stack([make_error(x, y, weight, steps, kind, seed, "model", "unit", "fixture")[0][:, 0]
                                    for seed in range(2048)]).double()
    shared = torch.cov(samples["gaussian_weight"].T)
    independent = torch.cov(samples["gaussian_output"].T)
    assert torch.allclose(shared, torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float64), rtol=0.1)
    assert torch.allclose(independent.diag(), torch.tensor([1., 4.], dtype=torch.float64), rtol=0.1)
    assert abs(float(independent[0, 1])) < 0.1


def test_distribution_change_and_added_reference_loss_have_distinct_signs():
    default, candidate = torch.tensor([[1., 0.]]), torch.tensor([[2., 0.]])
    result = compare_logits(default, candidate, candidate.numpy())
    assert result["default_kl"][0] > 0
    assert result["added_reference_kl"][0] < 0
    np.testing.assert_allclose(result["added_reference_kl"],
                               result["reference_candidate_kl"] - result["reference_default_kl"])
    assert compare_logits(default, default)["default_kl"][0] == 0
    with pytest.raises(ValueError, match="nonfinite"):
        compare_logits(default, torch.tensor([[float("nan"), 0.]]))


def test_orientation_reflection_preserves_local_summaries_and_changes_error():
    gen = torch.Generator().manual_seed(47)
    weight = torch.randn(12, 4, generator=gen)
    x = torch.randn(7, 4, generator=gen)
    y = x @ weight.T
    error = torch.randn(7, 12, generator=gen)
    reflected = reflect_error(error, weight, y)
    assert not torch.allclose(error, reflected)
    torch.testing.assert_close(error @ error.T, reflected @ reflected.T)
    torch.testing.assert_close(error.sum(1), reflected.sum(1))
    torch.testing.assert_close((error * y).sum(1), (reflected * y).sum(1))
    torch.testing.assert_close(error.mean(0).norm(), reflected.mean(0).norm())
    steps = torch.full_like(weight, 0.2)
    first, second = (subspace_correction(e, x, y, steps) for e in (error, reflected))
    assert first["shape_factor"] == pytest.approx(second["shape_factor"], rel=1e-6)
    assert first["power_ratios"] == pytest.approx(second["power_ratios"], rel=1e-6)


def test_subspace_rule_bounds_every_nonnegative_combination_of_its_projectors():
    gen = torch.Generator().manual_seed(54)
    x, w = torch.randn(7, 4, generator=gen).double(), torch.randn(12, 4, generator=gen).double()
    y = x @ w.T
    steps = torch.rand(w.shape, generator=gen).double() + 0.1
    error = torch.randn(7, 12, generator=gen).double()
    measured = subspace_correction(error, x, y, steps)
    m, n = error.shape
    covariance = torch.einsum("mk,tk,nk,nq->mntq", x, x, steps.square() / 12,
                              torch.eye(n, dtype=torch.float64)).reshape(m * n, m * n)
    directions = y / y.norm(dim=1, keepdim=True)
    projectors = {
        "total": torch.eye(m * n, dtype=torch.float64),
        "shared_tokens": torch.kron(torch.ones(m, m, dtype=torch.float64) / m,
                                      torch.eye(n, dtype=torch.float64)),
        "constant_channels": torch.kron(torch.eye(m, dtype=torch.float64),
                                          torch.ones(n, n, dtype=torch.float64) / n),
        "output_alignment": torch.block_diag(*(torch.outer(row, row) for row in directions)),
    }
    vector = error.flatten()
    for name, projector in projectors.items():
        ratio = (vector @ projector @ vector) / torch.trace(projector @ covariance)
        assert measured["power_ratios"][name] == pytest.approx(float(ratio), rel=1e-12)
    for coefficients in torch.rand(20, 4, generator=gen).double():
        sensitivity = sum(a * p for a, p in zip(coefficients, projectors.values()))
        actual = vector @ sensitivity @ vector
        expected = torch.trace(sensitivity @ covariance)
        assert actual <= measured["corrected_amplitude_ratio"]**2 * expected + 1e-10
    zero = subspace_correction(torch.zeros_like(error), x, y, steps)
    assert zero["corrected_amplitude_ratio"] == 0
    assert zero["shape_factor"] == 1
    unsupported = subspace_correction(error, x, y, torch.zeros_like(steps))
    assert unsupported["status"] == "outside_noise_support"


def test_reference_sanity_is_remeasured_not_trusted_from_a_flag(tmp_path):
    (tmp_path / "reference.json").write_text(json.dumps({"model": "model", "rendered_digest": "rendered",
        "implementation_digest": "implementation", "sanity": {"passed": True}}))
    header = {"model": "model", "rendered_digest": "rendered", "vocabulary": 2,
              "producer": "reference:test", "rows": [{"fixture_id": "fixture", "position": 0}]}
    (tmp_path / "logits.json").write_text(json.dumps(header))
    np.array([[0., 100.]], dtype="<f4").tofile(tmp_path / "logits.f32")
    fixture = SimpleNamespace(fixture_id="fixture", positions=lambda: [0], targets=lambda: [0])
    rendered = SimpleNamespace(model="model", digest="rendered", fixtures=[fixture])
    with pytest.raises(ValueError, match="sanity gate"):
        checked_reference(tmp_path, rendered, 2)
    np.array([[100., 0.]], dtype="<f4").tofile(tmp_path / "logits.f32")
    values, identity = checked_reference(tmp_path, rendered, 2)
    assert identity["remeasured_sanity"]["passed"]
    assert values["fixture"].shape == (1, 2)
    header["rows"][0]["position"] = 1
    (tmp_path / "logits.json").write_text(json.dumps(header))
    with pytest.raises(ValueError, match="rows do not match"):
        checked_reference(tmp_path, rendered, 2)


def test_cell_width_follows_payload_adjacency_of_each_float_type():
    from magnitude_references.precision import cell_width, lattice_ratio
    values = np.array([1.0, -1.0, 1.5, 3.0, 0.0, 1.001], dtype=np.float32)
    # At a binade boundary the gap below is half the gap above; elsewhere the cell is one unit in the last place.
    np.testing.assert_array_equal(cell_width(values, "f16"),
                                  [0.75 * 2.0**-10, 0.75 * 2.0**-10, 2.0**-10, 2.0**-9, 2.0**-24, 2.0**-10])
    np.testing.assert_array_equal(cell_width(values, "bf16"),
                                  [0.75 * 2.0**-7, 0.75 * 2.0**-7, 2.0**-7, 2.0**-6, 2.0**-133, 0.75 * 2.0**-7])
    np.testing.assert_array_equal(cell_width(values[:5], "f32"),
                                  [0.75 * 2.0**-23, 0.75 * 2.0**-23, 2.0**-23, 2.0**-22, 2.0**-149])
    # The smallest subnormal is adjacent to zero and to twice itself; a subnormal's cell is one step.
    np.testing.assert_array_equal(cell_width(np.array([2.0**-24, 3 * 2.0**-24], dtype=np.float32), "f16"), [2.0**-24] * 2)
    with pytest.raises(ValueError, match="no finite"):
        cell_width(np.array([65504.0], dtype=np.float32), "f16")
    with pytest.raises(ValueError, match="finite values"):
        cell_width(np.array([np.inf], dtype=np.float32), "f32")
    # Uniform rounding error over each cell has ratio one by construction.
    gen = np.random.default_rng(3)
    canonical = gen.standard_normal(200_000).astype(np.float32)
    error = (gen.random(canonical.shape) - 0.5) * cell_width(canonical, "bf16")
    assert lattice_ratio(error, canonical, "bf16") == pytest.approx(1.0, rel=0.01)
    assert lattice_ratio(2 * error, canonical, "bf16") == pytest.approx(2.0, rel=0.01)
    assert lattice_ratio(error, canonical, "f16") == pytest.approx(8.0, rel=0.02)


@pytest.mark.parametrize("kind,size,halves", [("Q4_0", 18, (0,)), ("Q8_0", 34, (0,)), ("Q4_K", 144, (0, 2)),
                                               ("Q5_K", 176, (0, 2)), ("Q6_K", 210, (208,))])
def test_storage_grid_decodes_each_integer_format_exactly(kind, size, halves):
    """Arbitrary stored bytes with finite half-precision scales, decoded by the GGUF library."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    from magnitude_references.gguf import TensorInfo, dequantize
    from magnitude_references.precision import storage_grid
    quantization = GGMLQuantizationType[kind]
    block = GGML_QUANT_SIZES[quantization][0]
    gen = np.random.default_rng(11)
    blocks = gen.integers(0, 256, (6, 512 // block, size), dtype=np.uint8)
    for offset in halves:
        scales = (gen.standard_normal(blocks.shape[:2]) * 0.01).astype(np.float16)
        blocks[..., offset:offset + 2] = scales[..., None].view(np.uint8)
    raw = blocks.reshape(-1)
    info = TensorInfo("unit", (512, 6), quantization, 0)
    package = SimpleNamespace(info=lambda name: info, raw=lambda name: raw)
    weight = torch.from_numpy(dequantize(info, raw))
    grid = storage_grid(package, "unit", weight)
    assert grid["codes"].shape == (6, 512 // grid["group"], grid["group"])
    decoded = grid["codes"].float() * grid["scale"][..., None] - grid["minimum"][..., None]
    torch.testing.assert_close(decoded.reshape(6, 512), weight, rtol=0, atol=1e-6 * float(weight.abs().max()))
    width = {"Q4_0": 16, "Q8_0": 256, "Q4_K": 16, "Q5_K": 32, "Q6_K": 64}[kind]
    assert int(grid["codes"].max()) - int(grid["codes"].min()) == width - 1
    with pytest.raises(ValueError, match="does not match"):
        storage_grid(package, "unit", weight * 1.01)
