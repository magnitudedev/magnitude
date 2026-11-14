import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import ROWS_EXACT, VOCAB, assert_rows_hold, storage_step, write_rendered
from magnitude_references import model as model_module
from magnitude_references.cli import main
from magnitude_references.contract import ContractError, read_logits
from magnitude_rig.host.identity import sha256
from magnitude_references.families import open_reference
from magnitude_references.run import Options, build_reference
from magnitude_references.sanity import Bounds

CPU = torch.device("cpu")
# The runs name their build: a rig host runs these tests in a staged tree, which is no git checkout (`record.build`)
BUILD = {"commit": "a" * 40, "dirty_digest": None}


def sequential(start: int, length: int) -> list[int]:
    return [(start + i) % VOCAB for i in range(length)]


FIXTURES = [
    (sequential(0, 40), [[10, 20], [30, 40]]),  # the last span ends at the last token
    (sequential(7, 25), [[5, 9]]),
    (sequential(3, 61), [[1, 2], [2, 6], [50, 60]]),  # the earliest scorable token, adjacent spans
]


def test_contract_logits_output(tmp_path: Path, tiny_llama: Path) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    output = tmp_path / "out"
    record = build_reference(rendered, artifacts, output, Options(CPU, build=BUILD, batch_tokens=64))
    header, values = read_logits(output, "logits")
    assert header["producer"] == "reference:llama"
    assert header["model"] == "tiny:gguf:f32"
    assert header["rendered_digest"] == sha256(rendered)
    assert header["vocabulary"] == VOCAB
    expected_rows = [{"fixture_id": f"test/{index}", "position": position}
                     for index, (_, spans) in enumerate(FIXTURES) for start, end in spans
                     for position in range(start - 1, end - 1)]
    assert header["rows"] == expected_rows
    assert values.shape == (len(expected_rows), VOCAB)
    assert (output / "logits.f32").stat().st_size == len(expected_rows) * VOCAB * 4
    assert set(record) == {"model", "rendered_digest", "family", "implementation_digest", "platform", "dtype", "sanity"}
    assert record["family"] == "llama" and record["dtype"] == "float32"
    assert set(record["sanity"]) == {"perplexity", "top1_accuracy", "bounds", "passed", "fixtures"}
    assert [fixture["fixture_id"] for fixture in record["sanity"]["fixtures"]] == ["test/0", "test/1", "test/2"]
    assert json.loads((output / "reference.json").read_text()) == record
    result = json.loads((output / "result.json").read_text())
    assert result["schema"] == "evaluation-result/1" and result["tool"] == "reference"
    assert result["build"] == BUILD
    assert result["rendered_digest"] == header["rendered_digest"]
    assert result["artifacts"]["target"] == str(tiny_llama)

    # Every row equals an unbatched, unpadded forward of its own fixture (padding and batching are
    # exact), at position i - 1 for scored token i.
    reference = open_reference(tiny_llama, CPU, cache=True)
    row = 0
    for tokens, spans in FIXTURES:
        with torch.no_grad():
            full = reference.forward(torch.tensor([tokens]))[0].numpy()
        for start, end in spans:
            np.testing.assert_allclose(values[row:row + end - start], full[start - 1:end - 1], rtol=1e-5, atol=1e-5)
            row += end - start


def test_batching_does_not_change_rows(tmp_path: Path, tiny_llama: Path) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    build_reference(rendered, artifacts, tmp_path / "one", Options(CPU, build=BUILD, batch_tokens=1))
    build_reference(rendered, artifacts, tmp_path / "all", Options(CPU, build=BUILD, batch_tokens=10_000))
    build_reference(rendered, artifacts, tmp_path / "moved",
                    Options(CPU, build=BUILD, batch_tokens=10_000, variant="perturbed:1"))
    (_, one), (_, together), (_, moved) = (read_logits(tmp_path / name, "logits") for name in ("one", "all", "moved"))
    assert_rows_hold(one, together, storage_step(together, moved))


def test_a_row_is_the_same_however_many_rows_share_its_products() -> None:
    """A row's linear and attention output has the same bits alone (M = 1, padded to the tile, never the small-M GEMM
    path), in one full row tile, and among 2 * ROW_TILE + 3 rows, where the platform promises it."""
    if not ROWS_EXACT:
        pytest.skip("this platform's BLAS rounds a GEMM row by its place in the call (digest.ROWS_EXACT)")
    tile = model_module.ROW_TILE
    rows = 2 * tile + 3
    g = torch.Generator().manual_seed(0)
    x, weight = torch.randn(rows, 96, generator=g), torch.randn(40, 96, generator=g)
    q, k = torch.randn(1, 4, rows, 16, generator=g), torch.randn(1, 2, rows, 16, generator=g)
    v = torch.randn(1, 2, rows, 8, generator=g)
    sinks = torch.randn(4, generator=g)
    positions = torch.arange(rows)

    def attend(count: int) -> torch.Tensor:
        visible = model_module.visibility(positions[:count], positions[:count])
        return model_module.attention(q[:, :, :count], k[:, :, :count], v[:, :, :count], visible, 0.25, sinks)

    whole, attended = model_module.linear(x, weight), attend(rows)
    for count in (1, tile, rows):
        assert torch.equal(model_module.linear(x[:count], weight), whole[:count]), count
        assert torch.equal(model_module.linear(x[count - 1:count], weight), whole[count - 1:count]), count
        assert torch.equal(attend(count), attended[:, :count]), count


@pytest.mark.parametrize("rows_exact", (True, False))
def test_products_run_in_tiles_exactly_where_rows_are_exact(monkeypatch, rows_exact: bool) -> None:
    """`linear` and `products` call their GEMMs in ROW_TILE-row (and KEY_TILE-key) tiles on a platform that makes a
    row's bits its own, and whole, as the released code does, on one that does not (`model.tiled`)."""
    from torch.overrides import TorchFunctionMode

    class Calls(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.rows: list[tuple[int, int]] = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is torch.nn.functional.linear:
                self.rows.append((args[0].shape[-2], args[1].shape[-2]))
            elif getattr(func, "__name__", "") in ("__matmul__", "matmul"):
                self.rows.append((args[0].shape[-2], args[1].shape[-1]))
            return func(*args, **(kwargs or {}))

    platform = model_module.digest.platform
    monkeypatch.setattr(model_module.digest, "platform", lambda device: {**platform(device), "rows_exact": rows_exact})
    model_module.tiled.cache_clear()
    try:
        g = torch.Generator().manual_seed(0)
        rows, keys = model_module.ROW_TILE + 3, model_module.KEY_TILE + 5
        x, weight = torch.randn(1, rows, 32, generator=g), torch.randn(24, 32, generator=g)
        with Calls() as linear:
            model_module.linear(x, weight)
        with Calls() as products:
            model_module.products(x, torch.randn(1, keys, 32, generator=g))
    finally:
        model_module.tiled.cache_clear()
    tile, key = model_module.ROW_TILE, model_module.KEY_TILE
    if rows_exact:
        assert linear.rows == [(tile, 24)] * 2
        assert products.rows == [(tile, key)] * 4
    else:
        assert linear.rows == [(rows, 24)]
        assert products.rows == [(rows, keys)]


def test_sane_model_passes_and_garbage_fails(tmp_path: Path, sharp_llama: Path, tiny_llama: Path) -> None:
    """The sharp model predicts the next id: sequential text is its distribution and passes; random
    text, or any text through a random model, is a broken regime and fails."""
    rendered, artifacts = write_rendered(tmp_path, sharp_llama, FIXTURES)
    sane = build_reference(rendered, artifacts, tmp_path / "sane", Options(CPU, build=BUILD))["sanity"]
    assert sane["passed"] and sane["perplexity"] < 2 and sane["top1_accuracy"] > 0.9

    rng = np.random.default_rng(5)
    noise = [(rng.integers(0, VOCAB, 40).tolist(), [[10, 39]])]
    (tmp_path / "noise").mkdir()
    rendered, artifacts = write_rendered(tmp_path / "noise", sharp_llama, noise)
    noise_out = build_reference(rendered, artifacts, tmp_path / "noise-out", Options(CPU, build=BUILD))
    assert not noise_out["sanity"]["passed"]

    (tmp_path / "random").mkdir()
    rendered, artifacts = write_rendered(tmp_path / "random", tiny_llama, FIXTURES)
    broken = build_reference(rendered, artifacts, tmp_path / "random-out", Options(CPU, build=BUILD))["sanity"]
    assert not broken["passed"] and broken["perplexity"] > 20


def test_cli_exit_status_follows_the_gate(tmp_path: Path, tiny_llama: Path) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    (tmp_path / "build.json").write_text(json.dumps(BUILD))
    common = ["run", "--rendered", str(rendered), "--artifacts", str(artifacts), "--device", "cpu",
              "--build-identity", str(tmp_path / "build.json")]
    assert main([*common, "--output", str(tmp_path / "strict")]) == 1
    assert json.loads((tmp_path / "strict" / "reference.json").read_text())["sanity"]["passed"] is False
    assert main([*common, "--output", str(tmp_path / "loose"), "--perplexity-max", "1e9", "--top1-min", "0"]) == 0
    loose = json.loads((tmp_path / "loose" / "reference.json").read_text())["sanity"]
    assert loose["passed"] is True and loose["bounds"] == Bounds(1e9, 0.0).to_json()


@pytest.mark.parametrize("change, message", [
    (lambda line: line | {"scored_spans": [[5, 10], [8, 12]]}, "overlapping"),
    (lambda line: line | {"scored_spans": [[5, 41]]}, "outside"),
    (lambda line: line | {"scored_spans": [[0, 4]]}, "token 0"),
    (lambda line: line | {"scored_spans": []}, "no scored span"),
    (lambda line: line | {"model": "other:gguf:q4"}, "different model"),
    (lambda line: {k: v for k, v in line.items() if k != "template_sha256"}, "missing"),
])
def test_contract_violations_are_refused(tmp_path: Path, tiny_llama: Path, change, message: str) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    lines = [json.loads(text) for text in rendered.read_text().splitlines()]
    lines[0] = change(lines[0])
    rendered.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ContractError, match=message):
        build_reference(rendered, artifacts, tmp_path / "out", Options(CPU, build=BUILD))


def test_light_modules_do_not_import_torch() -> None:
    import subprocess
    import sys
    code = ("import sys, magnitude_references, magnitude_references.digest, magnitude_references.contract, "
            "magnitude_references.sanity; assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_supplied_build_identity_is_recorded_verbatim(tmp_path: Path, tiny_llama: Path) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    identity = {"commit": "a" * 40, "dirty_digest": None}
    (tmp_path / "build.json").write_text(json.dumps(identity))
    main(["run", "--rendered", str(rendered), "--artifacts", str(artifacts), "--device", "cpu",
          "--output", str(tmp_path / "out"), "--build-identity", str(tmp_path / "build.json")])
    assert json.loads((tmp_path / "out" / "result.json").read_text())["build"] == identity


def test_rendered_for_another_target_is_refused(tmp_path: Path, tiny_llama: Path) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    record = json.loads(artifacts.read_text())
    artifacts.write_text(json.dumps(record | {"target_sha256": "f" * 64}))
    with pytest.raises(ContractError, match="target_sha256"):
        build_reference(rendered, artifacts, tmp_path / "out", Options(CPU, build=BUILD))
