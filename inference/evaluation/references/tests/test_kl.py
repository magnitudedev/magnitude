"""`magnitude-references kl`: a build against the cached reference, a pair of builds, a build
against a yardstick or the storage-resolution bar, and a slice judged without the sanity gate."""

import json

import logits_fake
import numpy as np
import pytest
from magnitude_rig.host.identity import sha256
from test_cache import obtain, references_cli, rendered  # noqa: F401 (fixtures)

from magnitude_references import kl, quality
from magnitude_references.cache import ReferenceRefused


def _logits(directory, rendered_file, values):
    directory.mkdir()
    (directory / "rendered.jsonl").write_text(rendered_file.read_text())
    lines = [json.loads(line) for line in rendered_file.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(directory, "harness", sha256(rendered_file), rows, values(rows))


def _acquired(tmp_path, sliced=False):
    """`tmp_path`'s acquisition object, of a target that is a `catalog-tool slice` output when `sliced`."""
    target = logits_fake.write_target(tmp_path / "model.gguf")
    provenance = target.with_suffix(".json")
    if sliced:
        provenance.write_text(json.dumps({"main_layers": [0], "sha256": sha256(target)}))
    elif provenance.exists():
        provenance.unlink()
    (tmp_path / "artifacts.json").write_text(json.dumps({
        "model": logits_fake.MODEL, "target": str(target), "target_sha256": sha256(target),
        "drafter": None, "projector": None}))


@pytest.fixture(autouse=True)
def acquired(tmp_path):
    _acquired(tmp_path)


def test_a_pair_is_judged_against_the_reference_and_each_other(tmp_path, rendered, references_cli):  # noqa: F811
    built = obtain(rendered, tmp_path)
    lines = [json.loads(line) for line in rendered.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(built.directory, "reference", sha256(rendered), rows, logits_fake.truth(rows))
    noisy = np.random.default_rng(1).normal(0, 0.3, logits_fake.truth(rows).shape)
    _logits(tmp_path / "base", rendered, lambda r: logits_fake.truth(r) + noisy)
    _logits(tmp_path / "candidate", rendered, lambda r: logits_fake.truth(r) + noisy)
    summary = kl.run(tmp_path / "candidate", tmp_path / "artifacts.json", tmp_path / "build.json",
                     tmp_path / "cache", tmp_path / "out" / "kl.json", tmp_path / "base", None, None)
    assert summary["reference"]["cached"] is True
    assert summary["gate"] == {"mode": "model"}
    assert summary["builds_identical"] is True
    assert {change["verdict"] for change in summary["changes"]} == {"within noise"}
    assert summary["build"]["mean_kl"] > 0
    assert not list(tmp_path.glob("*/logits.f32"))  # judged logits are deleted
    assert json.loads((tmp_path / "out" / "kl.json").read_text())["build"] == summary["build"]


def _judge(tmp_path, mode, yardstick=None, candidate="candidate"):
    _acquired(tmp_path, sliced=mode == "slice")
    return kl.run(tmp_path / candidate, tmp_path / "artifacts.json", tmp_path / "build.json",
                  tmp_path / "cache", tmp_path / "out" / "kl.json", None, None, yardstick)


def _noise(scale):
    return np.random.default_rng(4).normal(0, scale, (2 * logits_fake.SCORED, logits_fake.VOCABULARY))


@pytest.fixture
def failed_slice(tmp_path, rendered, references_cli):  # noqa: F811
    """A slice: its reference fails the sanity gate, and its logits are the fake's truth, the
    same in every build."""
    _, behaviour = references_cli
    behaviour["passed"] = False
    _logits(tmp_path / "candidate", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    return references_cli


def test_a_slice_is_judged_without_the_sanity_gate_and_refused_under_it(tmp_path, failed_slice):
    calls, _ = failed_slice
    with pytest.raises(ReferenceRefused, match="sanity gate"):
        _judge(tmp_path, "model")
    summary = _judge(tmp_path, "slice")
    assert summary["gate"] == {"mode": "slice"}
    assert summary["reference"]["sanity"]["passed"] is False  # recorded, not applied
    assert summary["reference"]["admission"]["agreement"] == "CPU-deterministic"
    assert summary["build"]["mean_kl"] > 0
    assert len(calls) == 1  # built once: the cached reference is judged as built


def test_a_slice_is_its_target_s_provenance_record(tmp_path):
    assert kl.mode_of(tmp_path / "artifacts.json") == "model"
    _acquired(tmp_path, sliced=True)
    assert kl.mode_of(tmp_path / "artifacts.json") == "slice"
    (tmp_path / "model.json").write_text(json.dumps({"sha256": "0" * 64}))  # a record of another file
    assert kl.mode_of(tmp_path / "artifacts.json") == "model"


def test_a_held_lone_slice_is_judged_as_a_slice_and_a_held_whole_set_as_a_model(tmp_path):
    """`kl --checkpoint`'s acquisition (`host/checkpoint.py`) names its target by the target file's own SHA-256, as
    a catalog acquisition does: a lone slice beside its provenance record is a slice, a split set a model."""
    from magnitude_rig.host import checkpoint
    held = tmp_path / "slice"
    held.mkdir()
    target = logits_fake.write_target(held / "GLM-layers.gguf")
    (held / "GLM-layers.json").write_text(json.dumps({"main_layers": [0], "sha256": sha256(target)}))
    checkpoint.acquire(str(held), str(tmp_path / "slice.json"))
    assert kl.mode_of(tmp_path / "slice.json") == "slice"
    whole = tmp_path / "whole"
    whole.mkdir()
    for index in (1, 2):
        (whole / f"GLM-{index:05d}-of-00002.gguf").write_bytes(b"GGUF shard %d" % index)
    artifacts = checkpoint.acquire(str(whole), str(tmp_path / "whole.json"))
    assert kl.mode_of(tmp_path / "whole.json") == "model"
    assert artifacts["target_sha256"] == sha256(artifacts["target"])  # what `precision.run` checks the target by


def test_a_build_is_judged_against_a_yardstick_on_the_same_tokens(tmp_path, rendered, failed_slice):  # noqa: F811
    _logits(tmp_path / "level", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    level = _judge(tmp_path, "slice", tmp_path / "level")
    assert level["yardstick"]["mean_kl"] == level["build"]["mean_kl"]
    assert level["within_yardstick"] is True
    assert (tmp_path / "level" / "logits.f32").is_file()  # the yardstick is the caller's
    _logits(tmp_path / "finer", rendered, lambda r: logits_fake.truth(r) + _noise(0.1))
    _logits(tmp_path / "drifted", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    drifted = _judge(tmp_path, "slice", tmp_path / "finer", candidate="drifted")
    assert drifted["within_yardstick"] is False
    assert {change["metric"]: change["verdict"] for change in drifted["yardstick_changes"]}["mean_kl"] == "regressed"


def test_a_build_is_judged_against_the_storage_resolution_bar(tmp_path, rendered, references_cli):  # noqa: F811
    """The perturbed references, each draw built and cached beside the reference, are the bar: a build at twice a
    draw's logit noise (about 4 times its KL) is within 64 times the draws' largest KL at every row, one at 16 times
    its noise (about 256 times) is not."""
    calls, behaviour = references_cli
    noise = {seed: np.random.default_rng(seed).normal(0, 0.01, (2 * logits_fake.SCORED, logits_fake.VOCABULARY))
             for seed in (3, 4)}
    for seed, values in noise.items():
        behaviour["variants"][f"perturbed:{seed}"] = lambda rows, v=values: logits_fake.truth(rows) + v.astype(np.float32)

    def judged(scale, name):
        _logits(tmp_path / name, rendered, lambda r: logits_fake.truth(r) + (scale * noise[3]).astype(np.float32))
        return kl.run(tmp_path / name, tmp_path / "artifacts.json", tmp_path / "build.json", tmp_path / "cache",
                      tmp_path / "out" / f"{name}.json", None, None, None, None, (3, 4))

    near = judged(2, "near")
    bar = near["resolution"]
    assert {draw["reference"]["implementation_digest"] for draw in bar["draws"]}.isdisjoint(
        {near["reference"]["implementation_digest"]}) and len(bar["draws"]) == 2
    assert all(draw["positions"] == near["build"]["positions"] and draw["median_kl"] > 0 for draw in bar["draws"])
    assert bar["rows"] == {"judgeable": near["build"]["positions"], "unjudgeable": 0}
    assert bar["within"] is True and bar["over"] == [] and bar["ratio"]["max"] < 4 / 64 * 1.5
    assert set(bar["worst"]) == {"fixture_id", "position"}
    far = judged(16, "far")
    assert far["resolution"]["within"] is False and far["resolution"]["over"]
    assert len(calls) == 3  # the reference and the two draws, each built once on the CPU
    with pytest.raises(ValueError, match="perturbs the model's numerics"):
        kl.run(tmp_path / "far", tmp_path / "artifacts.json", tmp_path / "build.json", tmp_path / "cache",
               tmp_path / "out" / "x.json", None, None, None, "activations", (3, 4))


VOCABULARY = 50


def _draws(exact, count, seed, scale):
    """`count` independent perturbations of `exact`'s logits, each value moved by Gaussian noise of `scale`."""
    rng = np.random.default_rng(seed)
    return [exact + rng.normal(0, scale, exact.shape).astype(np.float32) for _ in range(count)]


def _exact(rows, seed=3):
    """Reference logits over `VOCABULARY` from confident to flat rows: H from near 0 to near ln V."""
    rng = np.random.default_rng(seed)
    peak = np.linspace(14.0, 0.0, rows, dtype=np.float32)[:, None] * np.eye(VOCABULARY, dtype=np.float32)[0]
    return rng.normal(0, 1.0, (rows, VOCABULARY)).astype(np.float32) + peak


def _within(exact, draws, build):
    entropy = quality.entropy(exact)
    perturbed = np.stack([quality.compare_rows(exact, draw)[0] for draw in draws])
    kl_build, agree = quality.compare_rows(exact, build)
    rows = [{"fixture_id": "f", "position": position} for position in range(len(exact))]
    return kl.within_resolution(rows, entropy, VOCABULARY, perturbed, kl_build, agree)


def test_a_build_at_four_times_a_draw_s_kl_is_within_the_bar():
    """A build at twice one draw's Gaussian logit noise, about four times its KL, is within 64 times the largest KL of
    four draws at every judgeable row, from confident rows to flat ones."""
    exact = _exact(200)
    draws = _draws(exact, 4, 5, 0.05)
    judged = _within(exact, draws, exact + 2 * (draws[0] - exact))
    assert judged["within"] is True and judged["over"] == [], judged["ratio"]
    assert judged["rows"]["judgeable"] > 150


def test_one_row_over_its_bar_fails_the_build_and_is_listed():
    """The bar is each row's 64 times the draws' largest KL. A build within it at every row passes; one row past it
    fails the build, and each row over is listed by identity with its ratio and whether its top-1 flipped."""
    rows = [{"fixture_id": "f", "position": position} for position in range(10)]
    perturbed, entropy = np.full((3, 10), 1e-4) * np.linspace(0.5, 1.0, 10), np.ones(10)
    perturbed[0, :2] = 1e-3
    bar = 64 * perturbed.max(axis=0)
    assert np.array_equal(kl.bars(entropy, VOCABULARY, perturbed), bar)
    agree = np.ones(10, dtype=bool)
    agree[9] = False

    def build(over):
        built = bar.copy()
        built[10 - over:] *= 2
        return kl.within_resolution(rows, entropy, VOCABULARY, perturbed, built, agree)

    at = build(0)
    assert at["within"] is True and at["over"] == [] and at["ratio"]["max"] == 1.0
    past = build(2)
    assert past["within"] is False
    assert past["over"] == [{"fixture_id": "f", "position": 8, "ratio": 2.0, "top1_flipped": False},
                            {"fixture_id": "f", "position": 9, "ratio": 2.0, "top1_flipped": True}]


def test_a_uniform_guess_is_over_every_judgeable_bar():
    """A build guessing the uniform distribution, at KL ln V - H from the reference, is over every judgeable row's bar:
    a row is judgeable only while its bar is below ln V - H. Rows the draws move past it are unjudgeable, reported apart
    with the build's top-1 there."""
    exact = _exact(200)
    draws = _draws(exact, 3, 5, 0.05) + [exact + _rows_of(exact, 6.0)]
    uniform = np.zeros_like(exact)
    room = np.log(VOCABULARY) - quality.entropy(exact)
    assert np.allclose(quality.compare_rows(exact, uniform)[0], room)
    guessed = _within(exact, draws, uniform)
    assert guessed["within"] is False and len(guessed["over"]) == guessed["rows"]["judgeable"] > 0
    assert guessed["rows"]["unjudgeable"] > 0 and guessed["unjudgeable_top1_agreement"] is not None


def _rows_of(exact, size):
    """A shift of `size` onto each row's second token: on confident rows a KL past the room ln V - H."""
    shift = np.zeros_like(exact)
    shift[:, 1] = size * np.linspace(1.0, 0.0, len(exact))
    return shift


def test_a_zero_bar_admits_only_a_zero_kl_and_no_judgeable_row_is_no_pass():
    rows = [{"fixture_id": "f", "position": position} for position in range(2)]
    entropy, zero, agree = np.array([1.0, 1.0]), np.zeros(2), np.ones(2, dtype=bool)
    draws = np.zeros((3, 2))
    assert kl.within_resolution(rows, entropy, VOCABULARY, draws, zero, agree)["within"] is True
    over = kl.within_resolution(rows, entropy, VOCABULARY, draws, np.array([0.0, 1e-12]), agree)
    assert over["within"] is False and over["ratio"] == {"median": np.inf, "p90": np.inf, "p99": np.inf, "max": np.inf}
    assert over["worst"] == rows[1]
    past = np.full((3, 2), np.log(VOCABULARY) / 64)
    none = kl.within_resolution(rows, entropy, VOCABULARY, past, zero, agree)
    assert none["within"] is False and none["ratio"] is None and none["rows"] == {"judgeable": 0, "unjudgeable": 2}
