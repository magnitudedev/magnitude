import pytest
from bench_support import TREE
from magnitude_bench import cli
from magnitude_bench.models import catalog_reference, targets


def test_contexts_and_sections():
    assert cli.contexts("16k,4k,4096") == (4096, 16384)
    with pytest.raises(ValueError):
        cli.contexts("0")
    assert cli.choices("all", ("a", "b")) == ("a", "b")
    with pytest.raises(ValueError):
        cli.choices("context,nope", ("context",))


def test_models_come_only_from_the_catalog():
    assert catalog_reference("qwen3.5-4b:gguf:q4", TREE) == "qwen3.5-4b:gguf:q4"
    for reference in ("qwen3.5-4b", "qwen3.5-4b:gguf:q3", "not-a-model:gguf:q4"):
        with pytest.raises(ValueError):
            catalog_reference(reference, TREE)


def test_gguf_engines_serve_the_catalog_target_and_others_name_their_form(acquisition):
    selected = targets(
        ["magnitude", "llama.cpp", "mlx-vlm"],
        {"mlx-vlm": "hf:mlx-community/Qwen3.5-4B-4bit@" + "a" * 40},
        acquisition,
    )
    assert [t.reference for t in selected[:2]] == [str(acquisition.target)] * 2
    assert all(t.sha256 == acquisition.target_sha256 for t in selected[:2])
    assert selected[2].reference.startswith("hf:mlx-community/")
    with pytest.raises(ValueError, match="takes no --artifact"):
        targets(["llama.cpp"], {"llama.cpp": "/x.gguf"}, acquisition)
    with pytest.raises(ValueError, match="needs --artifact omlx"):
        targets(["omlx"], {}, acquisition)
    with pytest.raises(ValueError, match="not selected"):
        targets(["magnitude"], {"omlx": "/m"}, acquisition)
    with pytest.raises(ValueError, match="Hub snapshots"):
        targets(["mlx-vlm"], {"mlx-vlm": "hf:owner/repo@main"}, acquisition)


def test_measurement_options_round_trip_through_their_arguments():
    given = ["--mode", "tools", "--context", "4k,16k", "--engine", "llama.cpp", "--engine", "magnitude",
             "--llama-binary", "bin/llama-server", "--fresh-tuning", "--passes", "2",
             "--magnitude-build", '{"commit":"c","dirty_digest":null}', "--ollama-answer-prefill"]
    parser = cli.argparse.ArgumentParser()
    cli.measurement_options(parser)
    parsed = parser.parse_args(given)
    again = parser.parse_args(cli.arguments(parsed))
    assert vars(again) == vars(parsed)
    assert "--llama-gpu-layers" not in cli.arguments(parsed)  # defaults stay unsaid


def test_invalid_selection_fails_before_any_job():
    args = cli.parser().parse_args([
        "job", "--mode", "tools", "--model", "qwen3.5-4b:gguf:q3", "--context", "4k",
        "--inference", "i", "--artifacts", "a.json", "--output", "out", "--build-identity", "b.json",
    ])
    with pytest.raises(ValueError, match="no variant"):
        cli.check(args, TREE)


def test_a_held_set_is_served_under_the_name_its_acquisition_gives_it(tmp_path):
    """`job` takes no catalog reference for a set the host holds: the acquisition names it, and a catalog
    reference, where given, must be the one the acquisition names."""
    base = ["job", "--mode", "prose-repeat", "--context", "4k", "--inference", str(tmp_path), "--artifacts", str(tmp_path / "a.json"),
            "--output", str(tmp_path / "out"), "--build-identity", str(tmp_path / "b.json")]
    held = cli.parser().parse_args(base)
    assert held.model is None
    cli.check(held, TREE)
    with pytest.raises(ValueError, match="not a catalog model"):
        cli.check(cli.parser().parse_args([*base, "--model", "held-set:gguf:fp8"]), TREE)


def test_the_tool_imports_nothing_of_the_rig_and_reads_only_the_tree_it_is_given(tmp_path):
    """Bench is the measurement package and the rig drives it, so it neither finds a checkout nor imports the
    rig: the tree is an argument, and a tree without a catalog is no catalog."""
    from pathlib import Path

    with pytest.raises(OSError):
        catalog_reference("qwen3.5-4b:gguf:q4", tmp_path)
    source = Path(cli.__file__).parent
    assert not [path for path in source.rglob("*.py") if "magnitude_rig" in path.read_text()]
