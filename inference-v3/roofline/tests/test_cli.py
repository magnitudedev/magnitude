import pytest

from roofline.cli import experiment, parser


def test_named_selectors_are_exclusive_and_required():
    for argv in (["query"], ["query", "some-id"], ["query", "--model", "x", "--measurement", "y"]):
        with pytest.raises(SystemExit):
            parser().parse_args(argv)
    args = parser().parse_args(
        [
            "measure",
            "--model",
            "qwen:gguf:q4",
            "--context",
            "16k",
            "--targets",
            "local,remote",
            "--scope",
            "decode/x[0]",
            "--step",
            "0",
        ]
    )
    request = experiment(args)
    assert request.context == 16384
    assert request.targets == ("local", "remote")
    assert request.step == 0


def test_shared_inputs_require_an_explicit_producer_and_component_position():
    base = ["measure", "--model", "qwen:gguf:q4", "--input-source", "a" * 64]
    with pytest.raises(ValueError, match="both"):
        experiment(parser().parse_args(base))
    with pytest.raises(ValueError, match="component"):
        experiment(parser().parse_args([*base, "--input-target", "producer"]))
    selected = experiment(
        parser().parse_args(
            [
                *base,
                "--input-target",
                "producer",
                "--scope",
                "decode/linear[0]",
                "--step",
                "0",
            ]
        )
    )
    assert selected.input_source == "a" * 64 and selected.input_target == "producer"
