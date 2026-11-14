"""A build's profile (`profile.py`) reports where a batch's time went and changes nothing it computes."""
import json
from pathlib import Path

import numpy as np
import torch

from conftest import write_rendered
from magnitude_references import profile
from magnitude_references.contract import read_logits
from magnitude_references.run import Options, build_reference
from test_run import BUILD, CPU, FIXTURES


def test_a_profiled_build_reports_each_batch_and_computes_the_same_logits(tmp_path: Path, tiny_llama: Path, capsys) -> None:
    rendered, artifacts = write_rendered(tmp_path, tiny_llama, FIXTURES)
    options = Options(CPU, build=BUILD, batch_tokens=64)
    build_reference(rendered, artifacts, tmp_path / "plain", options)
    capsys.readouterr()
    uninstall = profile.install()
    try:
        build_reference(rendered, artifacts, tmp_path / "profiled", options)
    finally:
        uninstall()
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith('{"profile"')]
    assert lines and all(line["wall_s"] >= 0 and line["dequantize_thread_s"] > 0 and line["read_thread_s"] >= 0
                         for line in lines)
    assert sum(line["fixtures"] for line in lines) == len(FIXTURES)
    (_, plain), (_, profiled) = (read_logits(tmp_path / name, "logits") for name in ("plain", "profiled"))
    assert np.array_equal(plain, profiled)
