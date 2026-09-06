import json

import pytest

from benchmarks.contracts import Experiment, Observation, SubjectBlueprint
from benchmarks.runner import measure
from magnitude_engine.composition import component


class Subject:
    events = []

    def __init__(self, *, invalid: bool):
        self.invalid = invalid

    def reset(self):
        self.events.append("reset")

    def invoke(self):
        self.events.append("invoke")

    def complete(self):
        self.events.append("complete")

    def observe(self):
        self.events.append("validate")
        if self.invalid:
            raise ValueError("incorrect output")
        return Observation("same", {"count": 3}, {"requests": [{"elapsed_ns": 5}]})

    def close(self):
        self.events.append("close")


@component
class Trace(SubjectBlueprint):
    invalid: bool = False

    @staticmethod
    def implementation():
        return Subject


def experiment(*, invalid=False):
    return Experiment(
        identity="test",
        subject=Trace(invalid=invalid),
        characteristic="correctness",
        measurement_width="component",
        claim="test timing boundary",
        warmup=1,
        repetitions=2,
    )


def test_completion_is_timed_validation_is_excluded_and_raw_samples_are_preserved(tmp_path):
    events = Subject.events = []
    ticks = iter((0, 10, 20, 40, 50, 90))

    def clock():
        events.append("clock")
        return next(ticks)

    run = measure(experiment(), {}, clock=clock)
    assert events == ["reset", "clock", "invoke", "complete", "clock", "validate"] * 3 + ["close"]
    assert [sample.elapsed_ns for sample in run.samples] == [10, 20, 40]
    path = tmp_path / "run.json"
    run.write(path)
    record = json.loads(path.read_text())
    assert record["statistics_ms"]["count"] == 2
    assert len(record["samples"]) == 3
    assert record["status"] == "valid_characterization"
    assert record["samples"][0]["observation"]["evidence"] == {"requests": [{"elapsed_ns": 5}]}


def test_invalid_measurements_remain_rejected_records_and_still_close_the_subject():
    events = Subject.events = []
    run = measure(experiment(invalid=True), {})
    assert run.record()["status"] == "rejected"
    assert "incorrect output" in run.rejections[0]
    assert run.samples[0].rejection is not None
    assert run.record()["statistics_ms"]["count"] == 0
    assert events[-1] == "close"


def test_python_cases_import_without_device_dependencies_and_record_all_compositions():
    import subprocess
    import sys

    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'mlx_vlm', 'transformers', 'numpy'}:
            raise AssertionError('case import initialized ' + fullname)
sys.meta_path.insert(0, Block())
from importlib import import_module
from pkgutil import iter_modules
import benchmarks.cases
from benchmarks.contracts import Experiment
from benchmarks.loading import digest, load
seen = set()
for domain in iter_modules(benchmarks.cases.__path__):
    module = import_module('benchmarks.cases.' + domain.name)
    for name, value in vars(module).items():
        if isinstance(value, Experiment):
            selected = load(module.__name__ + ':' + name)
            assert selected is value
            assert len(digest(selected)) == 64
            assert selected.record()['subject'] == selected.subject.describe()
            seen.add(selected.identity)
assert seen, "no benchmark experiments were discovered"
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_cold_shape_subject_rejects_warmed_or_repeated_measurement():
    from dataclasses import replace

    import pytest

    from benchmarks.cases.recurrence import shapes

    for change in ({"warmup": 1}, {"repetitions": 2}):
        with pytest.raises(ValueError, match="warmup"):
            replace(shapes, **change)


def test_python_selector_requires_an_experiment():
    import pytest

    from benchmarks.loading import load

    with pytest.raises(ValueError):
        load("benchmarks/cases/state.toml")
    with pytest.raises(TypeError, match="Experiment"):
        load("benchmarks.cases.state:KVBranch")


def test_cli_describes_then_measures_in_child_and_preserves_existing_evidence(tmp_path):
    import os
    import subprocess
    import sys

    from benchmarks.cases.state import branch

    command = [sys.executable, "-m", "benchmarks", "benchmarks.cases.state:branch"]
    described = subprocess.run([*command, "--describe"], capture_output=True, text=True, timeout=10)
    assert described.returncode == 0, described.stderr
    assert json.loads(described.stdout) == json.loads(json.dumps(branch.record()))
    path = tmp_path / "branch.json"
    measured = subprocess.run(
        [*command, "--output", str(path)], capture_output=True, text=True, timeout=30
    )
    assert measured.returncode == 0, measured.stderr
    original = path.read_bytes()
    record = json.loads(original)
    assert record["status"] == "valid_characterization"
    assert record["environment"]["process"] == "fresh benchmark child"
    assert record["environment"]["pid"] != os.getpid()
    assert record["statistics_ms"]["count"] == branch.repetitions
    assert len(record["samples"]) == branch.warmup + branch.repetitions
    assert record["environment"]["composition"] == record["experiment"]["subject"]
    repeated = subprocess.run(
        [*command, "--output", str(path)], capture_output=True, text=True, timeout=10
    )
    assert repeated.returncode == 2 and "already exists" in repeated.stderr
    assert path.read_bytes() == original


def test_benchmark_api_checks_dependency_types_and_parameters(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    source = tmp_path / "consumer.py"
    source.write_text("""
from benchmarks.subjects import Attention, KVAppend
from magnitude_engine import blueprints as bp
valid = Attention(computation=bp.model.attention.metal.Paged(), prefix_tokens=16)
bad_dependency = Attention(computation=bp.resources.io.PositionalReader(), prefix_tokens=16)
bad_scalar = Attention(computation=valid.computation, prefix_tokens='large')
bad_variant = KVAppend(granularity='magic', prefix_tokens=16, append_tokens=2)
bad_field = Attention(computation=valid.computation, prefix_tokens=16, unknown=1)
""")
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "pyright"),
            "--pythonpath",
            sys.executable,
            "--project",
            str(project / "pyproject.toml"),
            "--outputjson",
            str(source),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    errors = [d for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert len(errors) == 4, report
    assert {e["range"]["start"]["line"] for e in errors} == {4, 5, 6, 7}


def test_model_subjects_use_injected_engine_and_release_all_reservations(tmp_path, monkeypatch):
    from dataclasses import replace

    from benchmarks.subjects import EngineWaves, ModelPrefill, PlainDecode
    from magnitude_engine import blueprints as bp
    from magnitude_engine.composition import build
    from tests.models.architectures.qwen35.test_construction import artifact_pair
    from tests.worker.test_worker import compose_engine

    target, head, _, _ = artifact_pair(tmp_path)
    engine = compose_engine(
        str(target),
        str(head),
        memory_bytes=64 << 20,
        context_tokens=32,
        output_capacity=2,
        max_active=2,
        prefill_tokens=4,
    )

    class Tokenizer:
        def encode(self, text):
            return [1, 2, 3]

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    native = replace(
        engine,
        generation=bp.generation.Generation(
            target=bp.model.auto(bp.model.artifacts.Local(path=str(target))),
        ),
    )
    plain = replace(
        engine, generation=replace(engine.generation, method=bp.generation.methods.Plain())
    )
    for subject in (
        ModelPrefill(engine=engine, prefix_tokens=3, input_tokens=4),
        ModelPrefill(engine=native, prefix_tokens=3, input_tokens=4),
        ModelPrefill(engine=native, prefix_tokens=3, input_tokens=4, rows=2),
        ModelPrefill(engine=native, prefix_tokens=3, input_tokens=4, rows=2,
                     execution="independent"),
        ModelPrefill(engine=plain, prefix_tokens=3, input_tokens=4, rows=2),
        PlainDecode(engine=native, prompt_tokens=7, output_tokens=4),
        PlainDecode(engine=plain, prompt_tokens=7, output_tokens=4),
        EngineWaves(engine=engine, prompt_text="test", prompt_tokens=7, output_tokens=4, rows=2),
    ):
        case = replace(experiment(), subject=subject)
        result = measure(case, {})
        assert not result.rejections, result.rejections
        assert len(result.samples) == 3
        with build(subject) as live:
            budget = live.budget
            live.reset()
            live.invoke()
            live.complete()
            live.observe()
            assert budget.snapshot().reserved > 0
        assert budget.snapshot().reserved == 0


def test_execution_and_cleanup_failures_both_remain_in_rejected_evidence(monkeypatch):
    events = Subject.events = []

    def fail_close(self):
        self.events.append("close")
        raise RuntimeError("cleanup did not complete")

    monkeypatch.setattr(Subject, "close", fail_close)
    result = measure(experiment(invalid=True), {})
    assert result.record()["status"] == "rejected"
    assert "incorrect output" in result.rejections[0]
    assert "cleanup did not complete" in result.rejections[0]
    assert events.count("close") == 1
    assert len(result.samples) == 1


@pytest.mark.parametrize('speculative', [False, True])
@pytest.mark.parametrize('execution', ['shared', 'independent'])
def test_generation_component_subject_validates_every_row_and_releases_state(
    speculative, execution
):
    from types import SimpleNamespace

    from benchmarks.subjects.generation.runtime import GenerationTrace
    from magnitude_engine.generation.methods.plain.runtime import PlainMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from tests.generation.test_mtp import setup

    target, method, budget, _, _, _ = setup()
    generation = GenerationRuntime(target, method if speculative else PlainMethod())
    engine = SimpleNamespace(engine=SimpleNamespace(generation=generation), budget=budget)
    subject = GenerationTrace(
        engine=engine, prompt_tokens=3, output_tokens=10, rows=2, execution=execution,
    )
    for _ in range(2):
        subject.reset()
        subject.invoke()
        subject.complete()
        observation = subject.observe()
        assert observation.counters['output_tokens'] == 20
        assert observation.counters['max_physical_batch'] == (2 if execution == 'shared' else 1)
    subject.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0
