"""Static enforcement of the Magnitude / Ops / TileLang boundary."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
ENGINE = ROOT / "engine"
MT = ROOT / "ops"

REMOVED_ENGINE_NUMERICS = {
    "engine.kernels",
    "engine.platform.binding",
    "engine.platform.driver",
    "engine.platform.execution",
    "engine.platform.specialization",
    "engine.weights.binding",
    "engine.weights.representation",
}


def _sources(root: Path):
    return sorted(root.rglob("*.py"))


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            result.append(node.module)
    return result


def test_removed_engine_numerical_layers_have_no_source_or_importers():
    for module in REMOVED_ENGINE_NUMERICS:
        path = ROOT.joinpath(*module.split("."))
        assert not path.with_suffix(".py").exists()
        assert not path.is_dir() or not any(path.rglob("*.py"))
    offenders = [
        (str(path.relative_to(ROOT)), module)
        for path in _sources(ENGINE)
        for module in _imports(path)
        if any(module == old or module.startswith(old + ".") for old in REMOVED_ENGINE_NUMERICS)
    ]
    assert offenders == []


def test_engine_model_and_service_code_contains_no_tilelang_or_torch_computation():
    offenders = []
    for area in ("models", "generation", "service", "weights"):
        for path in _sources(ENGINE / area):
            for module in _imports(path):
                if module == "tilelang" or module.startswith("tilelang."):
                    offenders.append((str(path.relative_to(ROOT)), module))
                if module == "torch" or module.startswith("torch."):
                    offenders.append((str(path.relative_to(ROOT)), module))
    assert offenders == []


def test_tilelang_is_reached_only_by_ops_kernel_and_runtime_realization():
    allowed = {MT / "runtime" / "tilelang.py", MT / "compiler" / "program.py", MT / "compiler" / "streaming.py"}
    offenders = []
    for path in _sources(ROOT):
        if any(
            module == "tilelang" or module.startswith("tilelang.") for module in _imports(path)
        ) and not (path.is_relative_to(MT / "kernels") or path in allowed):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_model_code_expresses_numerics_only_through_ops():
    offenders = [
        (str(path.relative_to(ROOT)), module)
        for path in _sources(ENGINE / "models")
        for module in _imports(path)
        if module.startswith("engine.operations")
        or module.startswith("engine.platform")
        and module != "engine.platform.storage"
    ]
    assert offenders == []


def test_public_recipe_roundtrips_the_new_integration_graph_without_construction():
    source = """
from engine.blueprints import execution, inputs, models, service, serving, weights
from engine.composition import dumps, loads
from engine.platform.backend import Backend
from engine.devices import DevicePlan, DeviceTopology, MemoryConstraint
container = weights.GGUF(path='/no-artifact-needed-for-inspection.gguf')
topology = DeviceTopology.model_validate_json('''{
  "devices":[{"id":"gpu","name":"test HIP","kind":"gpu"}],
  "memory":[{"id":"host","kind":"host","capacity_bytes":4096}],
  "constraints":[], "cpu_caches":[],
  "endpoints":[{"id":"hip:2","device":"gpu","backend":"hip","ordinal":2,
    "modes":[{"kind":"shared","domains":["host"],"constraints":[],
      "access":[{"device":"gpu","kind":"direct","host_synchronization_required":true}],
      "max_allocation_bytes":4096}]}]
}''')
device = execution.DeviceRuntime(plan=DevicePlan(topology=topology, endpoints=('hip:2',),
    constraints=(MemoryConstraint(domains=('host',), maximum_bytes=1024),)))
residency = weights.Weights(format=container, context=device)
model = models.Qwen35Dense(
    description=models.Qwen35DenseDescription(format=container),
    device=device,
    weights=residency,
    max_sequences=4,
)
engine = service.Continuous(
    model=model,
    limits=service.ServiceLimits(max_requests=12, max_batch=4, prefill_tokens=256),
)
chat = serving.ChatComponents(engine=engine, tokenizer=serving.ChatMetadata(artifact=container))
copy = loads(dumps(chat))
assert copy.engine.model.device is copy.engine.model.weights.context
assert copy.engine.model.description.format is copy.engine.model.weights.format
assert dumps(copy) == dumps(chat)
tokenizer = inputs.ByteBPE(config=inputs.Qwen35Tokenization(artifact=container))
assert dumps(loads(dumps(tokenizer))) == dumps(tokenizer)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
