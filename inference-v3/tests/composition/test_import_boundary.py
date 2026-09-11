"""Each axis has one owner, and inspecting a composition initializes nothing.

The static scan is the enforcement of the import rules: a package that tests a
value of an axis it does not own is exactly the leak this layout removed.
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "magnitude_engine"

# `kernels/` is portable TileLang. It may read the DType vocabulary, the resident
# representation it is given, and other kernels. `<op>/select.py` additionally
# declares its candidate rows, whose type lives with `realize`.
KERNEL_ALLOWED = {
    "magnitude_engine.kernels",
    "magnitude_engine.platform.execution",
    "magnitude_engine.weights.representation",
    "magnitude_engine.weights.descriptor",
}
SELECT_ALLOWED = KERNEL_ALLOWED | {"magnitude_engine.operations.candidates"}

# Backend selection is an endpoint choice. Only the driver, the machine
# inventory that opens it, and composition may name one.
BACKEND_OWNERS = ("platform/", "blueprints/", "composition/", "serving/")

# One schedule still reaches Metal directly: its cooperative matrix operations
# and `metal.simdgroup` scope have no portable spelling in the fork yet. Every
# other part of its selection is portable, and it is the only recorded
# exception; see design/inference/engine/kernels.md.
PASS_THROUGH_EXCEPTIONS = {"kernels/attention/decode_partitioned.py"}

PASS_THROUGH = ("call_extern", "call_pure_extern", "call_intrin")


def sources(*relative: str) -> list[Path]:
    return sorted(path for part in relative for path in (ROOT / part).rglob("*.py"))


def imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
    return names


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def permitted(module: str, allowed: set[str]) -> bool:
    if not module.startswith("magnitude_engine"):
        return True
    return any(module == name or module.startswith(name + ".") for name in allowed)


def test_kernels_import_only_the_language_dtypes_and_representations():
    offenders = []
    for path in sources("kernels"):
        allowed = SELECT_ALLOWED if path.name == "select.py" else KERNEL_ALLOWED
        offenders += [
            (relative(path), module)
            for module in imports(path)
            if not permitted(module, allowed)
        ]
    assert offenders == []


def test_kernels_contain_no_backend_pass_through():
    offenders = []
    for path in sources("kernels"):
        if relative(path) in PASS_THROUGH_EXCEPTIONS:
            continue
        text = path.read_text()
        if any(name in text for name in PASS_THROUGH):
            offenders.append((relative(path), "extern call"))
        if "tilelang.metal" in text or "tilelang.cuda" in text or "tilelang.tvm" in text:
            offenders.append((relative(path), "dialect import"))
    assert offenders == []


def names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def test_only_the_platform_and_composition_name_a_backend():
    """Everything else reads the capability the driver resolved from a backend."""
    offenders = [
        relative(path)
        for path in sources(
            "kernels", "weights", "operations", "models", "inputs", "state", "generation"
        )
        if "Backend" in names(path)
        or "magnitude_engine.platform.backend" in imports(path)
    ]
    assert offenders == []


# Residency compiles the repack and the conversion; a capability decides a
# representation. Nothing else in `weights/` reaches a schedule.
WEIGHT_KERNELS = {
    "magnitude_engine.kernels.capabilities",
    "magnitude_engine.kernels.copy.convert",
    "magnitude_engine.kernels.projection.planar_affine.pack",
}


def test_weights_reach_kernels_only_for_residency():
    offenders = [
        (relative(path), module)
        for path in sources("weights")
        for module in imports(path)
        if module.startswith("magnitude_engine.kernels") and module not in WEIGHT_KERNELS
    ]
    assert offenders == []


def test_models_never_import_a_kernel_or_another_models_container():
    offenders = []
    for path in sources("models"):
        for module in imports(path):
            if module.startswith("magnitude_engine.kernels.") and module not in (
                "magnitude_engine.kernels.precision",
                "magnitude_engine.kernels.semantics",
            ):
                offenders.append((relative(path), module))
            if module.startswith("magnitude_engine.weights.formats") and "formats/" not in relative(
                path
            ):
                offenders.append((relative(path), module))
    assert offenders == []


def test_operations_import_tables_not_schedules():
    offenders = [
        (relative(path), module)
        for path in sources("operations")
        for module in imports(path)
        if module.startswith("magnitude_engine.kernels.")
        and not module.endswith((".select", ".precision", ".semantics", ".capabilities"))
        and module
        not in ("magnitude_engine.kernels.pointwise.portable", "magnitude_engine.kernels.kv.append")
        and not module.endswith(".copy.copy")
        and not module.endswith(".rotary.portable")
        and not module.endswith(".recurrence.prepare")
        and not module.endswith(".sampling.finish")
        and not module.endswith(".sampling.select_tiles")
        and not module.endswith(".embedding.gather")
    ]
    assert offenders == []


def test_catalog_and_model_graph_roundtrip_are_native_runtime_free():
    source = """
import sys
from magnitude_engine.blueprints import execution, inputs, models, operations
from magnitude_engine.blueprints import service, serving, weights
from magnitude_engine.composition import dumps, loads
from magnitude_engine.platform.backend import Backend
container = weights.GGUF(path='/no-artifact-needed-for-inspection.gguf')
endpoint = execution.Device(backend=Backend.HIP, budget_bytes=1024)
arena = operations.Arena(context=endpoint)
binding = operations.Operations(
    weights=weights.Weights(format=container, context=endpoint), arena=arena
)
root = models.Qwen35Dense(
    description=models.Qwen35DenseDescription(format=container), operations=binding
)
copy = loads(dumps(root))
assert copy.description.format is copy.operations.weights.format
engine = service.Continuous(model=root, selector=operations.SampleSelector(context=endpoint))
restored = loads(dumps(engine))
assert restored.model.operations.weights.context is restored.selector.context
assert dumps(restored) == dumps(engine)
chat = serving.ChatComponents(engine=engine, tokenizer=serving.ChatMetadata(artifact=container))
assert dumps(loads(dumps(chat))) == dumps(chat)
tokenizer = inputs.ByteBPE(config=inputs.Qwen35Tokenization(artifact=container))
assert dumps(loads(dumps(tokenizer))) == dumps(tokenizer)
assert not {'tilelang', 'torch', 'tokenizers'}.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
