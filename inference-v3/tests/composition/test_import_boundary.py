"""Blueprint inspection must not initialize compiler, tensor, or tokenizer runtimes."""

import subprocess
import sys


def test_catalog_and_model_graph_roundtrip_are_native_runtime_free():
    source = """
import sys
from magnitude_engine.blueprints import artifacts, execution, inputs, models, operations
from magnitude_engine.blueprints import service, serving
from magnitude_engine.composition import dumps, loads
from magnitude_engine.platform.backend import Backend
artifact = artifacts.GGUF(path='/no-artifact-needed-for-inspection.gguf')
endpoint = execution.Device(backend=Backend.HIP, budget_bytes=1024)
root = models.Qwen35Dense(
    description=models.Qwen35DenseDescription(artifact=artifact),
    operations=operations.ResidentOperations(artifact=artifact, context=endpoint),
)
copy = loads(dumps(root))
assert copy.description.artifact is copy.operations.artifact
engine = service.Continuous(model=root, selector=operations.SampleSelector(context=endpoint))
restored = loads(dumps(engine))
assert restored.model.operations.context is restored.selector.context
assert dumps(restored) == dumps(engine)
chat = serving.ChatComponents(engine=engine, tokenizer=serving.ChatMetadata(artifact=artifact))
assert dumps(loads(dumps(chat))) == dumps(chat)
tokenizer = inputs.ByteBPE(config=inputs.Qwen35Tokenization(artifact=artifact))
assert dumps(loads(dumps(tokenizer))) == dumps(tokenizer)
assert not {'tilelang', 'torch', 'tokenizers'}.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
