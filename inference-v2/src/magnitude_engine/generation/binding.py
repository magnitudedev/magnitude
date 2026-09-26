from dataclasses import dataclass
from pathlib import Path

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.models.executor.contracts import ExecutorFactory
from magnitude_engine.models.residency import BoundExecutor, ModelResources

from .contracts import GenerationFactory, MethodFactory
from .guidance import GuidanceCompiler
from .runtime import GenerationRuntime


@dataclass(frozen=True)
class BoundGeneration:
    runtime: GenerationRuntime
    target: BoundExecutor
    draft_capacity: int
    speculative_backend: str | None


@dataclass(eq=False)
class Generation(GenerationFactory):
    target: ExecutorFactory
    method: MethodFactory

    def load(self, resources: ModelResources) -> BoundGeneration:
        target = self.target.load(resources)
        method, capacity, backend = self.method.bind(target, resources)
        path = target.program.descriptor.path

        def tokenizer():
            import llguidance.hf

            artifact = TokenizerArtifact.load(Path(path))
            return llguidance.hf.from_tokenizer(
                artifact.tokenizer, n_vocab=artifact.vocabulary, eos_token=list(artifact.eos_tokens)
            )

        return BoundGeneration(
            GenerationRuntime(target.model, method, GuidanceCompiler(tokenizer)),
            target,
            capacity,
            backend,
        )
