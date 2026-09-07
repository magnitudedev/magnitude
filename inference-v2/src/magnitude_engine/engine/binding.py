"""Build the entire live engine in one residency, from injected construction components."""

import mlx.core as mx

from magnitude_engine.generation.contracts import GenerationFactory
from magnitude_engine.models.residency import ModelResources

from .contracts import EngineInstance
from .memory.contracts import MemoryPolicy
from .prefixes.contracts import PrefixIndex
from .runtime import Engine
from .scheduler.contracts import Scheduler


class EngineResidency(EngineInstance):
    def __init__(
        self,
        *,
        generation: GenerationFactory,
        scheduler: Scheduler,
        memory: MemoryPolicy,
        prefixes: PrefixIndex,
        context_tokens: int,
        output_capacity: int,
    ):
        self.budget = memory
        self.output_capacity = output_capacity
        self.resources = ModelResources(
            budget=memory,
            context_tokens=context_tokens,
            max_active=scheduler.max_active,
        )
        self._closed = False
        try:
            mx.set_cache_limit(256 << 20)
            loaded = generation.load(self.resources)
            descriptor = loaded.target.program.descriptor
            self.selection = loaded.target.selection
            self.target = loaded.target
            self.engine = Engine(
                loaded.runtime,
                namespace=(
                    descriptor.path + descriptor.tokenizer_identity + loaded.runtime.method.identity
                ).encode(),
                scheduler=scheduler,
                prefixes=prefixes,
            )
            memory.bind_reclaimer(lambda: memory.pressure.relieve(prefixes))
            self.properties = {
                "context_tokens": min(context_tokens, descriptor.context_tokens),
                "vocab_size": descriptor.vocab_size,
                "target_path": descriptor.path,
                "tokenizer_identity": descriptor.tokenizer_identity,
                "speculative_backend": loaded.speculative_backend,
                "max_draft_tokens": loaded.draft_capacity,
                "parallel_sequences": scheduler.max_active,
                "memory_bytes": memory.limit,
                "prefill_tokens": scheduler.prefill_tokens,
                "output_capacity": output_capacity,
                "retained_prefixes": prefixes.retention.max_entries,
                "program_implementation": descriptor.implementation,
            }
        except BaseException:
            self.resources.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self.engine.close()
        self.budget.bind_reclaimer(lambda: False)
        self.resources.close()
        self._closed = True
