"""Artifact loading and scoped program construction for qwen35/mtp."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from magnitude_engine.artifacts.encoding import AffineMaterializer
from magnitude_engine.artifacts.layouts import logical_tensors
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.artifacts.tensors import TensorCatalog, read_json
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.models.loading.validation import (
    canonical_names,
    configure_affine_modules,
    encoded_shapes,
    validate_parameters,
)
from magnitude_engine.models.ownership import OwnedProgram, VocabularyLoan
from magnitude_engine.models.residency import (
    BoundProgram,
    DraftRequirements,
    ModelDescriptor,
    ModelResources,
    NativeRequirements,
)
from magnitude_engine.models.state.native import LibraryState, LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from ..loading import LoadedQwen35
from .program import MTPProgram


class MTPParameters(nn.Module):
    """Checkpoint naming and geometry only; execution belongs to MTPProgram."""

    def __init__(self, arguments: TextModelArgs):
        super().__init__()
        self.fc = nn.Linear(2 * arguments.hidden_size, arguments.hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(arguments.hidden_size, eps=arguments.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(arguments.hidden_size, eps=arguments.rms_norm_eps)
        self.norm = nn.RMSNorm(arguments.hidden_size, eps=arguments.rms_norm_eps)
        self.layers = [DecoderLayer(arguments, i) for i in range(arguments.num_hidden_layers)]


@dataclass(frozen=True)
class AttentionStep:
    layer: DecoderLayer

    def __call__(self, hidden: mx.array, cache: KVCache) -> mx.array:
        mask: Any = create_attention_mask(hidden, cache)
        return self.layer(hidden, mask=mask, cache=cache)


@dataclass(frozen=True)
class LoadedMTP:
    program: OwnedProgram[LibraryState]
    vocabulary: VocabularyLoan
    depth: int
    capacity: int
    kv_bytes_per_token: int
    target_feature: str

    def state_store(self, budget: MemoryBudget) -> LibraryStateStore:
        return LibraryStateStore(
            lambda: [KVCache() for _ in range(self.depth)],
            budget,
            lambda length: (
                ((length + KVCache.step - 1) // KVCache.step)
                * KVCache.step
                * self.kv_bytes_per_token
            ),
        )

    def close(self) -> None:
        self.program.close()


def load_mtp(
    directory: Path,
    target: LoadedQwen35,
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    quantize: bool = True,
    capacity: int | None = None,
) -> LoadedMTP:
    directory = directory.expanduser().resolve()
    config = read_json(directory / "config.json")
    text = config.get("text_config", config)
    head = TextModelArgs.from_dict(text)
    for name in (
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        if getattr(head, name) != getattr(target.arguments, name):
            raise ValueError(f"MTP artifact differs from target geometry: {name}")
    depth = text.get("mtp_num_hidden_layers", 1)
    block = config.get("block_size", depth + 2)
    if type(depth) is not int or depth < 1 or type(block) is not int or block < 2:
        raise ValueError("MTP depth and total block size must be positive integers")
    capacity = block - 1 if capacity is None else capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError("MTP proposal capacity must be a positive integer")
    args = replace(target.arguments, num_hidden_layers=depth, full_attention_interval=1)
    tensors = canonical_names(
        logical_tensors(TensorCatalog.inspect(directory), declaration=None), "mtp."
    )
    parameters = MTPParameters(args)
    parameters.eval()
    # Check the floating checkpoint before replacing any module. In particular,
    # released MTP norm weights are already sanitized and must not receive +1.
    validate_parameters(parameters, {name: tensor.shape for name, tensor in tensors.items()})
    encodings = {}
    if quantize:
        for name, tensor in tensors.items():
            if name.endswith(".weight") and len(tensor.shape) >= 2:
                router = name.endswith(("mlp.gate.weight", "shared_expert_gate.weight"))
                encodings[name] = AffineEncoding(8, 64) if router else target.encoding
    configure_affine_modules(parameters, encodings)
    validate_parameters(parameters, encoded_shapes(tensors, encodings))
    vocabulary = target.program.borrow_vocabulary()
    allocation = None
    try:
        allocation = AffineMaterializer(budget, reader, encodings, owner="mtp.weights").materialize(
            tensors
        )
        parameters.load_weights(list(allocation.arrays.items()), strict=True)
        program = MTPProgram(
            vocabulary,
            parameters.pre_fc_norm_embedding,
            parameters.pre_fc_norm_hidden,
            parameters.fc,
            tuple(AttentionStep(layer) for layer in parameters.layers),
            parameters.norm,
            vocabulary.project,
        )
        dtype = allocation.arrays["norm.weight"].dtype
        assert args.head_dim is not None
        kv_bytes = 2 * depth * args.num_key_value_heads * args.head_dim * dtype.size
        return LoadedMTP(
            OwnedProgram(program, (allocation, vocabulary)),
            vocabulary,
            depth,
            capacity,
            kv_bytes,
            f"residual:{target.arguments.num_hidden_layers}",
        )
    except BaseException:
        if allocation is not None:
            allocation.close()
        vocabulary.close()
        raise


@dataclass(eq=False)
class MTPSource(ProgramSource):
    artifact: LocalArtifact
    target: ProgramSource
    reader: PositionalReader

    def native_state_source(self) -> ProgramSource:
        return self

    def load(self, resources: ModelResources) -> BoundProgram:
        def construct():
            target = self.target.load(resources)
            binding = self.target.head_binding(resources)
            head = binding.load(self.artifact, self.reader, resources)
            caches = head.state_store(resources.budget)
            return BoundProgram(
                head.program,
                ModelDescriptor(
                    self.artifact.path,
                    target.descriptor.context_tokens,
                    target.descriptor.vocab_size,
                    target.descriptor.tokenizer_identity,
                    "mtp.Head",
                ),
                NativeRequirements(caches.make_cache, caches.capacity, self),
                DraftRequirements(
                    target.program,
                    head.target_feature,
                    head.capacity,
                    head.vocabulary,
                ),
            )

        return resources.once(self, construct)
