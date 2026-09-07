"""Artifact loading and scoped program construction for gemma4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma4_text.config import ModelConfig
from mlx_vlm.models.gemma4_text.language import LanguageModel

from magnitude_engine.artifacts.identity import tokenizer_identity
from magnitude_engine.artifacts.layouts import LogicalTensor, logical_tensors
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.artifacts.tensors import TensorCatalog, read_json
from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.models.embeddings.contracts import EmbeddingFactory
from magnitude_engine.models.experts.contracts import ExpertFactory
from magnitude_engine.models.loading.materialization import bind_operations, prepare_layout
from magnitude_engine.models.loading.parameters import affine_encodings
from magnitude_engine.models.loading.partitions import EmbeddingPartition, ExpertPartition
from magnitude_engine.models.loading.validation import canonical_names
from magnitude_engine.models.ownership import OwnedProgram
from magnitude_engine.models.residency import (
    BoundProgram,
    ModelDescriptor,
    ModelResources,
    PagedRequirements,
)
from magnitude_engine.models.state.arena import LayerGeometry
from magnitude_engine.models.state.pages import SequencePages
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from .binding import bind_gemma4
from .definition import DEFINITION


@dataclass(frozen=True)
class LoadedGemma4:
    program: OwnedProgram[SequencePages]
    arguments: ModelConfig
    attention: tuple[LayerGeometry, ...]
    state_dtype: mx.Dtype
    encoding: AffineEncoding
    tokenizer_identity: str
    vision_tensors: Mapping[str, LogicalTensor]

    def close(self) -> None:
        self.program.close()


def load_gemma4(
    directory: Path,
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    attention: PagedAttention,
    embedding_factory: EmbeddingFactory,
    per_layer_embedding_factory: EmbeddingFactory,
    expert_factory: ExpertFactory,
) -> LoadedGemma4:
    directory = directory.expanduser().resolve()
    config = read_json(directory / "config.json")
    text = config.get("text_config", config)
    if text.get("model_type") != "gemma4_text":
        raise ValueError("artifact is not a supported Gemma 4 text model")
    args = ModelConfig.from_dict(text)
    if not 0 <= args.num_kv_shared_layers < args.num_hidden_layers:
        raise ValueError("a standalone Gemma target requires its own KV producers")
    settings = config.get("quantization", text.get("quantization", {}))
    if not settings or settings.get("mode", "affine") != "affine":
        raise ValueError("Gemma construction requires converted MLX affine tensors")
    default = AffineEncoding(settings["bits"], settings["group_size"])
    tensors = canonical_names(
        logical_tensors(TensorCatalog.inspect(directory), declaration=None), "language_model."
    )
    vision = {
        name: tensor
        for name, tensor in tensors.items()
        if name.startswith(("vision_tower.", "embed_vision."))
    }
    if vision and not config.get("vision_config"):
        raise ValueError("vision tensor component has no declared configuration")
    tensors = {name: tensor for name, tensor in tensors.items() if name not in vision}
    identity = tokenizer_identity(directory)
    model = LanguageModel(args)
    model.eval()
    encodings = affine_encodings(tensors, settings, prefix="language_model.")
    prepare_layout(model, tensors, encodings)
    embeddings = {}
    for key, name, module, factory, tied in (
        (
            "tokens",
            "model.embed_tokens",
            model.model.embed_tokens,
            embedding_factory,
            args.tie_word_embeddings,
        ),
        (
            "per_layer",
            "model.embed_tokens_per_layer",
            getattr(model.model, "embed_tokens_per_layer", None),
            per_layer_embedding_factory,
            False,
        ),
    ):
        if module is None:
            continue
        partition = EmbeddingPartition(
            name,
            module,
            {key: tensor for key, tensor in tensors.items() if key.startswith(name + ".")},
            encodings.get(name + ".weight"),
            tied,
        )
        embeddings[key] = (partition, factory)
    partitions = {}
    for index, layer in enumerate(model.layers):
        if not layer.enable_moe:
            continue
        module = layer.experts.switch_glu
        name = f"model.layers.{index}.experts.switch_glu"
        partitions[index] = (
            ExpertPartition(
                name,
                module.up_proj,
                module.gate_proj,
                module.down_proj,
                {key: tensor for key, tensor in tensors.items() if key.startswith(name + ".")},
                encodings,
                lambda up, gate: nn.gelu_approx(gate) * up,
            ),
            expert_factory,
        )
    operations = bind_operations(
        model,
        tensors,
        embeddings=embeddings,
        experts=partitions,
        budget=budget,
        reader=reader,
    )
    try:
        embedding = operations.embeddings["tokens"]
        per_layer = operations.embeddings.get("per_layer")
        dtype = model.model.norm.weight.dtype
        experts = operations.experts
        binding = bind_gemma4(
            model,
            embedding=embedding,
            per_layer_embedding=per_layer,
            experts=experts,
            attention=attention,
        )

        # Vocabulary loans include the final softcap used by generation methods.
        def project(hidden: mx.array) -> mx.array:
            logits = binding.program.output(hidden)
            cap = binding.program.softcap
            return mx.tanh(logits / cap) * cap if cap is not None else logits

        program = OwnedProgram(
            binding.program, (operations,), (identity, args.vocab_size, embedding, project)
        )
        return LoadedGemma4(
            program, args, binding.attention, dtype, default, identity, MappingProxyType(vision)
        )
    except BaseException:
        operations.close()
        raise


@dataclass(eq=False)
class Gemma4Source(ProgramSource):
    artifact: LocalArtifact
    attention: PagedAttention
    reader: PositionalReader
    embedding: EmbeddingFactory
    per_layer_embedding: EmbeddingFactory
    experts: ExpertFactory

    def load(self, resources: ModelResources) -> BoundProgram:
        def construct():
            loaded = resources.own(
                load_gemma4(
                    self.artifact.directory,
                    budget=resources.budget,
                    reader=self.reader,
                    attention=self.attention,
                    embedding_factory=self.embedding,
                    per_layer_embedding_factory=self.per_layer_embedding,
                    expert_factory=self.experts,
                )
            )
            return BoundProgram(
                loaded.program,
                ModelDescriptor(
                    self.artifact.path,
                    loaded.arguments.max_position_embeddings,
                    loaded.arguments.vocab_size,
                    loaded.tokenizer_identity,
                    "gemma4.Program",
                    DEFINITION,
                ),
                PagedRequirements(loaded.attention, loaded.state_dtype),
            )

        return resources.once(self, construct)
