"""Artifact loading and scoped program construction for qwen35."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU
from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig
from mlx_vlm.utils import get_model_and_args

from magnitude_engine.artifacts.identity import tokenizer_identity
from magnitude_engine.artifacts.layouts import LogicalTensor, logical_tensors
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.artifacts.tensors import TensorCatalog, read_json
from magnitude_engine.models.contracts import HeadBinding, ProgramSource
from magnitude_engine.models.embeddings.contracts import EmbeddingFactory
from magnitude_engine.models.loading.materialization import bind_operations, prepare_layout
from magnitude_engine.models.loading.packing import ProjectionPack
from magnitude_engine.models.loading.parameters import affine_encodings
from magnitude_engine.models.loading.partitions import EmbeddingPartition, ExpertPartition
from magnitude_engine.models.loading.validation import canonical_names
from magnitude_engine.models.ownership import OwnedProgram
from magnitude_engine.models.residency import (
    BoundProgram,
    HybridRequirements,
    ModelDescriptor,
    ModelResources,
)
from magnitude_engine.models.state.arena import LayerGeometry
from magnitude_engine.models.state.hybrid import HybridState
from magnitude_engine.models.state.recurrent import RecurrentLayout
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from .binding import bind_qwen35
from .contracts import (
    AttentionFactory,
    FeedForwardFactory,
    RecurrentFactory,
)
from .definition import DEFINITION

if TYPE_CHECKING:
    from .mtp.loading import LoadedMTP


@dataclass(frozen=True)
class LoadedQwen35:
    program: OwnedProgram[HybridState]
    arguments: TextConfig
    attention: tuple[LayerGeometry, ...]
    recurrence: tuple[RecurrentLayout, ...]
    state_dtype: mx.Dtype
    encoding: AffineEncoding
    tokenizer_identity: str
    vision_tensors: Mapping[str, LogicalTensor]
    configuration: ModelConfig | None

    def close(self) -> None:
        self.program.close()


def load_qwen35(
    directory: Path,
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    attention: AttentionFactory,
    recurrence: RecurrentFactory,
    feedforward: FeedForwardFactory,
    embedding_factory: EmbeddingFactory,
) -> LoadedQwen35:
    directory = directory.expanduser().resolve()
    config = read_json(directory / "config.json")
    model_type = config.get("text_config", config).get("model_type")
    if model_type not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError("artifact is not a supported Qwen3.5-family text model")
    architecture, _ = get_model_and_args({"model_type": model_type.removesuffix("_text")})
    full = (
        architecture.ModelConfig.from_dict(
            {**deepcopy(config), "model_type": model_type.removesuffix("_text")}
        )
        if config.get("vision_config")
        else None
    )
    args = (
        full.text_config
        if full is not None
        else architecture.TextConfig.from_dict(deepcopy(config.get("text_config", config)))
    )
    quantization = config.get("quantization", config.get("text_config", {}).get("quantization", {}))
    if not quantization or quantization.get("mode", "affine") != "affine":
        raise ValueError("this construction path requires converted MLX affine tensors")
    default = AffineEncoding(quantization["bits"], quantization["group_size"])
    tensors = canonical_names(
        logical_tensors(TensorCatalog.inspect(directory), declaration=None), "language_model."
    )
    vision = {name: tensor for name, tensor in tensors.items() if name.startswith("vision_tower.")}
    if vision and not config.get("vision_config"):
        raise ValueError("vision tensor component has no declared configuration")
    # This constructor owns text execution. Keep other declared components as
    # artifact records for separate construction instead of loading unused weights.
    tensors = {name: tensor for name, tensor in tensors.items() if name not in vision}
    identity = tokenizer_identity(directory)
    model = architecture.LanguageModel(args, full)
    model.eval()
    encodings = affine_encodings(tensors, quantization, prefix="language_model.")
    # Configure and validate the full header layout before assigning ownership.
    prepare_layout(model, tensors, encodings)
    embedding_partition = EmbeddingPartition(
        "model.embed_tokens",
        model.model.embed_tokens,
        {
            name: tensor
            for name, tensor in tensors.items()
            if name.startswith("model.embed_tokens.")
        },
        encodings.get("model.embed_tokens.weight"),
        args.tie_word_embeddings,
    )
    expert_partitions = {}
    for index, layer in enumerate(model.layers):
        mlp = getattr(layer.mlp, "switch_mlp", None)
        if mlp is None:
            continue
        name = f"model.layers.{index}.mlp.switch_mlp"
        expert_partitions[index] = ExpertPartition(
            name,
            mlp.up_proj,
            mlp.gate_proj,
            mlp.down_proj,
            {key: tensor for key, tensor in tensors.items() if key.startswith(name + ".")},
            encodings,
            # Both pinned upstream Qwen SwiGLU implementations compute
            # silu(gate) * up. Bind the qualified numerical primitive used by
            # resident and streamed expert kernels, independently of module identity.
            SwiGLU(),
        )
    packs = []
    for index, layer in enumerate(model.layers):
        if layer.is_linear:
            prefix = f"model.layers.{index}.linear_attn."
            names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        else:
            prefix = f"model.layers.{index}.self_attn."
            names = ("q_proj", "k_proj", "v_proj")
        packs.append(ProjectionPack(tuple(prefix + name for name in names)))
        if index in expert_partitions:
            prefix = f"model.layers.{index}.mlp."
            packs.append(ProjectionPack((prefix + "gate", prefix + "shared_expert_gate")))
    operations = bind_operations(
        model,
        tensors,
        budget=budget,
        reader=reader,
        packs=tuple(packs),
        encodings=encodings,
        embeddings={"tokens": (embedding_partition, embedding_factory)},
        experts={
            index: (partition, feedforward.experts)
            for index, partition in expert_partitions.items()
        },
    )
    try:
        embedding = operations.embeddings["tokens"]
        dtype = model.model.norm.weight.dtype
        experts = operations.experts
        binding = bind_qwen35(
            model,
            embedding=embedding,
            experts=experts,
            attention=attention,
            recurrence=recurrence,
            feedforward=feedforward,
            state_dtype=dtype,
            projections=operations.parameters.projections,
        )
        project = model.model.embed_tokens.as_linear if args.tie_word_embeddings else model.lm_head
        program = OwnedProgram(
            binding.program, (operations,), (identity, args.vocab_size, embedding, project)
        )
        return LoadedQwen35(
            program,
            args,
            binding.attention,
            binding.recurrence,
            dtype,
            default,
            identity,
            MappingProxyType(vision),
            full,
        )
    except BaseException:
        operations.close()
        raise


@dataclass(eq=False)
class Qwen35Source(ProgramSource):
    artifact: LocalArtifact
    attention: AttentionFactory
    recurrence: RecurrentFactory
    embedding: EmbeddingFactory
    feedforward: FeedForwardFactory
    reader: PositionalReader

    def head_binding(self, resources: ModelResources):
        return QwenHeadBinding(self.loaded(resources))

    def loaded(self, resources: ModelResources):
        return resources.once(
            self,
            lambda: resources.own(
                load_qwen35(
                    self.artifact.directory,
                    budget=resources.budget,
                    reader=self.reader,
                    attention=self.attention,
                    recurrence=self.recurrence,
                    feedforward=self.feedforward,
                    embedding_factory=self.embedding,
                )
            ),
        )

    def load(self, resources: ModelResources) -> BoundProgram:
        loaded = self.loaded(resources)
        inputs = None
        if loaded.configuration is not None:
            from .vision import QwenVision

            inputs = resources.once(
                loaded,
                lambda: resources.own(
                    QwenVision(
                        self.artifact,
                        loaded.configuration,
                        loaded.vision_tensors,
                        self.reader,
                        resources,
                    )
                ),
            )
        return BoundProgram(
            loaded.program,
            ModelDescriptor(
                self.artifact.path,
                loaded.arguments.max_position_embeddings,
                loaded.arguments.vocab_size,
                loaded.tokenizer_identity,
                "qwen35.Program",
                DEFINITION,
            ),
            HybridRequirements(loaded.attention, loaded.state_dtype, loaded.recurrence),
            inputs=inputs,
        )


@dataclass(frozen=True)
class QwenHeadBinding(HeadBinding):
    target: LoadedQwen35

    def load(
        self,
        artifact: LocalArtifact,
        reader: PositionalReader,
        resources: ModelResources,
    ) -> LoadedMTP:
        from .mtp.loading import load_mtp

        return resources.own(
            load_mtp(
                artifact.directory,
                self.target,
                budget=resources.budget,
                reader=reader,
            )
        )
