"""Generic MLX-VLM text-program binding; upstream selects the architecture."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.utils import get_model_and_args

from magnitude_engine.artifacts.identity import tokenizer_identity
from magnitude_engine.artifacts.layouts import logical_tensors
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.artifacts.tensors import TensorCatalog
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.models.loading.parameters import (
    affine_encodings,
    load_resident_parameters,
    resident_embedding,
)
from magnitude_engine.models.loading.validation import canonical_names
from magnitude_engine.models.ownership import OwnedProgram
from magnitude_engine.models.residency import (
    BoundProgram,
    ModelDescriptor,
    ModelResources,
    NativeRequirements,
)
from magnitude_engine.models.state.native import _SUPPORTED, _arrays, _detach
from magnitude_engine.resources.io.reader import PositionalReader

from .program import LibraryProgram


@dataclass(eq=False)
class UpstreamForward(ProgramSource):
    source: ProgramSource

    def load(self, resources: ModelResources) -> BoundProgram:
        return self.source.load(resources)

    def native_state_source(self) -> ProgramSource | None:
        return self.source.native_state_source()


@dataclass(eq=False)
class UpstreamLoader(ProgramSource):
    artifact: LocalArtifact
    reader: PositionalReader

    def native_state_source(self) -> ProgramSource:
        return self

    def load(self, resources: ModelResources) -> BoundProgram:
        return resources.once(self, lambda: self._load(resources))

    def _load(self, resources: ModelResources) -> BoundProgram:
        config = self.artifact.configuration()
        text = config.get("text_config", config)
        # Text-only converted checkpoints can retain a language-config type. Resolve
        # its upstream package first; family execution never enters worker dispatch.
        discovery = dict(config)
        discovery.setdefault("model_type", text.get("model_type"))
        try:
            architecture, _ = get_model_and_args(discovery)
        except ValueError:
            model_type = discovery.get("model_type", "")
            if not model_type.endswith("_text"):
                raise
            architecture, _ = get_model_and_args({**discovery, "model_type": model_type[:-5]})
        language = getattr(architecture, "LanguageModel", None) or getattr(
            getattr(architecture, "language", None), "LanguageModel", None
        )
        configuration = getattr(architecture, "TextConfig", architecture.ModelConfig)
        if language is None:
            raise ValueError("upstream architecture does not expose a standalone language model")
        arguments = configuration.from_dict(text)
        model = language(arguments)
        model.eval()
        caches = make_prompt_cache(model)
        if not caches or any(type(cache) not in _SUPPORTED for cache in caches):
            raise ValueError("upstream cache requires an explicit state adapter")
        if any(True for _ in _arrays(caches)):
            raise ValueError("upstream cache construction must not allocate populated state")

        # Capture empty cache descriptions, not a bound model method that would
        # retain neural weights after the program owner closes.
        def make_cache() -> list:
            return _detach(caches)

        capacity = native_capacity(arguments, caches)
        settings = config.get("quantization", text.get("quantization", {}))
        if settings.get("mode", "affine") != "affine":
            raise ValueError("upstream materialization currently supports affine or float tensors")
        all_tensors = canonical_names(
            logical_tensors(TensorCatalog.inspect(self.artifact.directory), declaration=None),
            "language_model.",
        )
        # The source is explicitly a text program. Peer modality weights stay in
        # the artifact and are never advertised as executable conditioning inputs.
        tensors = {
            name: tensor
            for name, tensor in all_tensors.items()
            if not name.startswith(
                ("vision_tower.", "embed_vision.", "audio_tower.", "embed_audio.")
            )
        }
        allocation = load_resident_parameters(
            model,
            tensors,
            affine_encodings(tensors, settings, prefix="language_model."),
            budget=resources.budget,
            reader=self.reader,
            owner="upstream.weights",
        )
        try:

            def call(tokens: mx.array, cache: list) -> mx.array:
                offset = next((c.offset for c in cache if hasattr(c, "offset")), 0)
                if isinstance(offset, mx.array) and offset.ndim == 1:
                    offset = offset[:, None]
                positions = mx.arange(tokens.shape[1], dtype=mx.int32)[None, :] + offset
                output = model(tokens, cache=cache, position_ids=positions)
                return output if isinstance(output, mx.array) else output.logits

            identity = tokenizer_identity(self.artifact.directory)
            # Vocabulary sharing is advertised only when this binding can provide
            # the actual lookup/projection pair. Plain execution doesn't require it.
            vocabulary = None
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "embed_tokens"):
                embedding, _ = resident_embedding(inner.embed_tokens)
                project = (
                    inner.embed_tokens.as_linear if arguments.tie_word_embeddings else model.lm_head
                )
                vocabulary = (identity, arguments.vocab_size, embedding, project)
            owned = OwnedProgram(LibraryProgram(call), (allocation,), vocabulary)
            resources.own(owned)
            return BoundProgram(
                owned,
                ModelDescriptor(
                    self.artifact.path,
                    arguments.max_position_embeddings,
                    arguments.vocab_size,
                    identity,
                    f"{language.__module__}.{language.__qualname__}",
                ),
                NativeRequirements(make_cache, capacity, self),
            )
        except BaseException:
            allocation.close()
            raise


def native_capacity(arguments, caches):
    """Conservative reservations for recognized append, rotating and recurrent caches."""
    from mlx_vlm.models.cache import ArraysCache

    heads = max(
        getattr(arguments, "num_key_value_heads", 0),
        getattr(arguments, "num_global_key_value_heads", 0) or 0,
    )
    width = max(
        getattr(arguments, "head_dim", 0) or 0,
        getattr(arguments, "global_head_dim", 0) or 0,
        arguments.hidden_size // arguments.num_attention_heads,
    )
    if heads < 1 or width < 1:
        raise ValueError("upstream cache needs declared KV geometry")
    recurrent = sum(isinstance(cache, ArraysCache) for cache in caches)
    fixed = 0
    if recurrent:
        required = (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
        if any(not hasattr(arguments, name) for name in required):
            raise ValueError("recurrent cache requires a qualified geometry adapter")
        conv = (
            2 * arguments.linear_num_key_heads * arguments.linear_key_head_dim
            + arguments.linear_num_value_heads * arguments.linear_value_head_dim
        )
        fixed = (
            recurrent
            * 4
            * (
                (arguments.linear_conv_kernel_dim - 1) * conv
                + arguments.linear_num_value_heads
                * arguments.linear_key_head_dim
                * arguments.linear_value_head_dim
            )
        )
    kv_bytes = (len(caches) - recurrent) * heads * width * 2 * 4
    return lambda position: fixed + (((position + 255) // 256 + 1) * 256 * kv_bytes)
