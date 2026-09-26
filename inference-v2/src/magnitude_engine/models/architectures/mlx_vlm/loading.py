"""MLX-VLM composition with explicit bindings for conditional model semantics."""

from copy import deepcopy
from dataclasses import dataclass

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

from .definition import DEFINITION
from .program import LibraryForward, LibraryProgram


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
        if config.get("vision_config") and text.get("model_type", "").removesuffix("_text") not in (
            "qwen3_5",
            "qwen3_5_moe",
            "gemma4",
        ):
            raise ValueError(
                "this full vision architecture requires a qualified model input adapter"
            )
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
        # Materialize the decoder partition once. The qualified conditional binding
        # below owns its encoder/projector partition under the same resource scope.
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
            inputs = input_forward = None
            if config.get("vision_config") and text.get("model_type", "").removesuffix("_text") in (
                "qwen3_5",
                "qwen3_5_moe",
            ):
                from ..qwen35.library import QwenForward
                from ..qwen35.vision import QwenVision

                full = architecture.ModelConfig.from_dict(deepcopy(config))
                inputs = resources.own(
                    QwenVision(
                        self.artifact,
                        full,
                        {
                            name: tensor
                            for name, tensor in all_tensors.items()
                            if name.startswith("vision_tower.")
                        },
                        self.reader,
                        resources,
                    )
                )
                input_forward = QwenForward(model).forward_inputs
            elif config.get("vision_config") and text.get("model_type") == "gemma4_text":
                from ..gemma4.library import GemmaForward
                from ..gemma4.vision import GemmaVision
                from ..gemma4.vision import configuration as gemma_configuration

                inputs = resources.own(
                    GemmaVision(
                        self.artifact,
                        gemma_configuration(config),
                        all_tensors,
                        self.reader,
                        resources,
                    )
                )
                input_forward = GemmaForward(model).forward_inputs
            owned = OwnedProgram(
                LibraryProgram(LibraryForward(model), input_forward=input_forward),
                (allocation,),
                vocabulary,
            )
            resources.own(owned)
            return BoundProgram(
                owned,
                ModelDescriptor(
                    self.artifact.path,
                    arguments.max_position_embeddings,
                    arguments.vocab_size,
                    identity,
                    f"{language.__module__}.{language.__qualname__}",
                    DEFINITION,
                ),
                NativeRequirements(make_cache, capacity, self),
                inputs=inputs,
            )
        except BaseException:
            allocation.close()
            raise


def native_capacity(arguments, caches):
    """Bound retained history and the additional window required by a forward.

    Geometry and element size remain conservative. Rotating caches retain a
    bounded history, but a multi-token query needs window + query - 1 keys so
    every query can attend its complete window. Zero query tokens describes
    retained history for admission; transaction preparation supplies the actual
    query width before allocating its extension.
    """
    from mlx_vlm.models.cache import ArraysCache, RotatingKVCache

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
    kv_bytes = heads * width * 2 * 4
    layouts = tuple(
        (
            getattr(cache, "step", 256),
            cache.max_size if isinstance(cache, RotatingKVCache) else None,
        )
        for cache in caches
        if not isinstance(cache, ArraysCache)
    )

    def capacity(position: int, query_tokens: int) -> int:
        if not 0 <= query_tokens <= position:
            raise ValueError("cache query width must be within the declared end position")
        tokens = 0
        for step, window in layouts:
            allocated = ((position + step - 1) // step + 1) * step
            if window is not None:
                allocated = max(
                    min(allocated, window),
                    min(position, window + max(0, query_tokens - 1)),
                )
            tokens += allocated
        return fixed + tokens * kv_bytes

    return capacity
