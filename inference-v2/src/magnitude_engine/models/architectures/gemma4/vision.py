"""Gemma image preparation, atomic local-attention blocks, and feature ownership."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
from mlx_vlm.models.gemma4.vision import VisionModel

from magnitude_engine.artifacts.blueprint import Local
from magnitude_engine.artifacts.identity import processor_identity
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.components import component, component_id
from magnitude_engine.models.context import InputCheckpoint, InputContinuation, SpanContext
from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement
from magnitude_engine.models.features import FeatureLease, FeatureSet
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.loading.parameters import affine_encodings, load_resident_parameters
from magnitude_engine.models.preparation import PreparedMedia
from magnitude_engine.models.prompt import InputSpan, Prompt
from magnitude_engine.models.residency import ModelResources

from .blueprint import Images
from .inputs import GemmaInputs
from .preparation import GemmaImages


def configuration(values):
    return ModelConfig.from_dict(
        {
            **values,
            "text_config": TextConfig.from_dict(values["text_config"]),
            "vision_config": VisionConfig.from_dict(values["vision_config"]),
            "audio_config": None,
        }
    )


class VisionModules(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = MultimodalEmbedder(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.vision_config.rms_norm_eps,
        )


@dataclass(frozen=True)
class ImageInput:
    span: InputSpan
    pixels: np.ndarray
    positions: np.ndarray


@component("MODEL:GEMMA4.VISION:VLM:STANDARD")
class GemmaVision:
    def __init__(
        self, artifact: LocalArtifact, configuration, tensors, reader, resources: ModelResources
    ):
        self.owner, self.budget = resources.owner, resources.budget
        self.cache = resources.input_features
        self.config = configuration
        self.processor = Images(artifact=Local(path=artifact.path))
        self.identity = processor_identity(artifact.directory, component_id(GemmaImages))
        self.model = VisionModules(configuration)
        self.model.eval()
        vision = {
            name: t
            for name, t in tensors.items()
            if name.startswith(("vision_tower.", "embed_vision."))
        }
        self.parameters = load_resident_parameters(
            self.model,
            vision,
            affine_encodings(vision, artifact.configuration().get("quantization", {}), prefix=""),
            budget=self.budget,
            reader=reader,
            owner="gemma4.vision.weights",
        )
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.model = None
            self.parameters.close()

    def prepare(self, tokens: tuple[int, ...], media: PreparedMedia) -> tuple[Prompt, GemmaSource]:
        if self.closed or media.processor != self.identity:
            raise ValueError("Gemma media processor differs from the bound artifact")
        tensors = {t.name: t.array() for t in media.tensors}
        if set(tensors) != {"pixel_values", "image_position_ids", "num_soft_tokens_per_image"}:
            raise ValueError("Gemma image tensors differ from the model input contract")
        pixels, positions, counts = (
            tensors[name]
            for name in ("pixel_values", "image_position_ids", "num_soft_tokens_per_image")
        )
        cfg = self.config.vision_config
        if (
            pixels.dtype != np.float32
            or pixels.ndim != 3
            or pixels.shape[2] != cfg.patch_size**2 * 3
            or not 1 <= pixels.shape[0] <= 16
            or not np.isfinite(pixels).all()
            or positions.dtype != np.int64
            or positions.shape != (*pixels.shape[:2], 2)
            or counts.dtype != np.int64
            or counts.shape != (len(pixels),)
        ):
            raise ValueError("invalid Gemma image pixel or position geometry")
        images, cursor = [], 0
        atomic = self.config.text_config.use_bidirectional_attention == "vision"
        for values, coords, count in zip(pixels, positions, counts.tolist(), strict=True):
            real = np.all(coords >= 0, axis=1)
            valid = coords[real]
            pool = cfg.pooling_kernel_size
            if (
                not len(valid)
                or count < 1
                or count > cfg.default_output_length
                or not np.all(coords[~real] == -1)
                or not np.all(real[: len(valid)])
                or np.any(real[len(valid) :])
                or np.any(valid >= cfg.position_embedding_size)
            ):
                raise ValueError("invalid Gemma image patch positions")
            width, height = valid.max(axis=0) + 1
            expected = np.stack(
                np.meshgrid(np.arange(width), np.arange(height), indexing="xy"), -1
            ).reshape(-1, 2)
            if (
                height % pool
                or width % pool
                or not np.array_equal(valid, expected)
                or count != len(valid) // pool**2
            ):
                raise ValueError(
                    "Gemma patch positions do not describe complete pooled image features"
                )
            try:
                start = tokens.index(self.config.image_token_id, cursor)
            except ValueError as error:
                raise ValueError("Gemma image has no matching prompt span") from error
            end = start + count
            if (
                start == 0
                or tokens[start - 1] != self.config.boi_token_id
                or tokens[start:end] != (self.config.image_token_id,) * count
                or end >= len(tokens)
                or tokens[end] != self.config.eoi_token_id
            ):
                raise ValueError("Gemma image placeholder span differs from its feature geometry")
            identity = hashlib.sha256(
                self.identity.encode() + values.tobytes() + coords.tobytes()
            ).digest()
            images.append(
                ImageInput(InputSpan(start, end, identity, indivisible=atomic), values, coords)
            )
            cursor = end
        if self.config.image_token_id in tokens[cursor:]:
            raise ValueError("Gemma prompt contains an image span without source pixels")
        prompt = Prompt(tokens, tuple(image.span for image in images))
        return prompt, GemmaSource(self, prompt, tuple(images))


@dataclass
class EncodeImage:
    encoder: GemmaVision
    image: ImageInput
    output: FeatureLease

    @property
    def owner(self):
        return self.encoder.owner

    @property
    def batch_key(self):
        return id(self.encoder), self.image.pixels.shape

    def reserve(self, rows):
        cfg = self.encoder.config.vision_config
        patches = self.image.pixels.shape[0]
        scratch = (
            len(rows)
            * 4
            * (
                patches * (12 * cfg.hidden_size + 2 * cfg.intermediate_size)
                + cfg.num_attention_heads * patches**2
            )
        )
        return self.encoder.budget.reserve("gemma4.vision.scratch", scratch)

    def run_batch(self, rows, scope):
        if self.encoder.closed:
            raise RuntimeError("Gemma encoder is closed")
        for row in rows:
            scope.acquire(row.output.fork)
        pixels = mx.array(np.stack([row.image.pixels for row in rows]))
        positions = mx.array(np.stack([row.image.positions for row in rows]), mx.int32)
        model = self.encoder.model
        assert model is not None
        features = model.embed_vision(model.vision_tower(pixels, positions))
        outputs, start = [], 0
        for row in rows:
            count = row.image.span.end - row.image.span.start
            value = mx.array(features[:, start : start + count])
            if value.nbytes > row.output.feature.reservation.size:
                raise MemoryError("Gemma feature exceeded its declared allocation")
            row.output.feature.value = value
            outputs.append((value,))
            start += count
        if start != features.shape[1]:
            raise ValueError("Gemma encoder output does not align with image spans")
        return tuple(outputs)


@dataclass(frozen=True)
class GemmaSource:
    encoder: GemmaVision
    prompt: Prompt
    images: tuple[ImageInput, ...]

    def identities(self, position: int) -> tuple[bytes, ...]:
        return tuple(image.span.identity for image in self.images if image.span.start < position)

    def bind(self, checkpoint: InputCheckpoint | None) -> GemmaContext:
        if checkpoint is not None and (
            not isinstance(checkpoint, GemmaCheckpoint)
            or checkpoint.closed
            or not self.prompt.boundary(checkpoint.position)
            or checkpoint.identities != self.identities(checkpoint.position)
        ):
            raise ValueError("Gemma input continuation differs from the prepared prompt")
        return GemmaContext(self)


class GemmaContinuation(InputContinuation):
    def __init__(self, identities=()):
        self.identities = identities

    def assemble(self, inputs: ModelInputs, position: int) -> ModelInputs:
        return inputs

    def checkpoint(self, position: int) -> GemmaCheckpoint:
        return GemmaCheckpoint(position, self.identities)


class GemmaContext(SpanContext[ImageInput], GemmaContinuation):
    def __init__(self, source: GemmaSource):
        GemmaContinuation.__init__(self, source.identities(len(source.prompt.tokens)))
        self.source = source
        SpanContext.__init__(
            self,
            source.prompt,
            source.images,
            FeatureSet(source.encoder, source.encoder.budget, source.encoder.cache),
            source.encoder.config.text_config.hidden_size * 4,
            partial(EncodeImage, source.encoder),
        )

    def assemble(self, inputs: ModelInputs, position: int) -> ModelInputs:
        end = position + inputs.count
        embeddings = []
        language = np.ones((1, inputs.count), dtype=bool)
        key_ends = None
        for image in self.source.images:
            start, stop = max(position, image.span.start), min(end, image.span.end)
            if start >= stop:
                continue
            if image.span.indivisible and (start != image.span.start or stop != image.span.end):
                raise ValueError("Gemma bidirectional image blocks require a complete forward")
            value = self.features.value(image.span.identity)
            embeddings.append(
                EmbeddingReplacement(
                    start - position, value[:, start - image.span.start : stop - image.span.start]
                )
            )
            language[:, start - position : stop - position] = False
            if image.span.indivisible:
                if key_ends is None:
                    key_ends = np.arange(position + 1, end + 1, dtype=np.int32)[None]
                key_ends[:, start - position : stop - position] = image.span.end
        if not embeddings:
            return inputs
        return ModelInputs(
            inputs.tokens,
            inputs.conditioning,
            GemmaInputs(
                tuple(embeddings),
                mx.array(language),
                None if key_ends is None else mx.array(key_ends),
            ),
        )

    def checkpoint(self, position: int) -> GemmaCheckpoint:
        return GemmaCheckpoint(position, self.source.identities(position))


class GemmaCheckpoint:
    reclaimable = True

    def __init__(self, position, identities):
        self.position, self.identities, self.closed = position, identities, False

    def retained_storage(self):
        return ()

    def restore(self):
        if self.closed:
            raise ValueError("Gemma input checkpoint is closed")
        return GemmaContinuation(self.identities)

    def close(self):
        self.closed = True
